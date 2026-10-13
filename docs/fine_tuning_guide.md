# Laya System One Local Fine-Tuning & Distillation Guide

This guide documents the end-to-end active learning and knowledge distillation pipeline for **Laya**, an on-device, sub-40ms System One decision engine. It details how to extract boundary edge cases from live social data, distill teacher labels from frontier LLMs, train on consumer GPU hardware ($\le 8\text{ GB}$ VRAM), and verify multi-task performance before and after fine-tuning.

---

## 1. Architecture Overview

```text
[ Bluesky JetStream Firehose ] ──► (50-100 posts/sec)
               │
               ▼
       [ NATS JetStream ] (stream.raw)
               │
               ▼
     [ inference.py Worker ]
               │
      (HTTP /v1/systemone/batch)
               │
         ┌─────▼───────────────────────────────┐
         │     Laya System One Decision Engine │
         │   • Model: ModernBERT Backbone      │
         │   • Latency: ~37ms / post on CUDA   │
         │   • Heads: Category, Sentiment,     │
         │            Virality, Breaking Claim,│
         │            Geo-Region, NSFW         │
         └─────┬───────────────────────────────┘
               │ (Confidence Threshold Gate)
       ┌───────┴────────────────────────┐
       ▼ (p >= 0.85)                    ▼ (0.40 <= p <= 0.60 or 'other')
  [ Stream Enriched ]             [ Stream Escalation ]
       │                                │
       ▼                                ▼
 [ ClickHouse DB ] ◄────────────── [ ClickHouse DB ]
 (Anchor Cohort)                  (Boundary Cohort)
       │                                │
       └────────────────┬───────────────┘
                        │
                        ▼ [scripts/extract_dataset.py]
          data/distillation_candidates.jsonl (1,000 items)
                        │
                        ▼ [scripts/label_dataset.py]
                 Gemini 3.7 Flash Teacher (Structured Outputs)
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    data/train.jsonl        data/test.jsonl
       (800 rows)              (200 rows)
            │                       │
            ▼ [scripts/train_laya.py]
    checkpoints/laya_softce/ (Safetensors + Calibration Maps)
            │
            ▼
    [scripts/benchmark_laya.py] & [scripts/compare_metrics.py]
            │
            ▼
    data/scorecard_delta.md (Publication Scorecard)
```

---

## 2. Prerequisites & Environment

* **GPU Hardware**: NVIDIA GeForce RTX 2080 Ti (8 GB VRAM) or equivalent.
* **Storage & Isolation**: All datasets (`data/`) and model checkpoints (`checkpoints/`) are quarantined in `.gitignore`.
* **Dependencies**: Python virtual environment (`.venv`) with PyTorch (CUDA enabled), `requests`, `aiohttp`, `numpy`, and `google-genai`.

---

## 3. Step-by-Step Operations Manual

### Step 1: Slice Active Learning Cohorts from ClickHouse
Instead of random sampling, the dataset uses a **50/50 Anchor vs. Boundary** split:
* **Anchors (500 samples)**: High-confidence ($p \ge 0.85$) baseline decisions to prevent catastrophic forgetting.
* **Boundary Edges (500 samples)**: Ambiguous decisions ($0.40 \le p \le 0.60$) or generic fallback classifications (`category = 'other'`).

```bash
python scripts/extract_dataset.py \
  --anchors 500 \
  --boundary 500 \
  --output data/distillation_candidates.jsonl
```

### Step 2: Distill Ground Truth from Frontier Teacher (Gemini 3.7 Flash)
Labels are distilled via `gemini-3.7-flash` using strict Pydantic schemas and formatted into Laya's native multi-task sequence format:
```bash
python scripts/label_dataset.py \
  --input data/distillation_candidates.jsonl \
  --output-train data/train.jsonl \
  --output-test data/test.jsonl
```
* Generates 800 training rows and 200 evaluation rows with 0 schema errors.
* Preserves the 50/50 Anchor/Boundary cohort ratio in both train and test splits.

### Step 3: Run Baseline Pre-Fine-Tuning Benchmark
Run the benchmark against the un-fine-tuned base checkpoint to record the baseline metrics:
```bash
python scripts/benchmark_laya.py \
  --checkpoint baseline \
  --test-data data/test.jsonl \
  --output data/metrics_baseline.json
```

### Step 4: Execute Local Fine-Tuning
Fine-tuning on consumer 8 GB GPUs requires `--freeze-encoder` to keep memory consumption under 2 GB.
* **Objective Loss**: `--loss soft-ce` trains against teacher probability distributions directly.
* **Effective Batch**: Micro-batch 4 with 16 gradient accumulation steps (effective batch size: 64).
* **Option Shuffling**: `--shuffle-options` randomizes option order during training to eliminate positional bias.

```bash
python scripts/train_laya.py \
  --train-data data/train.jsonl \
  --eval-data data/test.jsonl \
  --output-dir checkpoints/laya_softce \
  --epochs 6 \
  --head-lr 0.00025 \
  --loss soft-ce
```
* The training script automatically pauses background Docker serving containers to avoid VRAM contention, executes fine-tuning in ~4.9 minutes, and restarts the service.

### Step 5: Hot Deploy Fine-Tuned Weights
1. Ensure world-readable permissions for the container:
   ```bash
   chmod -R a+rX checkpoints/laya_softce
   ```
2. Update `.env` to point to the new checkpoint path:
   ```ini
   LAYA_MODELS=/checkpoints/laya_softce
   ```
3. Restart the serving container:
   ```bash
   docker compose restart laya-serve
   ```
4. Verify health and model load:
   ```bash
   curl -s -H "Authorization: Bearer my-secret-laya-api-key" http://localhost:8111/health
   ```

### Step 6: Evaluate Fine-Tuned Model & Generate Scorecard
1. Run the benchmark harness against the fine-tuned server:
   ```bash
   python scripts/benchmark_laya.py \
     --checkpoint softce_finetuned \
     --test-data data/test.jsonl \
     --output data/metrics_finetuned_softce.json
   ```
2. Compute statistical deltas and generate the publication Markdown table:
   ```bash
   python scripts/compare_metrics.py \
     --baseline data/metrics_baseline.json \
     --finetuned data/metrics_finetuned_softce.json \
     --output data/scorecard_delta.md
   ```

---

## 4. Script Catalog

| Script | Purpose | Key Flags |
|---|---|---|
| [`scripts/extract_dataset.py`](file:///home/vinayb/CodeProjects/laya-fine-tune/scripts/extract_dataset.py) | Slices anchor & boundary cohorts from ClickHouse | `--anchors`, `--boundary`, `--output` |
| [`scripts/label_dataset.py`](file:///home/vinayb/CodeProjects/laya-fine-tune/scripts/label_dataset.py) | Distills teacher labels via Gemini 3.7 Flash | `--input`, `--output-train`, `--output-test` |
| [`scripts/benchmark_laya.py`](file:///home/vinayb/CodeProjects/laya-fine-tune/scripts/benchmark_laya.py) | Evaluates accuracy, F1, calibration, and latency | `--checkpoint`, `--test-data`, `--output` |
| [`scripts/train_laya.py`](file:///home/vinayb/CodeProjects/laya-fine-tune/scripts/train_laya.py) | Orchestrates local fine-tuning on CUDA | `--epochs`, `--loss`, `--head-lr`, `--output-dir` |
| [`scripts/compare_metrics.py`](file:///home/vinayb/CodeProjects/laya-fine-tune/scripts/compare_metrics.py) | Computes before/after deltas and generates scorecards | `--baseline`, `--finetuned`, `--output` |
| [`scripts/nats_stats.py`](file:///home/vinayb/CodeProjects/laya-fine-tune/scripts/nats_stats.py) | Inspects stream backlog and worker consumer lag | (None) |
