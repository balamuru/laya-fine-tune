# Fine-Tuning & Operational Scripts Suite

This directory contains the self-contained, reproducible scripts for the active learning distillation pipeline, model benchmarking, and stream operations.

All scripts are designed to be run from the repository root: `python scripts/<script_name>.py [OPTIONS]`.

---

## Script Manifest & Pipeline Workflow

```text
ClickHouse DB (95k+ posts)
       │
       ▼ [1. scripts/extract_dataset.py]
data/distillation_candidates.jsonl (1,000 posts: 500 Anchors + 500 Boundary Edges)
       │
       ▼ [2. scripts/label_dataset.py] (Gemini 3.7 Flash Teacher)
data/train.jsonl (800 posts)  &  data/test.jsonl (200 posts: 100 Anchors + 100 Edges)
       │
       ├──► [3. scripts/benchmark_laya.py] (Baseline Checkpoint) ──► data/metrics_baseline.json
       │
       ▼ [4. scripts/train_laya.py] (Local GPU Fine-Tuning: RLCD + Shuffling + Calibration)
checkpoints/laya_finetuned.pt
       │
       ├──► [5. scripts/benchmark_laya.py] (Fine-Tuned Checkpoint) ──► data/metrics_finetuned.json
       │
       ▼ [6. scripts/compare_metrics.py]
Publication Scorecard Table (Markdown, LaTeX & JSON Deltas for Technical Article)
```

---

## Detailed Script Descriptions

### 1. `scripts/extract_dataset.py`
* **Purpose**: Extracts balanced sample cohorts from ClickHouse `bluesky.enriched_posts`.
  * **Anchor Cohort (500 samples)**: High-confidence baseline predictions across all heads ($p \ge 0.85$). Used to guarantee zero catastrophic forgetting.
  * **Boundary Cohort (500 samples)**: Low-confidence ambiguity ($0.40 \le p \le 0.60$) and unclassified generic posts (`category = 'other'`).
* **Input**: ClickHouse server (`http://localhost:8123`)
* **Output**: `data/distillation_candidates.jsonl`
* **Usage**:
  ```bash
  python scripts/extract_dataset.py --anchors 500 --boundary 500 --output data/distillation_candidates.jsonl
  ```

---

### 2. `scripts/label_dataset.py`
* **Purpose**: Distills labels from frontier model `gemini-3.7-flash` via the official `google-genai` SDK using a structured output schema matching Laya's multi-task questions. Automatically splits data 80/20 into training and testing sets, preserving the 50/50 Anchor vs. Boundary ratio in both splits.
* **Input**: `data/distillation_candidates.jsonl` and `GEMINI_API_KEY`
* **Output**: `data/train.jsonl` (800 rows) and `data/test.jsonl` (200 rows)
* **Usage**:
  ```bash
  python scripts/label_dataset.py --input data/distillation_candidates.jsonl --train-out data/train.jsonl --test-out data/test.jsonl
  ```

---

### 3. `scripts/benchmark_laya.py`
* **Purpose**: Standalone, reproducible evaluation harness measuring:
  1. **Classification Accuracy & Macro F1** across all 6 heads.
  2. **Cohort Sliced Accuracy**: Anchor Slice Retention vs. Boundary Slice Disambiguation Gain.
  3. **Uncertainty Calibration**: Expected Calibration Error (ECE) and Brier Score.
  4. **Latency Distribution**: $p50$, $p90$, $p95$, $p99$ across micro-batch sizes (1, 16, 32, 64) on CUDA.
  5. **Peak VRAM Allocation**: Measured via `torch.cuda.max_memory_allocated()`.
* **Input**: Target checkpoint (or live local endpoint) and `data/test.jsonl`
* **Output**: Structured JSON metrics file (e.g., `data/metrics_baseline.json` or `data/metrics_finetuned.json`)
* **Usage**:
  ```bash
  # Benchmark baseline
  python scripts/benchmark_laya.py --checkpoint baseline --test-data data/test.jsonl --output data/metrics_baseline.json

  # Benchmark fine-tuned model
  python scripts/benchmark_laya.py --checkpoint checkpoints/laya_finetuned.pt --test-data data/test.jsonl --output data/metrics_finetuned.json
  ```

---

### 4. `scripts/train_laya.py`
* **Purpose**: Orchestrates local GPU fine-tuning using `laya-train` with support for both Soft Cross-Entropy (`soft-ce`) and RLCD loss, `--shuffle-options` to eliminate positional option bias, `--freeze-encoder` for memory-constrained GPUs ($\le 8\text{ GB}$ VRAM), and post-hoc temperature scaling.
* **Auto-VRAM Management**: Automatically detects running `laya-serve` containers and pauses them during the training run to prevent OOM conflicts with display servers, restoring the container upon completion.
* **Input**: `data/train.jsonl` (and optional `data/test.jsonl` for evaluation)
* **Output**: `checkpoints/<dir>/` (`model.safetensors`, `rl_agent_config.json`, tokenizer, and `train_report.json`)
* **Usage**:
  ```bash
  # Recommended high-accuracy fine-tuning on 8GB GPU (~4.9 mins)
  python scripts/train_laya.py \
    --train-data data/train.jsonl \
    --eval-data data/test.jsonl \
    --output-dir checkpoints/laya_softce \
    --epochs 6 \
    --head-lr 0.00025 \
    --loss soft-ce \
    --freeze-encoder
  ```

---

### 5. `scripts/compare_metrics.py`
* **Purpose**: Ingests baseline and post-fine-tuning metrics JSON files, computes exact deltas ($\Delta$), tests for statistical significance, and generates formatted Markdown and LaTeX tables ready for direct publication in the Medium article.
* **Input**: `data/metrics_baseline.json` and `data/metrics_finetuned.json`
* **Output**: Formatted summary to stdout and `data/scorecard_delta.md`
* **Usage**:
  ```bash
  python scripts/compare_metrics.py --baseline data/metrics_baseline.json --finetuned data/metrics_finetuned.json
  ```

---

### 6. `scripts/nats_stats.py`
* **Purpose**: Real-time diagnostic tool for monitoring NATS JetStream consumer queues, pending backlog, and worker delivery latencies across `stream.raw`, `stream.enriched`, and `stream.escalation`.
* **Usage**:
  ```bash
  python scripts/nats_stats.py
  ```
