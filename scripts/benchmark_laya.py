#!/usr/bin/env python3
"""
scripts/benchmark_laya.py
Standalone, reproducible evaluation harness measuring:
- Multi-task accuracy and Macro F1 against Gemini teacher labels
- Sliced Accuracy (Anchor Retention vs. Boundary Disambiguation)
- Expected Calibration Error (ECE) and Brier Score
- Inference Latency Distributions (p50, p90, p95, p99 across batch sizes 1, 16, 32, 64)
- Peak VRAM memory allocation

Inputs:  data/test.jsonl
Outputs: data/metrics_baseline.json (or data/metrics_finetuned.json)
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Criteria configuration matching Laya inference schema
LAYA_QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "What is the primary category of this post?",
        "criteria": {
            "tech": "technology, AI, programming, software",
            "politics": "government, elections, policy",
            "entertainment": "movies, music, pop culture",
            "finance": "markets, crypto, money",
            "lifestyle": "health, daily life, hobbies",
            "other": "everything else"
        }
    },
    "sentiment": {
        "type": "choice",
        "instructions": "What is the sentiment of this post?",
        "criteria": {
            "positive": "happy, excited, supportive",
            "negative": "angry, sad, critical",
            "neutral": "factual, objective, unopinionated"
        }
    },
    "virality": {
        "type": "score",
        "instructions": "How incendiary or viral is this post likely to be?",
        "criteria": ["mundane", "conversational", "incendiary and highly viral"]
    },
    "is_breaking_claim": {
        "type": "noul",
        "instructions": "Does this post make a verifiable factual claim or report breaking news?"
    },
    "geo_region": {
        "type": "choice",
        "instructions": "Which geographic region or continent does this post primarily reference or discuss?",
        "criteria": {
            "north_america": "United States, Canada, Mexico, North America",
            "europe": "UK, Germany, France, Ukraine, EU, Europe",
            "asia": "Japan, China, India, Korea, Taiwan, Asia",
            "middle_east": "Israel, Palestine, Gaza, Iran, Middle East",
            "latin_america": "Brazil, Argentina, Latin America, South America",
            "africa": "Nigeria, Kenya, South Africa, Africa",
            "oceania": "Australia, New Zealand",
            "global_none": "general, personal, online, or no specific geographic focus"
        }
    },
    "is_nsfw": {
        "type": "noul",
        "instructions": "Does this post contain sexually explicit, obscene, or adult NSFW material?"
    }
}

SYSTEM_ONE_API_KEY = os.getenv("SYSTEM_ONE_API_KEY", "")


def query_laya_batch(endpoint: str, texts: List[str]) -> List[dict]:
    states = [{"body": t} for t in texts]
    payload = {
        "states": states,
        "questions": LAYA_QUESTIONS
    }
    headers = {}
    if SYSTEM_ONE_API_KEY:
        headers["Authorization"] = f"Bearer {SYSTEM_ONE_API_KEY}"
    url = f"{endpoint.rstrip('/')}/v1/systemone/batch"
    res = requests.post(url, json=payload, headers=headers, timeout=30)
    res.raise_for_status()
    return res.json().get("results", [])


def compute_ece(confidences: List[float], accuracies: List[int], n_bins: int = 10) -> float:
    """Computes Expected Calibration Error across equal-width confidence bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = [
            (c, a) for c, a in zip(confidences, accuracies)
            if (bin_lower <= c < bin_upper) or (i == n_bins - 1 and bin_lower <= c <= bin_upper)
        ]

        if not in_bin:
            continue

        bin_size = len(in_bin)
        bin_acc = sum(a for _, a in in_bin) / bin_size
        bin_conf = sum(c for c, _ in in_bin) / bin_size
        ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_macro_f1(actuals: List[str], predictions: List[str]) -> float:
    """Computes unweighted macro F1 across unique classes."""
    classes = sorted(list(set(actuals) | set(predictions)))
    f1s = []
    for cls in classes:
        tp = sum(1 for a, p in zip(actuals, predictions) if a == cls and p == cls)
        fp = sum(1 for a, p in zip(actuals, predictions) if a != cls and p == cls)
        fn = sum(1 for a, p in zip(actuals, predictions) if a == cls and p != cls)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def compute_cohens_kappa(actuals: List[str], predictions: List[str]) -> float:
    """Computes Cohen's Kappa inter-rater agreement."""
    classes = sorted(list(set(actuals) | set(predictions)))
    n = len(actuals)
    if n == 0:
        return 0.0

    po = sum(1 for a, p in zip(actuals, predictions) if a == p) / n

    pe = 0.0
    for cls in classes:
        p_act = sum(1 for a in actuals if a == cls) / n
        p_pred = sum(1 for p in predictions if p == cls) / n
        pe += p_act * p_pred

    if (1 - pe) == 0:
        return 1.0
    return float((po - pe) / (1 - pe))


def benchmark_latency(endpoint: str, sample_text: str, batch_sizes=(1, 16, 32, 64), runs_per_batch=15) -> Dict:
    results = {}
    print("\nMeasuring micro-batch latency distributions...")

    # Warm-up call
    try:
        query_laya_batch(endpoint, [sample_text] * 4)
    except Exception as e:
        print(f"Warning: warm-up call failed ({e})")

    for bsz in batch_sizes:
        batch_texts = [sample_text] * bsz
        latencies = []
        for _ in range(runs_per_batch):
            t0 = time.perf_counter()
            query_laya_batch(endpoint, batch_texts)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms

        lat_arr = np.array(latencies)
        results[f"batch_{bsz}"] = {
            "p50": float(np.percentile(lat_arr, 50)),
            "p90": float(np.percentile(lat_arr, 90)),
            "p95": float(np.percentile(lat_arr, 95)),
            "p99": float(np.percentile(lat_arr, 99)),
            "mean": float(np.mean(lat_arr)),
            "std": float(np.std(lat_arr)),
            "per_item_p50_ms": float(np.percentile(lat_arr, 50) / bsz)
        }
        print(f"  • Batch {bsz:2d}: p50 = {results[f'batch_{bsz}']['p50']:5.2f} ms | p95 = {results[f'batch_{bsz}']['p95']:5.2f} ms | per-post = {results[f'batch_{bsz}']['per_item_p50_ms']:5.3f} ms")

    return results


def run_benchmark(
    endpoint: str = "http://localhost:8111",
    test_data_path: str = "data/test.jsonl",
    output_path: str = "data/metrics_baseline.json",
    checkpoint_name: str = "baseline"
):
    test_file = Path(test_data_path)
    if not test_file.exists():
        print(f"Error: Test dataset not found at {test_file}. Run label_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} test samples from {test_file}")
    texts = [r["text"] for r in records]

    # Evaluate predictions in micro-batches
    bsz = 32
    all_laya_results = []
    print(f"Running inference on {len(texts)} test samples via {endpoint}...")
    for i in range(0, len(texts), bsz):
        chunk = texts[i:i + bsz]
        chunk_res = query_laya_batch(endpoint, chunk)
        all_laya_results.extend(chunk_res)

    # Accumulate metrics
    heads = ["category", "sentiment", "virality", "is_breaking_claim", "geo_region", "is_nsfw"]
    head_metrics = {}

    anchor_correct = 0
    anchor_total = 0
    boundary_correct = 0
    boundary_total = 0

    confidences_all = []
    accuracies_all = []

    for head in heads:
        actuals = []
        preds = []
        head_confs = []
        head_accs = []
        brier_diffs = []

        for rec, laya_res in zip(records, all_laya_results):
            cohort = rec.get("cohort", "unknown")
            gold = rec.get("gold_label", {})
            ans = laya_res.get("answers", {}).get(head, {})

            if head in ["category", "sentiment", "geo_region"]:
                gold_val = gold.get(head)
                pred_val = ans.get("choice") or ans.get("answer_text") or ans.get("criteria_chosen")
                conf = ans.get("answer_confidence", 0.5)

                actuals.append(str(gold_val))
                preds.append(str(pred_val))
                is_corr = 1 if gold_val == pred_val else 0

                head_confs.append(conf)
                head_accs.append(is_corr)
                confidences_all.append(conf)
                accuracies_all.append(is_corr)

                if head == "category":
                    if cohort == "anchor":
                        anchor_total += 1
                        if is_corr: anchor_correct += 1
                    elif cohort == "boundary":
                        boundary_total += 1
                        if is_corr: boundary_correct += 1

            elif head in ["is_breaking_claim", "is_nsfw"]:
                gold_bool = gold.get(head, False)
                gold_num = 1.0 if gold_bool else 0.0
                pred_prob = float(ans.get("noul", ans.get("probability", 0.0)))

                brier_diffs.append((pred_prob - gold_num) ** 2)
                pred_bool = pred_prob >= 0.5
                is_corr = 1 if pred_bool == gold_bool else 0
                actuals.append(str(gold_bool))
                preds.append(str(pred_bool))
                conf = pred_prob if pred_bool else (1.0 - pred_prob)
                head_confs.append(conf)
                head_accs.append(is_corr)
                confidences_all.append(conf)
                accuracies_all.append(is_corr)

            elif head == "virality":
                gold_vir = gold.get("virality", "conversational")
                probs = ans.get("probabilities", {})
                legend = ans.get("legend", {"0": "mundane", "1": "conversational", "2": "incendiary and highly viral"})
                if probs:
                    top_k = max(probs.keys(), key=lambda k: probs[k])
                    pred_bin = legend.get(str(top_k), "conversational")
                    conf = float(probs[top_k])
                else:
                    pred_score = float(ans.get("score", 1.0))
                    pred_bin = "mundane" if pred_score < 0.67 else ("conversational" if pred_score < 1.33 else "incendiary and highly viral")
                    conf = float(ans.get("answer_confidence", 0.5))
                actuals.append(gold_vir)
                preds.append(pred_bin)
                is_corr = 1 if pred_bin == gold_vir else 0
                head_confs.append(conf)
                head_accs.append(is_corr)
                confidences_all.append(conf)
                accuracies_all.append(is_corr)

        acc = float(np.mean(head_accs)) if head_accs else 0.0
        f1 = compute_macro_f1(actuals, preds)
        kappa = compute_cohens_kappa(actuals, preds)

        metrics = {
            "accuracy": acc,
            "macro_f1": f1,
            "cohens_kappa": kappa
        }
        if brier_diffs:
            metrics["brier_score"] = float(np.mean(brier_diffs))
        if head_confs:
            metrics["head_ece"] = compute_ece(head_confs, head_accs)

        head_metrics[head] = metrics

    overall_ece = compute_ece(confidences_all, accuracies_all)
    anchor_acc = (anchor_correct / anchor_total) if anchor_total > 0 else 0.0
    boundary_acc = (boundary_correct / boundary_total) if boundary_total > 0 else 0.0
    macro_f1_avg = float(np.mean([m["macro_f1"] for m in head_metrics.values()]))
    overall_acc = float(np.mean([m["accuracy"] for m in head_metrics.values()]))

    # Latency benchmarking
    latency_stats = benchmark_latency(endpoint, texts[0])

    # Check CUDA memory if available
    vram_peak_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            vram_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        pass

    benchmark_record = {
        "checkpoint_name": checkpoint_name,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "total_test_samples": len(records),
            "anchor_samples": anchor_total,
            "boundary_samples": boundary_total
        },
        "classification": {
            "overall_accuracy": overall_acc,
            "macro_f1": macro_f1_avg,
            "anchor_slice_retention": anchor_acc,
            "boundary_slice_accuracy": boundary_acc,
            "per_head": head_metrics
        },
        "calibration": {
            "expected_calibration_error_ece": overall_ece
        },
        "latency": latency_stats,
        "compute": {
            "vram_peak_mb": vram_peak_mb
        }
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, indent=2)

    print(f"\nBenchmark completed successfully! Saved to {out_file.resolve()}")
    print("=" * 60)
    print(f"Overall Accuracy:           {overall_acc * 100:5.2f}%")
    print(f"Macro F1 Score:             {macro_f1_avg:5.3f}")
    print(f"Anchor Slice Retention:     {anchor_acc * 100:5.2f}% (Target: >= 98.0%)")
    print(f"Boundary Slice Accuracy:    {boundary_acc * 100:5.2f}%")
    print(f"Expected Calib Error (ECE): {overall_ece:5.3f} (Lower is better)")
    print(f"Batch 32 Latency (p50):     {latency_stats['batch_32']['p50']:5.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Laya performance before or after fine-tuning")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8111", help="Laya serve HTTP endpoint")
    parser.add_argument("--test-data", type=str, default="data/test.jsonl", help="Held-out test set JSONL")
    parser.add_argument("--output", type=str, default="data/metrics_baseline.json", help="Output metrics JSON file")
    parser.add_argument("--checkpoint", type=str, default="baseline", help="Checkpoint label")
    args = parser.parse_args()

    run_benchmark(
        endpoint=args.endpoint,
        test_data_path=args.test_data,
        output_path=args.output,
        checkpoint_name=args.checkpoint
    )
