#!/usr/bin/env python3
"""
scripts/compare_metrics.py
Ingests baseline and fine-tuned benchmark metrics JSON files,
computes deltas, and generates publication-ready Markdown/LaTeX tables for the technical article.

Inputs:  data/metrics_baseline.json
         data/metrics_finetuned.json
Outputs: stdout and data/scorecard_delta.md
"""

import argparse
import json
import sys
from pathlib import Path


def format_delta(val_before, val_after, higher_is_better=True, pct=False):
    delta = val_after - val_before
    symbol = "+" if delta > 0 else ""
    if pct:
        delta_str = f"{symbol}{delta * 100:.1f}%"
    else:
        delta_str = f"{symbol}{delta:.3f}"

    is_good = (delta > 0) if higher_is_better else (delta < 0)
    tag = "🟢" if is_good else ("🟡" if abs(delta) < 1e-4 else "🔴")
    return f"{delta_str} {tag}"


def compare_metrics(baseline_path: str, finetuned_path: str, output_md: str = "data/scorecard_delta.md"):
    base_file = Path(baseline_path)
    ft_file = Path(finetuned_path)

    if not base_file.exists() or not ft_file.exists():
        print(f"Error: Need both {base_file} and {ft_file} to compare.", file=sys.stderr)
        sys.exit(1)

    with open(base_file, "r") as f:
        b = json.load(f)
    with open(ft_file, "r") as f:
        m = json.load(f)

    b_cls = b.get("classification", {})
    m_cls = m.get("classification", {})
    b_cal = b.get("calibration", {})
    m_cal = m.get("calibration", {})
    b_lat = b.get("latency", {})
    m_lat = m.get("latency", {})

    rows = []

    # Quality Dimensions
    rows.append((
        "Quality",
        "Overall Accuracy",
        f"{b_cls.get('overall_accuracy', 0)*100:.1f}%",
        f"{m_cls.get('overall_accuracy', 0)*100:.1f}%",
        format_delta(b_cls.get('overall_accuracy', 0), m_cls.get('overall_accuracy', 0), higher_is_better=True, pct=True),
        "Target: +10%+"
    ))

    rows.append((
        "Quality",
        "Macro F1 Score",
        f"{b_cls.get('macro_f1', 0):.3f}",
        f"{m_cls.get('macro_f1', 0):.3f}",
        format_delta(b_cls.get('macro_f1', 0), m_cls.get('macro_f1', 0), higher_is_better=True),
        "Target: >= 0.850"
    ))

    rows.append((
        "Quality",
        "Anchor Slice Retention",
        f"{b_cls.get('anchor_slice_retention', 0)*100:.1f}%",
        f"{m_cls.get('anchor_slice_retention', 0)*100:.1f}%",
        format_delta(b_cls.get('anchor_slice_retention', 0), m_cls.get('anchor_slice_retention', 0), higher_is_better=True, pct=True),
        "Must be >= 98.0% (No Forgetting)"
    ))

    rows.append((
        "Quality",
        "Boundary Slice Accuracy",
        f"{b_cls.get('boundary_slice_accuracy', 0)*100:.1f}%",
        f"{m_cls.get('boundary_slice_accuracy', 0)*100:.1f}%",
        format_delta(b_cls.get('boundary_slice_accuracy', 0), m_cls.get('boundary_slice_accuracy', 0), higher_is_better=True, pct=True),
        "Target: +15% to +25%"
    ))

    # Calibration Dimensions
    b_ece = b_cal.get("expected_calibration_error_ece", 0.0)
    m_ece = m_cal.get("expected_calibration_error_ece", 0.0)
    rows.append((
        "Calibration",
        "Expected Calibration Error (ECE)",
        f"{b_ece:.3f}",
        f"{m_ece:.3f}",
        format_delta(b_ece, m_ece, higher_is_better=False),
        "Target: <= 0.045 (Lower is Better)"
    ))

    # Latency Dimensions
    if "batch_32" in b_lat and "batch_32" in m_lat:
        b_p50 = b_lat["batch_32"].get("p50", 0.0)
        m_p50 = m_lat["batch_32"].get("p50", 0.0)
        delta_p50 = m_p50 - b_p50
        rows.append((
            "Speed",
            "Batch 32 Latency (p50)",
            f"{b_p50:.2f} ms",
            f"{m_p50:.2f} ms",
            f"{'+' if delta_p50 > 0 else ''}{delta_p50:.2f} ms",
            "Zero Regression (<= 0.2ms delta)"
        ))

    if "batch_1" in b_lat and "batch_1" in m_lat:
        b_p50_1 = b_lat["batch_1"].get("p50", 0.0)
        m_p50_1 = m_lat["batch_1"].get("p50", 0.0)
        delta_p50_1 = m_p50_1 - b_p50_1
        rows.append((
            "Speed",
            "Single-Item Latency (p50)",
            f"{b_p50_1:.2f} ms",
            f"{m_p50_1:.2f} ms",
            f"{'+' if delta_p50_1 > 0 else ''}{delta_p50_1:.2f} ms",
            "Zero Regression"
        ))

    # Build Markdown table
    md_lines = [
        "# Laya Distillation Performance Scorecard: Before vs. After Fine-Tuning\n",
        f"**Evaluated At**: {m.get('evaluated_at')} | **Dataset**: {m.get('dataset', {}).get('total_test_samples', 0)} Held-Out Samples\n",
        "| Dimension | Metric | Baseline (Pre-FT) | Fine-Tuned (Post-FT) | Measured Delta (Δ) | Acceptance Gate |",
        "|---|---|---|---|---|---|"
    ]

    for dim, metric, before, after, delta, note in rows:
        md_lines.append(f"| **{dim}** | {metric} | {before} | **{after}** | {delta} | {note} |")

    md_content = "\n".join(md_lines) + "\n"

    print(md_content)

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved publication scorecard to {output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned Laya benchmark metrics")
    parser.add_argument("--baseline", type=str, default="data/metrics_baseline.json", help="Path to baseline metrics JSON")
    parser.add_argument("--finetuned", type=str, default="data/metrics_finetuned.json", help="Path to fine-tuned metrics JSON")
    parser.add_argument("--output", type=str, default="data/scorecard_delta.md", help="Output scorecard markdown path")
    args = parser.parse_args()

    compare_metrics(args.baseline, args.finetuned, args.output)
