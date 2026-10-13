#!/usr/bin/env python3
"""
scripts/extract_dataset.py
Extracts balanced cohorts from ClickHouse bluesky.enriched_posts:
1. Cohort A (Anchors): High-confidence baseline predictions (p >= 0.85 or p <= 0.10)
2. Cohort B (Boundary Edges): Ambiguous predictions (0.35 <= p <= 0.65 or category == 'other')

Outputs: data/distillation_candidates.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import clickhouse_connect

load_dotenv()

CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
if CH_HOST == "clickhouse" and not os.path.exists("/.dockerenv"):
    CH_HOST = "localhost"
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "default")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "bluesky_secret")
CH_DB = os.getenv("CLICKHOUSE_DATABASE", "bluesky")


def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DB,
    )


def extract_cohorts(anchors_n: int = 500, boundary_n: int = 500, output_path: str = "data/distillation_candidates.jsonl"):
    client = get_clickhouse_client()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to ClickHouse ({CH_HOST}:{CH_PORT}/{CH_DB})...")

    # 1. Extract Anchors (High-confidence, decisive predictions)
    # Clear categorical assignment (not 'other'), high/low breaking and nsfw scores
    anchor_query = f"""
    SELECT 
        id, 
        text, 
        category, 
        sentiment, 
        virality, 
        is_breaking_claim, 
        geo_region, 
        is_nsfw
    FROM {CH_DB}.enriched_posts
    WHERE length(text) >= 30 AND length(text) <= 400
      AND category != 'other'
      AND (is_breaking_claim <= 0.10 OR is_breaking_claim >= 0.85)
      AND (is_nsfw <= 0.05 OR is_nsfw >= 0.85)
      AND primary_lang = 'en'
    ORDER BY rand()
    LIMIT {anchors_n}
    """
    
    print(f"Fetching {anchors_n} Anchor posts (high confidence)...")
    anchor_results = client.query(anchor_query)
    anchor_rows = anchor_results.result_rows
    print(f"Retrieved {len(anchor_rows)} Anchor samples.")

    # 2. Extract Boundary Edges (Ambiguous, borderline, or 'other')
    boundary_query = f"""
    SELECT 
        id, 
        text, 
        category, 
        sentiment, 
        virality, 
        is_breaking_claim, 
        geo_region, 
        is_nsfw
    FROM {CH_DB}.enriched_posts
    WHERE length(text) >= 30 AND length(text) <= 400
      AND (
            category = 'other' 
            OR (is_breaking_claim >= 0.35 AND is_breaking_claim <= 0.65)
            OR (is_nsfw >= 0.30 AND is_nsfw <= 0.70)
            OR (virality >= 0.35 AND virality <= 0.65)
          )
      AND primary_lang = 'en'
    ORDER BY rand()
    LIMIT {boundary_n}
    """

    print(f"Fetching {boundary_n} Boundary posts (low confidence / ambiguous)...")
    boundary_results = client.query(boundary_query)
    boundary_rows = boundary_results.result_rows
    print(f"Retrieved {len(boundary_rows)} Boundary samples.")

    records = []

    def format_row(row, cohort_label):
        return {
            "id": row[0],
            "text": row[1],
            "cohort": cohort_label,
            "baseline_predictions": {
                "category": row[2],
                "sentiment": row[3],
                "virality": float(row[4]),
                "is_breaking_claim": float(row[5]),
                "geo_region": row[6],
                "is_nsfw": float(row[7])
            }
        }

    for r in anchor_rows:
        records.append(format_row(r, "anchor"))

    for r in boundary_rows:
        records.append(format_row(r, "boundary"))

    with open(output_file, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item) + "\n")

    print(f"Successfully extracted {len(records)} total records to {output_file.resolve()}")
    print(f"  - Anchors: {len(anchor_rows)}")
    print(f"  - Boundaries: {len(boundary_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract distillation dataset cohorts from ClickHouse")
    parser.add_argument("--anchors", type=int, default=500, help="Number of anchor samples (default: 500)")
    parser.add_argument("--boundary", type=int, default=500, help="Number of boundary edge samples (default: 500)")
    parser.add_argument("--output", type=str, default="data/distillation_candidates.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    extract_cohorts(anchors_n=args.anchors, boundary_n=args.boundary, output_path=args.output)
