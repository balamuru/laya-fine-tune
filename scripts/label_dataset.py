#!/usr/bin/env python3
"""
scripts/label_dataset.py
Distills ground-truth multi-task labels using Gemini 3.7 Flash via the official google-genai SDK.
Splits into stratified training (80%) and held-out evaluation (20%) sets.

Inputs:  data/distillation_candidates.jsonl
Outputs: data/train.jsonl (800 rows)
         data/test.jsonl (200 rows: 100 Anchors + 100 Boundary Edges)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class LayaGoldLabel(BaseModel):
    category: Literal["tech", "politics", "entertainment", "finance", "lifestyle", "other"] = Field(
        description="Primary topic category: tech (technology, AI, software), politics (government, elections), entertainment (movies, music, culture), finance (markets, crypto, economy), lifestyle (hobbies, everyday life), other (general chat)."
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Sentiment: positive (excited, supportive), negative (critical, angry), neutral (factual, objective)."
    )
    virality: Literal["mundane", "conversational", "incendiary and highly viral"] = Field(
        description="Predicted viral spread: mundane (low engagement), conversational (normal banter), incendiary and highly viral (polarizing or explosive)."
    )
    is_breaking_claim: bool = Field(
        description="True if the post makes a verifiable factual claim or reports breaking news, events, or unverified announcements. False if personal opinion or chit-chat."
    )
    geo_region: Literal[
        "north_america", "europe", "asia", "middle_east", "latin_america", "africa", "oceania", "global_none"
    ] = Field(
        description="Geographic region referenced: north_america (US/Canada/Mexico), europe (EU/UK/Ukraine), asia (China/Japan/India), middle_east (Israel/Gaza/Iran), latin_america, africa, oceania (Australia/NZ), global_none (no specific region)."
    )
    is_nsfw: bool = Field(
        description="True if the post contains sexually explicit, obscene, or adult NSFW material. False otherwise."
    )

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


def build_gemini_client():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set in environment or .env file.", file=sys.stderr)
        print("Please export GEMINI_API_KEY='your-key' before running this script.", file=sys.stderr)
        sys.exit(1)

    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


def label_post(client, text: str, max_retries: int = 3) -> Optional[dict]:
    prompt = f"""
You are a master semantic classifier distilling gold-standard annotations for a high-speed System One decision engine.
Carefully analyze this social post and output structured classifications according to the schema.

Post text:
\"\"\"{text}\"\"\"
"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.7-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": LayaGoldLabel,
                    "temperature": 0.1,
                },
            )
            data = json.loads(response.text)
            return data
        except Exception as e:
            time.sleep(1.0 * (attempt + 1))
            if attempt == max_retries - 1:
                print(f"Failed to label post after {max_retries} attempts: {e}", file=sys.stderr)
                return None


def run_labeling(
    input_file: str = "data/distillation_candidates.jsonl",
    train_out: str = "data/train.jsonl",
    test_out: str = "data/test.jsonl",
    test_ratio: float = 0.20,
    seed: int = 42,
    limit: Optional[int] = None,
):
    random.seed(seed)
    client = build_gemini_client()

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_path}. Run extract_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit:
        records = records[:limit]

    print(f"Loaded {len(records)} candidate posts from {input_path}")
    print(f"Teacher model: gemini-3.7-flash (via google-genai SDK)")

    labeled_anchors = []
    labeled_boundaries = []

    start_time = time.time()
    for idx, rec in enumerate(records, 1):
        text = rec["text"]
        cohort = rec.get("cohort", "unknown")
        
        gold = label_post(client, text)
        if not gold:
            continue

        rec["gold_label"] = gold

        # Format Laya training native question answers
        # virality mapped to score [0.0, 0.5, 1.0]
        virality_score_map = {
            "mundane": 0.0,
            "conversational": 0.5,
            "incendiary and highly viral": 1.0
        }

        rec["state"] = {"body": text}
        rec["questions"] = LAYA_QUESTIONS
        rec["expected"] = {
            "category": gold["category"],
            "sentiment": gold["sentiment"],
            "virality": virality_score_map.get(gold["virality"], 0.5),
            "is_breaking_claim": 1.0 if gold["is_breaking_claim"] else 0.0,
            "geo_region": gold["geo_region"],
            "is_nsfw": 1.0 if gold["is_nsfw"] else 0.0,
        }

        if cohort == "anchor":
            labeled_anchors.append(rec)
        else:
            labeled_boundaries.append(rec)

        if idx % 10 == 0 or idx == len(records):
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"Progress: [{idx}/{len(records)}] labeled ({rate:.1f} posts/sec)...")

    # Stratified Split: Guarantee exact test_ratio for both anchors and boundaries
    random.shuffle(labeled_anchors)
    random.shuffle(labeled_boundaries)

    n_test_anchors = int(len(labeled_anchors) * test_ratio)
    n_test_boundaries = int(len(labeled_boundaries) * test_ratio)

    test_anchors = labeled_anchors[:n_test_anchors]
    train_anchors = labeled_anchors[n_test_anchors:]

    test_boundaries = labeled_boundaries[:n_test_boundaries]
    train_boundaries = labeled_boundaries[n_test_boundaries:]

    train_set = train_anchors + train_boundaries
    test_set = test_anchors + test_boundaries
    random.shuffle(train_set)
    random.shuffle(test_set)

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")

    Path(test_out).parent.mkdir(parents=True, exist_ok=True)
    with open(test_out, "w", encoding="utf-8") as f:
        for item in test_set:
            f.write(json.dumps(item) + "\n")

    print("\nDataset generation and stratification complete:")
    print(f"  • Training set: {len(train_set)} samples -> {train_out}")
    print(f"    - Anchors: {len(train_anchors)} | Boundaries: {len(train_boundaries)}")
    print(f"  • Held-out test set: {len(test_set)} samples -> {test_out}")
    print(f"    - Anchors: {len(test_anchors)} | Boundaries: {len(test_boundaries)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label candidate posts using Gemini 3.7 Flash")
    parser.add_argument("--input", type=str, default="data/distillation_candidates.jsonl", help="Candidate JSONL file")
    parser.add_argument("--train-out", type=str, default="data/train.jsonl", help="Training set JSONL output")
    parser.add_argument("--test-out", type=str, default="data/test.jsonl", help="Test set JSONL output")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Test set fraction (default: 0.20)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of posts (for testing)")
    args = parser.parse_args()

    run_labeling(
        input_file=args.input,
        train_out=args.train_out,
        test_out=args.test_out,
        test_ratio=args.test_ratio,
        limit=args.limit
    )
