# Pipeline Architecture: Tiered Prefiltering & Laya Stream Analytics

## 1. Overview & Objective

This document defines the ingestion and prefiltering architecture for processing the high-velocity Bluesky Jetstream firehose (~50–200+ msg/sec) using **Laya** as the primary System 1 decision engine.

The core objective is to maximize throughput, minimize unnecessary GPU inference, and extract rich, structured analytics by establishing a **4-Tier Ingestion Funnel**.

---

## 2. The 4-Tier Analytics Funnel

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Raw AT-Protocol Firehose (Relay)                     │
│                (~1,000+ total events/s across all collections)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    TIER 0: Protocol / Wire Level   │  Query params: ?wantedCollections=app.bsky.feed.post
    (Zero Compute Cost)             │  Drops likes, follows, reposts before arrival
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Post Creation Stream (~100 posts/s)                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    TIER 1: Deterministic Filtering │  Sub-millisecond CPU metadata checks:
    (< 0.01 ms / event)             │  • Non-create operations dropped
                                    │  • Short / low-information text dropped (<15 chars)
                                    │  • Pure emojis / bare URLs dropped
                                    │  • Target language matching (record.langs)
                                    │  • Thread position: Root vs. Reply separation
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             High-Signal Posts (~30–40 posts/s, 60–70% reduction)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    TIER 2: Laya System 1 Inference │  Single forward-pass micro-batches (~25 ms on GPU):
    (Typed Semantic Decisions)      │  • Topic Category (choice)
                                    │  • Sentiment / Emotional Valence (choice)
                                    │  • Virality / Controversy (score)
                                    │  • Breaking News / Factual Claim (noul: 0.0 - 1.0)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Enriched Stream with Calibrated Metrics                │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
    95% Standard Event Flow                          │ Top 5% Outlier / High-Value Events
    (Normal Distribution)                            │ (factual_claim > 0.85 & virality == high)
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│  Real-Time Analytics & Storage       │  │  TIER 3: Frontier LLM Escalation     │
│  • Rolling Window Metrics            │  │  • Gemini / Claude System 2          │
│  • Time-Series DB (DuckDB/ClickHouse)│  │  • Deep Fact-Checking & Synthesis    │
│  • Category & Sentiment Dashboards   │  │  • Real-Time Critical Event Alerting │
└──────────────────────────────────────┘  └──────────────────────────────────────┘
```

---

## 3. Tier 1: Deterministic Metadata & Structural Filtering

### Why Not Start with an LLM?
Evaluating natural language with neural networks incurs tensor memory overhead, tokenization, and compute cycles. A substantial percentage of social stream events are low-entropy noise (e.g., `"lol"`, `"yes"`, `"+1"`, bare hyperlinks, or single emojis).

Running an LLM on these events is wasteful. A deterministic Python rule-set executes in **<10 microseconds** and eliminates **60% to 70%** of total volume before GPU scheduling.

### Filter Rules & Criteria

| Field | Rule / Condition | Rationale |
| :--- | :--- | :--- |
| `commit.operation` | `operation == "create"` | Discards post deletions and metadata updates. |
| `record.langs` | `set(record.langs) & target_langs` | Focuses on intended language corpora (e.g., `en`, `es`, `ja`) without spending model compute on unmonitored languages. |
| `record.text` | `len(text.strip()) >= 20` and word count $\ge 4$ | Filters out low-entropy reactions (*"thank you"*, *"awesome"*, *"good night"*). |
| `record.text` | Must contain alphanumeric characters | Rejects pure emoji strings, symbols, and malformed characters. |
| `record.reply` | Configurable: Separate `reply is None` (Root) vs. `reply is not None` (Reply) | Root posts generate original topics and claims; replies represent conversational reactions. Depending on the analysis, replies can be routed to a lighter queue or skipped. |
| `did` | Max $N$ posts per minute per account | Eliminates automated RSS bots flooding the firehose. |

---

## 4. Tier 2: Laya System 1 Semantic Analysis

For posts that pass deterministic filtering, Laya evaluates multiple orthogonal semantic dimensions in a **single forward pass** (~25 ms).

### Multi-Dimensional Question Schema

```python
STREAM_DECISION_SCHEMA = {
    # 1. Topic Taxonomy (Categorical routing)
    "category": {
        "type": "choice",
        "instructions": "What is the primary topic of this post?",
        "criteria": {
            "tech": "artificial intelligence, software, gadgets, engineering, science",
            "politics": "government, elections, public policy, global affairs",
            "entertainment": "gaming, movies, music, celebrities, pop culture, art",
            "finance": "markets, economics, cryptocurrency, business, investing",
            "lifestyle": "food, travel, fitness, personal hobbies, daily life",
            "other": "everything else that does not clearly fit the above",
        },
    },
    # 2. Emotional Valence (Fine-grained sentiment)
    "sentiment": {
        "type": "choice",
        "instructions": "What is the emotional tone of this post?",
        "criteria": {
            "enthusiastic": "excitement, praise, optimism, celebration",
            "frustrated": "complaints, anger, exasperation, criticism",
            "neutral": "matter-of-fact, observational, informational",
        },
    },
    # 3. Urgency / Virality Potential
    "virality": {
        "type": "score",
        "instructions": "How discussion-provoking or polarizing is this post?",
        "criteria": ["mundane", "conversational", "incendiary/polarizing"],
    },
    # 4. Strictly Proper Calibrated Probability
    "is_breaking_claim": {
        "type": "noul",
        "instructions": "Does this post make a verifiable factual claim about a breaking real-world event?",
    },
}
```

### Why This Schema Leverages Laya’s Strengths
1. **Zero Marginal Latency for Extra Questions:** Because Laya is non-autoregressive, evaluating 4 questions takes the same single forward pass as evaluating 1.
2. **Proper Scoring Calibration (`noul`):** Trained via RLCD, Laya's `noul` outputs represent true Bayesian probabilities. If `is_breaking_claim` returns `0.88`, there is an ~88% likelihood of a real factual claim.
3. **Multilingual Consistency:** The exact same questions defined in English work on foreign text without requiring a preliminary translation step.

---

## 5. Tier 3: Selective Escalation to System 2

Only the top **~1–5%** of events that satisfy strict threshold criteria are escalated to an expensive frontier LLM (e.g., Gemini 1.5/2.5 Flash, Claude 3.5 Sonnet):

$$\text{Escalate if: } P(\text{is\_breaking\_claim}) \ge 0.85 \quad \text{AND} \quad \text{virality} == \text{"incendiary/polarizing"}$$

### Escalation Tasks for System 2:
* Automated entity and relationship extraction.
* Cross-referencing against real-time search or news APIs.
* Synthesizing multi-post event summaries for real-time alerting dashboards.

---

## 6. Performance & Resource Impact

| Metric | Without Prefiltering (100% to Model) | With Tiered Prefiltering Funnel | Delta |
| :--- | :--- | :--- | :--- |
| **Inference Load** | 100 requests / sec | 30–40 requests / sec | **60–70% reduction** |
| **GPU Utilization** | Near saturation on single GPU | Comfortable batch headroom | **~3× capacity gain** |
| **Socket Drop Risk** | Moderate to High (consumer lag) | Zero (lightweight CPU filter) | **High stability** |
| **System 2 Costs** | Prohibitive | Capped to top 1–5% of items | **>95% cost savings** |
