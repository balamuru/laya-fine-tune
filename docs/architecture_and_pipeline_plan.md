# Real-Time Stream Analytics with Laya: Master System Architecture & Pipeline Plan

## 1. Executive Summary & System Vision

This document serves as the single source of truth for the real-time stream analytics system. 

The system ingests high-velocity natural language events from public WebSocket firehoses (specifically the **Bluesky Jetstream** feed at 50–200+ msg/sec), filters out low-entropy noise at microsecond wire speeds, and performs rich, multi-dimensional semantic analysis using **Laya**, a non-autoregressive multilingual System 1 decision engine.

Unlike autoregressive frontier LLMs (System 2) that incur high latency (500–3,000 ms) and compute costs, Laya executes typed decisions (`choice`, `score`, `noul`) across 100+ languages in a single forward pass (~10–30 ms). High-value, complex outliers are selectively escalated to System 2 models.

---

## 2. Architecture Decision: Serving Laya via HTTP (`laya serve`)

- **Decision:** Run `laya serve` as an independent, decoupled HTTP daemon and dispatch requests via `/v1/systemone/batch`.
- **Status:** **Accepted**

### Strategic Justifications

1. **Backpressure Elimination & Zero Socket Drops:**
   * Public WebSocket firehoses drop connections if the client's TCP socket buffer fills due to downstream latency.
   * Running in-process tightly couples network I/O with model execution: any GPU stall, memory compaction, or PyTorch initialization risks a disconnect.
   * Decoupling ensures the ingestion worker remains purely non-blocking and never drops connections.

2. **Negligible Amortized Latency:**
   * Local loopback HTTP (`localhost` or container network) introduces only ~1.5–3.0 ms per batch.
   * When processing micro-batches of 32 items, network overhead amortizes to **<0.1 ms per event**, well within real-time streaming budgets.

3. **Fault Isolation & Independent Scaling:**
   * Model crashes (e.g., CUDA OOM) do not crash the ingestion process.
   * Ingestion (I/O-bound) and model inference (compute-bound) can scale independently across CPU and GPU nodes behind standard load balancers.

4. **Zero-Downtime Model Updates:**
   * Model checkpoints can be reloaded or hot-swapped without restarting the 24/7 stream ingestor.

---

## 3. End-to-End System Architecture

```
[WebSocket Firehose (Bluesky Jetstream)]
                   │  Raw JSON events (~100+ posts/s)
                   ▼
┌──────────────────────────────────────────────────────────┐
│             Reactive Ingestion Producer                  │
│  • Async WebSocket client with automatic reconnection    │
│  • Non-blocking push to in-memory queue                  │
└──────────────────────────┬───────────────────────────────┘
                           │  asyncio.Queue / Rx Subject
                           ▼
┌──────────────────────────────────────────────────────────┐
│      Tier 1: Deterministic Metadata Filter               │
│  • Executes in <0.01 ms on CPU                           │
│  • Checks operation type, text length, character set,    │
│    target languages, and thread context                  │
│  • Drops 60–70% of low-entropy noise                     │
└──────────────────────────┬───────────────────────────────┘
                           │  Passes ~30–40 high-signal posts/s
                           ▼
┌──────────────────────────────────────────────────────────┐
│              Micro-Batch Accumulator                     │
│  • Gathers events into dynamic batches                   │
│  • Flushes on: Batch size = 32 OR Timeout = 50 ms        │
└──────────────────────────┬───────────────────────────────┘
                           │  List[dict] batch of states
                           ▼
┌──────────────────────────────────────────────────────────┐
│      Tier 2: Laya System 1 Inference Client              │
│  • Dispatches POST /v1/systemone/batch over HTTP         │
│  • Evaluates 4 typed questions in a single forward pass  │
│  • Returns: Category, Sentiment, Virality, Breaking Claim│
└──────────────────────────┬───────────────────────────────┘
                           │  Enriched event objects
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  Stream Dispatcher                       │
└──────────────┬────────────────────────────┬──────────────┘
               │ (Standard Enriched Flow)   │ (Outliers: factual_claim > 0.85 &
               │                            │            virality == high)
               ▼                            ▼
┌──────────────────────────────┐  ┌────────────────────────────────────┐
│      Analytics Sink          │  │     Tier 3 Placeholder Handler     │
│  • Rolling window metrics    │  │  • def handle_tier3_candidate(evt):│
│  • Real-time console display │  │      # Placeholder for future      │
│  • DuckDB / ClickHouse store │  │      # Frontier LLM (Gemini/Claude)│
└──────────────────────────────┘  │      pass                          │
                                  └────────────────────────────────────┘
```

---

## 4. Multi-Tier Filtering & Decision Specifications

### Tier 0: Wire-Level Protocol Filtering
* **Mechanism:** WebSocket query parameter `?wantedCollections=app.bsky.feed.post`.
* **Cost:** 0 ms (filtered by the remote relay server).
* **Impact:** Drops likes, follows, and reposts before they enter local network bandwidth.

### Tier 1: Deterministic Metadata Filtering (Sub-Millisecond CPU)
Before scheduling model inference, every post passes through deterministic Python rules:

| Field | Rule | Purpose |
| :--- | :--- | :--- |
| `commit.operation` | `operation == "create"` | Discards post deletions and metadata updates. |
| `record.langs` | `set(record.langs) & target_langs` | Focuses on intended language corpora (e.g., `["en"]` or target set). |
| `record.text` | $\text{len}(\text{text}) \ge 20$ AND $\text{words} \ge 4$ | Filters out low-entropy noise (*"lol"*, *"nice"*, *"good night"*). |
| `record.text` | Must contain alphanumeric characters | Drops pure emoji reactions and symbol spam. |
| `record.reply` | Configurable: Root (`None`) vs. Reply (`not None`) | Separates original topic seeds from conversational replies. |
| `did` | Rate-limit check per author DID | Discards automated bots flooding the feed. |

### Tier 2: Laya Multi-Dimensional Decision Schema
For the 30–40% of posts that pass Tier 1, Laya evaluates four orthogonal semantic questions in a **single forward pass**:

```python
STREAM_DECISION_SCHEMA = {
    # 1. Topic Taxonomy (Categorical Routing)
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
    # 2. Emotional Tone (Fine-Grained Sentiment)
    "sentiment": {
        "type": "choice",
        "instructions": "What is the dominant emotional tone of this post?",
        "criteria": {
            "positive": "excitement, praise, optimism, celebration",
            "negative": "complaints, anger, exasperation, criticism",
            "neutral": "matter-of-fact, observational, informational",
        },
    },
    # 3. Urgency & Discussion Potential
    "virality": {
        "type": "score",
        "instructions": "How discussion-provoking or polarizing is this post?",
        "criteria": ["mundane", "conversational", "incendiary/polarizing"],
    },
    # 4. Strictly Proper Calibrated Probability (0.0 to 1.0)
    "is_breaking_claim": {
        "type": "noul",
        "instructions": "Does this post make a verifiable factual claim about a breaking real-world event?",
    },
}
```

### Tier 3: Placeholder for Frontier LLM Escalation
Only the top ~1–5% of posts meeting strict criteria are routed to the Tier 3 handler:
$$\text{Escalate if: } P(\text{is\_breaking\_claim}) \ge 0.85 \quad \text{AND} \quad \text{virality} == \text{"incendiary/polarizing"}$$

```python
async def handle_tier3_candidate(event: dict) -> None:
    """Placeholder handler for high-signal events requiring Frontier LLM reasoning.
    
    Future capabilities:
    - Route to Gemini 1.5/2.5 Flash/Pro for deep fact-checking.
    - Automated entity and knowledge graph extraction.
    - Real-time alert dispatch to Slack, webhooks, or notification systems.
    """
    pass
```

---

## 5. Model Strategy: Zero-Shot Baseline & Fine-Tuning Flywheel

### Should We Fine-Tune Laya?
**Yes, for production accuracy—but we start Zero-Shot.**

* **The Baseline (Zero-Shot):**
  Laya's base checkpoints (`english`, `multilingual`, `laya-typed-decisions`) evaluate our schema prompt zero-shot immediately. This allows us to validate the pipeline, refine criteria definitions, and establish throughput baselines without any upfront training.
* **The Production Gap:**
  On Laya's typed-decisions benchmark, fine-tuned checkpoints achieve **0.766 accuracy** vs. **0.362** for zero-shot models. Social media contains internet slang, sarcasm, and evolving jargon where domain fine-tuning produces significant accuracy gains.

### The "Active Learning" Distillation Flywheel

Instead of manual human labeling, we implement a **Teacher-Student Distillation Flywheel**:

```
                  [Live Stream (Zero-Shot Laya)]
                                │
                                ▼
                     Confidence / Entropy Filter
             (Identify ambiguous posts: confidence near 0.50)
                                │
                                ▼  (~1,000–2,000 curated real posts)
              Frontier Teacher LLM (Gemini 2.5)
              Labels ground-truth offline via rich reasoning
                                │
                                ▼  (High-quality domain dataset)
              Laya Fine-Tuning Pipeline (RLCD)
              Trained against strictly proper scoring rules
                                │
                                ▼
              Deploy `laya-bluesky-stream`
              (Accuracy jumps from ~36% to ~80%+)
```

1. **Phase 1 (Stream Logging):** During live zero-shot inference, log incoming texts and Laya's prediction probabilities to a local SQLite/JSONL cache.
2. **Phase 2 (Candidate Selection):** Extract the most ambiguous decisions (entropy > threshold, or confidence between 0.40 and 0.60).
3. **Phase 3 (Teacher Labeling):** Pass these candidates to a frontier model (Gemini) with full Chain-of-Thought reasoning to establish authoritative ground-truth labels.
4. **Phase 4 (RLCD Training):** Run Laya’s reinforcement learning fine-tuning workflow (using `notebooks/laya_finetune_typed_decisions_mps.py` or Kaggle T4 notebook) to train and export `laya-bluesky-stream`.

---

## 6. Implementation Roadmap

| Milestone | Deliverables | Status |
| :--- | :--- | :--- |
| **Milestone 1: Project & Ingestion Foundation** | Project setup, virtual environment, `StreamAnalytics` class, WebSocket client, signal handling. | **Completed** |
| **Milestone 2: Architecture & Decision Plan** | Consolidated master architecture document, ADR, and pipeline specification. | **Completed** |
| **Milestone 3: Tier 1 Filtering & Micro-Batching** | Implement deterministic prefilter module and async queue batch accumulator (32 items / 50 ms timeout). | **Next** |
| **Milestone 4: Laya HTTP Client & Tier 3 Placeholder** | HTTP batch client to `laya serve` (`/v1/systemone/batch`), result unpacker, and `handle_tier3_candidate()` hook. | Pending |
| **Milestone 5: Stream Logging & Distillation Pipeline** | Zero-shot confidence logger for active-learning fine-tuning dataset generation. | Pending |
| **Milestone 6: Real-Time Analytics Dashboard** | Rolling window metrics, category distributions, and DuckDB time-series storage. | Pending |
