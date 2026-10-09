# Architecture & Analytics Proposals: Real-Time Stream Processing with Laya

## 1. Executive Summary

This proposal outlines the architectural strategies and analytics activities for processing high-volume, real-time event streams (such as the Bluesky Jetstream firehose at 50–200+ msg/sec) using **Laya**, a non-autoregressive, multilingual System 1 decision engine.

Unlike traditional autoregressive LLMs (System 2) that generate text token-by-token with latencies of 500–2000 ms, Laya performs typed decision-making (`choice`, `score`, `noul`) in a single forward pass (~10–30 ms). This document details how to deploy Laya for stream processing and recommends high-value analytics activities tailored to its System 1 architecture.

---

## 2. Architecture Comparison: In-Process vs. Delegated Service

```
                    ┌─────────────────────────────────────────┐
                    │       WebSocket Firehose Ingestor       │
                    │      (e.g., Bluesky Jetstream)          │
                    └────────────────────┬────────────────────┘
                                         │
                         Async Micro-Batch Buffer (Queue)
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               ▼                                                   ▼
     [Option A: Direct In-Process]                       [Option B: Delegated Service]
     • Zero network serialization                        • Decoupled microservices
     • Calls router.predict_batch()                      • Calls HTTP POST /v1/systemone/batch
     • Maximum single-node GPU efficiency                • Independent horizontal scaling
```

### Option A: Direct In-Process (`laya.Router`)

In this topology, the streaming client and model inference execute within the same Python application runtime.

* **Latency:** **~10–30 ms** (zero HTTP round-trip, no JSON serialization).
* **Throughput:** Maximized per GPU by gathering incoming WebSocket events into micro-batches (e.g., 32–64 items or 50 ms time windows) and executing `router.predict_batch()`.
* **Pros:**
  * Minimum computational overhead.
  * No inter-process network latency or serialization cost.
  * Direct memory access to tensors.
* **Cons & Risks:**
  * Tight coupling: A slow inference pass or GPU OOM error directly blocks the WebSocket consumer, risking socket buffer overflow and disconnects.
  * Bound to a single machine/GPU.

### Option B: Delegated Service (`laya serve`)

In this topology, the ingestion layer is decoupled from inference. The ingestion worker buffers events and dispatches batches over HTTP to one or more `laya serve` daemon instances.

* **Latency:** **~15–35 ms** (+3–5 ms network and JSON serialization overhead).
* **Throughput:** Highly scalable across multi-GPU or multi-node clusters.
* **Pros:**
  * **Failure Isolation:** Ingestion never stalls if model workers reload, scale, or failover.
  * **Horizontal Scalability:** Allows load-balancing across multiple inference replicas behind NGINX or an API gateway.
  * **Language Agnostic:** Ingestion can be implemented in Go, Rust, or Node.js while Laya serves model decisions.
* **Cons & Risks:**
  * Additional HTTP and JSON payload serialization overhead.
  * Requires managing and monitoring dedicated server processes.

### Recommended Strategy
* **Prototyping & Development:** Use **Option A (Direct In-Process)** with an asynchronous queue worker to keep setup lightweight.
* **Production Deployment:** Use **Option B (Delegated Service)** with a durable intermediate queue (e.g., NATS JetStream, Redis Streams, or Kafka) buffering between the firehose and a pool of `laya serve` workers.

---

## 3. System 1 vs. System 2 Paradigm in Stream Analytics

| Characteristic | System 1 (Laya / Jev) | System 2 (Autoregressive LLMs) |
| :--- | :--- | :--- |
| **Execution Model** | Single forward pass (non-autoregressive) | Token-by-token generation |
| **Typical Latency** | ~10–30 ms | 500–3,000+ ms |
| **Cost Profile** | Extremely low (local GPU / CPU) | High (GPU memory bandwidth / API pricing) |
| **Output Type** | Structured types (`choice`, `score`, `noul`) | Free-form text / JSON via prompting |
| **Cognitive Mode** | Intuitive, reflexive perception & classification | Multi-step reasoning & Chain-of-Thought |

### Is Sentiment Analysis the Right Use Case?
Standard 3-way sentiment (*positive / negative / neutral*) is naturally a System 1 task, but relying *only* on basic sentiment underutilizes Laya's capabilities. A traditional small transformer (such as DistilBERT) or dictionary analyzer can perform basic polarity checks.

Laya's architectural advantage lies in **evaluating multiple typed, orthogonal questions simultaneously in a single forward pass without accuracy penalty**, natively across 100+ languages.

---

## 4. Proposed High-Value Analytics Activities

### Activity 1: Multi-Dimensional Fast Triage
Instead of single-attribute labeling, pass a comprehensive schema per event to categorize intent, emotional valence, virality potential, and claim presence in one forward pass:

```python
TRIAGE_SCHEMA = {
    "intent": {
        "type": "choice",
        "instructions": "What is the primary intent of this post?",
        "criteria": {
            "opinion": "personal commentary, reactions, or thoughts",
            "news": "announcements or reporting of events and facts",
            "question": "seeking information, answers, or troubleshooting",
            "promotion": "marketing, affiliate links, self-promotion, or ads",
        },
    },
    "sentiment": {
        "type": "choice",
        "instructions": "What is the emotional tone?",
        "criteria": {
            "positive": "praise, optimism, joy, celebration",
            "frustrated": "complaints, anger, exasperation, annoyance",
            "neutral": "matter-of-fact, objective, observational",
        },
    },
    "virality": {
        "type": "score",
        "instructions": "How discussion-provoking or controversial is this post?",
        "criteria": ["mundane", "conversational", "incendiary/provocative"],
    },
    "factual_claim": {
        "type": "noul",
        "instructions": "Does this post make a verifiable factual or political claim?",
    },
}
```

### Activity 2: Intelligent Gating / Cascade Router for System 2 LLMs
Because >90% of public social streams are low-signal chatter, passing every post to an expensive frontier LLM (e.g., Gemini, Claude) is cost-prohibitive.

Laya serves as a high-speed line-rate gatekeeper:
```
[100% Raw Stream (100 msg/s)]
            │
            ▼
   [Laya System 1 Filter]
   • is_high_signal_insight (noul)
   • factual_claim (noul)
            │
     ┌──────┴───────────────────────────┐
     ▼                                  ▼
[p < 0.85 (Discard / Metric Aggregate)] [p ≥ 0.85 (High Signal ~5-10%)]
                                        │
                                        ▼
                           [System 2 LLM Deep Analysis]
                           • Deep Fact-Checking
                           • Automated Synthesis & Summarization
                           • Entity Knowledge-Graph Insertion
```

### Activity 3: Real-Time Stream Shielding & Content Moderation
* **Spam & Abuse Defense:** Score incoming messages with calibrated probabilities (`noul`) using criteria for commercial link farm spam or targeted harassment.
* **Calibrated Confidence:** Because Laya models are trained with RLCD against strictly proper scoring rules, the output probabilities are well-calibrated (a score of 0.82 reflects ~82% real likelihood), enabling reliable automated thresholding.

### Activity 4: Zero-Shot Multilingual Stream Harmonization
* The Bluesky Jetstream firehose contains significant volume in Japanese, Portuguese, German, Arabic, and French.
* Rather than maintaining separate translation pipelines, `laya-multilingual` allows defining questions in English while evaluating inputs across 100+ languages directly with native script support.

---

## 5. Next Steps & Implementation Milestones

1. **Phase 1 (Ingestion Micro-Batching):** Extend `main.py` with an async `asyncio.Queue` consumer that batches events into chunks of 32 items or 50 ms windows.
2. **Phase 2 (In-Process Integration):** Connect `laya.Router().predict_batch()` to the queue and benchmark end-to-end throughput against live Jetstream volume.
3. **Phase 3 (Service Evaluation):** Run `laya serve` in Docker/CUDA and benchmark HTTP batch requests (`/v1/systemone/batch`) to measure network overhead under sustained load.
4. **Phase 4 (Analytics Dashboards):** Stream enriched outputs into a rolling time-series storage (e.g., DuckDB / ClickHouse) for live metric visualization.
