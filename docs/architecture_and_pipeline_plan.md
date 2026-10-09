# Real-Time Stream Analytics with Laya: Master System Architecture & Pipeline Plan

## 1. Executive Summary & System Vision

This document serves as the single source of truth for the real-time stream analytics system. 

The system ingests high-velocity natural language events from public WebSocket firehoses (specifically the **Bluesky Jetstream** feed at 50–200+ msg/sec), filters out low-entropy noise at microsecond wire speeds, and performs rich, multi-dimensional semantic analysis using **Laya**, a non-autoregressive multilingual System 1 decision engine.

To ensure resilience against micro-outages and to enable efficient batching, the pipeline utilizes **NATS JetStream** as an elastic, memory-backed shock absorber. The entire stack is deployed as composable Docker containers configured via a central `.env` file.

---

## 2. Architecture Decision: Serving Laya via HTTP (`laya serve`)

- **Decision:** Run `laya serve` as an independent, decoupled HTTP daemon and dispatch requests via `/v1/systemone/batch`.
- **Status:** **Accepted**

### Strategic Justifications
1. **Backpressure Elimination & Zero Socket Drops:** Decoupling ensures the ingestion worker remains purely non-blocking.
2. **Negligible Amortized Latency:** Network overhead amortizes to **<0.1 ms per event** over micro-batches of 32 items.
3. **Fault Isolation & Independent Scaling:** Model crashes do not crash the ingestion process.
4. **Zero-Downtime Model Updates:** Hot-swap models without disconnecting the 24/7 stream ingestor.

---

## 3. End-to-End System Architecture

```text
[WebSocket Firehose (Bluesky Jetstream)]
                   │  Raw JSON events (~100+ posts/s)
                   ▼
┌──────────────────────────────────────────────────────────┐
│      1. Ingestion Service (Python Worker)                │
│  • Async WebSocket client with auto-reconnect            │
│  • Tier 1: Deterministic Metadata Filter (<0.01ms CPU)   │
│  • Pushes high-signal posts (~30-40/s) to NATS           │
└──────────────────────────┬───────────────────────────────┘
                           │  Publish to `stream.raw`
                           ▼
┌──────────────────────────────────────────────────────────┐
│      2. NATS JetStream Broker (Elastic Shock Absorber)   │
│  • Memory-backed Ring Buffer (No disk I/O)               │
│  • Short TTL (e.g., 60s) & Max Msg limits                │
│  • Absorbs bursts; drops worthless stale data            │
└──────────────────────────┬───────────────────────────────┘
                           │  Batch Pull (e.g., 32 items)
                           ▼
┌──────────────────────────────────────────────────────────┐
│      3. Inference Service (Python Worker)                │
│  • Pulls micro-batches from NATS JetStream               │
│  • Tier 2: Dispatches POST /v1/systemone/batch to Laya   │
│  • Publishes enriched events to `stream.enriched`        │
└──────────────────────────┬───────────────────────────────┘
                           │  Publish to `stream.enriched`
                           ▼
┌──────────────────────────────────────────────────────────┐
│      4. Stream Dispatcher & Sinks                        │
└──────────────┬────────────────────────────┬──────────────┘
               │                            │ (Outliers: factual_claim > 0.85 &
               ▼                            ▼            virality == high)
┌──────────────────────────────┐  ┌────────────────────────────────────┐
│      Analytics Sink          │  │     Tier 3 Placeholder Handler     │
│  • DuckDB / ClickHouse       │  │  • Subscribes to outliers          │
│  • Real-time Dashboards      │  │  • Future: Gemini/Claude reasoning │
└──────────────────────────────┘  └────────────────────────────────────┘
```

---

## 4. Infrastructure & Docker Deployment Strategy

The pipeline is provisioned as composable Docker applications using a unified `docker-compose.yml` that wraps both our custom pipeline workers and the official Laya inference server. All tunable parameters are isolated in a `.env` file.

### Composable Services
1. **`nats`**: The NATS server running with JetStream enabled (`-js`).
2. **`laya-serve`**: The Laya HTTP REST API service. We will wrap the existing configuration found in `/home/vinayb/apps/laya/docker-compose/docker-compose.yml` (using the `vinaybalamuru/laya:cuda` image and mapping the local model cache volume) into our project's compose file.
3. **`ingest`**: The Python application running the WebSocket consumer and Tier 1 filter.
4. **`inference`**: The Python application pulling from JetStream, calling `laya-serve`, and publishing results.
5. **`analytics`**: (Phase 3) The ClickHouse database and Grafana layer.

### Environment Configuration (`.env`)
All services rely on the `.env` file for topology and limits. This includes the Laya runtime configurations imported from the official repository:
```env
# NATS JetStream Configuration
NATS_URL=nats://nats:4222
JETSTREAM_RAW_SUBJECT=stream.raw
JETSTREAM_ENRICHED_SUBJECT=stream.enriched
JETSTREAM_TTL=60s             # Drop data older than 60 seconds
JETSTREAM_MAX_MSGS=10000      # Ring buffer limit

# Laya Inference Server Configuration (Wrapped)
LAYA_PORT=8111
LAYA_DEVICE=cuda
LAYA_CUDA_AMP=fp16
LAYA_BATCH_SIZE=32
LAYA_SERVE_URL=http://laya-serve:8111

# Ingestion Target
JETSTREAM_WS_URL=wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post

# Stage 2 Sieve Configuration
SIEVE_ROOT_POSTS_ONLY=true
```

---

## 5. Multi-Tier Filtering & Decision Specifications

### Tier 1: Deterministic Metadata Filtering (Sub-Millisecond CPU)
Executed by the **Ingestion Service** before hitting NATS:
* `operation == "create"`
* Target language matching (`record.langs`).
* Minimum character count ($\ge 20$) and word count ($\ge 4$).
* Regex validation (drop pure emoji/symbol spam).

### Tier 2: Laya Multi-Dimensional Decision Schema
Executed by the **Inference Service** using Laya's single forward pass:
* **`category`** (choice): Tech, politics, entertainment, finance, lifestyle.
* **`sentiment`** (choice): Positive, negative, neutral.
* **`virality`** (score): Mundane, conversational, incendiary.
* **`is_breaking_claim`** (noul): 0.0 to 1.0 probability of a verifiable factual claim.

---

## 6. Phased Implementation Plan (Capstone Quality)

This project is structured as a capstone, moving from raw ingestion to a flexible, production-grade analytics dashboard with hooks for future AI research workflows.

### Phase 1: Core Infrastructure & Ingestion (The Firehose)
**Goal:** Establish the composable Docker environment, connect to the Bluesky firehose, and prove we can route messages through NATS JetStream without dropping connections.

*   **Functional Requirements (FRs):**
    *   **FR1.1:** Docker Compose must spin up NATS JetStream, Laya Serve, and Python worker containers seamlessly.
    *   **FR1.2:** Ingestion script must maintain a persistent, auto-reconnecting WebSocket connection to Bluesky Jetstream.
    *   **FR1.3:** Ingestion script must push raw messages to NATS JetStream under the `stream.raw` subject.
*   **Evaluations (Evals):**
    *   **Eval 1.1:** Run `docker-compose up -d`. Verify all containers reach a "healthy" state.
    *   **Eval 1.2:** Disconnect network temporarily; verify Python worker auto-reconnects to Bluesky within 5 seconds without crashing.
    *   **Eval 1.3:** Use `nats sub stream.raw` in the CLI to visually confirm message throughput matches firehose volume.

### Phase 2: The Sieve & Laya Inference (The Brain)
**Goal:** Implement the deterministic Stage 2 filter and the Stage 3 Laya batch-inference logic, driven by the `.env` configuration.

*   **Functional Requirements (FRs):**
    *   **FR2.1:** Stage 2 Sieve must filter posts based on `.env` toggles (e.g., `SIEVE_ROOT_POSTS_ONLY=true`).
    *   **FR2.2:** Stage 2 must extract and append `#hashtags` to the message metadata before NATS publishing.
    *   **FR2.3:** Inference worker must pull micro-batches (e.g., size 32) from `stream.raw`, hit the `laya serve` `/v1/systemone/batch` endpoint, and publish to `stream.enriched`.
*   **Evaluations (Evals):**
    *   **Eval 2.1:** Toggle `.env` variables and verify log output reflects the new filtering rules immediately upon restart.
    *   **Eval 2.2:** Inspect NATS `stream.raw` payload to confirm `#hashtags` array exists.
    *   **Eval 2.3:** Measure end-to-end latency. The Laya inference step must process a batch of 32 in < 50ms total.

### Phase 3: The Capstone Analytics Sink & Dashboard (The Eyes)
**Goal:** Ingest the enriched data into an OLAP database and visualize it in real-time, building a flexible monitoring dashboard.

*   **Functional Requirements (FRs):**
    *   **FR3.1:** Deploy a Time-Series/OLAP database (e.g., ClickHouse) via Docker Compose.
    *   **FR3.2:** Create a Sink Worker that subscribes to `stream.enriched` and batch-inserts records into the database.
    *   **FR3.3:** Deploy Grafana (or Streamlit) connected to the database to visualize live metrics (sentiment velocity, trending hashtags, topic distribution).
*   **Evaluations (Evals):**
    *   **Eval 3.1:** Run SQL queries directly against the database to verify data integrity and schema types.
    *   **Eval 3.2:** Stress test: simulate 500 msgs/sec for 5 minutes and verify 0 dropped inserts in the database.
    *   **Eval 3.3:** Visually confirm the dashboard updates in real-time (sub-5-second latency from firehose to chart).

### Phase 4: Flexible Triggers & Escalation (The Future)
**Goal:** Implement the Tier 3 Placeholder to ensure the architecture is flexible enough to trigger heavy LangGraph research agents or external data collection later on.

*   **Functional Requirements (FRs):**
    *   **FR4.1:** Establish a NATS consumer specifically listening for outlier thresholds (e.g., `laya.is_breaking_claim > 0.95`).
    *   **FR4.2:** Design a flexible "Trigger Schema" allowing messages to be pushed to a webhook or a separate queue (`stream.escalation`).
    *   **FR4.3:** Ensure the trigger mechanism does not block or slow down the main `stream.enriched` pipeline.
*   **Evaluations (Evals):**
    *   **Eval 4.1:** Inject a synthetic "perfect storm" message into NATS. Verify it triggers the specific escalation consumer.
    *   **Eval 4.2:** Check the empty LangGraph placeholder block to ensure the payload arrives fully structured and ready for future heavy LLM processing.
