# Real-Time Stream Analytics (Agentic AI Capstone)

A production-grade, real-time streaming analytics pipeline that ingests the public Bluesky Jetstream firehose, runs sub-millisecond AI semantic inference using **Laya System One**, stores enriched data in **ClickHouse**, visualizes it in **Grafana**, and features a fully automated **Active Learning Distillation Loop**.

## Architecture & Pipeline

This project is built using a composable Docker-based architecture with NATS JetStream acting as the highly concurrent messaging backbone.

1. **Ingestion (The Sieve):** Connects to the Bluesky firehose via WebSockets. Deterministically filters non-English posts and extracts `#hashtags`. Pushes raw data to the `stream.raw` NATS queue.
2. **AI Inference (The Brain):** Pulls micro-batches from NATS and runs a single forward-pass through the local **Laya System One** model to extract: Category, Sentiment, Virality Score, and Factual Claims. Pushes enriched data to `stream.enriched`.
3. **Analytics Sink (The Eyes):** Pulls from the enriched queue and batch-inserts records into **ClickHouse**. A pre-configured **Grafana** dashboard visualizes the data in real-time.
4. **Flexible Triggers (The Escalation):** Scans the enriched feed for outliers (e.g., highly viral breaking news claims) and routes them to a dedicated `stream.escalation` queue for heavy LangGraph agent research.
5. **Active Learning (The Feedback Loop):** A Distillation worker samples "edge cases" where Laya's confidence is low, sends them to Gemini 1.5 for a "gold standard" label, and saves them to a persistent JSONL dataset for future fine-tuning.

## Prerequisites
- Docker & Docker Compose
- (Optional) NVIDIA GPU for hardware-accelerated Laya inference
- (Optional) Gemini API Key for the Distillation worker

## Quick Start

### 1. Configuration
The entire pipeline is controlled via the `.env` file. A template `.env.example` is provided:
```bash
cp .env.example .env
```
If you plan to use the Active Learning distillation loop, add your Gemini API key:
```env
GEMINI_API_KEY=your_key_here
```

### 2. Launch the Pipeline
Spin up the entire stack using Docker Compose:
```bash
docker compose up -d --build
```
This will start: `nats`, `laya-serve`, `ingest`, `inference`, `clickhouse`, `sink`, `grafana`, `escalation`, and `distillation`.

### 3. Verify the Services

**View the Live Logs:**
Check the ingestion metrics or the inference worker:
```bash
docker compose logs -f ingest
docker compose logs -f inference
```

**View the Raw & Enriched Data via NATS:**
You can tap directly into the live firehose streams using the NATS CLI:
```bash
# View Raw Ingested Data
docker run --rm --network stream-analytics_default natsio/nats-box nats sub stream.raw -s nats://nats:4222

# View Enriched Data (with Laya AI classifications)
docker run --rm --network stream-analytics_default natsio/nats-box nats sub stream.enriched -s nats://nats:4222
```

### 4. Access the Dashboard
The pipeline automatically provisions a **ClickHouse** database and a **Grafana** instance.
1. Navigate to [http://localhost:3000](http://localhost:3000)
2. Login with `admin` / `admin`
3. The ClickHouse datasource is pre-configured. Create your dashboards using the `bluesky.enriched_posts` table!

## Managing the Active Learning Loop

By default, the Distillation Worker (Phase 5) is disabled to prevent unnecessary API usage. To turn it on and start generating a fine-tuning dataset:

1. Open `.env` and set `DISTILLATION_ENABLED=true`
2. Ensure `GEMINI_API_KEY` is set.
3. Restart the distillation worker:
   ```bash
   docker compose restart distillation
   ```
4. The worker will save its sampled datasets to the local Docker volume `laya_finetune_cache` as `distillation_dataset.jsonl`.

## License
This project is licensed under the [Apache License 2.0](LICENSE).
