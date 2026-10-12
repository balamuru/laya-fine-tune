# Real-Time Stream Analytics on the Bluesky Firehose with Laya

A production-grade, real-time streaming intelligence pipeline that ingests the public Bluesky Jetstream firehose (~50+ ops/sec), executes sub-millisecond local AI semantic inference using **Laya** (on-device local inference instead of hosted models like Jev), indexes structured analytics in **ClickHouse**, visualizes multi-dimensional insights across 18 panels in **Grafana**, and features an automated **Active Learning Distillation Loop**.

```
🌐 Bluesky Jetstream (WebSocket)
       │
       ▼
⚙️ Ingestion Sieve (ingest.py) ──► Microsecond Filters (Language, Anti-Spoofing, Replies, Spam, Cashtags)
       │
       ▼
🚀 NATS JetStream (stream.raw)
       │
       ▼
⚡ Laya System One AI (inference.py) ──► Multi-Task Local Inference (Category, Sentiment, Virality, Claims, Geo, Safety)
       │
       ▼
🚀 NATS JetStream (stream.enriched)
       ├──► 📊 ClickHouse Sink (sink.py) ──► bluesky.enriched_posts, funnel_stats, queue_stats
       │           │
       │           ▼
       │      📈 Grafana Live Dashboard (18 Panels, Dynamic Filters, Queue Telemetry)
       │
       ├──► 🚨 Escalation Worker (escalation.py) ──► Breaking news claims (prob >= 0.85)
       │
       └──► 🔁 Distillation Worker (distillation.py) ──► Active learning dataset generation
```

---

## Key Features & Highlights

* **Local Laya Inference (Instead of Hosted Jev)**: Complete data sovereignty with zero external API bills. Rather than depending on hosted System One APIs like TypeSafe AI's Jev or slow cloud LLMs, all classification runs entirely in the local container via Laya on GPU tensor cores.
* **Deterministic Edge Sieve**: Pre-filters 75%+ of low-signal events (child replies, non-English posts, adversarial metadata-spoofed Japanese kana scripts, affiliate spam domains, and adult labels) in microseconds before touching GPU resources.
* **Automatic Financial & Cashtag Extraction**: Regex-driven microsecond extraction of stock and crypto symbols (`$NVDA`, `$BTC`, `$TSLA`, `$SPY`, `$META`).
* **Columnar Vectorized OLAP**: ClickHouse stores millions of events with native array types, sub-millisecond in-database tokenization with stopword exclusion, and automatic partition indexing.
* **18-Panel Grafana Live Dashboard (`bluesky_stream_01`)**:
  * **Throughput & Funnel Velocity**: Firehose vs. Sieve Passed vs. Final Output rates.
  * **Stage-by-Stage Funnel Audit Table**: Exact sequential mathematical attribution across all 7 pipeline stages.
  * **NATS Worker Queue Lag Monitoring**: Real-time health states and lag curves for AI GPU, database sink, and escalation workers.
  * **Trending Topics & Cashtags**: Stacked 100% sentiment breakdown, viral tags, and stock ticker volume.
  * **Stopword-Filtered Word Cloud**: In-database token extraction and frequency mapping per topic category.
  * **Breaking Claims Radar**: Alerts on high-confidence breaking news reports (`is_breaking_claim >= 0.85`).
* **Multi-Dimensional Filter Bar**:
  * **Geographic Region**: North America, Europe, Asia, Middle East, Latin America, Africa, Oceania.
  * **Topic Category**: Tech, Finance, Politics, Entertainment, Lifestyle, Other.
  * **Sentiment**: Positive (Blue), Neutral (Yellow), Negative (Red).
  * **Trending Hashtags**: Prepopulated Top 50 trending tags with fuzzy-search combobox.
  * **Custom Hashtag / Keyword Search**: Arbitrary free-text query box matching tags and post text.
  * **Account Activity / Bot Filter**: Tiers separating casual human posters from automated news wire feeds.
  * **↺ Reset Filters to Default**: 1-click toolbar action restoring all filters to baseline without resetting your time zoom.

---

## Architecture Components

| Service | Technology | Role |
|---|---|---|
| **`nats`** | NATS JetStream 2.x | High-throughput, memory-backed messaging buffer (`stream.raw`, `stream.enriched`, `stream.escalation`). |
| **`laya-serve`** | Laya System One (CUDA) | Local, on-device decision engine providing zero-shot multi-task semantic scoring (`/v1/systemone/batch`). |
| **`ingest`** | Python / WebSockets | Consumes Bluesky Jetstream, enforces deterministic sieve, extracts `#hashtags` & `$cashtags`, logs funnel/queue metrics. |
| **`inference`** | Python / AsyncIO | Pulls batches of 32 from NATS, dispatches to Laya, enriches events with schema attributes. |
| **`sink`** | Python / ClickHouse | Micro-batches 100 enriched records into ClickHouse with automatic ACK/NAK delivery guarantees. |
| **`clickhouse`** | ClickHouse Server | High-performance columnar database hosting analytical tables and telemetry stats. |
| **`grafana`** | Grafana 11+ | Real-time monitoring and exploratory dashboard with pre-provisioned data sources and dashboard JSON. |
| **`escalation`** | Python | Listens for high-virality breaking news claims (`prob >= 0.85`) for autonomous downstream research agents. |
| **`distillation`** | Python / Gemini 1.5 | Samples boundary edge-cases (low Laya confidence) for synthetic gold labeling and model fine-tuning. |

---

## Laya Declarative Inference Schema

The local Laya System One daemon evaluates posts using structured zero-shot primitives (`choice`, `score`, and `noul`). In [`inference.py`](file:///home/vinayb/CodeProjects/laya-fine-tune/inference.py), the worker dispatches micro-batches to the local `/v1/systemone/batch` endpoint:

```python
questions = {
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
```

---

## ClickHouse Data Schemas

The pipeline provisions three primary tables in the `bluesky` database:

### 1. `bluesky.enriched_posts`
```sql
CREATE TABLE IF NOT EXISTS bluesky.enriched_posts (
    id String,
    created_at DateTime,
    text String,
    author_did String,
    hashtags Array(String),
    cashtags Array(String),
    category String,
    sentiment String,
    virality Float32,
    is_breaking_claim Float32,
    geo_region LowCardinality(String),
    primary_lang LowCardinality(String),
    is_nsfw Float32 DEFAULT 0.0
) ENGINE = MergeTree()
ORDER BY (created_at, category, sentiment);
```

### 2. `bluesky.funnel_stats`
Tracks volume and drop-off counts across 3-second rolling windows for exact pipeline funnel accounting:
* `timestamp`, `window_seconds`, `raw_received`, `dropped_replies`, `dropped_non_en`, `dropped_spam_nsfw`, `passed_sieve`, `raw_rate`, `passed_rate`.

### 3. `bluesky.queue_stats`
Monitors consumer lag and message states across all NATS JetStream workers sampled every 3 seconds:
* `timestamp`, `consumer`, `backlog` (pending), `in_flight` (processing), `redelivered` (failures/stalls).

---

## Quick Start

### 1. Prerequisites
* Docker & Docker Compose v2+
* NVIDIA GPU with NVIDIA Container Toolkit (recommended for CUDA acceleration, CPU fallback supported)
* (Optional) Gemini API key if enabling the distillation fine-tuning loop

### 2. Environment Configuration
Copy the sample environment file:
```bash
cp .env.example .env
```
Ensure your ports and Laya settings match your host:
```env
# Laya Inference Engine
SYSTEM_ONE_URL=http://laya-serve:8111
LAYA_DEVICE=cuda
LAYA_CUDA_AMP=fp16

# ClickHouse Credentials
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=bluesky_secret
CLICKHOUSE_DATABASE=bluesky

# Grafana
GF_SECURITY_ADMIN_PASSWORD=admin
```

### 3. Launch the Stack
Build and start all services in detached mode:
```bash
docker compose up -d --build
```

### 4. Access the Live Dashboard
1. Open your browser to [http://localhost:3000](http://localhost:3000)
2. Log in with `admin` / `admin` (or the password configured in `.env`).
3. Open the **Bluesky Live Stream Analytics** dashboard (`bluesky_stream_01`).
4. Watch real-time events, sentiment curves, trending cashtags, and worker queues refresh live!

---

## Verification & Operational CLI Tools

### Check Pipeline Logs
```bash
docker compose logs -f --tail=20 ingest sink inference
```

### Check NATS Queue Status via Script
Run the built-in diagnostic CLI tool:
```bash
python scripts/nats_stats.py
```

### Query ClickHouse Directly
```bash
docker compose exec -it clickhouse clickhouse-client -u default --password bluesky_secret --database bluesky
```

Example analytical queries:
```sql
-- Top 10 Stock & Crypto Cashtags in the last hour
SELECT cashtag, count(*) AS mentions
FROM bluesky.enriched_posts
ARRAY JOIN cashtags AS cashtag
WHERE created_at >= now() - INTERVAL 1 HOUR
GROUP BY cashtag
ORDER BY mentions DESC
LIMIT 10;

-- Live Pipeline Funnel Yield
SELECT
    sum(raw_received) AS total_raw,
    sum(passed_sieve) AS total_passed,
    round(100.0 * sum(passed_sieve) / sum(raw_received), 1) AS yield_pct
FROM bluesky.funnel_stats
WHERE timestamp >= now() - INTERVAL 15 MINUTE;
```

---

## Documentation Index

* [`docs/blog_realtime_bluesky_laya.md`](docs/blog_realtime_bluesky_laya.md): In-depth technical blog post detailing the architecture, pipeline stages, filters, and production insights.
* [`docs/lessons_learned_stream_analytics.md`](docs/lessons_learned_stream_analytics.md): Comprehensive architectural war stories (anti-spoofing heuristics, Sybil botnet defenses, Grafana subquery alignment, and worker observability).
* [`docs/architecture_and_pipeline_plan.md`](docs/architecture_and_pipeline_plan.md): Original system design specification and technical requirements.

---

## License
Licensed under the [Apache License 2.0](LICENSE).
