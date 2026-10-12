# Real-Time Stream Analytics with Laya: Master System Architecture & Production Specification

## 1. Executive Summary & System Vision

This document serves as the master technical specification and single source of truth for the real-time stream analytics system.

The system ingests high-velocity natural language events from public WebSocket firehoses (specifically the **Bluesky Jetstream** feed at 50–200+ msg/sec), filters out low-entropy noise and adversarial spam at microsecond wire speeds, and performs rich, multi-dimensional semantic analysis using **Laya**, a non-autoregressive multilingual System One decision engine.

To ensure resilience against micro-outages and to enable efficient micro-batching, the pipeline utilizes **NATS JetStream** as an elastic, memory-backed shock absorber. All enriched data is indexed into **ClickHouse** and surfaced via an 18-panel **Grafana** dashboard with live telemetry and interactive multi-dimensional drilldowns.

---

## 2. End-to-End System Architecture

```text
[WebSocket Firehose (Bluesky Jetstream ~50 ops/s)]
                   │  Raw JSON events
                   ▼
┌──────────────────────────────────────────────────────────┐
│      1. Ingestion Service (ingest.py)                    │
│  • Async WebSocket client with auto-reconnect            │
│  • Tier 1 Deterministic Sieve (<0.01ms CPU):             │
│    - Root post extraction (drops replies)                │
│    - Non-English metadata check                          │
│    - Adversarial script check (Kana regex anti-spoof)    │
│    - Blocked tags (#nsfw, #dlsite, #r18) & domains       │
│    - AT Protocol adult/warning label inspection          │
│    - Regex-based Cashtag extraction ($TICKER)            │
│  • Pushes clean events to NATS (~10-15 msgs/s)           │
│  • Emits funnel_stats & queue_stats to ClickHouse        │
└──────────────────────────┬───────────────────────────────┘
                           │  Publish to `stream.raw`
                           ▼
┌──────────────────────────────────────────────────────────┐
│      2. NATS JetStream Broker (Elastic Buffer)           │
│  • Memory-backed Ring Buffer (`stream.raw`)              │
│  • Short TTL (60s) & Max Message caps (10,000)           │
│  • Prevents data loss during downstream GPU/DB spikes    │
└──────────────────────────┬───────────────────────────────┘
                           │  Batch Pull (e.g., 32 items)
                           ▼
┌──────────────────────────────────────────────────────────┐
│      3. Inference Service (inference.py)                 │
│  • Pulls micro-batches from NATS JetStream               │
│  • Tier 2: Calls Laya Serve (/v1/systemone/batch):       │
│    - category (tech, finance, politics, etc.)            │
│    - sentiment (positive, neutral, negative)             │
│    - virality (0.0 to 1.0 probability)                   │
│    - is_breaking_claim (0.0 to 1.0 confidence)           │
│    - geo_region (north_america, europe, asia, etc.)      │
│    - is_nsfw (0.0 to 1.0 safety probability)             │
│  • Publishes enriched events to `stream.enriched`        │
└──────────────────────────┬───────────────────────────────┘
                           │  Publish to `stream.enriched`
                           ▼
┌──────────────────────────────────────────────────────────┐
│      4. Analytics Sink & Dispatcher (sink.py)            │
│  • Pulls batches of 100 from NATS                        │
│  • Inserts columnar records into ClickHouse              │
│  • Employs ACK on insert success, NAK on failure         │
└──────────────┬────────────────────────────┬──────────────┘
               │                            │ (Breaking Claims: is_breaking >= 0.85)
               ▼                            ▼
┌──────────────────────────────┐  ┌────────────────────────────────────┐
│      ClickHouse Database     │  │      Escalation Service            │
│  • bluesky.enriched_posts    │  │  • Subscribes to stream.enriched   │
│  • bluesky.funnel_stats      │  │  • Emits to stream.escalation      │
│  • bluesky.queue_stats       │  │  • Dispatches autonomous agents    │
└──────────────┬───────────────┘  └────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Grafana Dashboard       │
│  • 18 Live Visual Panels     │
│  • 5-Second Auto-Refresh     │
│  • 6 Multi-Dimensional Drill │
│  • 1-Click Baseline Reset    │
└──────────────────────────────┘
```

---

## 3. Two-Tier Filtering Architecture

### Tier 1: Deterministic Edge Sieve (`ingest.py`)
Executed at wire speed in pure Python before events enter the message bus:
1. **Reply Exclusion**: Evaluates `"reply" not in record` to focus on root conversational seeds and headlines.
2. **Language Gating**: Evaluates declared `langs` metadata for `"en"`.
3. **Adversarial Anti-Spoofing**: Inspects post text for Japanese Kana characters (`[\u3040-\u309F\u30A0-\u30FF]`). Catches automated botnets that spoof `langs: ["en"]` metadata.
4. **Signature Matching**: Discards known spam domains (`dlaf.jp`, `dlsite.com`) and adult tags (`#nsfw`, `#r18`, `#porn`, `#hentai`).
5. **Protocol Label Inspection**: Evaluates AT Protocol `labels` dictionary for adult, graphic, or sexual flags.
6. **Cashtag Extraction**: Regex `r'\$([A-Za-z]{1,6})\b'` extracts normalized financial and crypto symbols (`$NVDA`, `$BTC`).

### Tier 2: System One Semantic Inference (`inference.py`)
Executed by the local Laya server (`/v1/systemone/batch`) in micro-batches of 32. 

**Local Laya vs. Hosted Jev System Design Decision**:
Rather than depending on hosted System One APIs like TypeSafe AI's Jev or slow cloud LLMs, this architecture runs **Laya on-device** inside the local Docker environment (`vinaybalamuru/laya:cuda`). This eliminates external network transit latency, prevents cloud API egress costs, guarantees data privacy, and enables deterministic throughput during firehose spikes.

The single forward pass evaluates 6 concurrent criteria:
* **`category`** (Choice): `tech`, `finance`, `politics`, `entertainment`, `lifestyle`, `other`.
* **`sentiment`** (Choice): `positive`, `neutral`, `negative`.
* **`virality`** (Score): Float rating predicted viral spread.
* **`is_breaking_claim`** (Probability): 0.0 to 1.0 confidence that the text reports breaking, urgent, or unverified claims.
* **`geo_region`** (Choice): `north_america`, `europe`, `asia`, `middle_east`, `latin_america`, `africa`, `oceania`, `global_none`.
* **`is_nsfw`** (Probability): Semantic safety score identifying contextual toxicity or adult content that evades static blocklists.

---

## 4. ClickHouse Schema & In-Database Analytics

ClickHouse serves as the vectorized analytical engine hosting three core tables:

### 1. `bluesky.enriched_posts`
Stores individual classified social events with columnar array indexing:
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
    geo_region LowCardinality(String) DEFAULT 'global_none',
    primary_lang LowCardinality(String) DEFAULT 'en',
    is_nsfw Float32 DEFAULT 0.0
) ENGINE = MergeTree()
ORDER BY (created_at, category, sentiment);
```

### 2. `bluesky.funnel_stats`
Provides mathematical attribution for the ingestion sieve in 3-second windows:
* `timestamp`, `window_seconds`, `raw_received`, `dropped_replies`, `dropped_non_en`, `dropped_spam_nsfw`, `passed_sieve`, `raw_rate`, `passed_rate`.

### 3. `bluesky.queue_stats`
Logs NATS JetStream consumer lag every 3 seconds for real-time queue health:
* `timestamp`, `consumer`, `backlog`, `in_flight`, `redelivered`.

### 4. Vectorized Stopword Exclusion & Keyword Tokenization
Word clouds are generated dynamically in ClickHouse using native array functions without Python or external NLP overhead:
```sql
SELECT
    word,
    count() AS freq
FROM bluesky.enriched_posts
ARRAY JOIN arrayFilter(
    w -> length(w) >= 3 
         AND match(w, '^[a-z]+$') 
         AND w NOT IN ('the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'just', 'like', 'https', 'http', 'com', 'www', 'bsky'),
    tokens(lower(text))
) AS word
WHERE $__timeFilter(created_at)
  AND is_nsfw < 0.5
  AND ${category_filter}
GROUP BY word
ORDER BY freq DESC
LIMIT 50;
```

---

## 5. Grafana Dashboard Specification (`bluesky_stream_01`)

The dashboard features 18 real-time panels organized into functional operational zones:

1. **KPI Stat Cards**: Raw Firehose Total, Pipeline Yield %, Ingestion Speed (ops/s), Breaking Claims Alerts.
2. **Velocity & Funnel Audit**: 3-series throughput speed comparison (Firehose vs. Sieve vs. Final Output) and 7-stage sequential attribution table.
3. **Queue Health & Worker Lag**: Real-time worker health status table (`HEALTHY`, `MODERATE`, `OVERWORKED`, `STALLED`) and consumer lag curves.
4. **Topic & Sentiment Distribution**: Trending hashtags bar chart, 100% stacked sentiment breakdown (Blue/Yellow/Red), and viral hashtags table.
5. **Audience & Geographic Demographics**: Most active posters table (with bot activity tiering), regional mentions bar chart, category share of voice donut chart, and live category volume curves.
6. **Financial & Emerging Signals**: Breaking Claims radar table, Trending Stock & Crypto Cashtags (`$TICKER`), and Stopword-filtered Category Word Cloud.

### Template Filter Variables
* **`region_filter`**: Custom geographic region selector.
* **`category_filter`**: Topic category selector.
* **`sentiment_filter`**: Sentiment selector (Positive, Neutral, Negative).
* **`top_hashtag`**: Dynamic query variable loading the top 50 trending tags from the last 2 hours with search-as-you-type support.
* **`custom_hashtag`**: Free-text textbox matching arbitrary hashtags and post keywords.
* **`bot_filter`**: Account posting activity tier (`All Accounts`, `Humans <= 15 posts/hr`, `Casual <= 5 posts/hr`, `Automated Feeds > 15 posts/hr`).
* **`↺ Reset Filters to Default`**: Header toolbar link restoring all variables to default in one click.

---

## 6. Feedback & Active Learning Loops

* **Phase 4 (Escalation)**: Posts with `is_breaking_claim >= 0.85` and `virality >= 0.70` are routed to `stream.escalation` for autonomous multi-agent deep research.
* **Phase 5 (Distillation)**: A background worker samples ambiguous posts where Laya prediction confidence is between 0.4 and 0.6, obtains gold-standard classifications from a frontier teacher model (Gemini 1.5), and writes to a persistent dataset volume for downstream fine-tuning.
