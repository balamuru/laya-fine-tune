# Lessons Learned: Real-Time Stream Analytics & AI Filtering

This document records architectural, data engineering, and dashboard lessons learned while building the Bluesky real-time firehose analytics and Laya System One AI enrichment pipeline.

---

## 1. Firehose Metadata Spoofing & Adversarial Bypass
* **Symptom:** Automated affiliate spambots (such as `#DLsite` adult manga/audio bots) dominated the stream despite strict English-only filtering (`langs: ["en"]`).
* **Root Cause:** Spammers deliberately tag their AT Protocol post metadata with `langs: ["en"]` to game Bluesky algorithmic feeds and bypass language sieves, even though 100% of the actual post text is Japanese.
* **Lesson & Fix:** Never rely solely on client-declared metadata for language filtering. Implementing a fast deterministic script consistency check at ingestion (e.g. detecting Japanese Kana `[\u3040-\u309F\u30A0-\u30FF]`) drops 100% of these spoofed posts in microseconds before they consume any downstream queue or AI resources.

---

## 2. Distributed Sybil Networks vs. Single-Account Rate Limits
* **Symptom:** Naive bot-detection filters (e.g., `post_count > 15 posts/hour`) completely missed the `#DLsite` spam wave.
* **Root Cause:** The campaign operated as a distributed botnet across 38+ distinct Decentralized Identifiers (DIDs), each publishing only ~8–12 posts per hour.
* **Lesson & Fix:** Per-account rate limiting is insufficient against coordinated spam rings. Effective defenses require:
  1. Content signature matching (affiliate domains like `dlaf.jp`, explicit spam tags).
  2. Protocol-level label inspection (AT Protocol `labels` dictionary for adult/warning flags).
  3. Pre-inference deterministic dropping at the ingestion sieve.

---

## 3. Two-Stage Filtering Architecture: Deterministic Sieve vs. Laya AI Criteria
* **Design Decision:** Should content moderation happen deterministically or via AI?
* **Trade-off:**
  * **Pre-AI Deterministic Sieve (`ingest.py`):**
    * Checks known blocked tags (`#nsfw`, `#dlsite`, `#r18`), affiliate domains, script mismatches, and AT Protocol self-labels.
    * **Cost:** Zero inference latency, zero GPU/CPU cost.
    * **Impact:** Protects the NATS queue, inference workers, and ClickHouse storage from obvious garbage.
  * **Post-AI Laya System One Criterion (`inference.py`):**
    * Evaluates subtle, contextual, or English-language content (`is_nsfw` probability).
    * **Cost:** Small compute footprint per passed post.
    * **Impact:** Flags borderline content that evades keyword blocklists without false-positive keyword bans.
* **Takeaway:** Always employ a **hybrid architecture**—use deterministic sieves to filter out the bulk of obvious spam, and reserve System One AI inference for semantic, nuanced judgments.

---

## 4. Grafana Cross-Panel Query Alignment
* **Symptom:** The "Top 10 Trending Hashtags" panel did not match the hashtags shown in the adjacent "Sentiment Breakdown per Top 10 Hashtags" panel.
* **Root Cause:**
  1. Panel A ranked hashtags by raw volume (`count(*)`).
  2. Panel B independently ranked hashtags by enriched sentiment volume (`Positive + Neutral + Negative`), which excluded posts ingested before sentiment enrichment, causing different sets of hashtags to be selected.
* **Lesson & Fix:** When two panels represent different dimensions of the same top-$N$ entity, never rank them independently in separate queries. Use an explicit SQL subquery in Panel B:
  ```sql
  WHERE hashtag IN (
    SELECT arrayJoin(hashtags) AS h
    FROM bluesky.enriched_posts
    WHERE $__timeFilter(created_at) AND h != ''
    GROUP BY h
    ORDER BY count(*) DESC
    LIMIT 10
  )
  ```
  This guarantees 100% alignment across both visualizations.

---

## 5. Grafana Bar Chart Orientation & Multi-Series Pivoting
* **Horizontal Bar Charts:** In Grafana `barchart` panels, horizontal bar orientation (placing string labels on the vertical Y-axis and numeric values on the horizontal X-axis) is configured via:
  ```json
  "options": {
    "orientation": "horizontal",
    "xField": "dimension_name"
  }
  ```
  Placing `orientation` inside `fieldConfig.defaults.custom` is ignored by Grafana.
* **Multi-Category Time-Series:** Grafana ClickHouse datasource does not automatically pivot rows into multiple colored curves when returning `(time, category, count)`. To produce distinct, multi-series curves with shared tooltips, the SQL query must explicitly pivot categories using `countIf(...)`:
  ```sql
  SELECT
    toStartOfInterval(created_at, INTERVAL 15 SECOND) AS time,
    countIf(category = 'tech') AS Tech,
    countIf(category = 'entertainment') AS Entertainment,
    ...
  FROM bluesky.enriched_posts
  GROUP BY time
  ORDER BY time
  ```
* **Color Overrides for Stacked Bar Charts:** Rather than relying on default palette cycling, semantic metrics (like Sentiment) require explicit color overrides mapped by field name:
  ```json
  "overrides": [
    { "matcher": { "id": "byName", "options": "Positive" }, "properties": [{ "id": "color", "value": { "mode": "fixed", "fixedColor": "green" } }] },
    { "matcher": { "id": "byName", "options": "Neutral" }, "properties": [{ "id": "color", "value": { "mode": "fixed", "fixedColor": "#8e8e8e" } }] },
    { "matcher": { "id": "byName", "options": "Negative" }, "properties": [{ "id": "color", "value": { "mode": "fixed", "fixedColor": "red" } }] }
  ]
  ```

---

## 6. Pipeline Funnel Accounting & Transparency
* **Symptom:** Presenting drop stats across disconnected cards and pie charts caused confusion over whether numbers were relative or absolute.
* **Lesson & Fix:** A pipeline funnel must reflect exact sequential attribution where each stage is traceable:
  $$\text{Raw Firehose Ingested (100\%)} = \text{Replies Drop} + \text{Non-EN Drop} + \text{Spam/NSFW Drop} + \text{Passed Sieve}$$
  Representing this as an ordered, stage-by-stage audit table showing `Stage`, `Volume`, `Relative Share`, and `Pipeline Meaning` eliminates all ambiguity for operators.

---

## 7. Pipeline Worker Telemetry & Consumer Lag Monitoring
* **Challenge:** Monitoring whether individual downstream stages (Laya AI GPU micro-batches vs. ClickHouse batch inserts vs. alert escalation) are keeping up with stream velocity or silently accumulating consumer lag.
* **Architecture:** Rather than running separate heavy monitoring daemons, the ingestion worker periodically samples JetStream consumer metadata (`js.consumers_info("stream")`) and writes micro-rows to ClickHouse table `bluesky.queue_stats` every 3 seconds:
  * `backlog` (`num_pending`): Unprocessed messages waiting in queue.
  * `in_flight` (`num_ack_pending`): Messages currently being evaluated by workers.
  * `redelivered` (`num_redelivered`): Failed or timed-out deliveries.
* **Dashboard Representation:**
  1. **Live Queue Health Table:** Dynamic health categorization:
     * `HEALTHY` (backlog <= 50, redeliveries = 0)
     * `MODERATE` (50 < backlog <= 500)
     * `OVERWORKED` (backlog > 500)
     * `STALLED / ERROR` (redeliveries > 50)
  2. **Consumer Backlog Over Time (TimeSeries):** Direct multi-series visibility into queue growth, enabling instant detection of GPU or database bottlenecks during firehose spikes.

