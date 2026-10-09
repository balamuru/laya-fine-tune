# Real-Time Stream Analytics

A Python project for streaming analytics and real-time event processing using the public Bluesky Jetstream firehose WebSocket feed.

## Overview

This project connects to public, unauthenticated real-time WebSocket feeds to demonstrate:
- High-throughput streaming ingestion
- Sliding window aggregations (message rates, counts)
- Real-time categorical distributions (language frequencies)
- Graceful connection recovery and backoff

## Project Structure

```text
.
├── .gitignore          # Git ignore rules for Python & venv
├── LICENSE             # Apache License 2.0
├── README.md           # Project documentation
├── pyproject.toml      # Project metadata and dependencies
├── requirements.txt    # Frozen pip dependencies
├── main.py             # Stream consumer & runtime coordinator
├── stream_analytics.py # StreamAnalytics processor & window aggregations
├── docs/
│   ├── proposals.md                     # Proposals for stream analytics & Laya System 1
│   ├── decisions.md                     # ADR for serving Laya via HTTP
│   └── filtering_and_analytics_pipeline.md # 4-Tier prefiltering & Laya decision schema
└── .venv/                               # Python virtual environment
```

## Quick Start

### 1. Activate the Virtual Environment

```bash
source .venv/bin/activate
```

Or run directly using the environment's python binary:

```bash
.venv/bin/python main.py
```

### 2. Run the Stream Processor

```bash
python main.py
```

Sample output:
```text
Connecting to live feed: wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post
Press Ctrl+C to terminate cleanly.

Connected! Streaming real-time events...
[13:52:20] Total:    133 | Rate:  44.0 msg/s | Top Langs: [en:93, unknown:14, pt:8, es:7, ar:3]
   ↳ Latest: "Hello world from the firehose..."
```

## Dependencies

- Python 3.10+
- [`websockets`](https://websockets.readthedocs.io/)

Install dependencies into your venv:
```bash
.venv/bin/pip install -r requirements.txt
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
