# Architecture Decision Record (ADR): Serving Laya via HTTP (`laya serve`)

- **Status:** Accepted
- **Date:** 2026-10-09
- **Deciders:** Engineering Team
- **Related Documents:** [`docs/proposals.md`](proposals.md)

---

## 1. Context & Problem Statement

We are designing a real-time analytics pipeline to process high-velocity WebSocket streams (e.g., Bluesky Jetstream at 50–200+ msg/sec) using Laya as the System 1 decision engine. 

We must decide between two deployment topologies:
1. **Direct In-Process Embedding:** Running `laya.Router` directly inside the Python WebSocket consumer loop.
2. **Delegated HTTP Service:** Running `laya serve` as an independent service and calling `/v1/systemone/batch` over HTTP.

The core question is whether the operational benefits of an independent HTTP service justify the additional network latency and serialization overhead.

---

## 2. Decision: Adopt Delegated HTTP Service (`laya serve`)

**We agree that serving the model via HTTP is decisively worth the trade-off.**

We will decouple the stream ingestion client from the inference engine, running `laya serve` as a standalone daemon (or containerized service) and dispatching micro-batches via HTTP.

---

## 3. Justification & Why the Trade-Off Is Favorable

### 3.1. Elimination of Ingestion Backpressure & Socket Drops
Public WebSocket firehoses push events continuously. If the client fails to read from the socket in time, the TCP buffer fills and the server forcibly terminates the connection.
* In an **in-process** setup, any GPU stall, memory compaction, or PyTorch initialization blocks the event loop, causing immediate consumer lag and disconnects.
* In an **HTTP-decoupled** setup, the ingestion process only enqueues raw events to an in-memory buffer or message broker. It never blocks on model computation, ensuring 100% connection uptime.

### 3.2. Latency Penalty Amortizes to Negligible Levels over Batches
* Local loopback HTTP (`localhost` or container network) introduces approximately **1.5 to 3.5 ms** of round-trip latency.
* Because events are sent in micro-batches (e.g., 32 items per call), that 3 ms penalty is distributed across the entire batch:
  $$\frac{3\text{ ms overhead}}{32\text{ items}} \approx 0.09\text{ ms per event}$$
* Compared to the model's forward pass (~20–35 ms), a sub-millisecond per-item overhead is negligible for real-time streaming analytics.

### 3.3. Failure Isolation & Fault Tolerance
* Machine learning inference components are prone to out-of-memory (OOM) errors, driver crashes, or CUDA lockups.
* Running `laya serve` out-of-process ensures that a model crash does **not** crash the stream ingestion client. The client can retry with exponential backoff or buffer events locally while the model service restarts automatically.

### 3.4. Independent Horizontal Scalability
* Ingestion is I/O-bound (network WebSockets).
* Inference is compute-bound (GPU / SIMD CPU).
* Decoupling allows these two tiers to scale independently. If stream volume doubles, we can deploy additional `laya serve` instances behind a load balancer without modifying or restarting the ingestion code.

### 3.5. Seamless Model Checkpoint Lifecycle & Zero-Downtime Updates
* When swapping models, fine-tuning checkpoints, or updating routing rules, we can perform a rolling reload of `laya serve` without disconnecting the 24/7 stream ingestor.

---

## 4. Accepted Trade-offs & Mitigations

| Trade-off / Cost | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **HTTP & JSON Overhead** | ~2–4 ms per network request | Use persistent HTTP Keep-Alive connections (`httpx.AsyncClient(http2=True)` or `aiohttp`) to eliminate TCP/TLS handshake latency. |
| **Request Serialization** | CPU cycles for JSON encode/decode | Use fast JSON parsers (`orjson` or `msgspec`) in both ingestion and server. |
| **Inter-Process Latency** | Network stack overhead | In Linux environments, configure `laya serve` to bind to a **Unix Domain Socket (UDS)** rather than TCP loopback for zero-network-stack overhead if needed. |
| **Process Management** | Need to run two distinct services | Manage via `docker compose` or `systemd` with health check endpoints (`/health`). |

---

## 5. Implementation Guidelines

1. **Client Ingestion:**
   * Ingestion service maintains an in-memory queue (`asyncio.Queue`).
   * A dedicated batcher task flushes the queue every **50 ms** or when **32 items** accumulate.
   * Dispatches `POST /v1/systemone/batch` using a shared, persistent connection pool.
2. **Server Configuration:**
   * Launch `laya serve` with GPU support and preloaded checkpoints (`--preload` or appropriate workers).
   * Ensure dynamic batching parameters align with stream throughput.
