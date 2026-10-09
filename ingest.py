#!/usr/bin/env python3
"""Ingestion worker: Reads from Bluesky Jetstream, filters, and publishes to NATS JetStream."""

import asyncio
import json
import os
import signal
import websockets
from dotenv import load_dotenv

import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError
from nats.js.api import StreamConfig

from stream_analytics import StreamAnalytics, WindowMetrics

load_dotenv()

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
JETSTREAM_RAW_SUBJECT = os.getenv("JETSTREAM_RAW_SUBJECT", "stream.raw")
JETSTREAM_WS_URL = os.getenv("JETSTREAM_WS_URL", "wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post")
SIEVE_ROOT_POSTS_ONLY = os.getenv("SIEVE_ROOT_POSTS_ONLY", "true").lower() == "true"


def is_root_post(record: dict) -> bool:
    """Check if a post is a root post (not a reply)."""
    return "reply" not in record


async def setup_nats():
    """Connect to NATS and ensure JetStream stream exists."""
    print(f"Connecting to NATS at {NATS_URL}...")
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()

    stream_name = JETSTREAM_RAW_SUBJECT.split('.')[0]
    
    try:
        # Try to add the stream. If it exists, this will update it or no-op if identical.
        await js.add_stream(name=stream_name, subjects=[f"{stream_name}.*"], max_msgs=10000, max_age=60)
        print(f"JetStream '{stream_name}' configured successfully.")
    except Exception as e:
        print(f"Warning setting up JetStream (it might already exist): {e}")

    return nc, js


def display_metrics(metrics: WindowMetrics) -> None:
    """Format and print windowed metrics."""
    top_langs = ", ".join(f"{lang}:{count}" for lang, count in metrics.top_languages)
    print(
        f"[{time.strftime('%H:%M:%S')}] "
        f"Total: {metrics.total_events:6d} | "
        f"Rate: {metrics.rate_per_sec:5.1f} msg/s | "
        f"Top Langs: [{top_langs}]"
    )
    if metrics.latest_text_preview:
        print(f"   ↳ Latest: \"{metrics.latest_text_preview}\"")


import time

async def run_stream(stop_event: asyncio.Event, analytics: StreamAnalytics, nc, js) -> None:
    print(f"Connecting to live feed: {JETSTREAM_WS_URL}")
    print("Press Ctrl+C to terminate cleanly.\n")

    backoff = 1.0
    while not stop_event.is_set():
        try:
            async with websockets.connect(
                JETSTREAM_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_queue=256,
            ) as ws:
                backoff = 1.0
                print("Connected to Bluesky! Streaming real-time events...")
                while not stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        
                        # Use our analytics parser
                        parsed_data = analytics.process_raw(message)
                        
                        if parsed_data:
                            # Apply Stage 2 Sieve logic
                            record = parsed_data.get("commit", {}).get("record", {})
                            if SIEVE_ROOT_POSTS_ONLY and not is_root_post(record):
                                continue # Skip replies
                            
                            # Publish to NATS JetStream
                            payload = json.dumps(parsed_data).encode("utf-8")
                            await js.publish(JETSTREAM_RAW_SUBJECT, payload)
                        
                        metrics = analytics.check_window()
                        if metrics:
                            display_metrics(metrics)
                            
                    except asyncio.TimeoutError:
                        continue
        except (websockets.ConnectionClosed, OSError) as exc:
            if stop_event.is_set():
                break
            print(f"WebSocket Connection lost ({exc}). Reconnecting in {backoff:.1f}s...")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2.0, 30.0)


async def async_main() -> None:
    nc = None
    try:
        nc, js = await setup_nats()
    except Exception as e:
        print(f"Failed to connect to NATS: {e}")
        return

    analytics = StreamAnalytics(window_seconds=3.0)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: stop_event.set())
        except NotImplementedError:
            pass

    try:
        await run_stream(stop_event, analytics, nc, js)
    finally:
        print("\nShutdown initiated...")
        if nc and not nc.is_closed:
            await nc.drain()
        print("Shutdown complete.")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
