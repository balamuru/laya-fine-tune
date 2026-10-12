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
SIEVE_EN_ONLY = os.getenv("SIEVE_EN_ONLY", "true").lower() == "true"


import clickhouse_connect
from datetime import datetime, timezone
import time

CH_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "default")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CH_DB = os.getenv("CLICKHOUSE_DATABASE", "bluesky")


def setup_clickhouse():
    """Connect to ClickHouse and ensure funnel_stats table exists."""
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD
        )
        client.command(f"CREATE DATABASE IF NOT EXISTS {CH_DB}")
        table_schema = f"""
        CREATE TABLE IF NOT EXISTS {CH_DB}.funnel_stats (
            timestamp DateTime,
            window_seconds Float32,
            raw_received UInt32,
            dropped_replies UInt32,
            dropped_non_en UInt32,
            passed_sieve UInt32,
            raw_rate Float32,
            passed_rate Float32
        ) ENGINE = MergeTree()
        ORDER BY timestamp
        """
        client.command(table_schema)
        print("ClickHouse funnel_stats verified.")
        return client
    except Exception as e:
        print(f"Warning: Could not connect to ClickHouse for funnel stats: {e}")
        return None


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
        f"Raw: {metrics.raw_received:4d} ({metrics.raw_rate_per_sec:5.1f}/s) | "
        f"Replies Drop: {metrics.dropped_replies:3d} | "
        f"Non-EN Drop: {metrics.dropped_non_en:3d} | "
        f"Passed: {metrics.passed_sieve:3d} ({metrics.passed_rate_per_sec:5.1f}/s)"
    )
    if metrics.latest_text_preview:
        print(f"   ↳ Latest: \"{metrics.latest_text_preview}\"")


import time

async def run_stream(stop_event: asyncio.Event, analytics: StreamAnalytics, nc, js, ch_client) -> None:
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
                                analytics.record_dropped_reply()
                                continue # Skip replies
                                
                            langs = record.get("langs", [])
                            if SIEVE_EN_ONLY:
                                if not langs or "en" not in langs:
                                    analytics.record_dropped_non_en()
                                    continue # Skip non-English posts
                            
                            analytics.record_passed_sieve()

                            primary_lang = langs[0] if (langs and isinstance(langs, list)) else "en"
                            record["primary_lang"] = primary_lang

                            # Extract hashtags (FR2.2)
                            text = record.get("text", "")
                            hashtags = [word for word in text.split() if word.startswith("#")]
                            record["extracted_hashtags"] = hashtags
                            
                            # Publish to NATS JetStream
                            payload = json.dumps(parsed_data).encode("utf-8")
                            await js.publish(JETSTREAM_RAW_SUBJECT, payload)
                        
                        metrics = analytics.check_window()
                        if metrics:
                            display_metrics(metrics)
                            if ch_client:
                                row = [
                                    datetime.now(timezone.utc),
                                    metrics.window_duration,
                                    metrics.raw_received,
                                    metrics.dropped_replies,
                                    metrics.dropped_non_en,
                                    metrics.passed_sieve,
                                    metrics.raw_rate_per_sec,
                                    metrics.passed_rate_per_sec,
                                ]
                                try:
                                    await asyncio.to_thread(
                                        ch_client.insert,
                                        f"{CH_DB}.funnel_stats",
                                        [row],
                                        column_names=[
                                            "timestamp", "window_seconds", "raw_received", "dropped_replies",
                                            "dropped_non_en", "passed_sieve", "raw_rate", "passed_rate"
                                        ]
                                    )
                                except Exception as err:
                                    print(f"Error logging funnel stats to ClickHouse: {err}")
                            
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

    ch_client = setup_clickhouse()
    analytics = StreamAnalytics(window_seconds=3.0)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: stop_event.set())
        except NotImplementedError:
            pass

    try:
        await run_stream(stop_event, analytics, nc, js, ch_client)
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
