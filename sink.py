#!/usr/bin/env python3
"""Sink worker: Pulls from NATS stream.enriched and batch inserts into ClickHouse."""

import asyncio
import json
import os
import signal
from dotenv import load_dotenv

import nats
from nats.errors import TimeoutError as NatsTimeoutError
import clickhouse_connect

load_dotenv()

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
ENRICHED_SUBJECT = os.getenv("JETSTREAM_ENRICHED_SUBJECT", "stream.enriched")
BATCH_SIZE = int(os.getenv("SINK_BATCH_SIZE", "100"))

CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "default")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CH_DB = os.getenv("CLICKHOUSE_DATABASE", "bluesky")


def setup_clickhouse():
    """Connect to ClickHouse and ensure database and table exist."""
    print(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD)
    
    # Create DB
    client.command(f"CREATE DATABASE IF NOT EXISTS {CH_DB}")
    
    # Create table for enriched posts
    table_schema = f"""
    CREATE TABLE IF NOT EXISTS {CH_DB}.enriched_posts (
        id String,
        created_at DateTime,
        text String,
        author_did String,
        hashtags Array(String),
        category String,
        sentiment String,
        virality Float32,
        is_breaking_claim Float32
    ) ENGINE = MergeTree()
    ORDER BY (created_at, category, sentiment)
    """
    client.command(table_schema)
    print("ClickHouse schema verified.")
    return client


async def process_batch(client, messages):
    if not messages:
        return

    data_rows = []
    for msg in messages:
        try:
            payload = json.loads(msg.data.decode())
            commit = payload.get("commit", {})
            record = commit.get("record", {})
            laya = payload.get("laya", {})
            
            # Map Laya output format (assuming choice/score/noul types output values)
            category = laya.get("category", {}).get("value", "unknown")
            sentiment = laya.get("sentiment", {}).get("value", "unknown")
            virality = float(laya.get("virality", {}).get("value", 0.0))
            is_breaking = float(laya.get("is_breaking_claim", {}).get("value", 0.0))
            
            from datetime import datetime, timezone
            
            # Handle timestamps (Bluesky timestamps are ISO8601 strings)
            created_at_str = record.get("createdAt", "1970-01-01T00:00:00Z").replace('Z', '+00:00')
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except ValueError:
                created_at = datetime.now(timezone.utc)
            
            row = (
                commit.get("rkey", ""),
                created_at,
                record.get("text", ""),
                payload.get("did", ""),
                record.get("extracted_hashtags", []),
                category,
                sentiment,
                virality,
                is_breaking
            )
            data_rows.append(row)
        except Exception as e:
            print(f"Parse error for ClickHouse: {e}")

    if data_rows:
        try:
            # Batch insert
            client.insert(f"{CH_DB}.enriched_posts", data_rows, column_names=[
                "id", "created_at", "text", "author_did", "hashtags", 
                "category", "sentiment", "virality", "is_breaking_claim"
            ])
            # ACK all if insert succeeds
            for msg in messages:
                await msg.ack()
            print(f"Inserted {len(data_rows)} rows into ClickHouse.")
        except Exception as e:
            print(f"ClickHouse insert failed: {e}")
            for msg in messages:
                await msg.nak()


async def run_sink(stop_event):
    # Setup ClickHouse (sync)
    ch_client = setup_clickhouse()
    
    # Setup NATS
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()
    
    stream_name = ENRICHED_SUBJECT.split('.')[0]
    sub = await js.pull_subscribe(ENRICHED_SUBJECT, "sink_worker", stream=stream_name)
    
    print(f"Sink worker started. Pulling batches of {BATCH_SIZE} from {ENRICHED_SUBJECT}...")
    
    while not stop_event.is_set():
        try:
            messages = await sub.fetch(BATCH_SIZE, timeout=1.0)
            await process_batch(ch_client, messages)
        except NatsTimeoutError:
            continue
        except Exception as e:
            print(f"Error pulling from NATS: {e}")
            await asyncio.sleep(1)

    if nc and not nc.is_closed:
        await nc.drain()


async def async_main():
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: stop_event.set())
        except NotImplementedError:
            pass

    await run_sink(stop_event)


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
