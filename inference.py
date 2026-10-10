#!/usr/bin/env python3
"""Inference worker: Pulls batches from NATS, queries Laya, and publishes enriched results."""

import asyncio
import json
import os
import signal
import aiohttp
from dotenv import load_dotenv

import nats
from nats.errors import TimeoutError as NatsTimeoutError

load_dotenv()

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
RAW_SUBJECT = os.getenv("JETSTREAM_RAW_SUBJECT", "stream.raw")
ENRICHED_SUBJECT = os.getenv("JETSTREAM_ENRICHED_SUBJECT", "stream.enriched")
LAYA_SERVE_URL = os.getenv("LAYA_SERVE_URL", "http://localhost:8111")
LAYA_BATCH_SIZE = int(os.getenv("LAYA_BATCH_SIZE", "32"))
LAYA_API_KEY = os.getenv("LAYA_API_KEY", "")


async def setup_nats():
    """Connect to NATS and ensure the enriched stream exists."""
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()
    
    stream_name = ENRICHED_SUBJECT.split('.')[0]
    try:
        await js.add_stream(name=stream_name, subjects=[f"{stream_name}.*"], max_msgs=10000, max_age=3600)
    except Exception:
        pass # Already exists
    return nc, js


async def process_batch(session, js, messages):
    """Send a batch to Laya, then publish the enriched output."""
    if not messages:
        return

    # Extract text from raw Bluesky JSON into Laya state objects
    states = []
    for msg in messages:
        try:
            data = json.loads(msg.data.decode())
            text = data.get("commit", {}).get("record", {}).get("text", "")
            states.append({"body": text})
        except Exception:
            states.append({"body": ""})
            
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
        }
    }
    
    headers = {"Content-Type": "application/json"}
    if LAYA_API_KEY:
        headers["Authorization"] = f"Bearer {LAYA_API_KEY}"
        
    payload = {"states": states, "questions": questions}
        
    # Send micro-batch to Laya
    try:
        async with session.post(f"{LAYA_SERVE_URL}/v1/systemone/batch", json=payload, headers=headers) as resp:
            if resp.status == 200:
                results = await resp.json()
                
                # Zip the original messages with the Laya predictions
                for msg, laya_result in zip(messages, results.get("results", [])):
                    try:
                        data = json.loads(msg.data.decode())
                        # Enrich the payload with Laya's semantic insights
                        data["laya"] = laya_result
                        
                        # Publish to stream.enriched
                        await js.publish(ENRICHED_SUBJECT, json.dumps(data).encode())
                        
                        # Acknowledge the message so NATS doesn't redeliver it
                        await msg.ack()
                    except Exception as e:
                        print(f"Error publishing enriched data: {e}")
                
                print(f"Successfully enriched and published batch of {len(messages)} posts.")
            else:
                error_text = await resp.text()
                print(f"Laya HTTP error {resp.status}: {error_text}")
                # NACK (Negative Acknowledge) so NATS redelivers them later
                for msg in messages:
                    await msg.nak()
    except Exception as e:
        print(f"Failed to call Laya Serve: {e}")
        for msg in messages:
            await msg.nak()


async def run_inference(stop_event):
    nc, js = await setup_nats()
    
    # We use a pull subscription on stream.raw to manually control batch sizes
    stream_raw_name = RAW_SUBJECT.split('.')[0]
    
    print(f"Subscribing to NATS JetStream: {RAW_SUBJECT}")
    sub = await js.pull_subscribe(RAW_SUBJECT, "inference_worker", stream=stream_raw_name)
    
    print(f"Inference worker started. Pulling batches of {LAYA_BATCH_SIZE}...")
    
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                # Fetch up to LAYA_BATCH_SIZE messages at once
                messages = await sub.fetch(LAYA_BATCH_SIZE, timeout=1.0)
                await process_batch(session, js, messages)
            except NatsTimeoutError:
                # Normal timeout if no new messages arrived in the last 1.0 second
                continue
            except Exception as e:
                print(f"Error pulling from NATS: {e}")
                await asyncio.sleep(1)

    if nc and not nc.is_closed:
        print("Draining NATS connection...")
        await nc.drain()


async def async_main():
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: stop_event.set())
        except NotImplementedError:
            pass

    try:
        await run_inference(stop_event)
    finally:
        print("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
