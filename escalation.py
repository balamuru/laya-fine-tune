#!/usr/bin/env python3
"""
Escalation Worker (Phase 4):
Subscribes to stream.enriched and listens for outlier posts.
If a post exceeds certain thresholds (e.g., is_breaking_claim > 0.95),
it publishes it to a stream.escalation queue for heavy LangGraph agents to pick up.
"""

import asyncio
import json
import os
import signal
from dotenv import load_dotenv
import nats
from nats.errors import TimeoutError as NatsTimeoutError

load_dotenv()

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
ENRICHED_SUBJECT = os.getenv("JETSTREAM_ENRICHED_SUBJECT", "stream.enriched")
ESCALATION_SUBJECT = os.getenv("JETSTREAM_ESCALATION_SUBJECT", "stream.escalation")
THRESHOLD_BREAKING = float(os.getenv("TRIGGER_THRESHOLD_BREAKING", "0.95"))
BATCH_SIZE = 50

async def setup_nats():
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()
    
    stream_name = ESCALATION_SUBJECT.split('.')[0]
    try:
        await js.add_stream(name=stream_name, subjects=[f"{stream_name}.*"], max_msgs=5000, max_age=3600)
    except Exception:
        pass
    return nc, js

async def process_messages(js, messages):
    if not messages:
        return

    escalations = []
    for msg in messages:
        try:
            payload = json.loads(msg.data.decode())
            laya_answers = payload.get("laya", {}).get("answers", {})
            
            # Check threshold
            breaking_score = laya_answers.get("is_breaking_claim", {}).get("noul", 0.0)
            
            if breaking_score >= THRESHOLD_BREAKING:
                # Construct Trigger Schema
                trigger_payload = {
                    "trigger_type": "breaking_claim",
                    "confidence": breaking_score,
                    "original_payload": payload,
                    "status": "pending_langgraph_review"
                }
                escalations.append(trigger_payload)
                
            await msg.ack()
        except Exception as e:
            print(f"Error processing message: {e}")

    for item in escalations:
        try:
            await js.publish(ESCALATION_SUBJECT, json.dumps(item).encode())
            print(f"🔥 Escalated post (Score: {item['confidence']:.2f}) -> {ESCALATION_SUBJECT}")
        except Exception as e:
            print(f"Error publishing to escalation stream: {e}")

async def run_escalation(stop_event):
    nc, js = await setup_nats()
    stream_name = ENRICHED_SUBJECT.split('.')[0]
    
    sub = await js.pull_subscribe(ENRICHED_SUBJECT, "escalation_worker", stream=stream_name)
    
    print(f"Escalation worker started. Listening to {ENRICHED_SUBJECT} for breaking claims (>{THRESHOLD_BREAKING})...")
    
    while not stop_event.is_set():
        try:
            messages = await sub.fetch(BATCH_SIZE, timeout=1.0)
            await process_messages(js, messages)
        except NatsTimeoutError:
            continue
        except Exception as e:
            print(f"Error: {e}")
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

    await run_escalation(stop_event)

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
