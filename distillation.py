#!/usr/bin/env python3
"""
Distillation Worker (Phase 5):
Subscribes to stream.enriched. When Laya's confidence is low, 
it samples the post and sends it to Gemini for a ground-truth label.
Saves the result to a JSONL dataset for future fine-tuning.
"""

import asyncio
import json
import os
import random
import signal
from dotenv import load_dotenv

import nats
from nats.errors import TimeoutError as NatsTimeoutError
from google import genai
from google.genai import types

load_dotenv()

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
ENRICHED_SUBJECT = os.getenv("JETSTREAM_ENRICHED_SUBJECT", "stream.enriched")
DISTILLATION_ENABLED = os.getenv("DISTILLATION_ENABLED", "false").lower() == "true"
CONFIDENCE_MIN = float(os.getenv("DISTILLATION_CONFIDENCE_MIN", "0.2"))
CONFIDENCE_MAX = float(os.getenv("DISTILLATION_CONFIDENCE_MAX", "0.6"))
SAMPLE_RATE = float(os.getenv("DISTILLATION_SAMPLE_RATE", "0.1"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DATASET_PATH = "/data/distillation_dataset.jsonl"
BATCH_SIZE = 50

async def setup_nats():
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()
    return nc, js

def get_gemini_label(text: str) -> dict:
    if not GEMINI_API_KEY:
        return {"error": "No Gemini API Key"}
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    Analyze the following social media post.
    Post: "{text}"
    
    Provide a JSON response with the following keys:
    - "category": (tech, politics, entertainment, finance, lifestyle, or other)
    - "sentiment": (positive, negative, neutral)
    - "virality": (0 for mundane, 1 for conversational, 2 for incendiary and highly viral)
    - "is_breaking_claim": (0.0 to 1.0 probability that it makes a verifiable factual claim)
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"error": str(e)}

async def process_messages(js, messages):
    if not messages:
        return

    for msg in messages:
        try:
            payload = json.loads(msg.data.decode())
            laya_answers = payload.get("laya", {}).get("answers", {})
            text = payload.get("commit", {}).get("record", {}).get("text", "")
            
            # Find the lowest confidence among Laya's answers to determine if it's an edge case
            confidences = [
                ans.get("answer_confidence", 1.0) 
                for key, ans in laya_answers.items() 
                if isinstance(ans, dict) and "answer_confidence" in ans
            ]
            min_conf = min(confidences) if confidences else 1.0
            
            if CONFIDENCE_MIN <= min_conf <= CONFIDENCE_MAX:
                if random.random() <= SAMPLE_RATE:
                    print(f"Sampling edge case (Confidence: {min_conf:.2f})...")
                    
                    # Call Gemini (synchronous call inside an executor to not block asyncio)
                    loop = asyncio.get_running_loop()
                    gold_label = await loop.run_in_executor(None, get_gemini_label, text)
                    
                    if "error" not in gold_label:
                        dataset_entry = {
                            "text": text,
                            "laya_predictions": laya_answers,
                            "gold_label": gold_label
                        }
                        
                        # Append to dataset
                        with open(DATASET_PATH, "a") as f:
                            f.write(json.dumps(dataset_entry) + "\n")
                        print("Saved to distillation dataset!")
                        
        except Exception as e:
            print(f"Error processing message: {e}")
            
        await msg.ack()

async def run_distillation(stop_event):
    if not DISTILLATION_ENABLED:
        print(f"Distillation is disabled in .env (DISTILLATION_ENABLED=false). Sleeping...")
        while not stop_event.is_set():
            await asyncio.sleep(1)
        return

    if not GEMINI_API_KEY:
        print("WARNING: DISTILLATION_ENABLED=true but GEMINI_API_KEY is missing. Dataset generation will fail.")

    nc, js = await setup_nats()
    stream_name = ENRICHED_SUBJECT.split('.')[0]
    sub = await js.pull_subscribe(ENRICHED_SUBJECT, "distillation_worker", stream=stream_name)
    
    print(f"Distillation worker started. Listening for edge cases (Conf: {CONFIDENCE_MIN}-{CONFIDENCE_MAX})...")
    
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

    await run_distillation(stop_event)

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
