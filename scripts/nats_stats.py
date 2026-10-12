#!/usr/bin/env python3
"""NATS JetStream topic & consumer health monitoring utility."""

import asyncio
import os
import sys
import nats


async def print_nats_stats():
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    # If running inside docker, default to nats://nats:4222
    if os.path.exists("/.dockerenv"):
        nats_url = os.getenv("NATS_URL", "nats://nats:4222")

    try:
        nc = await nats.connect(nats_url, connect_timeout=3)
        js = nc.jetstream()
    except Exception as e:
        print(f"Error connecting to NATS at {nats_url}: {e}")
        sys.exit(1)

    print("\n" + "=" * 78)
    print("                 NATS JETSTREAM QUEUE & CONSUMER HEALTH AUDIT")
    print("=" * 78)

    streams = await js.streams_info()
    if not streams:
        print("No JetStream streams found.")
        await nc.close()
        return

    for s in streams:
        cfg = s.config
        state = s.state
        print(f"\n📂 Stream Name : {cfg.name}")
        print(f"   Subjects    : {', '.join(cfg.subjects)}")
        print(f"   Retention   : {cfg.retention} | Storage: {cfg.storage}")
        print(f"   Messages    : {state.messages:,} msgs ({state.bytes / 1024 / 1024:.2f} MB)")
        print(f"   First Msg   : Seq #{state.first_seq} | Last Msg: Seq #{state.last_seq}")
        print(f"   Consumers   : {state.consumer_count}")
        print("\n   {:<22} {:<12} {:<12} {:<12} {:<14}".format(
            "Consumer Name", "Backlog", "In-Flight", "Redelivered", "Queue Health"
        ))
        print("   " + "-" * 74)

        consumers = await js.consumers_info(cfg.name)
        if not consumers:
            print("   (No active consumers attached to this stream)")

        for c in consumers:
            pending = c.num_pending
            ack_pending = c.num_ack_pending
            redelivered = c.num_redelivered

            # Health classification
            if redelivered > 50:
                health = "⚠️ STALLED/ERR"
            elif pending > 500:
                health = "🐢 OVERWORKED"
            elif pending > 50:
                health = "⚡ MODERATE"
            else:
                health = "✅ HEALTHY"

            print("   {:<22} {:<12,d} {:<12,d} {:<12,d} {:<14}".format(
                c.name, pending, ack_pending, redelivered, health
            ))

    print("\n" + "=" * 78 + "\n")
    await nc.close()


if __name__ == "__main__":
    asyncio.run(print_nats_stats())
