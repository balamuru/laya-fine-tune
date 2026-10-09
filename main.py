#!/usr/bin/env python3
"""Real-time stream analytics consumer for Bluesky Jetstream firehose."""

import asyncio
import signal
import time
import websockets

from stream_analytics import StreamAnalytics, WindowMetrics

JETSTREAM_URL = (
    "wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
)


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


async def run_stream(stop_event: asyncio.Event, analytics: StreamAnalytics) -> None:
    print(f"Connecting to live feed: {JETSTREAM_URL}")
    print("Press Ctrl+C to terminate cleanly.\n")

    backoff = 1.0
    while not stop_event.is_set():
        try:
            async with websockets.connect(
                JETSTREAM_URL,
                ping_interval=20,
                ping_timeout=20,
                max_queue=256,
            ) as ws:
                backoff = 1.0
                print("Connected! Streaming real-time events...")
                while not stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        analytics.process_raw(message)
                        metrics = analytics.check_window()
                        if metrics:
                            display_metrics(metrics)
                    except asyncio.TimeoutError:
                        continue
        except (websockets.ConnectionClosed, OSError) as exc:
            if stop_event.is_set():
                break
            print(f"Connection lost ({exc}). Reconnecting in {backoff:.1f}s...")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2.0, 30.0)


async def async_main() -> None:
    analytics = StreamAnalytics(window_seconds=3.0)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: stop_event.set())
        except NotImplementedError:
            pass

    try:
        await run_stream(stop_event, analytics)
    finally:
        print("\nShutdown complete.")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
