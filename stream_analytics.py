"""Stream analytics processor and windowed metric aggregations."""

from collections import Counter
from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional


@dataclass
class WindowMetrics:
    """Summary metrics computed for a time window."""

    window_duration: float
    total_events: int
    window_events: int
    raw_received: int
    dropped_replies: int
    dropped_non_en: int
    passed_sieve: int
    raw_rate_per_sec: float
    passed_rate_per_sec: float
    top_languages: List[tuple[str, int]]
    latest_text_preview: str = ""


class StreamAnalytics:
    """Processes stream messages and maintains rolling window analytics."""

    def __init__(self, window_seconds: float = 3.0) -> None:
        self.window_seconds = window_seconds
        self.total_events: int = 0
        self.window_events: int = 0
        self.window_raw: int = 0
        self.window_dropped_replies: int = 0
        self.window_dropped_non_en: int = 0
        self.window_passed_sieve: int = 0
        self.lang_counter: Counter[str] = Counter()
        self.last_window_start: float = time.time()
        self.latest_text: str = ""

    def process_raw(self, raw_message: str) -> Optional[Dict[str, Any]]:
        """Parse raw JSON message and update streaming counters.

        Returns:
            The parsed record payload if successfully parsed, or None.
        """
        try:
            data = json.loads(raw_message)
        except (json.JSONDecodeError, TypeError):
            return None

        commit = data.get("commit", {})
        record = commit.get("record", {})
        langs = record.get("langs") or ["unknown"]
        text = record.get("text", "").replace("\n", " ").strip()

        self.total_events += 1
        self.window_events += 1
        self.window_raw += 1
        self.latest_text = text

        for lang in langs:
            self.lang_counter[lang] += 1

        return data

    def record_dropped_reply(self) -> None:
        """Increment dropped reply counter for the window."""
        self.window_dropped_replies += 1

    def record_dropped_non_en(self) -> None:
        """Increment dropped non-English counter for the window."""
        self.window_dropped_non_en += 1

    def record_passed_sieve(self) -> None:
        """Increment passed sieve counter for the window."""
        self.window_passed_sieve += 1

    def check_window(self) -> Optional[WindowMetrics]:
        """Check if the current window interval has elapsed.

        If elapsed, returns a WindowMetrics snapshot and resets the window counter.
        Otherwise, returns None.
        """
        now = time.time()
        elapsed = now - self.last_window_start
        if elapsed < self.window_seconds:
            return None

        raw_rate = self.window_raw / elapsed if elapsed > 0 else 0.0
        passed_rate = self.window_passed_sieve / elapsed if elapsed > 0 else 0.0
        preview = (self.latest_text[:60] + "...") if len(self.latest_text) > 60 else self.latest_text

        metrics = WindowMetrics(
            window_duration=elapsed,
            total_events=self.total_events,
            window_events=self.window_events,
            raw_received=self.window_raw,
            dropped_replies=self.window_dropped_replies,
            dropped_non_en=self.window_dropped_non_en,
            passed_sieve=self.window_passed_sieve,
            raw_rate_per_sec=raw_rate,
            passed_rate_per_sec=passed_rate,
            top_languages=self.lang_counter.most_common(5),
            latest_text_preview=preview,
        )

        self.window_events = 0
        self.window_raw = 0
        self.window_dropped_replies = 0
        self.window_dropped_non_en = 0
        self.window_passed_sieve = 0
        self.last_window_start = now
        return metrics
