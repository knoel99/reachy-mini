"""Timestamped logger shared across the voice bridge modules.

Every event log goes through `log()` so we get a wall-clock timestamp
and a delta from the previous log line — handy to estimate the
duration of each action (mic→tool, tool→response.done, etc.).

Format: `[HH:MM:SS.mmm +0.123s] <message>`
"""

from __future__ import annotations

import threading
import time
from datetime import datetime


_lock = threading.Lock()
_last_t: float | None = None


def log(*parts, end: str = "\n", lead: str = "") -> None:
    """Print a timestamped log line with delta-since-previous-log.

    `lead` is emitted before the timestamp — pass `\\n` to break out of
    a mid-line streamed write (e.g. token deltas with no newline).
    """
    global _last_t
    msg = " ".join(str(p) for p in parts)
    with _lock:
        now = time.monotonic()
        delta = 0.0 if _last_t is None else (now - _last_t)
        _last_t = now
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{lead}[{ts} +{delta:5.2f}s] {msg}", end=end, flush=True)
