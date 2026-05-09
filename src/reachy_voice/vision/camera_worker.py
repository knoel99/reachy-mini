"""Camera frame buffer thread.

Pulls frames from ``mini.media.get_frame()`` at a low fixed rate and
keeps the latest one available behind a lock so that on-demand vision
tool calls can grab a fresh frame without blocking on an actual
capture. We don't do head-tracking here — that's intentionally left to
``reachy_mini_toolbox`` if you want it later. This worker only exists
to keep one warm frame ready for the LLM-triggered ``look_and_describe``
and ``find_object`` tools.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._log import log

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


class CameraWorker:
    """Thread keeping the latest BGR frame in a buffer."""

    def __init__(self, mini: ReachyMini, fps: float = 5.0) -> None:
        self._mini = mini
        self._period = 1.0 / max(0.5, fps)
        self._latest: NDArray[np.uint8] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log(f"[camera] worker started ({1.0 / self._period:.1f} Hz)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log(f"[camera] worker stopped (captured {self._frames} frame(s))")

    def get_latest(self) -> NDArray[np.uint8] | None:
        """Thread-safe latest-frame snapshot (returns a copy, or None)."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                frame = self._mini.media.get_frame()
            except Exception as e:
                log(f"[camera] get_frame failed: {e}")
                frame = None
            if frame is not None:
                with self._lock:
                    self._latest = frame
                self._frames += 1
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._period - elapsed))
