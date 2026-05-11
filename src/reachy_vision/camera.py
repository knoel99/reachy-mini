"""Background camera capture exposing the latest BGR frame."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from reachy_voice._log import log

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


class Camera:
    def __init__(self, mini: ReachyMini, fps: float = 5.0) -> None:
        self._mini = mini
        self._period = 1.0 / max(0.5, fps)
        self._frame: np.ndarray | None = None
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

    def get_latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

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
                    self._frame = frame
                self._frames += 1
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._period - elapsed))
