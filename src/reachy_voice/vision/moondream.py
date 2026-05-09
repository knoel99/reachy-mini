"""Moondream backend (cloud API or local SDK).

Cloud mode (default): set ``MOONDREAM_API_KEY``. ~400 ms/query, free
tier available, no local deps beyond ``moondream``.

Local mode: set ``MOONDREAM_LOCAL=1`` (and leave the API key unset).
Loads the int4 model in-process via ``moondream.vl()`` — runs anywhere
the SDK supports (CPU/CUDA/MPS), but **not on the Pi 5** for real-time
use (8–20 s/query, see Core Electronics bench).

Both modes expose the same ``query``/``detect``/``point`` interface
documented at https://docs.moondream.ai/.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._log import log
from .base import BoundingBox, Point, VisionBackend, VisionResult


class MoondreamBackend(VisionBackend):
    name = "moondream"
    supports_grounding = True

    def __init__(self, api_key: str | None = None, local: bool = False) -> None:
        try:
            import moondream as md
        except ImportError as e:
            raise RuntimeError(
                "moondream package missing. Install with `pip install moondream` "
                "(or add to pyproject deps)."
            ) from e

        if local:
            log("[vision] moondream: loading local model (this may take a while)…")
            self._model = md.vl()  # type: ignore[no-untyped-call]
            log("[vision] moondream: local model ready")
        else:
            if not api_key:
                raise RuntimeError(
                    "MoondreamBackend cloud mode requires MOONDREAM_API_KEY."
                )
            self._model = md.vl(api_key=api_key)  # type: ignore[no-untyped-call]
            log("[vision] moondream: cloud client ready")

        self._calls = 0

    @staticmethod
    def _bgr_to_pil(frame_bgr: NDArray[np.uint8]):
        from PIL import Image
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        return Image.fromarray(rgb)

    def _timed(self, fn) -> tuple[Any, float]:
        t0 = time.monotonic()
        try:
            out = fn()
        finally:
            self._calls += 1
        return out, (time.monotonic() - t0) * 1000.0

    def query(self, frame_bgr: NDArray[np.uint8], question: str) -> VisionResult:
        img = self._bgr_to_pil(frame_bgr)
        result, ms = self._timed(lambda: self._model.query(img, question))
        answer = (result or {}).get("answer") if isinstance(result, dict) else str(result)
        log(f"[vision] moondream.query {ms:.0f}ms: {str(answer)[:120]}")
        return VisionResult(text=str(answer or "").strip(), latency_ms=ms)

    def detect(self, frame_bgr: NDArray[np.uint8], target: str) -> VisionResult:
        img = self._bgr_to_pil(frame_bgr)
        result, ms = self._timed(lambda: self._model.detect(img, target))
        boxes: list[BoundingBox] = []
        for obj in (result or {}).get("objects", []) if isinstance(result, dict) else []:
            boxes.append(
                BoundingBox(
                    x_min=float(obj.get("x_min", 0.0)),
                    y_min=float(obj.get("y_min", 0.0)),
                    x_max=float(obj.get("x_max", 0.0)),
                    y_max=float(obj.get("y_max", 0.0)),
                    label=str(obj.get("label", target)),
                    confidence=(
                        float(obj["confidence"]) if "confidence" in obj else None
                    ),
                )
            )
        log(f"[vision] moondream.detect '{target}' {ms:.0f}ms: {len(boxes)} hit(s)")
        return VisionResult(boxes=boxes, latency_ms=ms)

    def point(self, frame_bgr: NDArray[np.uint8], target: str) -> VisionResult:
        img = self._bgr_to_pil(frame_bgr)
        result, ms = self._timed(lambda: self._model.point(img, target))
        points: list[Point] = []
        for pt in (result or {}).get("points", []) if isinstance(result, dict) else []:
            points.append(
                Point(x=float(pt.get("x", 0.0)), y=float(pt.get("y", 0.0)), label=target)
            )
        log(f"[vision] moondream.point '{target}' {ms:.0f}ms: {len(points)} hit(s)")
        return VisionResult(points=points, latency_ms=ms)


def from_env() -> MoondreamBackend:
    """Build a MoondreamBackend from env vars."""
    local = os.environ.get("MOONDREAM_LOCAL", "").lower() in ("1", "true", "yes")
    api_key = os.environ.get("MOONDREAM_API_KEY")
    return MoondreamBackend(api_key=api_key, local=local)
