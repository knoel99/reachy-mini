"""Moondream backend (cloud API or local SDK).

Cloud mode (default): pass ``api_key``. ~400 ms/query, free tier
available. Local mode: pass ``local=True`` and load the int4 model
in-process. Local is not viable on a Pi 5 (8-20 s/query) and is
intended for laptops with CUDA/MPS.
"""

from __future__ import annotations

import time

import moondream as md
import numpy as np
from PIL import Image

from reachy_voice._log import log


class Moondream:
    supports_grounding = True

    def __init__(self, api_key: str | None = None, local: bool = False) -> None:
        if local:
            log("[vision] moondream: loading local model...")
            self._model = md.vl()
            log("[vision] moondream: local model ready")
        else:
            if not api_key:
                raise ValueError("Moondream cloud mode requires api_key")
            self._model = md.vl(api_key=api_key)
            log("[vision] moondream: cloud client ready")

    def caption(self, image_bgr: np.ndarray, question: str) -> str:
        t0 = time.monotonic()
        result = self._model.query(self._to_pil(image_bgr), question)
        ms = (time.monotonic() - t0) * 1000.0
        answer = result.get("answer") if isinstance(result, dict) else result
        text = str(answer or "").strip()
        log(f"[vision] moondream.caption {ms:.0f}ms: {text[:120]}")
        return text

    def point(self, image_bgr: np.ndarray, target: str) -> list[tuple[float, float]]:
        t0 = time.monotonic()
        result = self._model.point(self._to_pil(image_bgr), target)
        ms = (time.monotonic() - t0) * 1000.0
        raw = result.get("points", []) if isinstance(result, dict) else result
        points = [(float(p["x"]), float(p["y"])) for p in raw]
        log(f"[vision] moondream.point '{target}' {ms:.0f}ms: {len(points)} hit(s)")
        return points

    @staticmethod
    def _to_pil(image_bgr: np.ndarray) -> Image.Image:
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        return Image.fromarray(rgb)
