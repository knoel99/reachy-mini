"""HTTP client for a remote FastVLM-7B server.

The model itself runs on a GPU host (typically a Colab notebook
exposing /query via a cloudflared tunnel; see colab/fastvlm_7b_server.ipynb).
FastVLM has no native grounding, so this backend exposes caption()
only and `find_object` will report `grounding_not_supported`.
"""

from __future__ import annotations

import base64
import io
import time

import httpx
import numpy as np
from PIL import Image

from reachy_voice._log import log


class FastVLM:
    supports_grounding = False

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        if not base_url:
            raise ValueError("FastVLM requires a non-empty base_url")
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        log(f"[vision] fastvlm: target {self._base_url}")

    def caption(self, image_bgr: np.ndarray, question: str) -> str:
        payload = {"image_b64": self._to_b64(image_bgr), "question": question}
        t0 = time.monotonic()
        response = self._client.post(f"{self._base_url}/query", json=payload)
        response.raise_for_status()
        data = response.json()
        ms = (time.monotonic() - t0) * 1000.0
        answer = str(data.get("answer", "")).strip()
        log(f"[vision] fastvlm.caption {ms:.0f}ms: {answer[:120]}")
        return answer

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_b64(image_bgr: np.ndarray, quality: int = 85) -> str:
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
