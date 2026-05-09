"""FastVLM backend — HTTP client to a remote FastVLM-7B server.

The model itself runs on a beefy GPU (typically Colab — see
``colab/fastvlm_7b_server.ipynb``) which exposes a ``/query`` endpoint
via a cloudflared quick tunnel (``*.trycloudflare.com``). This client
posts the JPEG-encoded frame + question and returns the answer.

Why HTTP / why Colab: FastVLM-7B is ~14 GB in fp16, far too heavy for
the Pi or a typical laptop CPU. Apple's Core ML build is Apple Silicon
only; the PyTorch checkpoint runs on CUDA but needs serious VRAM. A
Colab T4/L4/A100 is the path of least resistance for "highest config"
(7B) without any local hardware investment.

Set ``FASTVLM_URL`` to the base URL printed by the Colab notebook
(e.g. ``https://abc123.trycloudflare.com``). FastVLM does not natively
expose detect/point, so those fall back to the base class no-op.
"""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
from numpy.typing import NDArray

from .._log import log
from .base import VisionBackend, VisionResult


class FastVLMBackend(VisionBackend):
    name = "fastvlm"
    supports_grounding = False

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        if not base_url:
            raise RuntimeError("FastVLMBackend requires FASTVLM_URL.")
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "httpx missing. Install with `pip install httpx` "
                "(or add to pyproject deps)."
            ) from e
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        log(f"[vision] fastvlm: target {self._base_url}")

    @staticmethod
    def _bgr_to_jpeg_b64(frame_bgr: NDArray[np.uint8], quality: int = 85) -> str:
        from PIL import Image
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def query(self, frame_bgr: NDArray[np.uint8], question: str) -> VisionResult:
        import httpx

        b64 = self._bgr_to_jpeg_b64(frame_bgr)
        payload = {"image_b64": b64, "question": question}
        t0 = time.monotonic()
        try:
            r = httpx.post(
                f"{self._base_url}/query",
                json=payload,
                timeout=self._timeout,
            )
            r.raise_for_status()
            data = r.json()
            answer = str(data.get("answer", "")).strip()
        except Exception as e:
            log(f"[vision] fastvlm.query failed: {e}")
            return VisionResult(text=f"vision error: {e}")
        ms = (time.monotonic() - t0) * 1000.0
        log(f"[vision] fastvlm.query {ms:.0f}ms: {answer[:120]}")
        return VisionResult(text=answer, latency_ms=ms)


def from_env() -> FastVLMBackend:
    base_url = os.environ.get("FASTVLM_URL", "").strip()
    timeout = float(os.environ.get("FASTVLM_TIMEOUT", "30"))
    return FastVLMBackend(base_url=base_url, timeout=timeout)
