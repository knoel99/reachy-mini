"""Vision backends for the Reachy Mini voice agent.

Pluggable image-to-text providers exposed to the LLM as tool calls
(`look_and_describe`, `find_object`). The backend is selected via the
`VISION_BACKEND` env var; if unset, vision tools are disabled and the
agent keeps its audio-only behavior.

Backends:

- :class:`MoondreamBackend` — Moondream cloud API or local SDK. Native
  ``query``/``detect``/``point`` returning structured JSON.
- :class:`FastVLMBackend` — HTTP client to a remote FastVLM-7B server
  (typically run on Colab via ``colab/fastvlm_7b_server.ipynb``).
  Supports ``query`` only.
"""

from .base import VisionBackend, VisionResult, BoundingBox, Point
from .camera_worker import CameraWorker
from .factory import build_vision_backend

__all__ = [
    "BoundingBox",
    "CameraWorker",
    "Point",
    "VisionBackend",
    "VisionResult",
    "build_vision_backend",
]
