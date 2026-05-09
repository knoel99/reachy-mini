"""VISION_BACKEND env-var dispatcher."""

from __future__ import annotations

import os

from .base import VisionBackend


def build_vision_backend() -> VisionBackend | None:
    """Read VISION_BACKEND and instantiate. Returns ``None`` when unset
    or set to ``"none"`` so vision tools are simply not registered."""
    name = os.environ.get("VISION_BACKEND", "").strip().lower()
    if not name or name == "none":
        return None

    if name == "moondream":
        from . import moondream
        return moondream.from_env()

    if name == "fastvlm":
        from . import fastvlm
        return fastvlm.from_env()

    raise RuntimeError(
        f"Unknown VISION_BACKEND={name!r}. Use one of: moondream, fastvlm, none."
    )
