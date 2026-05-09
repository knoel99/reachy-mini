"""Abstract VisionBackend interface.

All backends accept a BGR frame (the format returned by
``ReachyMini.media.get_frame()``) and a natural-language prompt, and
return text. Optional structured operations (``detect``, ``point``)
let the LLM ground objects in pixel coordinates so the robot can
orient its head toward them.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    """Axis-aligned bbox in normalized image coords (0..1)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: float | None = None

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


@dataclass
class Point:
    """A point in normalized image coords (0..1)."""

    x: float
    y: float
    label: str = ""


@dataclass
class VisionResult:
    """Common result wrapper. Either ``text`` is set (free-form answer)
    or ``boxes``/``points`` is set (structured grounding)."""

    text: str | None = None
    boxes: list[BoundingBox] | None = None
    points: list[Point] | None = None
    latency_ms: float | None = None


class VisionBackend(abc.ABC):
    """Backend that turns a camera frame + prompt into text or grounded
    coordinates."""

    name: str = "abstract"
    supports_grounding: bool = False

    @abc.abstractmethod
    def query(self, frame_bgr: NDArray[np.uint8], question: str) -> VisionResult:
        """Free-form VQA. Returns ``VisionResult`` with ``.text`` set."""

    def detect(self, frame_bgr: NDArray[np.uint8], target: str) -> VisionResult:  # noqa: ARG002
        """Locate objects matching ``target``. Default: not supported."""
        return VisionResult(text=f"detect not supported by backend '{self.name}'")

    def point(self, frame_bgr: NDArray[np.uint8], target: str) -> VisionResult:  # noqa: ARG002
        """Pinpoint instances of ``target``. Default: not supported."""
        return VisionResult(text=f"point not supported by backend '{self.name}'")

    def close(self) -> None:
        """Release backend resources. Default: no-op."""
