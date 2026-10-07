from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Image = NDArray[Any]


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame of a sequence. `index` is 1-based, as in the MOT format."""

    index: int
    image: Image | None = None


@dataclass(frozen=True, slots=True)
class Detections:
    """Detector output for one frame. Boxes are (N, 4) in x1, y1, x2, y2 pixels."""

    boxes: FloatArray
    scores: FloatArray
    classes: IntArray

    def __post_init__(self) -> None:
        n = len(self.boxes)
        if self.boxes.shape != (n, 4):
            raise ValueError(f"boxes must have shape (N, 4), got {self.boxes.shape}")
        if self.scores.shape != (n,) or self.classes.shape != (n,):
            raise ValueError("scores and classes must have one entry per box")

    def __len__(self) -> int:
        return len(self.boxes)

    def select(self, mask: NDArray[np.bool_] | IntArray) -> Detections:
        return Detections(self.boxes[mask], self.scores[mask], self.classes[mask])

    @staticmethod
    def empty() -> Detections:
        return Detections(
            np.empty((0, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )


@dataclass(frozen=True, slots=True)
class TrackedObject:
    """One confirmed track in one frame."""

    track_id: int
    box: tuple[float, float, float, float]
    score: float
    class_id: int
