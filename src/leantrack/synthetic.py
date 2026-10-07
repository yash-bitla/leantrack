from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leantrack._types import Detections, FloatArray


@dataclass(frozen=True, slots=True)
class SyntheticObject:
    """An object that moves at constant velocity. Frames are 1-based."""

    start_box: tuple[float, float, float, float]
    velocity: tuple[float, float]
    first_frame: int = 1
    last_frame: int | None = None
    hidden: tuple[tuple[int, int], ...] = ()
    """Inclusive frame ranges in which the detector does not see the object."""
    score: float = 0.9

    def box_at(self, frame: int) -> FloatArray:
        step = frame - self.first_frame
        dx, dy = self.velocity[0] * step, self.velocity[1] * step
        x1, y1, x2, y2 = self.start_box
        return np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy], dtype=np.float64)

    def exists_at(self, frame: int) -> bool:
        return frame >= self.first_frame and (self.last_frame is None or frame <= self.last_frame)

    def visible_at(self, frame: int) -> bool:
        return self.exists_at(frame) and not any(a <= frame <= b for a, b in self.hidden)


@dataclass(frozen=True, slots=True)
class SyntheticScene:
    """Generates detections and ground truth for a set of objects."""

    objects: tuple[SyntheticObject, ...]
    length: int
    jitter: float = 0.0
    """Standard deviation of the box coordinate noise, in pixels."""
    seed: int = 0
    _noise: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.jitter, (self.length + 1, len(self.objects), 4))
        object.__setattr__(self, "_noise", noise)

    def ground_truth(self, frame: int) -> dict[int, FloatArray]:
        """Object id (1-based) to true box, for each object that exists in `frame`."""
        return {
            i: obj.box_at(frame)
            for i, obj in enumerate(self.objects, start=1)
            if obj.exists_at(frame)
        }

    def detections(self, frame: int) -> Detections:
        visible = [(i, o) for i, o in enumerate(self.objects) if o.visible_at(frame)]
        if not visible:
            return Detections.empty()
        return Detections(
            np.stack([o.box_at(frame) + self._noise[frame, i] for i, o in visible]),
            np.array([o.score for _, o in visible], dtype=np.float64),
            np.zeros(len(visible), dtype=np.int64),
        )
