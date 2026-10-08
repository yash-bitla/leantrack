from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from leantrack._types import FloatArray, TrackedObject
from leantrack.boxes import cxcywh_to_xyxy


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


_ALLOWED: dict[TrackState, frozenset[TrackState]] = {
    TrackState.TENTATIVE: frozenset({TrackState.CONFIRMED, TrackState.REMOVED}),
    TrackState.CONFIRMED: frozenset({TrackState.LOST, TrackState.REMOVED}),
    TrackState.LOST: frozenset({TrackState.CONFIRMED, TrackState.REMOVED}),
    TrackState.REMOVED: frozenset(),
}


@dataclass(slots=True)
class Track:
    track_id: int
    mean: FloatArray
    covariance: FloatArray
    score: float
    class_id: int
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    frames_since_update: int = 0
    """Frames since the last detection match."""
    motion_since_update: float = 0.0
    """Path length of the box center since the last detection match, in box sizes."""
    reliability: float = 1.0
    """Reliability that the propagator reported in the last frame. 1.0 after a detection."""
    embedding: FloatArray | None = None
    """Appearance vector with unit length. None until the first clean detection match."""
    pending: FloatArray | None = None
    """Propagated box (cx, cy, w, h) of this frame that the filter did not use yet."""

    def transition(self, new_state: TrackState) -> None:
        if new_state not in _ALLOWED[self.state]:
            raise ValueError(f"track {self.track_id}: {self.state.value} -> {new_state.value}")
        self.state = new_state

    @property
    def box(self) -> FloatArray:
        return cxcywh_to_xyxy(self.mean[:4])

    def as_output(self) -> TrackedObject:
        x1, y1, x2, y2 = (float(v) for v in self.box)
        return TrackedObject(self.track_id, (x1, y1, x2, y2), self.score, self.class_id)
