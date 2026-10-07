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
