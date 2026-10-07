from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from leantrack._types import Detections, FloatArray, Propagated, TrackedObject
from leantrack.associate.matching import assign
from leantrack.boxes import iou_matrix, xyxy_to_cxcywh
from leantrack.propagate.kalman import KalmanFilter
from leantrack.tracks.track import Track, TrackState


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    high_score: float = 0.6
    """Detections at or above this score go into the first association pass."""
    low_score: float = 0.1
    """Detections below this score are discarded."""
    new_track_score: float = 0.7
    """Minimum score for a detection to start a track."""
    match_iou: float = 0.2
    """Minimum IoU for the first pass (confirmed and lost tracks, high-score detections)."""
    low_match_iou: float = 0.5
    """Minimum IoU for the second pass (confirmed tracks, low-score detections)."""
    tentative_match_iou: float = 0.3
    """Minimum IoU to confirm a tentative track."""
    max_lost_frames: int = 30
    """A lost track is removed after this number of frames without a match."""

    def __post_init__(self) -> None:
        if not 0 <= self.low_score <= self.high_score <= 1:
            raise ValueError("scores must satisfy 0 <= low_score <= high_score <= 1")
        if self.max_lost_frames < 0:
            raise ValueError("max_lost_frames must not be negative")


class Tracker:
    """Two-pass IoU association in the style of ByteTrack, with an explicit track lifecycle."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._kalman = KalmanFilter()
        self._tracks: list[Track] = []
        self._next_id = 1
        self._frame = 0

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def predict(self, moved: Mapping[int, Propagated] | None = None) -> list[TrackedObject]:
        """Advance one frame without a detector run. Returns the predicted confirmed tracks."""
        self.advance(moved)
        return self.coast()

    def update(
        self, detections: Detections, moved: Mapping[int, Propagated] | None = None
    ) -> list[TrackedObject]:
        """Advance one frame. Returns the confirmed tracks that got a detection in this frame."""
        self.advance(moved)
        return self.associate(detections)

    def advance(self, moved: Mapping[int, Propagated] | None = None) -> None:
        """Move each track to the next frame. Follow it with `associate` or `coast`.

        `moved` maps a track id to the result of a propagator for this frame. The filter
        uses the moved box as a measurement only if the frame gets no detector run
        (`coast`). A detection is a better measurement of the same frame. If the filter
        used the two, the detection would correct only a part of the propagation drift.
        A lost track ignores `moved`, because the detector did not see the object and
        the pixels in its box can be an occluder.
        """
        self._frame += 1
        for track in self._tracks:
            if track.state is TrackState.LOST:
                # A lost track keeps its position velocity but its size stays constant.
                # An unobserved size velocity makes the box collapse or grow without limit.
                track.mean[6:] = 0.0
            center = track.mean[:2].copy()
            track.mean, track.covariance = self._kalman.predict(track.mean, track.covariance)
            track.frames_since_update += 1

            track.pending = None
            result = moved.get(track.track_id) if moved else None
            if result is not None and track.state is not TrackState.LOST:
                track.reliability = result.reliability
                if result.box is not None:
                    track.pending = xyxy_to_cxcywh(result.box)
            target = track.mean[:2] if track.pending is None else track.pending[:2]
            size = float(np.sqrt(max(track.mean[2] * track.mean[3], 1.0)))
            track.motion_since_update += float(np.linalg.norm(target - center)) / size

    def coast(self) -> list[TrackedObject]:
        """Complete a frame without a detector run. Returns the predicted confirmed tracks."""
        for track in self._tracks:
            if track.pending is not None:
                track.mean, track.covariance = self._kalman.update(
                    track.mean, track.covariance, track.pending
                )
                track.pending = None
        self._remove_expired()
        return [t.as_output() for t in self._tracks if t.state is TrackState.CONFIRMED]

    def associate(self, detections: Detections) -> list[TrackedObject]:
        """Complete a frame with detections. Returns the confirmed tracks that got a match."""
        cfg = self.config
        for track in self._tracks:
            track.pending = None

        high = detections.select(detections.scores >= cfg.high_score)
        low = detections.select(
            (detections.scores >= cfg.low_score) & (detections.scores < cfg.high_score)
        )

        established = [t for t in self._tracks if t.state is not TrackState.TENTATIVE]
        tentative = [t for t in self._tracks if t.state is TrackState.TENTATIVE]

        # Pass 1: confirmed and lost tracks against high-score detections.
        first = assign(_iou_cost(established, high.boxes), 1 - cfg.match_iou)
        for row, col in first.matches:
            self._apply(established[row], high, col)

        # Pass 2: tracks that were confirmed in the previous frame against low-score
        # detections. A low score on a known track usually means partial occlusion.
        remaining = [
            established[r]
            for r in first.unmatched_rows
            if established[r].state is TrackState.CONFIRMED
        ]
        second = assign(_iou_cost(remaining, low.boxes), 1 - cfg.low_match_iou)
        for row, col in second.matches:
            self._apply(remaining[row], low, col)
        for row in second.unmatched_rows:
            remaining[row].transition(TrackState.LOST)

        # Pass 3: tentative tracks against the high-score detections that are left.
        left = high.select(np.array(first.unmatched_cols, dtype=np.int64))
        third = assign(_iou_cost(tentative, left.boxes), 1 - cfg.tentative_match_iou)
        for row, col in third.matches:
            self._apply(tentative[row], left, col)
        for row in third.unmatched_rows:
            tentative[row].transition(TrackState.REMOVED)

        for col in third.unmatched_cols:
            if left.scores[col] >= cfg.new_track_score:
                self._start(left, col)

        self._remove_expired()

        return [
            t.as_output()
            for t in self._tracks
            if t.state is TrackState.CONFIRMED and t.frames_since_update == 0
        ]

    def _remove_expired(self) -> None:
        for track in self._tracks:
            if (
                track.state is TrackState.LOST
                and track.frames_since_update > self.config.max_lost_frames
            ):
                track.transition(TrackState.REMOVED)
        self._tracks = [t for t in self._tracks if t.state is not TrackState.REMOVED]

    def _apply(self, track: Track, detections: Detections, index: int) -> None:
        measurement = xyxy_to_cxcywh(detections.boxes[index])
        track.mean, track.covariance = self._kalman.update(
            track.mean, track.covariance, measurement
        )
        track.score = float(detections.scores[index])
        track.class_id = int(detections.classes[index])
        track.hits += 1
        track.frames_since_update = 0
        track.motion_since_update = 0.0
        track.reliability = 1.0
        if track.state is not TrackState.CONFIRMED:
            track.transition(TrackState.CONFIRMED)

    def _start(self, detections: Detections, index: int) -> None:
        mean, covariance = self._kalman.initiate(xyxy_to_cxcywh(detections.boxes[index]))
        track = Track(
            track_id=self._next_id,
            mean=mean,
            covariance=covariance,
            score=float(detections.scores[index]),
            class_id=int(detections.classes[index]),
        )
        self._next_id += 1
        # No earlier frame exists to confirm against, so first-frame tracks start confirmed.
        if self._frame == 1:
            track.transition(TrackState.CONFIRMED)
        self._tracks.append(track)


def _iou_cost(tracks: list[Track], boxes: FloatArray) -> FloatArray:
    if not tracks:
        return np.zeros((0, len(boxes)), dtype=np.float64)
    return 1.0 - iou_matrix(np.stack([t.box for t in tracks]), boxes)
