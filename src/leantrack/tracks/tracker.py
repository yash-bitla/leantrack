from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from leantrack._types import Detections, FloatArray, IntArray, Propagated, TrackedObject
from leantrack.associate.matching import assign
from leantrack.boxes import iou_matrix, xyxy_to_cxcywh
from leantrack.propagate.kalman import KalmanFilter
from leantrack.tracks.track import Track, TrackState

Embed = Callable[[IntArray], FloatArray]
"""Returns unit-length appearance vectors for the detections at the given indices."""


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
    max_appearance_distance: float = 0.3
    """Maximum cosine distance for the recovery of a lost track by appearance."""
    max_lost_match_distance: float = 0.4
    """Maximum cosine distance for the IoU match of a lost track. The predicted box of a
    lost track drifts, and it can overlap a different object."""
    recovery_gate: float = 9.21
    """Maximum squared Mahalanobis distance between a detection center and the prediction
    of a lost track. The default is the 99% point of the chi-square distribution with
    2 degrees of freedom."""
    embedding_momentum: float = 0.5
    """Weight of the stored appearance vector when a new vector updates it."""
    embedding_refresh: int = 5
    """A track updates its appearance vector on each N-th detection match."""
    max_crop_overlap: float = 0.4
    """A detection that overlaps another detection by more than this IoU does not update
    an appearance vector, because its pixels contain a second object."""

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
        self,
        detections: Detections,
        moved: Mapping[int, Propagated] | None = None,
        embed: Embed | None = None,
    ) -> list[TrackedObject]:
        """Advance one frame. Returns the confirmed tracks that got a detection in this frame."""
        self.advance(moved)
        return self.associate(detections, embed)

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

    def associate(self, detections: Detections, embed: Embed | None = None) -> list[TrackedObject]:
        """Complete a frame with detections. Returns the confirmed tracks that got a match.

        `embed` returns the appearance vectors of the detections at the given indices.
        With it, a lost track can recover by appearance when its predicted box no longer
        overlaps the object. The tracker calls it only for the detections that it needs.
        """
        cfg = self.config
        for track in self._tracks:
            track.pending = None

        high_index = np.flatnonzero(detections.scores >= cfg.high_score)
        high = detections.select(high_index)
        low = detections.select(
            (detections.scores >= cfg.low_score) & (detections.scores < cfg.high_score)
        )

        established = [t for t in self._tracks if t.state is not TrackState.TENTATIVE]
        tentative = [t for t in self._tracks if t.state is TrackState.TENTATIVE]

        if embed is not None:
            embed = _Memo(embed)

        # Pass 1: confirmed and lost tracks against high-score detections.
        cost = _iou_cost(established, high.boxes)
        if embed is not None:
            self._veto_lost_matches(established, cost, high_index, embed)
        first = assign(cost, 1 - cfg.match_iou)
        for row, col in first.matches:
            self._apply(established[row], high, col)
        if embed is not None:
            self._refresh_embeddings(
                [(established[row], col) for row, col in first.matches], high, high_index, embed
            )

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

        left_cols = list(first.unmatched_cols)
        if embed is not None:
            lost = [established[r] for r in first.unmatched_rows if established[r] not in remaining]
            left_cols = self._recover(lost, high, high_index, left_cols, embed)

        # Pass 3: tentative tracks against the high-score detections that are left.
        left = high.select(np.array(left_cols, dtype=np.int64))
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

    def _veto_lost_matches(
        self, tracks: list[Track], cost: FloatArray, high_index: IntArray, embed: Embed
    ) -> None:
        """Forbid the IoU match of a lost track with a detection that looks different."""
        lost = [
            (row, t.embedding)
            for row, t in enumerate(tracks)
            if t.state is TrackState.LOST and t.embedding is not None
        ]
        if not lost or cost.shape[1] == 0:
            return
        rows = [row for row, _ in lost]
        possible = cost[rows] <= 1 - self.config.match_iou
        cols = np.flatnonzero(possible.any(axis=0))
        if len(cols) == 0:
            return
        vectors = embed(high_index[cols])
        stored = np.stack([vector for _, vector in lost])
        different = (1.0 - stored @ vectors.T) > self.config.max_lost_match_distance
        for row, row_different in zip(rows, different, strict=True):
            cost[row, cols[row_different]] = np.inf

    def _recover(
        self,
        lost: list[Track],
        high: Detections,
        high_index: IntArray,
        cols: list[int],
        embed: Embed,
    ) -> list[int]:
        """Match lost tracks to unmatched detections by appearance. Returns the columns left."""
        lost = [t for t in lost if t.embedding is not None]
        if not lost or not cols:
            return cols
        centers = xyxy_to_cxcywh(high.boxes[cols])[:, :2]
        near = np.stack(
            [
                self._kalman.gating_distance(t.mean, t.covariance, centers)
                <= self.config.recovery_gate
                for t in lost
            ]
        )
        # Embed only the detections that are near a lost track.
        candidates = np.flatnonzero(near.any(axis=0))
        if len(candidates) == 0:
            return cols
        vectors = embed(high_index[np.array(cols)[candidates]])
        stored = np.stack([t.embedding for t in lost if t.embedding is not None])
        cost = 1.0 - stored @ vectors.T
        cost[~near[:, candidates]] = np.inf

        result = assign(cost, self.config.max_appearance_distance)
        matched = set()
        for row, candidate in result.matches:
            col = cols[int(candidates[candidate])]
            self._apply(lost[row], high, col)
            self._blend(lost[row], vectors[candidate])
            matched.add(col)
        return [c for c in cols if c not in matched]

    def _refresh_embeddings(
        self,
        matches: list[tuple[Track, int]],
        high: Detections,
        high_index: IntArray,
        embed: Embed,
    ) -> None:
        due = [
            (track, col)
            for track, col in matches
            if track.embedding is None or track.hits % self.config.embedding_refresh == 0
        ]
        if not due:
            return
        overlap = iou_matrix(high.boxes, high.boxes)
        np.fill_diagonal(overlap, 0.0)
        clean = [
            (track, col)
            for track, col in due
            if overlap[col].max(initial=0.0) <= self.config.max_crop_overlap
        ]
        if not clean:
            return
        vectors = embed(high_index[[col for _, col in clean]])
        for (track, _), vector in zip(clean, vectors, strict=True):
            self._blend(track, vector)

    def _blend(self, track: Track, vector: FloatArray) -> None:
        if track.embedding is None:
            track.embedding = vector
            return
        momentum = self.config.embedding_momentum
        blended = momentum * track.embedding + (1 - momentum) * vector
        track.embedding = blended / max(float(np.linalg.norm(blended)), 1e-12)

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


class _Memo:
    """Calls an `Embed` function at most one time for each detection of a frame."""

    def __init__(self, embed: Embed) -> None:
        self._embed = embed
        self._known: dict[int, FloatArray] = {}

    def __call__(self, index: IntArray) -> FloatArray:
        missing = [int(i) for i in index if int(i) not in self._known]
        if missing:
            vectors = self._embed(np.array(missing, dtype=np.int64))
            self._known.update(zip(missing, vectors, strict=True))
        return np.stack([self._known[int(i)] for i in index])


def _iou_cost(tracks: list[Track], boxes: FloatArray) -> FloatArray:
    if not tracks:
        return np.zeros((0, len(boxes)), dtype=np.float64)
    return 1.0 - iou_matrix(np.stack([t.box for t in tracks]), boxes)
