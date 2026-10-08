from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import numpy as np

from leantrack._types import FloatArray, Frame, Image, IntArray, TrackedObject
from leantrack.confidence.features import track_features
from leantrack.confidence.predictor import FailurePredictor
from leantrack.detect.base import Detector
from leantrack.propagate.flow import FlowPropagator
from leantrack.reid.base import Embedder
from leantrack.schedule.policy import FixedInterval, ScheduleContext, SchedulePolicy
from leantrack.tracks.track import TrackState
from leantrack.tracks.tracker import Tracker


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    objects: list[TrackedObject]
    detected: bool
    """True if the detector ran on this frame."""
    detect_ms: float
    propagate_ms: float
    """Time of the optical flow. Zero without a propagator."""
    embed_ms: float
    """Time of the embedder. Zero without an embedder and on frames that need no vector."""
    track_ms: float

    @property
    def total_ms(self) -> float:
        return self.detect_ms + self.propagate_ms + self.embed_ms + self.track_ms


def run(
    frames: Iterable[Frame],
    detector: Detector,
    tracker: Tracker,
    policy: SchedulePolicy | None = None,
    propagator: FlowPropagator | None = None,
    embedder: Embedder | None = None,
    predictor: FailurePredictor | None = None,
) -> Iterator[FrameResult]:
    """Track a frame sequence.

    Each frame has three steps. First, the tracks move to the frame: a Kalman prediction
    and, with a propagator, a correction from optical flow. Second, the policy decides
    on a detector run. Third, the detections update the tracks, or the tracks coast.

    With an embedder, a lost track can recover by appearance. The embedder runs only on
    frames with a detector run, and only for the detections that the tracker requests.
    The propagator and the embedder need the pixels.

    With a failure predictor, each reported object has the probability that its box is
    wrong. The pipeline does not remove objects. The caller selects a limit.
    """
    policy = policy or FixedInterval(1)
    since_detection: int | None = None
    for frame in frames:
        start = time.perf_counter()
        moved = None
        if propagator is not None:
            if frame.image is None:
                raise ValueError("a propagator needs the frame pixels")
            propagator.observe(frame.image)
            active = [t for t in tracker.tracks if t.state is not TrackState.LOST]
            if active:
                results = propagator.propagate(np.stack([t.box for t in active]))
                moved = {t.track_id: r for t, r in zip(active, results, strict=True)}
        propagated = time.perf_counter()
        tracker.advance(moved)
        if since_detection is not None:
            since_detection += 1
        detect = policy.should_detect(ScheduleContext(frame.index, since_detection, tracker.tracks))
        decided = time.perf_counter()

        detect_ms = 0.0
        embed_ms = 0.0
        if detect:
            detections = detector.detect(frame)
            detect_ms = (time.perf_counter() - decided) * 1000.0
            embed = None
            if embedder is not None:
                if frame.image is None:
                    raise ValueError("an embedder needs the frame pixels")
                embed = _TimedEmbed(embedder, frame.image, detections.boxes)
            objects = tracker.associate(detections, embed)
            if embed is not None:
                embed_ms = embed.seconds * 1000.0
            since_detection = 0
        else:
            objects = tracker.coast()
        if predictor is not None and objects:
            tracks = tracker.tracks
            probability = dict(
                zip(
                    (t.track_id for t in tracks),
                    predictor.probability(track_features(tracks)),
                    strict=True,
                )
            )
            objects = [
                replace(o, failure_probability=float(probability[o.track_id])) for o in objects
            ]
        end = time.perf_counter()
        yield FrameResult(
            frame.index,
            objects,
            detect,
            detect_ms=detect_ms,
            propagate_ms=(propagated - start) * 1000.0 if propagator else 0.0,
            embed_ms=embed_ms,
            track_ms=(end - propagated) * 1000.0 - detect_ms - embed_ms,
        )


class _TimedEmbed:
    """The `Embed` function of one frame. It records the time that the embedder takes."""

    def __init__(self, embedder: Embedder, image: Image, boxes: FloatArray) -> None:
        self._embedder = embedder
        self._image = image
        self._boxes = boxes
        self.seconds = 0.0

    def __call__(self, index: IntArray) -> FloatArray:
        start = time.perf_counter()
        vectors = self._embedder.embed(self._image, self._boxes[index])
        self.seconds += time.perf_counter() - start
        return vectors
