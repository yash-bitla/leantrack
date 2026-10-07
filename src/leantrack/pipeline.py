from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from leantrack._types import Frame, TrackedObject
from leantrack.detect.base import Detector
from leantrack.propagate.flow import FlowPropagator
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
    track_ms: float


def run(
    frames: Iterable[Frame],
    detector: Detector,
    tracker: Tracker,
    policy: SchedulePolicy | None = None,
    propagator: FlowPropagator | None = None,
) -> Iterator[FrameResult]:
    """Track a frame sequence.

    Each frame has three steps. First, the tracks move to the frame: a Kalman prediction
    and, with a propagator, a correction from optical flow. Second, the policy decides
    on a detector run. Third, the detections update the tracks, or the tracks coast.
    The propagator needs the pixels.
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
        if detect:
            detections = detector.detect(frame)
            detect_ms = (time.perf_counter() - decided) * 1000.0
            objects = tracker.associate(detections)
            since_detection = 0
        else:
            objects = tracker.coast()
        end = time.perf_counter()
        yield FrameResult(
            frame.index,
            objects,
            detect,
            detect_ms=detect_ms,
            propagate_ms=(propagated - start) * 1000.0 if propagator else 0.0,
            track_ms=(end - propagated) * 1000.0 - detect_ms,
        )
