from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from leantrack._types import Frame, TrackedObject
from leantrack.detect.base import Detector
from leantrack.propagate.flow import FlowPropagator
from leantrack.schedule.policy import FixedInterval, SchedulePolicy
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

    Without a propagator, a frame without a detector run gets the Kalman prediction. With
    a propagator, optical flow corrects that prediction. The propagator needs the pixels.
    """
    policy = policy or FixedInterval(1)
    for frame in frames:
        start = time.perf_counter()
        if propagator is not None:
            if frame.image is None:
                raise ValueError("a propagator needs the frame pixels")
            propagator.observe(frame.image)
        observed = time.perf_counter()

        if policy.should_detect(frame.index):
            detections = detector.detect(frame)
            detected = time.perf_counter()
            objects = tracker.update(detections)
            end = time.perf_counter()
            yield FrameResult(
                frame.index,
                objects,
                True,
                detect_ms=(detected - observed) * 1000.0,
                propagate_ms=(observed - start) * 1000.0 if propagator else 0.0,
                track_ms=(end - detected) * 1000.0,
            )
            continue

        moved = None
        if propagator is not None:
            active = [t for t in tracker.tracks if t.state is not TrackState.LOST]
            if active:
                boxes = propagator.propagate(np.stack([t.box for t in active]))
                moved = {
                    t.track_id: box for t, box in zip(active, boxes, strict=True) if box is not None
                }
        propagated = time.perf_counter()
        objects = tracker.predict(moved)
        end = time.perf_counter()
        yield FrameResult(
            frame.index,
            objects,
            False,
            detect_ms=0.0,
            propagate_ms=(propagated - start) * 1000.0 if propagator else 0.0,
            track_ms=(end - propagated) * 1000.0,
        )
