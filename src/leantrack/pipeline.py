from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from leantrack._types import Frame, TrackedObject
from leantrack.detect.base import Detector
from leantrack.schedule.policy import FixedInterval, SchedulePolicy
from leantrack.tracks.tracker import Tracker


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    objects: list[TrackedObject]
    detected: bool
    """True if the detector ran on this frame."""
    detect_ms: float
    track_ms: float


def run(
    frames: Iterable[Frame],
    detector: Detector,
    tracker: Tracker,
    policy: SchedulePolicy | None = None,
) -> Iterator[FrameResult]:
    policy = policy or FixedInterval(1)
    for frame in frames:
        start = time.perf_counter()
        if policy.should_detect(frame.index):
            detections = detector.detect(frame)
            detected = time.perf_counter()
            objects = tracker.update(detections)
            end = time.perf_counter()
            detect_ms, track_ms = (detected - start) * 1000.0, (end - detected) * 1000.0
            yield FrameResult(frame.index, objects, True, detect_ms, track_ms)
        else:
            objects = tracker.predict()
            yield FrameResult(
                frame.index, objects, False, 0.0, (time.perf_counter() - start) * 1000.0
            )
