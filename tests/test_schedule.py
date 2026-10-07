from __future__ import annotations

import pytest

from leantrack._types import Detections, Frame
from leantrack.io.sources import index_frames
from leantrack.pipeline import run
from leantrack.schedule.policy import FixedInterval
from leantrack.synthetic import SyntheticObject, SyntheticScene
from leantrack.tracks.tracker import Tracker


class _CountingDetector:
    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene
        self.frames: list[int] = []

    def detect(self, frame: Frame) -> Detections:
        self.frames.append(frame.index)
        return self.scene.detections(frame.index)


def test_fixed_interval_schedule() -> None:
    policy = FixedInterval(4)
    assert [i for i in range(1, 12) if policy.should_detect(i)] == [1, 5, 9]
    assert all(FixedInterval(1).should_detect(i) for i in range(1, 5))
    with pytest.raises(ValueError):
        FixedInterval(0)


def test_pipeline_runs_the_detector_only_on_scheduled_frames() -> None:
    scene = SyntheticScene((SyntheticObject((100, 100, 140, 180), (2.0, 0.0)),), length=21)
    detector = _CountingDetector(scene)
    results = list(run(index_frames(scene.length), detector, Tracker(), FixedInterval(10)))

    assert detector.frames == [1, 11, 21]
    assert [r.frame_index for r in results if r.detected] == [1, 11, 21]
    assert all(r.detect_ms == 0.0 for r in results if not r.detected)
    # The track is reported on each frame, also between detector runs.
    assert all([o.track_id for o in r.objects] == [1] for r in results)


def test_pipeline_default_detects_on_each_frame() -> None:
    scene = SyntheticScene((), length=5)
    detector = _CountingDetector(scene)
    list(run(index_frames(scene.length), detector, Tracker()))
    assert detector.frames == [1, 2, 3, 4, 5]
