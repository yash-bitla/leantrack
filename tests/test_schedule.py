from __future__ import annotations

import numpy as np
import pytest

from leantrack._types import Detections, Frame
from leantrack.io.sources import index_frames
from leantrack.pipeline import run
from leantrack.schedule.policy import ConfidenceTrigger, FixedInterval, ScheduleContext
from leantrack.synthetic import SyntheticObject, SyntheticScene
from leantrack.tracks.track import Track, TrackState
from leantrack.tracks.tracker import Tracker


class _CountingDetector:
    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene
        self.frames: list[int] = []

    def detect(self, frame: Frame) -> Detections:
        self.frames.append(frame.index)
        return self.scene.detections(frame.index)


def _at(frame_index: int) -> ScheduleContext:
    return ScheduleContext(frame_index, None, [])


def test_fixed_interval_schedule() -> None:
    policy = FixedInterval(4)
    assert [i for i in range(1, 12) if policy.should_detect(_at(i))] == [1, 5, 9]
    assert all(FixedInterval(1).should_detect(_at(i)) for i in range(1, 5))
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


def _track(*, reliability: float = 1.0, motion: float = 0.0, confirmed: bool = True) -> Track:
    track = Track(1, np.zeros(8), np.eye(8), 0.9, 0)
    if confirmed:
        track.transition(TrackState.CONFIRMED)
    track.reliability = reliability
    track.motion_since_update = motion
    return track


def _context(since: int | None, tracks: list[Track]) -> ScheduleContext:
    return ScheduleContext(10, since, tracks)


def test_trigger_detects_before_the_first_run_and_at_the_maximum_interval() -> None:
    policy = ConfidenceTrigger(max_interval=8)
    assert policy.should_detect(_context(None, []))
    assert not policy.should_detect(_context(7, [_track()]))
    assert policy.should_detect(_context(8, [_track()]))


def test_trigger_respects_the_minimum_interval() -> None:
    policy = ConfidenceTrigger(min_interval=3, max_motion=0.1)
    moved = [_track(motion=5.0)]
    assert not policy.should_detect(_context(2, moved))
    assert policy.should_detect(_context(3, moved))


def test_trigger_on_motion() -> None:
    policy = ConfidenceTrigger(max_motion=0.4)
    assert not policy.should_detect(_context(5, [_track(motion=0.3), _track(motion=0.1)]))
    assert policy.should_detect(_context(5, [_track(motion=0.5), _track(motion=0.1)]))
    # A tentative track does not trigger a run.
    assert not policy.should_detect(_context(5, [_track(motion=0.5, confirmed=False)]))


def test_trigger_on_unreliable_fraction() -> None:
    policy = ConfidenceTrigger(min_reliability=0.5, max_unreliable_fraction=0.25)
    one_of_four = [_track(reliability=0.2), _track(), _track(), _track()]
    two_of_four = [_track(reliability=0.2), _track(reliability=0.4), _track(), _track()]
    assert not policy.should_detect(_context(5, one_of_four))
    assert policy.should_detect(_context(5, two_of_four))


def test_trigger_without_conditions_uses_only_the_maximum_interval() -> None:
    policy = ConfidenceTrigger(max_interval=6)
    bad = [_track(reliability=0.0, motion=9.0)]
    assert not policy.should_detect(_context(5, bad))
    assert policy.should_detect(_context(6, bad))


def test_trigger_rejects_inverted_intervals() -> None:
    with pytest.raises(ValueError):
        ConfidenceTrigger(min_interval=5, max_interval=2)


def test_pipeline_gives_the_policy_the_frames_since_the_last_run() -> None:
    seen: list[int | None] = []

    class _Recorder:
        def should_detect(self, context: ScheduleContext) -> bool:
            seen.append(context.frames_since_detection)
            return context.frame_index in (1, 4)

    scene = SyntheticScene((SyntheticObject((100, 100, 140, 180), (2.0, 0.0)),), length=6)
    list(run(index_frames(scene.length), _CountingDetector(scene), Tracker(), _Recorder()))
    assert seen == [None, 1, 2, 3, 1, 2]


def test_motion_accumulates_between_detections_and_resets_on_a_match() -> None:
    tracker = Tracker()
    for step in range(10):
        box = (100.0 + 4 * step, 100.0, 140.0 + 4 * step, 180.0)
        tracker.update(Detections(np.array([box]), np.array([0.9]), np.array([0])))
    assert tracker.tracks[0].motion_since_update == 0.0

    for _ in range(5):
        tracker.predict()
    # About 4 px per frame for 5 frames, in units of sqrt(40 * 80) = 56.6 px.
    assert tracker.tracks[0].motion_since_update == pytest.approx(20 / 56.6, rel=0.15)

    box = (160.0, 100.0, 200.0, 180.0)
    tracker.update(Detections(np.array([box]), np.array([0.9]), np.array([0])))
    assert tracker.tracks[0].motion_since_update == 0.0
