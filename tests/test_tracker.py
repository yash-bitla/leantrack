from __future__ import annotations

import numpy as np
import pytest

from leantrack._types import Detections
from leantrack.synthetic import SyntheticObject, SyntheticScene
from leantrack.tracks.track import Track, TrackState
from leantrack.tracks.tracker import Tracker, TrackerConfig


def _run(scene: SyntheticScene, tracker: Tracker) -> list[dict[int, tuple[float, ...]]]:
    return [
        {obj.track_id: obj.box for obj in tracker.update(scene.detections(frame))}
        for frame in range(1, scene.length + 1)
    ]


def _one(box: tuple[float, float, float, float], score: float) -> Detections:
    return Detections(np.array([box]), np.array([score]), np.array([0]))


def test_one_object_keeps_one_id() -> None:
    scene = SyntheticScene(
        (SyntheticObject((100, 100, 140, 180), (4.0, 1.0)),), length=60, jitter=1.0
    )
    frames = _run(scene, Tracker())
    assert all(list(f) == [1] for f in frames)


def test_parallel_objects_keep_separate_ids() -> None:
    scene = SyntheticScene(
        (
            SyntheticObject((100, 100, 140, 180), (3.0, 0.0)),
            SyntheticObject((100, 300, 140, 380), (3.0, 0.0)),
            SyntheticObject((600, 200, 640, 280), (-3.0, 0.0)),
        ),
        length=80,
        jitter=1.0,
    )
    frames = _run(scene, Tracker())
    assert all(sorted(f) == [1, 2, 3] for f in frames)
    # Each id stays on its object: id 2 stays in the lower row.
    assert all(f[2][1] > 250 for f in frames)


def test_short_occlusion_keeps_the_id() -> None:
    obj = SyntheticObject((100, 100, 140, 180), (3.0, 0.0), hidden=((30, 44),))
    frames = _run(SyntheticScene((obj,), length=70), Tracker(TrackerConfig(max_lost_frames=30)))
    assert all(f == {} for f in frames[29:44])
    assert all(list(f) == [1] for f in frames[44:])


def test_long_occlusion_starts_a_new_id() -> None:
    obj = SyntheticObject((100, 100, 140, 180), (3.0, 0.0), hidden=((20, 39),))
    frames = _run(SyntheticScene((obj,), length=60), Tracker(TrackerConfig(max_lost_frames=10)))
    assert list(frames[18]) == [1]
    assert list(frames[-1]) == [2]


def test_lost_track_keeps_its_size() -> None:
    tracker = Tracker()
    # The box grows 2 px per frame, so the filter learns a size velocity.
    for step in range(20):
        tracker.update(_one((100, 100, 140 + 2 * step, 180 + 2 * step), 0.9))
    width = tracker.tracks[0].mean[2]
    for _ in range(15):
        tracker.update(Detections.empty())
    track = tracker.tracks[0]
    assert track.state is TrackState.LOST
    assert track.mean[2] == pytest.approx(width, abs=3.0)


def test_single_frame_false_positive_is_not_reported() -> None:
    tracker = Tracker()
    box = (100.0, 100.0, 140.0, 180.0)
    tracker.update(_one(box, 0.9))
    assert tracker.update(Detections(*_two(box, (500, 500, 540, 580)))) != []
    reported = tracker.update(_one(box, 0.9))
    assert [o.track_id for o in reported] == [1]
    assert len(tracker.tracks) == 1


def _two(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.array([a, b], dtype=np.float64), np.array([0.9, 0.9]), np.array([0, 0])


def test_new_object_is_reported_from_its_second_frame() -> None:
    tracker = Tracker()
    tracker.update(Detections.empty())
    box = (100.0, 100.0, 140.0, 180.0)
    assert tracker.update(_one(box, 0.9)) == []
    assert [o.track_id for o in tracker.update(_one(box, 0.9))] == [1]


def test_low_score_detection_keeps_a_confirmed_track() -> None:
    tracker = Tracker()
    box = (100.0, 100.0, 140.0, 180.0)
    for _ in range(5):
        tracker.update(_one(box, 0.9))
    reported = tracker.update(_one(box, 0.3))
    assert [o.track_id for o in reported] == [1]
    assert tracker.tracks[0].state is TrackState.CONFIRMED


def test_low_score_detection_does_not_start_a_track() -> None:
    tracker = Tracker()
    for _ in range(5):
        tracker.update(_one((100, 100, 140, 180), 0.3))
    assert tracker.tracks == []


def test_illegal_transition_raises() -> None:
    track = Track(1, np.zeros(8), np.eye(8), 0.9, 0)
    with pytest.raises(ValueError, match="tentative -> lost"):
        track.transition(TrackState.LOST)


def test_config_rejects_inverted_scores() -> None:
    with pytest.raises(ValueError):
        TrackerConfig(high_score=0.2, low_score=0.5)


def test_predict_reports_the_extrapolated_box() -> None:
    tracker = Tracker()
    for step in range(20):
        tracker.update(_one((100 + 5 * step, 100, 140 + 5 * step, 180), 0.9))
    for step in range(20, 25):
        (reported,) = tracker.predict()
        assert reported.track_id == 1
        assert reported.box[0] == pytest.approx(100 + 5 * step, abs=2.0)
    assert tracker.tracks[0].state is TrackState.CONFIRMED


def test_sparse_detection_keeps_the_id() -> None:
    scene = SyntheticScene((SyntheticObject((100, 100, 140, 180), (3.0, 0.0)),), length=90)
    tracker = Tracker()
    ids: set[int] = set()
    for frame in range(1, scene.length + 1):
        detect = (frame - 1) % 5 == 0
        objects = tracker.update(scene.detections(frame)) if detect else tracker.predict()
        ids.update(o.track_id for o in objects)
    assert ids == {1}


def test_predict_removes_an_expired_lost_track() -> None:
    tracker = Tracker(TrackerConfig(max_lost_frames=3))
    box = (100.0, 100.0, 140.0, 180.0)
    tracker.update(_one(box, 0.9))
    tracker.update(Detections.empty())
    assert tracker.tracks[0].state is TrackState.LOST
    for _ in range(3):
        assert tracker.predict() == []
    assert tracker.tracks == []
