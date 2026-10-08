from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from leantrack._types import Detections, FloatArray, IntArray
from leantrack.propagate.kalman import KalmanFilter
from leantrack.reid.base import crop
from leantrack.reid.histogram import HistogramEmbedder
from leantrack.reid.onnx import OnnxEmbedder
from leantrack.tracks.track import TrackState
from leantrack.tracks.tracker import Tracker, TrackerConfig

_MODEL = Path(__file__).parent.parent / "models" / "osnet_x0_25_msmt17.onnx"

RED = np.array([1.0, 0.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])


def _image() -> np.ndarray:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[:, :100] = (0, 0, 255)
    image[:, 100:200] = (255, 0, 0)
    image[:100, 200:] = (0, 255, 0)
    image[100:, 200:] = (0, 0, 255)
    return image


def test_crop_clips_the_box_to_the_image() -> None:
    image = _image()
    assert crop(image, np.array([-50.0, -50.0, 50.0, 50.0]), (8, 16)).shape == (16, 8, 3)
    outside = crop(image, np.array([400.0, 400.0, 450.0, 450.0]), (8, 16))
    assert outside.shape == (16, 8, 3) and not outside.any()


def test_histogram_separates_colors_and_layouts() -> None:
    boxes = np.array(
        [
            [10.0, 10.0, 90.0, 190.0],  # red
            [20.0, 30.0, 80.0, 150.0],  # red, other position and size
            [110.0, 10.0, 190.0, 190.0],  # blue
            [210.0, 10.0, 290.0, 190.0],  # green above red
        ]
    )
    vectors = HistogramEmbedder().embed(_image(), boxes)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.ones(4))
    distance = 1.0 - vectors @ vectors.T
    assert distance[0, 1] < 0.01
    assert distance[0, 2] > 0.5
    # Half of the stripes of the two-color box agree with the red box.
    assert 0.3 < distance[0, 3] < 0.7


def test_gating_distance_grows_with_the_offset_and_shrinks_with_uncertainty() -> None:
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 40.0, 80.0]))
    centers = np.array([[100.0, 100.0], [110.0, 100.0], [140.0, 100.0]])
    near = kf.gating_distance(mean, cov, centers)
    assert near[0] == pytest.approx(0.0)
    assert near[0] < near[1] < near[2]

    for _ in range(30):
        mean, cov = kf.predict(mean, cov)
    later = kf.gating_distance(mean, cov, centers)
    assert later[2] < near[2]


class _Appearance:
    """Gives each detection the vector of its color. Counts the vectors that it returns."""

    def __init__(self) -> None:
        self.colors: list[FloatArray] = []
        self.requested = 0

    def frame(self, boxes: Sequence[tuple[float, ...]], colors: list[FloatArray]) -> Detections:
        self.colors = colors
        count = len(boxes)
        return Detections(
            np.array(boxes, dtype=np.float64).reshape(count, 4),
            np.full(count, 0.9),
            np.zeros(count, dtype=np.int64),
        )

    def embed(self, index: IntArray) -> FloatArray:
        self.requested += len(index)
        return np.stack([self.colors[i] for i in index])


def _lose_track(appearance: _Appearance, tracker: Tracker, use_embed: bool) -> None:
    """Track a red object for 10 frames, then give 20 frames without detections."""
    embed = appearance.embed if use_embed else None
    for _ in range(10):
        tracker.update(appearance.frame([(100, 100, 140, 180)], [RED]), embed=embed)
    for _ in range(20):
        tracker.update(appearance.frame([], []), embed=embed)
    assert tracker.tracks[0].state is TrackState.LOST


# The object returns 70 px to the right: no overlap with the predicted box.
_RETURN = (170.0, 100.0, 210.0, 180.0)


def test_lost_track_recovers_by_appearance() -> None:
    appearance = _Appearance()
    tracker = Tracker(TrackerConfig(max_lost_frames=60))
    _lose_track(appearance, tracker, use_embed=True)
    reported = tracker.update(appearance.frame([_RETURN], [RED]), embed=appearance.embed)
    assert [o.track_id for o in reported] == [1]
    assert tracker.tracks[0].box == pytest.approx(_RETURN, abs=15.0)


def test_without_appearance_the_object_gets_a_new_id() -> None:
    appearance = _Appearance()
    tracker = Tracker(TrackerConfig(max_lost_frames=60))
    _lose_track(appearance, tracker, use_embed=False)
    for _ in range(3):
        reported = tracker.update(appearance.frame([_RETURN], [RED]))
    assert [o.track_id for o in reported] == [2]


def test_a_different_appearance_does_not_recover_the_track() -> None:
    appearance = _Appearance()
    tracker = Tracker(TrackerConfig(max_lost_frames=60))
    _lose_track(appearance, tracker, use_embed=True)
    for _ in range(3):
        reported = tracker.update(appearance.frame([_RETURN], [BLUE]), embed=appearance.embed)
    assert [o.track_id for o in reported] == [2]


def test_a_distant_detection_does_not_recover_the_track() -> None:
    appearance = _Appearance()
    tracker = Tracker(TrackerConfig(max_lost_frames=60))
    _lose_track(appearance, tracker, use_embed=True)
    far = (900.0, 600.0, 940.0, 680.0)
    for _ in range(3):
        reported = tracker.update(appearance.frame([far], [RED]), embed=appearance.embed)
    assert [o.track_id for o in reported] == [2]


def test_embeddings_are_requested_only_when_they_are_necessary() -> None:
    appearance = _Appearance()
    tracker = Tracker(TrackerConfig(embedding_refresh=5))
    for _ in range(20):
        tracker.update(appearance.frame([(100, 100, 140, 180)], [RED]), embed=appearance.embed)
    # One vector for the first match and one for each 5th match after it.
    assert appearance.requested <= 5
    assert tracker.tracks[0].embedding == pytest.approx(RED)


def test_an_overlapped_detection_does_not_update_the_appearance() -> None:
    appearance = _Appearance()
    tracker = Tracker(TrackerConfig(embedding_refresh=1))
    boxes = [(100.0, 100.0, 140.0, 180.0), (110.0, 100.0, 150.0, 180.0)]
    for _ in range(5):
        tracker.update(appearance.frame(boxes, [RED, BLUE]), embed=appearance.embed)
    assert all(t.embedding is None for t in tracker.tracks)


@pytest.mark.skipif(not _MODEL.is_file(), reason="models/osnet_x0_25_msmt17.onnx is not present")
def test_onnx_embedder_returns_unit_vectors() -> None:
    vectors = OnnxEmbedder(_MODEL).embed(
        _image(), np.array([[10.0, 10.0, 90.0, 190.0], [110.0, 10.0, 190.0, 190.0]])
    )
    assert vectors.shape == (2, 512)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.ones(2))


def test_lost_track_does_not_take_an_overlapping_object_that_looks_different() -> None:
    same_place = (100.0, 100.0, 140.0, 180.0)

    appearance = _Appearance()
    tracker = Tracker(TrackerConfig(max_lost_frames=60))
    _lose_track(appearance, tracker, use_embed=True)
    for _ in range(3):
        reported = tracker.update(appearance.frame([same_place], [BLUE]), embed=appearance.embed)
    assert [o.track_id for o in reported] == [2]

    # Without appearance, the IoU match gives the old ID to the new object.
    appearance = _Appearance()
    tracker = Tracker(TrackerConfig(max_lost_frames=60))
    _lose_track(appearance, tracker, use_embed=False)
    reported = tracker.update(appearance.frame([same_place], [BLUE]))
    assert [o.track_id for o in reported] == [1]
