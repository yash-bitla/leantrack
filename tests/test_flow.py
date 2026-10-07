from __future__ import annotations

import cv2
import numpy as np
import pytest

from leantrack._types import Detections, Frame
from leantrack.pipeline import run
from leantrack.propagate.flow import FlowPropagator
from leantrack.schedule.policy import FixedInterval
from leantrack.tracks.tracker import Tracker


def _texture(height: int = 240, width: int = 320) -> np.ndarray:
    noise = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (7, 7), 0)


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(image, (dy, dx), axis=(0, 1))


def test_box_follows_a_known_image_shift() -> None:
    image = _texture()
    propagator = FlowPropagator()
    propagator.observe(image)
    propagator.observe(_shift(image, 4, -3))

    boxes = np.array([[100.0, 80.0, 160.0, 160.0], [200.0, 60.0, 240.0, 140.0]])
    moved = propagator.propagate(boxes)
    for box, new in zip(boxes, moved, strict=True):
        assert new is not None
        assert new == pytest.approx(box + np.array([4, -3, 4, -3]), abs=0.3)


def test_downscaled_flow_reports_full_resolution_pixels() -> None:
    image = cv2.resize(_texture(), (1280, 960), interpolation=cv2.INTER_CUBIC)
    propagator = FlowPropagator(max_side=640)
    propagator.observe(image)
    propagator.observe(_shift(image, 8, 0))
    (moved,) = propagator.propagate(np.array([[400.0, 300.0, 600.0, 600.0]]))
    assert moved is not None
    assert moved == pytest.approx([408.0, 300.0, 608.0, 600.0], abs=1.0)


def test_box_without_texture_is_unreliable() -> None:
    flat = np.full((240, 320, 3), 128, np.uint8)
    propagator = FlowPropagator()
    propagator.observe(flat)
    propagator.observe(flat)
    assert propagator.propagate(np.array([[100.0, 80.0, 160.0, 160.0]])) == [None]


def test_no_result_before_the_second_frame() -> None:
    propagator = FlowPropagator()
    assert propagator.propagate(np.empty((0, 4))) == []
    propagator.observe(_texture())
    assert propagator.propagate(np.array([[100.0, 80.0, 160.0, 160.0]])) == [None]


class _FirstFrameDetector:
    """Sees one object on frame 1 only, at a fixed position."""

    box = (100.0, 80.0, 160.0, 160.0)

    def detect(self, frame: Frame) -> Detections:
        if frame.index != 1:
            return Detections.empty()
        return Detections(np.array([self.box]), np.array([0.9]), np.array([0]))


def _moving_frames(count: int, dx: int) -> list[Frame]:
    image = _texture()
    return [Frame(i + 1, _shift(image, dx * i, 0)) for i in range(count)]


def test_pipeline_follows_motion_that_the_kalman_filter_cannot_know() -> None:
    # The track has one detection, so its velocity estimate is zero. The image moves 3 px
    # per frame. Only the flow can report that.
    frames = _moving_frames(8, dx=3)
    policy = FixedInterval(100)

    without = list(run(frames, _FirstFrameDetector(), Tracker(), policy))
    with_flow = list(run(frames, _FirstFrameDetector(), Tracker(), policy, FlowPropagator()))

    assert without[-1].objects[0].box[0] == pytest.approx(100.0, abs=0.5)
    # The filter smooths the first flow measurements, so the box lags a few pixels.
    assert with_flow[-1].objects[0].box[0] == pytest.approx(100.0 + 3 * 7, abs=4.0)
    assert with_flow[-1].propagate_ms > 0.0
    assert without[-1].propagate_ms == 0.0


def test_pipeline_with_propagator_needs_pixels() -> None:
    with pytest.raises(ValueError, match="pixels"):
        list(run([Frame(1)], _FirstFrameDetector(), Tracker(), None, FlowPropagator()))


def test_lost_track_ignores_a_propagated_box() -> None:
    tracker = Tracker()
    box = np.array([100.0, 80.0, 160.0, 160.0])
    tracker.update(Detections(box[None], np.array([0.9]), np.array([0])))
    tracker.update(Detections.empty())
    before = tracker.tracks[0].box.copy()
    tracker.predict({1: box + 50.0})
    assert tracker.tracks[0].box == pytest.approx(before, abs=1e-6)
