from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from leantrack._types import Frame
from leantrack.detect.ultralytics import UltralyticsDetector
from leantrack.detect.yolox import YoloxDetector, _grid, decode

_MODEL = Path(__file__).parent.parent / "models" / "yolox_nano.onnx"


def _row(
    grid_xy: tuple[int, int],
    stride: int,
    box: tuple[float, float, float, float],
    scores: list[float],
) -> list[float]:
    """Raw head output for a box (cx, cy, w, h) in network pixels, from one grid cell."""
    cx, cy, w, h = box
    return [
        cx / stride - grid_xy[0],
        cy / stride - grid_xy[1],
        float(np.log(w / stride)),
        float(np.log(h / stride)),
        *scores,
    ]


def test_grid_layout() -> None:
    grid, stride = _grid(64, 96)
    # Strides 8, 16, 32 give 8x12 + 4x6 + 2x3 cells.
    assert len(grid) == len(stride) == 96 + 24 + 6
    assert grid[13].tolist() == [1.0, 1.0]
    assert stride[96] == 16


def test_decode_scales_boxes_to_image_pixels_and_suppresses_duplicates() -> None:
    grid, stride = _grid(64, 96)
    output = np.zeros((len(grid), 7))
    output[:, 4] = 0.0
    # Two cells predict almost the same person. One cell predicts a class that is filtered.
    output[13] = _row((1, 1), 8, (12.0, 12.0, 8.0, 16.0), [0.9, 1.0, 0.0])
    output[14] = _row((2, 1), 8, (12.5, 12.0, 8.0, 16.0), [0.6, 1.0, 0.0])
    output[40] = _row((4, 3), 8, (36.0, 28.0, 8.0, 8.0), [0.9, 0.0, 1.0])

    detections = decode(
        output,
        grid,
        stride,
        ratio=0.5,
        image_shape=(128, 192),
        score_threshold=0.1,
        nms_iou=0.7,
        classes=np.array([0]),
    )
    # Network box (8, 4, 16, 20) divided by the ratio 0.5 gives (16, 8, 32, 40).
    assert detections.boxes == pytest.approx(np.array([[16.0, 8.0, 32.0, 40.0]]))
    assert detections.scores == pytest.approx([0.9])
    assert detections.classes.tolist() == [0]


def test_decode_keeps_overlapping_boxes_of_different_classes() -> None:
    grid, stride = _grid(64, 96)
    output = np.zeros((len(grid), 7))
    output[13] = _row((1, 1), 8, (12.0, 12.0, 8.0, 16.0), [0.9, 1.0, 0.0])
    output[14] = _row((2, 1), 8, (12.0, 12.0, 8.0, 16.0), [0.8, 0.0, 1.0])
    detections = decode(
        output, grid, stride, 1.0, (64, 96), score_threshold=0.1, nms_iou=0.7, classes=None
    )
    assert sorted(detections.classes.tolist()) == [0, 1]


def test_decode_with_no_detection() -> None:
    grid, stride = _grid(64, 96)
    detections = decode(
        np.zeros((len(grid), 7)),
        grid,
        stride,
        1.0,
        (64, 96),
        score_threshold=0.1,
        nms_iou=0.7,
        classes=None,
    )
    assert len(detections) == 0


@pytest.mark.skipif(not _MODEL.is_file(), reason="models/yolox_nano.onnx is not present")
def test_model_runs_on_an_image() -> None:
    detector = YoloxDetector(_MODEL)
    image = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    detections = detector.detect(Frame(1, image))
    assert detections.boxes.shape == (len(detections), 4)
    assert np.all(detections.boxes[:, 2] <= 640) and np.all(detections.boxes[:, 3] <= 360)
    with pytest.raises(ValueError):
        detector.detect(Frame(1))


def test_ultralytics_backend_reports_a_missing_package() -> None:
    pytest.importorskip("onnxruntime")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"leantrack\[ultralytics\]"):
            UltralyticsDetector("yolo11n.pt")
    else:
        pytest.skip("ultralytics is installed")
