from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from leantrack._types import Detections, FloatArray, Frame, Image, IntArray

_STRIDES = (8, 16, 32)
_PAD_VALUE = 114


class YoloxDetector:
    """YOLOX through ONNX Runtime. Works with the ONNX files of the YOLOX 0.1.1 release."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.1,
        nms_iou: float = 0.7,
        classes: Sequence[int] | None = (0,),
        providers: Sequence[str] = ("CPUExecutionProvider",),
        threads: int | None = None,
    ) -> None:
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(model_path), options, providers=list(providers))
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._height, self._width = int(model_input.shape[2]), int(model_input.shape[3])
        self._score_threshold = score_threshold
        self._nms_iou = nms_iou
        self._classes = None if classes is None else np.array(classes, dtype=np.int64)
        self._grid, self._stride = _grid(self._height, self._width)

    def detect(self, frame: Frame) -> Detections:
        if frame.image is None:
            raise ValueError("YoloxDetector needs the frame pixels")
        blob, ratio = letterbox(frame.image, self._height, self._width)
        (output,) = self._session.run(None, {self._input_name: blob})
        return decode(
            output[0].astype(np.float64),
            self._grid,
            self._stride,
            ratio,
            (frame.image.shape[0], frame.image.shape[1]),
            score_threshold=self._score_threshold,
            nms_iou=self._nms_iou,
            classes=self._classes,
        )


def letterbox(image: Image, height: int, width: int) -> tuple[Image, float]:
    """Resize an image into the model input, with padding at the right and the bottom.

    Returns the (1, 3, H, W) input and the scale from image pixels to input pixels.
    """
    ratio = min(height / image.shape[0], width / image.shape[1])
    new_h, new_w = round(image.shape[0] * ratio), round(image.shape[1] * ratio)
    padded = np.full((height, width, 3), _PAD_VALUE, dtype=np.uint8)
    padded[:new_h, :new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # These models take BGR pixel values in 0..255 without normalization.
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    return blob, ratio


def decode(
    output: FloatArray,
    grid: FloatArray,
    stride: FloatArray,
    ratio: float,
    image_shape: tuple[int, int],
    *,
    score_threshold: float,
    nms_iou: float,
    classes: IntArray | None,
) -> Detections:
    """Convert the raw YOLOX head output of one image to boxes in image pixels.

    Each output row is (dx, dy, log w, log h, objectness, class scores...) for one grid cell.
    """
    class_ids = output[:, 5:].argmax(axis=1)
    scores = output[:, 4] * output[np.arange(len(output)), 5 + class_ids]
    keep = scores >= score_threshold
    if classes is not None:
        keep &= np.isin(class_ids, classes)
    if not keep.any():
        return Detections.empty()

    centers = (output[keep, :2] + grid[keep]) * stride[keep, None]
    sizes = np.exp(output[keep, 2:4]) * stride[keep, None]
    boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_shape[0])
    scores, class_ids = scores[keep], class_ids[keep]

    xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
    # An offset per class keeps boxes of different classes apart in one NMS call.
    offset = class_ids[:, None] * (max(image_shape) + 1.0) * np.array([1.0, 1.0, 0.0, 0.0])
    kept = cv2.dnn.NMSBoxes((xywh + offset).tolist(), scores.tolist(), score_threshold, nms_iou)
    index = np.asarray(kept, dtype=np.int64).reshape(-1)
    return Detections(boxes[index], scores[index], class_ids[index].astype(np.int64))


def _grid(height: int, width: int) -> tuple[FloatArray, FloatArray]:
    grids, strides = [], []
    for stride in _STRIDES:
        rows, cols = height // stride, width // stride
        xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
        grids.append(np.stack([xs, ys], axis=2).reshape(-1, 2))
        strides.append(np.full(rows * cols, stride))
    return (
        np.concatenate(grids).astype(np.float64),
        np.concatenate(strides).astype(np.float64),
    )
