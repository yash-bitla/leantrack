from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from leantrack._types import Detections, Frame


class UltralyticsDetector:
    """Optional backend for Ultralytics YOLO models.

    The `ultralytics` package and its pretrained weights have the AGPL-3.0 license.
    This module is the only place that imports the package, and the core install does
    not include it. Install it with `pip install leantrack[ultralytics]`.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.1,
        nms_iou: float = 0.7,
        classes: Sequence[int] | None = (0,),
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "the Ultralytics backend is not installed: pip install leantrack[ultralytics]"
            ) from exc
        self._model = YOLO(str(model_path))
        self._score_threshold = score_threshold
        self._nms_iou = nms_iou
        self._classes = None if classes is None else list(classes)
        self._device = device

    def detect(self, frame: Frame) -> Detections:
        if frame.image is None:
            raise ValueError("UltralyticsDetector needs the frame pixels")
        (result,) = self._model.predict(
            frame.image,
            conf=self._score_threshold,
            iou=self._nms_iou,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )
        boxes = result.boxes
        return Detections(
            boxes.xyxy.cpu().numpy().astype(np.float64),
            boxes.conf.cpu().numpy().astype(np.float64),
            boxes.cls.cpu().numpy().astype(np.int64),
        )
