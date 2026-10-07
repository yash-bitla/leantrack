from __future__ import annotations

from typing import Protocol

from leantrack._types import Detections, Frame


class Detector(Protocol):
    def detect(self, frame: Frame) -> Detections: ...
