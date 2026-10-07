from __future__ import annotations

from pathlib import Path

from leantrack._types import Detections, Frame
from leantrack.io.mot import read_detections


class MotFileDetector:
    """Replays the detections of a MOT `det.txt` file. It does not read the pixels.

    This backend makes tracker experiments deterministic and independent of a model.
    """

    def __init__(self, path: str | Path) -> None:
        self._by_frame = read_detections(path)

    def detect(self, frame: Frame) -> Detections:
        return self._by_frame.get(frame.index, Detections.empty())
