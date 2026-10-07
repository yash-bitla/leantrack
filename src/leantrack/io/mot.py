from __future__ import annotations

import configparser
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

import numpy as np

from leantrack._types import Detections, TrackedObject


@dataclass(frozen=True, slots=True)
class MotSequence:
    """A sequence directory in the MOTChallenge layout."""

    root: Path
    name: str
    length: int
    frame_rate: float
    image_dir: Path

    @property
    def detections_path(self) -> Path:
        return self.root / "det" / "det.txt"

    @property
    def ground_truth_path(self) -> Path:
        return self.root / "gt" / "gt.txt"

    @staticmethod
    def load(root: str | Path) -> MotSequence:
        root = Path(root)
        info_path = root / "seqinfo.ini"
        if not info_path.is_file():
            raise FileNotFoundError(f"{info_path} not found: not a MOT sequence directory")
        parser = configparser.ConfigParser()
        parser.read(info_path)
        info = parser["Sequence"]
        return MotSequence(
            root=root,
            name=info.get("name", root.name),
            length=int(info["seqLength"]),
            frame_rate=float(info["frameRate"]),
            image_dir=root / info.get("imDir", "img1"),
        )


def read_detections(path: str | Path) -> dict[int, Detections]:
    """Read a MOT detection file: `frame, id, left, top, width, height, score, ...`."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    if rows.shape[1] < 7:
        raise ValueError(f"{path}: expected at least 7 columns, got {rows.shape[1]}")
    by_frame: dict[int, Detections] = {}
    frames = rows[:, 0].astype(np.int64)
    for frame in np.unique(frames):
        part = rows[frames == frame]
        boxes = part[:, 2:6].copy()
        boxes[:, 2:] += boxes[:, :2]
        by_frame[int(frame)] = Detections(
            boxes, part[:, 6].copy(), np.zeros(len(part), dtype=np.int64)
        )
    return by_frame


class MotWriter:
    """Writes tracks in the MOTChallenge result format."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file: TextIO | None = None

    def __enter__(self) -> MotWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, frame_index: int, objects: Iterable[TrackedObject]) -> None:
        if self._file is None:
            raise RuntimeError("MotWriter must be used as a context manager")
        for obj in objects:
            x1, y1, x2, y2 = obj.box
            self._file.write(
                f"{frame_index},{obj.track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                f"{obj.score:.4f},-1,-1,-1\n"
            )
