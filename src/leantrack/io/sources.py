from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from leantrack._types import Frame

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def index_frames(length: int) -> Iterator[Frame]:
    """Frames without pixels, for detectors that replay stored detections."""
    for index in range(1, length + 1):
        yield Frame(index)


def image_dir_frames(directory: str | Path) -> Iterator[Frame]:
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no images in {directory}")
    for index, path in enumerate(paths, start=1):
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"cannot decode {path}")
        yield Frame(index, image)


def video_frames(path: str | Path) -> Iterator[Frame]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                return
            index += 1
            yield Frame(index, image)
    finally:
        capture.release()
