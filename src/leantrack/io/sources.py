from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
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


def paced(frames: Iterable[Frame], fps: float) -> Iterator[Frame]:
    """Give the frames at `fps`, as a camera does. A slow consumer gets no extra delay."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    start = time.perf_counter()
    for count, frame in enumerate(frames):
        delay = start + count / fps - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        yield frame


def video_fps(path: str | Path, default: float = 30.0) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return fps if fps > 0 else default
