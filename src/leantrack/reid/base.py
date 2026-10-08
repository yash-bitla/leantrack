from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from leantrack._types import FloatArray, Image


class Embedder(Protocol):
    def embed(self, image: Image, boxes: FloatArray) -> FloatArray:
        """Appearance vectors for xyxy boxes of one image: shape (N, D), each with unit length."""
        ...


def crop(image: Image, box: FloatArray, size: tuple[int, int]) -> Image:
    """The pixels of an xyxy box, resized to `size` (width, height).

    The box is clipped to the image. A box without pixels in the image gives a black crop.
    """
    height, width = image.shape[:2]
    x1, y1 = max(0, round(float(box[0]))), max(0, round(float(box[1])))
    x2, y2 = min(width, round(float(box[2]))), min(height, round(float(box[3])))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(image[y1:y2, x1:x2], size, interpolation=cv2.INTER_LINEAR)


def normalize(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
