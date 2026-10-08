from __future__ import annotations

import cv2
import numpy as np

from leantrack._types import FloatArray, Image
from leantrack.reid.base import crop, normalize

_SIZE = (32, 64)


class HistogramEmbedder:
    """Hue and saturation histograms of horizontal stripes of the box.

    This embedder needs no model. It is the low-cost baseline for the learned embedders.
    The stripes keep a part of the vertical layout, for example the colors of the upper
    and the lower clothes of a person.
    """

    def __init__(self, stripes: int = 4, hue_bins: int = 12, saturation_bins: int = 6) -> None:
        self._stripes = stripes
        self._bins = [hue_bins, saturation_bins]

    def embed(self, image: Image, boxes: FloatArray) -> FloatArray:
        size = self._stripes * self._bins[0] * self._bins[1]
        vectors = np.zeros((len(boxes), size), dtype=np.float64)
        for i, box in enumerate(boxes):
            hsv = cv2.cvtColor(crop(image, box, _SIZE), cv2.COLOR_BGR2HSV)
            parts = []
            for stripe in np.array_split(hsv, self._stripes, axis=0):
                hist = cv2.calcHist([stripe], [0, 1], None, self._bins, [0, 180, 0, 256])
                parts.append(np.sqrt(hist.ravel() / max(float(hist.sum()), 1.0)))
            vectors[i] = np.concatenate(parts)
        return normalize(vectors)
