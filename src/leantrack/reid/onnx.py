from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from leantrack._types import FloatArray, Image
from leantrack.reid.base import crop, normalize

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxEmbedder:
    """A re-identification network in ONNX format, for example OSNet.

    The model input is (1, 3, H, W): RGB with ImageNet normalization. The model runs
    one time for each box, because the common exports have a fixed batch size of 1.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        threads: int | None = None,
    ) -> None:
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(model_path), options, providers=list(providers))
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._size = (int(model_input.shape[3]), int(model_input.shape[2]))

    def embed(self, image: Image, boxes: FloatArray) -> FloatArray:
        vectors = []
        for box in boxes:
            rgb = cv2.cvtColor(crop(image, box, self._size), cv2.COLOR_BGR2RGB)
            blob = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
            (output,) = self._session.run(None, {self._input_name: blob})
            vectors.append(output[0].astype(np.float64))
        if not vectors:
            return np.empty((0, 0), dtype=np.float64)
        return normalize(np.stack(vectors))
