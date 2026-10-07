from __future__ import annotations

import numpy as np

from leantrack._types import FloatArray


def xyxy_to_cxcywh(boxes: FloatArray) -> FloatArray:
    out = np.empty_like(boxes, dtype=np.float64)
    out[..., 2:] = boxes[..., 2:] - boxes[..., :2]
    out[..., :2] = boxes[..., :2] + out[..., 2:] / 2
    return out


def cxcywh_to_xyxy(boxes: FloatArray) -> FloatArray:
    out = np.empty_like(boxes, dtype=np.float64)
    out[..., :2] = boxes[..., :2] - boxes[..., 2:] / 2
    out[..., 2:] = boxes[..., :2] + boxes[..., 2:] / 2
    return out


def iou_matrix(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise IoU of xyxy boxes. Returns an array of shape (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
