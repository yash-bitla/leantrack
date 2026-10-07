from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from leantrack.boxes import cxcywh_to_xyxy, iou_matrix, xyxy_to_cxcywh

_coord = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False)
_size = st.floats(min_value=1.0, max_value=1e3, allow_nan=False)


@st.composite
def boxes(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(min_value=0, max_value=6))
    rows = []
    for _ in range(n):
        x, y, w, h = draw(_coord), draw(_coord), draw(_size), draw(_size)
        rows.append([x, y, x + w, y + h])
    return np.array(rows, dtype=np.float64).reshape(n, 4)


def test_iou_known_values() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    # Identical: 1. Half overlap: 50 / (100 + 100 - 50) = 1/3. Disjoint: 0.
    assert iou_matrix(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_iou_of_zero_area_box_is_zero() -> None:
    a = np.array([[5.0, 5.0, 5.0, 5.0]])
    assert iou_matrix(a, a)[0, 0] == 0.0


@given(boxes(), boxes())
def test_iou_is_bounded_and_symmetric(a: np.ndarray, b: np.ndarray) -> None:
    iou = iou_matrix(a, b)
    assert iou.shape == (len(a), len(b))
    assert np.all((iou >= 0.0) & (iou <= 1.0 + 1e-12))
    assert np.allclose(iou, iou_matrix(b, a).T)


@given(boxes())
def test_box_conversion_round_trip(a: np.ndarray) -> None:
    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(a)), a, atol=1e-6)
