from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from leantrack._types import FloatArray


@dataclass(frozen=True, slots=True)
class Assignment:
    matches: list[tuple[int, int]]
    unmatched_rows: list[int]
    unmatched_cols: list[int]


def assign(cost: FloatArray, max_cost: float) -> Assignment:
    """Minimum-cost one-to-one assignment. Pairs with a cost above `max_cost` stay unmatched."""
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return Assignment([], list(range(n_rows)), list(range(n_cols)))

    # Forbidden pairs get a finite cost that is larger than each permitted pair. Thus the
    # solver uses a forbidden pair only when no permitted pair remains for that row.
    gated = np.where(cost > max_cost, max_cost + 1.0, cost)
    rows, cols = linear_sum_assignment(gated)

    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= max_cost]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return Assignment(
        matches,
        [r for r in range(n_rows) if r not in matched_rows],
        [c for c in range(n_cols) if c not in matched_cols],
    )
