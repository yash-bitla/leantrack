from __future__ import annotations

import numpy as np

from leantrack.associate.matching import assign


def test_assignment_is_optimal_where_greedy_is_not() -> None:
    # Greedy takes (0, 0) at cost 0.1 and then must take (1, 1) at 0.9: total 1.0.
    # The optimum is (0, 1) + (1, 0): total 0.5.
    cost = np.array([[0.1, 0.2], [0.3, 0.9]])
    result = assign(cost, max_cost=1.0)
    assert sorted(result.matches) == [(0, 1), (1, 0)]
    assert result.unmatched_rows == []
    assert result.unmatched_cols == []


def test_pairs_above_max_cost_stay_unmatched() -> None:
    cost = np.array([[0.2, 0.95], [0.95, 0.9]])
    result = assign(cost, max_cost=0.8)
    assert result.matches == [(0, 0)]
    assert result.unmatched_rows == [1]
    assert result.unmatched_cols == [1]


def test_gate_does_not_remove_a_permitted_match() -> None:
    # Row 0 can use only column 0. Row 1 can use both. Each row must get a column.
    cost = np.array([[0.5, 0.99], [0.1, 0.6]])
    result = assign(cost, max_cost=0.8)
    assert sorted(result.matches) == [(0, 0), (1, 1)]


def test_empty_inputs() -> None:
    assert assign(np.zeros((0, 3)), 0.5).unmatched_cols == [0, 1, 2]
    assert assign(np.zeros((2, 0)), 0.5).unmatched_rows == [0, 1]


def test_rectangular_cost() -> None:
    cost = np.array([[0.1, 0.5, 0.7]])
    result = assign(cost, max_cost=0.8)
    assert result.matches == [(0, 0)]
    assert result.unmatched_cols == [1, 2]
