from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from leantrack._types import FloatArray, Image


class FlowPropagator:
    """Moves boxes from the previous frame to the current frame with sparse optical flow.

    Each box gets a grid of points. Lucas-Kanade flow tracks the points forward and then
    backward. A point is reliable if the backward track returns to its start. The box
    moves by the median displacement of its reliable points, as in MedianFlow
    (Kalal et al., 2010). The box size does not change.
    """

    def __init__(
        self,
        *,
        max_side: int = 960,
        grid: int = 5,
        max_backward_error: float = 1.0,
        min_points: int = 6,
        window: int = 15,
        levels: int = 3,
    ) -> None:
        if grid < 2:
            raise ValueError("grid must be at least 2")
        self._max_side = max_side
        self._grid = grid
        self._max_backward_error = max_backward_error
        self._min_points = min_points
        self._window = (window, window)
        self._levels = levels
        self._previous: Image | None = None
        self._current: Image | None = None
        self._scale = 1.0

    def observe(self, image: Image) -> None:
        """Give the propagator the next frame. Call this one time for each frame."""
        self._scale = min(1.0, self._max_side / max(image.shape[:2]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._scale < 1.0:
            gray = cv2.resize(gray, None, fx=self._scale, fy=self._scale)
        self._previous, self._current = self._current, gray

    def propagate(self, boxes: FloatArray) -> list[FloatArray | None]:
        """Move xyxy boxes of the previous frame to the current frame.

        An entry is None if the flow of that box is not reliable.
        """
        if len(boxes) == 0:
            return []
        if self._previous is None or self._current is None:
            return [None] * len(boxes)

        # Use the central 80% of each box. The border contains more background.
        fractions = np.linspace(0.1, 0.9, self._grid)
        grid_x, grid_y = np.meshgrid(fractions, fractions)
        fx, fy = grid_x.ravel(), grid_y.ravel()
        x = boxes[:, None, 0] + fx[None, :] * (boxes[:, None, 2] - boxes[:, None, 0])
        y = boxes[:, None, 1] + fy[None, :] * (boxes[:, None, 3] - boxes[:, None, 1])
        start = (np.stack([x, y], axis=2) * self._scale).astype(np.float32).reshape(-1, 1, 2)

        forward, ok_forward = self._track(self._previous, self._current, start)
        backward, ok_backward = self._track(self._current, self._previous, forward)
        error = np.linalg.norm((backward - start).reshape(-1, 2), axis=1)
        reliable = (ok_forward & ok_backward & (error <= self._max_backward_error)).reshape(
            len(boxes), -1
        )
        shift = ((forward - start).reshape(len(boxes), -1, 2) / self._scale).astype(np.float64)

        moved: list[FloatArray | None] = []
        for box, box_shift, box_reliable in zip(boxes, shift, reliable, strict=True):
            if box_reliable.sum() < self._min_points:
                moved.append(None)
                continue
            dx, dy = np.median(box_shift[box_reliable], axis=0)
            moved.append(box + np.array([dx, dy, dx, dy]))
        return moved

    def _track(
        self, source: Image, target: Image, points: Image
    ) -> tuple[Image, NDArray[np.bool_]]:
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            source,
            target,
            points,
            None,  # type: ignore[call-overload]
            winSize=self._window,
            maxLevel=self._levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        return moved, status.ravel() == 1
