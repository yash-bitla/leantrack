from __future__ import annotations

import numpy as np
import scipy.linalg

from leantrack._types import FloatArray

_NDIM = 4


class KalmanFilter:
    """Constant-velocity Kalman filter on (cx, cy, w, h).

    The state is (cx, cy, w, h, vcx, vcy, vw, vh). Velocities are per step, and one step
    is one frame. Process and measurement noise scale with the box size, because a
    large box moves more pixels per frame than a small box at the same real speed.
    """

    def __init__(self, std_position: float = 1 / 20, std_velocity: float = 1 / 160) -> None:
        self._std_position = std_position
        self._std_velocity = std_velocity
        self._motion = np.eye(2 * _NDIM)
        self._motion[:_NDIM, _NDIM:] = np.eye(_NDIM)
        self._project = np.eye(_NDIM, 2 * _NDIM)

    def _size_scale(self, mean: FloatArray) -> FloatArray:
        w, h = mean[2], mean[3]
        return np.array([w, h, w, h], dtype=np.float64)

    def initiate(self, measurement: FloatArray) -> tuple[FloatArray, FloatArray]:
        mean = np.concatenate([measurement, np.zeros(_NDIM)])
        scale = self._size_scale(mean)
        std = np.concatenate([2 * self._std_position * scale, 10 * self._std_velocity * scale])
        return mean, np.diag(std**2)

    def predict(self, mean: FloatArray, covariance: FloatArray) -> tuple[FloatArray, FloatArray]:
        scale = self._size_scale(mean)
        std = np.concatenate([self._std_position * scale, self._std_velocity * scale])
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + np.diag(std**2)
        return mean, covariance

    def update(
        self, mean: FloatArray, covariance: FloatArray, measurement: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        noise = np.diag((self._std_position * self._size_scale(mean)) ** 2)
        projected_mean = self._project @ mean
        projected_cov = self._project @ covariance @ self._project.T + noise

        # Solve for the gain with a Cholesky factor. This avoids an explicit inverse.
        factor = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve(
            factor, (covariance @ self._project.T).T, check_finite=False
        ).T
        mean = mean + gain @ (measurement - projected_mean)
        covariance = covariance - gain @ projected_cov @ gain.T
        return mean, covariance

    def gating_distance(
        self, mean: FloatArray, covariance: FloatArray, centers: FloatArray
    ) -> FloatArray:
        """Squared Mahalanobis distance from the predicted center to each (cx, cy) in `centers`.

        Only the position is used. The size of a box after an occlusion is not reliable.
        """
        difference = centers - mean[:2]
        factor = scipy.linalg.cho_factor(covariance[:2, :2], lower=True, check_finite=False)
        solved = scipy.linalg.cho_solve(factor, difference.T, check_finite=False)
        return np.einsum("ij,ji->i", difference, solved)
