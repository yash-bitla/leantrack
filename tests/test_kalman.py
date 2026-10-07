from __future__ import annotations

import numpy as np

from leantrack.propagate.kalman import KalmanFilter


def test_filter_learns_constant_velocity() -> None:
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 40.0, 80.0]))
    for step in range(1, 31):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + 3.0 * step, 100.0, 40.0, 80.0]))
    assert abs(mean[4] - 3.0) < 0.05
    assert abs(mean[5]) < 0.05

    predicted, _ = kf.predict(mean, cov)
    assert abs(predicted[0] - (100.0 + 3.0 * 31)) < 0.5


def test_update_decreases_uncertainty_and_predict_increases_it() -> None:
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.0, 0.0, 50.0, 50.0]))
    predicted_mean, predicted_cov = kf.predict(mean, cov)
    assert np.trace(predicted_cov) > np.trace(cov)
    _, updated_cov = kf.update(predicted_mean, predicted_cov, predicted_mean[:4])
    assert np.trace(updated_cov) < np.trace(predicted_cov)


def test_covariance_stays_symmetric_positive_definite() -> None:
    kf = KalmanFilter()
    rng = np.random.default_rng(0)
    mean, cov = kf.initiate(np.array([200.0, 200.0, 30.0, 60.0]))
    for _ in range(200):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, mean[:4] + rng.normal(0, 2, 4))
        assert np.allclose(cov, cov.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(cov) > 0)
