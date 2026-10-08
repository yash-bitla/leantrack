from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.failure import auc, average_precision
from leantrack._types import Detections, Frame
from leantrack.confidence.features import FEATURE_NAMES, track_features
from leantrack.confidence.predictor import FailurePredictor, expected_calibration_error
from leantrack.io.sources import index_frames
from leantrack.pipeline import run
from leantrack.schedule.policy import FixedInterval
from leantrack.synthetic import SyntheticObject, SyntheticScene
from leantrack.tracks.tracker import Tracker


def _sample(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels from a known logistic model: logit = 2 * x0 - 1 * x1 - 0.5."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, 2))
    probability = 1 / (1 + np.exp(-(2 * features[:, 0] - features[:, 1] - 0.5)))
    return features, (rng.random(count) < probability).astype(np.float64)


def test_fit_recovers_a_known_model_and_is_calibrated() -> None:
    features, labels = _sample(20000, seed=0)
    predictor = FailurePredictor.fit(features, labels, ("a", "b"))
    # The features have unit scale, so the weights are near the true coefficients.
    assert predictor.weights == pytest.approx([2.0, -1.0], abs=0.1)
    assert predictor.bias == pytest.approx(-0.5, abs=0.1)

    test_features, test_labels = _sample(20000, seed=1)
    probabilities = predictor.probability(test_features)
    assert expected_calibration_error(probabilities, test_labels) < 0.02
    assert auc(probabilities, test_labels) > 0.85


def test_predictor_round_trip(tmp_path: Path) -> None:
    features, labels = _sample(500, seed=0)
    predictor = FailurePredictor.fit(features, labels, ("a", "b"))
    predictor.save(tmp_path / "model.json")
    loaded = FailurePredictor.load(tmp_path / "model.json")
    assert loaded.names == ("a", "b")
    assert loaded.probability(features) == pytest.approx(predictor.probability(features))


def test_predictor_rejects_a_wrong_feature_count() -> None:
    features, labels = _sample(100, seed=0)
    predictor = FailurePredictor.fit(features, labels, ("a", "b"))
    with pytest.raises(ValueError, match="expected 2 features"):
        predictor.probability(np.zeros((3, 5)))
    with pytest.raises(ValueError):
        FailurePredictor.fit(features, labels[:-1], ("a", "b"))


def test_expected_calibration_error_known_value() -> None:
    # Each prediction is 0.8 and 60% of the labels are 1: the error is 0.2.
    probabilities = np.full(10, 0.8)
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float64)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.2)


def test_auc_and_average_precision_known_values() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert auc(np.array([0.9, 0.1, 0.8, 0.2]), labels) == 1.0
    assert auc(np.array([0.1, 0.9, 0.2, 0.8]), labels) == 0.0
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5
    # Order by score: 1, 0, 1, 0. Precision at the two hits: 1/1 and 2/3.
    assert average_precision(np.array([0.9, 0.8, 0.7, 0.1]), labels[[0, 1, 2, 3]]) == (
        pytest.approx((1 + 2 / 3) / 2)
    )


def _detections(boxes: list[tuple[float, float, float, float]]) -> Detections:
    return Detections(np.array(boxes), np.full(len(boxes), 0.9), np.zeros(len(boxes), np.int64))


def test_track_features() -> None:
    assert track_features([]).shape == (0, len(FEATURE_NAMES))

    tracker = Tracker()
    # Tracks 1 and 2 overlap. Track 3 is alone.
    boxes = [
        (100.0, 100.0, 140.0, 180.0),
        (120.0, 100.0, 160.0, 180.0),
        (400.0, 100.0, 440.0, 180.0),
    ]
    for _ in range(4):
        tracker.update(_detections(boxes))
    for _ in range(3):
        tracker.predict()
    features = track_features(tracker.tracks)
    column = {name: features[:, i] for i, name in enumerate(FEATURE_NAMES)}

    assert features.shape == (3, len(FEATURE_NAMES))
    assert column["frames_since_update"].tolist() == [3.0, 3.0, 3.0]
    assert column["log_hits"] == pytest.approx(np.log(4))
    # IoU of two 40 px wide boxes with a 20 px shift: 20 / 60.
    assert column["crowding"] == pytest.approx([1 / 3, 1 / 3, 0.0], abs=0.02)
    assert column["log_size"] == pytest.approx(np.log(np.sqrt(40 * 80)), abs=0.05)
    assert np.all(column["position_sigma"] > 0)


class _SceneDetector:
    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene

    def detect(self, frame: Frame) -> Detections:
        return self.scene.detections(frame.index)


def test_pipeline_reports_a_failure_probability() -> None:
    scene = SyntheticScene((SyntheticObject((100, 100, 140, 180), (2.0, 0.0)),), length=12)
    # A model with one active feature: the probability grows with the frames since a detection.
    weights = np.zeros(len(FEATURE_NAMES))
    weights[FEATURE_NAMES.index("frames_since_update")] = 1.0
    ones = np.ones(len(FEATURE_NAMES))
    predictor = FailurePredictor(FEATURE_NAMES, 0 * ones, ones, weights, bias=-2.0)

    policy = FixedInterval(6)
    plain = list(run(index_frames(scene.length), _SceneDetector(scene), Tracker(), policy))
    assert all(o.failure_probability is None for r in plain for o in r.objects)

    results = list(
        run(
            index_frames(scene.length),
            _SceneDetector(scene),
            Tracker(),
            policy,
            predictor=predictor,
        )
    )
    probability = [r.objects[0].failure_probability or 0.0 for r in results]
    assert all(r.objects[0].failure_probability is not None for r in results)
    # Frames 1 and 7 have a detector run. The probability grows between them.
    assert probability[6] == pytest.approx(probability[0])
    assert probability[0] < probability[1] < probability[5]
