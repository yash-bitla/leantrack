from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from leantrack._types import FloatArray


@dataclass(frozen=True, slots=True)
class FailurePredictor:
    """Logistic regression that gives the probability that a track box is wrong.

    "Wrong" is the label of the training data. In the experiments of this repository, it
    means an IoU below 0.5 with each annotated box. The model is small: one weight for
    each feature. Thus the run-time cost is one dot product for each track.
    """

    names: tuple[str, ...]
    mean: FloatArray
    scale: FloatArray
    weights: FloatArray
    bias: float

    def probability(self, features: FloatArray) -> FloatArray:
        if features.shape[1] != len(self.names):
            raise ValueError(f"expected {len(self.names)} features, got {features.shape[1]}")
        return expit((features - self.mean) / self.scale @ self.weights + self.bias)

    @staticmethod
    def fit(
        features: FloatArray, labels: FloatArray, names: tuple[str, ...], l2: float = 1.0
    ) -> FailurePredictor:
        """Fit by maximum likelihood with an L2 penalty on the weights.

        The loss has no class weights. Class weights change the predicted probabilities,
        and then the output is no longer a calibrated probability.
        """
        if len(features) != len(labels) or features.shape[1] != len(names):
            raise ValueError("features, labels and names do not agree in size")
        mean = features.mean(axis=0)
        scale = np.maximum(features.std(axis=0), 1e-9)
        x = (features - mean) / scale
        y = labels.astype(np.float64)

        def loss(params: FloatArray) -> tuple[float, FloatArray]:
            logits = x @ params[:-1] + params[-1]
            # log(1 + exp(z)) - y z is the negative log-likelihood of one sample.
            value = (
                np.sum(np.logaddexp(0.0, logits) - y * logits)
                + 0.5 * l2 * params[:-1] @ params[:-1]
            )
            residual = expit(logits) - y
            gradient = np.append(x.T @ residual + l2 * params[:-1], residual.sum())
            return float(value), gradient

        result = minimize(loss, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B")
        return FailurePredictor(names, mean, scale, result.x[:-1], float(result.x[-1]))

    def save(self, path: str | Path) -> None:
        data = {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @staticmethod
    def load(path: str | Path) -> FailurePredictor:
        data = json.loads(Path(path).read_text())
        return FailurePredictor(
            tuple(data["names"]),
            np.array(data["mean"], dtype=np.float64),
            np.array(data["scale"], dtype=np.float64),
            np.array(data["weights"], dtype=np.float64),
            float(data["bias"]),
        )


def expected_calibration_error(
    probabilities: FloatArray, labels: FloatArray, bins: int = 10
) -> float:
    """Mean absolute difference between the predicted and the observed rate, over equal bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(error)
