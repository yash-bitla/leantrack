"""Experiment 6: a learned predictor of wrong track boxes.

The script runs the pipeline with a fixed detection interval and optical flow. For each
reported box it stores the track features and a label: the box is wrong if its IoU with
each annotated box is below 0.5. A logistic regression learns on sequences 02, 04 and 09.
The report uses sequences 05, 10, 11 and 13.

    python -m bench.failure yolox_s --jobs 6
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench._eval import METRICS, evaluate, markdown_table
from leantrack._types import FloatArray, TrackedObject
from leantrack.boxes import iou_matrix
from leantrack.confidence.features import FEATURE_NAMES, track_features
from leantrack.confidence.predictor import FailurePredictor, expected_calibration_error
from leantrack.detect.mot_file import MotFileDetector
from leantrack.io.mot import MotSequence, MotWriter
from leantrack.io.sources import image_dir_frames
from leantrack.pipeline import run
from leantrack.propagate.flow import FlowPropagator
from leantrack.schedule.policy import FixedInterval
from leantrack.tracks.tracker import Tracker

TUNE = ("02", "04", "09")
TEST = ("05", "10", "11", "13")
WRONG_IOU = 0.5
# Columns of a stored row before the features.
_META = ("sequence", "interval", "frame", "track_id", "x1", "y1", "x2", "y2", "score", "wrong")


def collect(args: tuple[Path, Path, str, int]) -> FloatArray:
    """Rows of (_META columns, features) for each reported box of one sequence."""
    data, cache, number, interval = args
    sequence = MotSequence.load(data / f"MOT17-{number}-FRCNN")
    # Each annotated box counts, also the classes that MOT17 does not score. A box on a
    # person in a vehicle is a correct box.
    truth = np.loadtxt(sequence.ground_truth_path, delimiter=",")
    truth_boxes = truth[:, 2:6].copy()
    truth_boxes[:, 2:] += truth_boxes[:, :2]
    truth_frames = truth[:, 0].astype(np.int64)

    tracker = Tracker()
    detector = MotFileDetector(cache / f"{sequence.name}.txt")
    results = run(
        image_dir_frames(sequence.image_dir),
        detector,
        tracker,
        FixedInterval(interval),
        FlowPropagator(),
    )
    rows = []
    for result in results:
        if not result.objects:
            continue
        tracks = tracker.tracks
        features = dict(zip((t.track_id for t in tracks), track_features(tracks), strict=True))
        boxes = np.array([o.box for o in result.objects])
        annotated = truth_boxes[truth_frames == result.frame_index]
        best = iou_matrix(boxes, annotated).max(axis=1) if len(annotated) else np.zeros(len(boxes))
        for obj, iou in zip(result.objects, best, strict=True):
            meta = [int(number), interval, result.frame_index, obj.track_id, *obj.box, obj.score]
            rows.append([*meta, float(iou < WRONG_IOU), *features[obj.track_id]])
    return np.array(rows, dtype=np.float64)


def auc(scores: FloatArray, labels: FloatArray) -> float:
    """Area under the ROC curve: the chance that a wrong box has a higher score than a good one."""
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Equal scores get their mean rank.
    for value in np.unique(scores[np.diff(np.sort(scores), prepend=np.nan) == 0]):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    positive = labels == 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: FloatArray, labels: FloatArray) -> float:
    order = (-scores).argsort(kind="stable")
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def _plot(probabilities: FloatArray, labels: FloatArray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.linspace(0, 1, 11)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, 9)
    predicted = [probabilities[index == b].mean() for b in range(10) if (index == b).any()]
    observed = [labels[index == b].mean() for b in range(10) if (index == b).any()]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(predicted, observed, marker="o", label="failure predictor")
    ax.set_xlabel("Predicted probability that the box is wrong")
    ax.set_ylabel("Observed rate of wrong boxes")
    ax.set_title("MOT17 train, sequences 05, 10, 11, 13")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def _filter_experiment(
    rows: FloatArray, probabilities: FloatArray, data: Path, out: Path, limits: list[float]
) -> dict[str, dict[str, float]]:
    """MOT scores of the test sequences when boxes above a probability limit are removed."""
    sequences = [MotSequence.load(data / f"MOT17-{n}-FRCNN") for n in TEST]
    summary: dict[str, dict[str, float]] = {}
    for interval in np.unique(rows[:, 1]).astype(int):
        names = []
        removed = {}
        for limit in limits:
            name = f"n{interval:02d}-limit-{limit:.2f}"
            names.append(name)
            keep = (rows[:, 1] == interval) & (probabilities <= limit)
            removed[name] = 100 * (1 - keep.sum() / (rows[:, 1] == interval).sum())
            for sequence in sequences:
                part = rows[keep & (rows[:, 0] == int(sequence.name[6:8]))]
                with MotWriter(out / name / "data" / f"{sequence.name}.txt") as writer:
                    for frame in np.unique(part[:, 2]):
                        objects = [
                            TrackedObject(int(r[3]), (r[4], r[5], r[6], r[7]), float(r[8]), 0)
                            for r in part[part[:, 2] == frame]
                        ]
                        writer.write(int(frame), objects)
        scores = evaluate(names, sequences, data, out)
        for name in names:
            summary[name] = {"removed_pct": removed[name], **scores[name]}
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("model", help="model name, a folder in the detection cache")
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--cache", type=Path, default=Path("runs/detections"))
    p.add_argument("--out", type=Path, default=Path("runs/failure"))
    p.add_argument("--intervals", type=int, nargs="+", default=[5, 10])
    p.add_argument("--limits", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--plot", type=Path, help="write the calibration chart to this file")
    args = p.parse_args()

    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)
    stored = out / "rows.npy"
    if stored.is_file():
        rows = np.load(stored)
    else:
        jobs = [
            (args.data, args.cache / args.model, number, interval)
            for number in (*TUNE, *TEST)
            for interval in args.intervals
        ]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = np.concatenate(list(pool.map(collect, jobs)))
        np.save(stored, rows)

    features, labels = rows[:, len(_META) :], rows[:, _META.index("wrong")]
    tune = np.isin(rows[:, 0], [int(n) for n in TUNE])
    test = ~tune
    predictor = FailurePredictor.fit(features[tune], labels[tune], FEATURE_NAMES)
    predictor.save(out / "predictor.json")
    probabilities = predictor.probability(features)

    print(f"MOT17 train, {args.model}, intervals {args.intervals}, optical flow")
    print(f"tune: {tune.sum()} boxes, {100 * labels[tune].mean():.1f}% wrong")
    print(f"test: {test.sum()} boxes, {100 * labels[test].mean():.1f}% wrong\n")

    quality: dict[str, dict[str, float]] = {
        "logistic regression": {
            "AUC": auc(probabilities[test], labels[test]),
            "AP": average_precision(probabilities[test], labels[test]),
            "weight": float("nan"),
        }
    }
    for i, name in enumerate(FEATURE_NAMES):
        # A single feature is a score in the direction that the model gives to it.
        sign = 1.0 if predictor.weights[i] >= 0 else -1.0
        quality[f"only {name}"] = {
            "AUC": auc(sign * features[test, i], labels[test]),
            "AP": average_precision(sign * features[test, i], labels[test]),
            "weight": float(predictor.weights[i]),
        }
    print(markdown_table(quality, ["AUC", "AP", "weight"], "predictor"))
    ece = expected_calibration_error(probabilities[test], labels[test])
    print(f"\nexpected calibration error on test: {ece:.3f}\n")

    filtered = _filter_experiment(rows[test], probabilities[test], args.data, out, args.limits)
    print(markdown_table(filtered, ["removed_pct", *METRICS], "limit"))

    summary = {"quality": quality, "ece": ece, "filter": filtered}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot:
        _plot(probabilities[test], labels[test], args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
