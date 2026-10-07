from __future__ import annotations

from pathlib import Path

import numpy as np

from leantrack.io.mot import MotSequence

METRICS = ("HOTA", "AssA", "DetA", "MOTA", "IDF1", "IDSW")


def mot17_sequences(data: Path, detector: str = "FRCNN") -> list[MotSequence]:
    return [MotSequence.load(root) for root in sorted(data.glob(f"MOT17-*-{detector}"))]


def evaluate(
    trackers: list[str], sequences: list[MotSequence], data: Path, results: Path
) -> dict[str, dict[str, float]]:
    """Score `results/<tracker>/data/<sequence>.txt` against the ground truth with TrackEval."""
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        USE_PARALLEL=False,
        PRINT_CONFIG=False,
        PRINT_RESULTS=False,
        OUTPUT_SUMMARY=False,
        OUTPUT_DETAILED=False,
        PLOT_CURVES=False,
        TIME_PROGRESS=False,
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        GT_FOLDER=str(data),
        TRACKERS_FOLDER=str(results),
        TRACKERS_TO_EVAL=trackers,
        SKIP_SPLIT_FOL=True,
        SEQ_INFO={s.name: s.length for s in sequences},
        PRINT_CONFIG=False,
    )
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ]
    raw, _ = trackeval.Evaluator(eval_config).evaluate([dataset], metrics)

    scores: dict[str, dict[str, float]] = {}
    for name in trackers:
        combined = raw["MotChallenge2DBox"][name]["COMBINED_SEQ"]["pedestrian"]
        scores[name] = {
            "HOTA": 100 * float(np.mean(combined["HOTA"]["HOTA"])),
            "AssA": 100 * float(np.mean(combined["HOTA"]["AssA"])),
            "DetA": 100 * float(np.mean(combined["HOTA"]["DetA"])),
            "MOTA": 100 * float(combined["CLEAR"]["MOTA"]),
            "IDF1": 100 * float(combined["Identity"]["IDF1"]),
            "IDSW": float(combined["CLEAR"]["IDSW"]),
        }
    return scores


def markdown_table(rows: dict[str, dict[str, float]], columns: list[str], label: str) -> str:
    integer = {"IDSW", "interval"}
    lines = [f"| {label} | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    for name, row in rows.items():
        cells = [f"{row[c]:.0f}" if c in integer else f"{row[c]:.2f}" for c in columns]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
