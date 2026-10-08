"""Experiment 7: a slow detector on a live stream.

Frames arrive at the frame rate of the sequence. The detector latency is the stored
latency of the model on that frame, on a simulated clock. Thus a run is repeatable.

- offline: the detector runs on each frame and time does not count. This is a reference.
- blocking: the loop waits for the detector. Frames that arrive during the wait are
  dropped. Each frame shows the newest result that is complete before the next frame.
- background: the detector runs in the background. Each frame gets an output from the
  optical flow. Flow moves the late detections to the current frame.
- background-raw: as background, but the late detections are used without a correction.

    python -m bench.realtime yolox_s --jobs 4
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench._eval import METRICS, evaluate, markdown_table, mot17_sequences
from leantrack._types import Frame, TrackedObject
from leantrack.detect.mot_file import MotFileDetector
from leantrack.io.mot import MotSequence, MotWriter
from leantrack.io.sources import image_dir_frames
from leantrack.propagate.flow import FlowPropagator
from leantrack.realtime import SimulatedExecutor, run_realtime
from leantrack.tracks.tracker import Tracker

MODES = ("offline", "blocking", "background", "background-raw")


def _blocking(
    sequence: MotSequence, detector: MotFileDetector, latency: np.ndarray, offline: bool
) -> tuple[list[tuple[int, list[TrackedObject]]], dict[str, list[float]]]:
    """The detector on each frame that the loop can process. `offline` processes each frame.

    In the live mode, a result is on the screen for a frame only if it is complete before
    the next frame arrives. Thus the output of a frame is the newest complete result,
    which comes from an earlier frame.
    """
    period = 1000.0 / sequence.frame_rate
    tracker = Tracker()
    stats: dict[str, list[float]] = {"latency_ms": [], "age": [], "detected": []}
    # (frame index, time at which the result is complete, objects)
    complete: list[tuple[int, float, list[TrackedObject]]] = []
    clock = 0.0
    next_index = 1
    for index in range(1, sequence.length + 1):
        arrival = (index - 1) * period
        if index != next_index:
            stats["detected"].append(0.0)
            continue
        objects = tracker.update(detector.detect(Frame(index)))
        # The tracker step is less than 1 ms. The simulated clock ignores it.
        finish = arrival if offline else max(clock, arrival) + float(latency[index - 1])
        clock = finish
        complete.append((index, finish, objects))
        # The loop takes the newest frame that is there when it becomes free.
        next_index = index + 1 if offline else max(index + 1, int(finish // period) + 1)
        stats["latency_ms"].append(finish - arrival)
        stats["detected"].append(1.0)

    outputs: list[tuple[int, list[TrackedObject]]] = []
    shown = -1
    for index in range(1, sequence.length + 1):
        deadline = index * period
        while shown + 1 < len(complete) and complete[shown + 1][1] < deadline:
            shown += 1
        if shown < 0:
            outputs.append((index, []))
            continue
        outputs.append((index, complete[shown][2]))
        stats["age"].append(float(index - complete[shown][0]))
    return outputs, stats


def _background(
    sequence: MotSequence, detector: MotFileDetector, latency: np.ndarray, compensate: bool
) -> tuple[list[tuple[int, list[TrackedObject]]], dict[str, list[float]]]:
    period = 1000.0 / sequence.frame_rate
    # The simulated clock of the executor starts at frame index 0, so use index - 1.
    executor = SimulatedExecutor(detector, lambda i: float(latency[i - 1]), period)
    results = run_realtime(
        image_dir_frames(sequence.image_dir),
        executor,
        Tracker(),
        FlowPropagator(),
        compensate=compensate,
    )
    outputs = []
    stats: dict[str, list[float]] = {"latency_ms": [], "age": [], "detected": []}
    for result in results:
        outputs.append((result.frame_index, result.objects))
        stats["latency_ms"].append(result.total_ms)
        stats["detected"].append(float(result.detected))
        if result.detected:
            stats["age"].append(float(result.detection_age))
    return outputs, stats


def _job(args: tuple[Path, Path, str, str, Path]) -> dict[str, list[float]]:
    root, cache, mode, name, out = args
    sequence = MotSequence.load(root)
    detector = MotFileDetector(cache / f"{sequence.name}.txt")
    latency = np.load(cache / f"{sequence.name}.latency.npy")
    if mode in ("offline", "blocking"):
        outputs, stats = _blocking(sequence, detector, latency, offline=mode == "offline")
    else:
        outputs, stats = _background(sequence, detector, latency, mode == "background")
    with MotWriter(out / name / "data" / f"{sequence.name}.txt") as writer:
        for index, objects in outputs:
            writer.write(index, objects)
    stats["period_ms"] = [1000.0 / sequence.frame_rate] * len(stats["latency_ms"])
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("model", help="model name, a folder in the detection cache")
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--cache", type=Path, default=Path("runs/detections"))
    p.add_argument("--out", type=Path, default=Path("runs/realtime"))
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--jobs", type=int, default=1, help="worker processes; more than 1 adds noise")
    args = p.parse_args()

    sequences = mot17_sequences(args.data)
    out = args.out / args.model
    cache = args.cache / args.model
    costs: dict[str, dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for mode in args.modes:
            jobs = [(s.root, cache, mode, mode, out) for s in sequences]
            merged: dict[str, list[float]] = {}
            for stats in pool.map(_job, jobs):
                for key, values in stats.items():
                    merged.setdefault(key, []).extend(values)
            latency = np.array(merged["latency_ms"])
            costs[mode] = {
                "detector_runs_pct": 100 * float(np.mean(merged["detected"])),
                "mean_age_frames": float(np.mean(merged["age"])),
                "max_age_frames": float(np.max(merged["age"])),
                "latency_mean_ms": float(latency.mean()),
                "latency_p99_ms": float(np.percentile(latency, 99)),
                "over_period_pct": 100 * float(np.mean(latency > np.array(merged["period_ms"]))),
            }

    scores = evaluate(list(args.modes), sequences, args.data, out)
    summary = {mode: {**costs[mode], **scores[mode]} for mode in args.modes}
    print(f"MOT17 train, {args.model}, live stream at the frame rate of each sequence")
    columns = [
        "detector_runs_pct",
        "mean_age_frames",
        "max_age_frames",
        "latency_mean_ms",
        "latency_p99_ms",
        "over_period_pct",
        *METRICS,
    ]
    print(markdown_table(summary, columns, "mode"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
