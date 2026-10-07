"""Experiment 1: tracking quality and latency against the detection interval.

Replays the stored detections of one model (see `cache_detections.py`). On a frame
without a detector run, the tracker reports its Kalman prediction.

    python -m bench.interval yolox_s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench._eval import METRICS, evaluate, markdown_table, mot17_sequences
from leantrack.detect.mot_file import MotFileDetector
from leantrack.io.mot import MotSequence, MotWriter
from leantrack.io.sources import image_dir_frames, index_frames
from leantrack.pipeline import run
from leantrack.propagate.flow import FlowPropagator
from leantrack.schedule.policy import FixedInterval
from leantrack.tracks.tracker import Tracker


def _run(
    interval: int, sequences: list[MotSequence], cache: Path, out: Path, flow: bool
) -> dict[str, float]:
    frame_ms: list[float] = []
    detector_runs = 0
    for sequence in sequences:
        detector = MotFileDetector(cache / f"{sequence.name}.txt")
        detect_ms = np.load(cache / f"{sequence.name}.latency.npy")
        # The flow needs the pixels. The Kalman prediction does not, so it skips the decode.
        frames = image_dir_frames(sequence.image_dir) if flow else index_frames(sequence.length)
        propagator = FlowPropagator() if flow else None
        results = run(frames, detector, Tracker(), FixedInterval(interval), propagator)
        with MotWriter(out / "data" / f"{sequence.name}.txt") as writer:
            for result in results:
                writer.write(result.frame_index, result.objects)
                # The replay costs no time, so use the latency that the model had on this frame.
                cost = result.track_ms + result.propagate_ms
                if result.detected:
                    cost += float(detect_ms[result.frame_index - 1])
                    detector_runs += 1
                frame_ms.append(cost)
    p50, p99 = np.percentile(frame_ms, [50, 99])
    return {
        "interval": float(interval),
        "detector_runs_pct": 100 * detector_runs / len(frame_ms),
        "mean_ms": float(np.mean(frame_ms)),
        "p50_ms": float(p50),
        "p99_ms": float(p99),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("model", help="model name, a folder in the detection cache")
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--cache", type=Path, default=Path("runs/detections"))
    p.add_argument("--out", type=Path, default=Path("runs/interval"))
    p.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20])
    p.add_argument("--flow", action="store_true", help="correct predictions with optical flow")
    args = p.parse_args()

    sequences = mot17_sequences(args.data)
    out = args.out / (f"{args.model}-flow" if args.flow else args.model)
    names = [f"n{n:02d}" for n in args.intervals]
    costs = {
        name: _run(n, sequences, args.cache / args.model, out / name, args.flow)
        for name, n in zip(names, args.intervals, strict=True)
    }
    scores = evaluate(names, sequences, args.data, out)
    summary = {name: {**costs[name], **scores[name]} for name in names}

    mode = "Kalman + optical flow" if args.flow else "Kalman"
    print(f"MOT17 train, {args.model}, {mode}, {len(sequences)} sequences")
    columns = ["detector_runs_pct", "mean_ms", "p99_ms", *METRICS]
    print(markdown_table(summary, columns, "interval"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
