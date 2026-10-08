"""Experiment 1: tracking quality and latency against the detection interval.

Replays the stored detections of one model (see `cache_detections.py`). On a frame
without a detector run, the tracker reports its Kalman prediction.

    python -m bench.interval yolox_s
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from bench._eval import (
    HISTOGRAM_CONFIG,
    METRICS,
    evaluate,
    markdown_table,
    mot17_sequences,
    run_policy,
)
from leantrack.reid.base import Embedder
from leantrack.reid.histogram import HistogramEmbedder
from leantrack.reid.onnx import OnnxEmbedder
from leantrack.schedule.policy import FixedInterval


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("model", help="model name, a folder in the detection cache")
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--cache", type=Path, default=Path("runs/detections"))
    p.add_argument("--out", type=Path, default=Path("runs/interval"))
    p.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20])
    p.add_argument("--flow", action="store_true", help="correct predictions with optical flow")
    p.add_argument("--reid", default="none", help="none, histogram, or a ReID model in ONNX format")
    args = p.parse_args()

    sequences = mot17_sequences(args.data)
    embedder: Embedder | None = None
    config = None
    if args.reid == "histogram":
        embedder = HistogramEmbedder()
        config = HISTOGRAM_CONFIG
    elif args.reid != "none":
        embedder = OnnxEmbedder(args.reid)
    suffix = ("-flow" if args.flow else "") + (
        "" if embedder is None else f"-{Path(args.reid).stem}"
    )
    out = args.out / f"{args.model}{suffix}"
    names = [f"n{n:02d}" for n in args.intervals]
    costs = {}
    for name, n in zip(names, args.intervals, strict=True):
        cost = run_policy(
            partial(FixedInterval, n),
            sequences,
            args.cache / args.model,
            out / name,
            args.flow,
            embedder,
            config,
        )
        costs[name] = {"interval": float(n), **cost}
    scores = evaluate(names, sequences, args.data, out)
    summary = {name: {**costs[name], **scores[name]} for name in names}

    mode = "Kalman + optical flow" if args.flow else "Kalman"
    print(f"MOT17 train, {args.model}, {mode}, ReID {args.reid}, {len(sequences)} sequences")
    columns = ["detector_runs_pct", "mean_ms", "embed_ms_per_run", "p99_ms", *METRICS]
    print(markdown_table(summary, columns, "interval"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
