"""Evaluate trackers on MOT17 train with the public detections of the dataset.

Each tracker gets the same detection file, so the scores compare association only.

    python -m bench.mot17 --data data/MOT17/train --out runs/mot17
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from bench._eval import (
    HISTOGRAM_CONFIG,
    METRICS,
    evaluate,
    load_embeddings,
    markdown_table,
    mot17_sequences,
)
from leantrack._types import Detections, TrackedObject
from leantrack.io.mot import MotSequence, MotWriter, read_detections
from leantrack.tracks.tracker import Tracker, TrackerConfig

Step = Callable[[int, Detections], list[TrackedObject]]
"""Takes the 1-based frame index and the detections of that frame."""

EMBEDDINGS = Path("runs/embeddings")
_LONG = TrackerConfig(max_lost_frames=90)
_HISTOGRAM = replace(HISTOGRAM_CONFIG, max_lost_frames=90)


def _leantrack(config: TrackerConfig, embedder: str | None, sequence: MotSequence) -> Step:
    tracker = Tracker(config)
    if embedder is None:
        return lambda frame, detections: tracker.update(detections)
    # Stored vectors, see `cache_embeddings.py`. They exist for the SDP detections only.
    vectors = load_embeddings(
        EMBEDDINGS / embedder / f"{sequence.name}.npy", sequence.detections_path
    )

    def step(frame: int, detections: Detections) -> list[TrackedObject]:
        stored = vectors.get(frame)
        return tracker.update(detections, embed=None if stored is None else stored.__getitem__)

    return step


def _boxmot_bytetrack(sequence: MotSequence) -> Step:
    from boxmot import ByteTrack

    tracker = ByteTrack(frame_rate=round(sequence.frame_rate))

    def step(frame: int, detections: Detections) -> list[TrackedObject]:
        rows = np.column_stack([detections.boxes, detections.scores, detections.classes])
        out = np.asarray(tracker.update(rows.reshape(-1, 6)))
        # Output columns: x1, y1, x2, y2, id, score, class, detection index.
        return [
            TrackedObject(int(r[4]), (r[0], r[1], r[2], r[3]), float(r[5]), int(r[6]))
            for r in out.reshape(-1, 8)
        ]

    return step


TRACKERS: dict[str, Callable[[MotSequence], Step]] = {
    "leantrack": partial(_leantrack, TrackerConfig(), None),
    "leantrack-lost-90": partial(_leantrack, _LONG, None),
    "leantrack-histogram": partial(_leantrack, _HISTOGRAM, "histogram"),
    "leantrack-osnet": partial(_leantrack, _LONG, "osnet"),
    "leantrack-osnet-lost-30": partial(_leantrack, TrackerConfig(), "osnet"),
    "leantrack-osnet-lost-60": partial(_leantrack, TrackerConfig(max_lost_frames=60), "osnet"),
    "boxmot-bytetrack": _boxmot_bytetrack,
}
DEFAULT_TRACKERS = ["boxmot-bytetrack", "leantrack"]


def _run(name: str, sequences: list[MotSequence], out: Path) -> dict[str, float]:
    step_ms: list[float] = []
    for sequence in sequences:
        by_frame = read_detections(sequence.detections_path)
        step = TRACKERS[name](sequence)
        with MotWriter(out / name / "data" / f"{sequence.name}.txt") as writer:
            for frame in range(1, sequence.length + 1):
                detections = by_frame.get(frame, Detections.empty())
                start = time.perf_counter()
                objects = step(frame, detections)
                step_ms.append((time.perf_counter() - start) * 1000.0)
                writer.write(frame, objects)
    p50, p99 = np.percentile(step_ms, [50, 99])
    return {"step_ms_p50": float(p50), "step_ms_p99": float(p99)}


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--detector", choices=["DPM", "FRCNN", "SDP"], default="FRCNN")
    p.add_argument("--out", type=Path, default=Path("runs/mot17"))
    p.add_argument("--trackers", nargs="+", choices=sorted(TRACKERS), default=DEFAULT_TRACKERS)
    args = p.parse_args()

    sequences = mot17_sequences(args.data, args.detector)
    if not sequences:
        p.error(f"no MOT17-*-{args.detector} sequences in {args.data}")

    timing = {name: _run(name, sequences, args.out) for name in args.trackers}
    scores = evaluate(args.trackers, sequences, args.data, args.out)
    summary = {name: {**scores[name], **timing[name]} for name in args.trackers}

    print(f"MOT17 train, {args.detector} detections, {len(sequences)} sequences")
    print(markdown_table(summary, [*METRICS, "step_ms_p50", "step_ms_p99"], "tracker"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
