"""Experiment 5: recovery of a track after a synthetic occlusion.

For one ground-truth object, the script removes its detections for D frames. Then it
checks if the track that followed the object before the gap follows it again after the
gap. The detector runs on each frame, so the result measures the tracker and not the
schedule. Each event has its own tracker run.

    python -m bench.occlusion --detector SDP --jobs 6
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np

from bench._eval import HISTOGRAM_CONFIG, load_embeddings, markdown_table
from leantrack._types import Detections, FloatArray, TrackedObject
from leantrack.boxes import iou_matrix
from leantrack.io.mot import MotSequence, read_detections
from leantrack.tracks.tracker import Tracker, TrackerConfig

PRE_FRAMES = 10
"""Frames before the gap in which the object must be visible and tracked."""
HORIZON = 30
"""Frames after the gap in which the track can recover."""
MIN_VISIBILITY = 0.5
MATCH_IOU = 0.5

SPLITS = {
    "tune": ("02", "04", "09"),
    "test": ("05", "10", "11", "13"),
    "all": ("02", "04", "05", "09", "10", "11", "13"),
}
"""The appearance thresholds are selected on "tune". The reported results use "test"."""


_LONG = TrackerConfig(max_lost_frames=90)

# name -> (tracker configuration, embedder name or None)
VARIANTS: dict[str, tuple[TrackerConfig, str | None]] = {
    "baseline": (TrackerConfig(), None),
    "lost-90": (_LONG, None),
    "histogram": (replace(HISTOGRAM_CONFIG, max_lost_frames=90), "histogram"),
    "osnet": (_LONG, "osnet"),
    "osnet-each-match": (replace(_LONG, embedding_refresh=1), "osnet"),
}


@dataclass(frozen=True, slots=True)
class Event:
    sequence: str
    object_id: int
    start: int
    """First frame without detections of the object."""
    duration: int


@dataclass(frozen=True, slots=True)
class Outcome:
    result: str
    """One of: recovered, switched, missed, not_tracked."""
    frames_to_recover: int | None = None


@cache
def _ground_truth(root: str) -> dict[int, dict[int, tuple[FloatArray, float]]]:
    """Object id -> frame -> (xyxy box, visibility), for the pedestrians that MOT17 scores."""
    rows = np.loadtxt(Path(root) / "gt" / "gt.txt", delimiter=",")
    rows = rows[(rows[:, 6] == 1) & (rows[:, 7] == 1)]
    objects: dict[int, dict[int, tuple[FloatArray, float]]] = {}
    for row in rows:
        box = np.array([row[2], row[3], row[2] + row[4], row[3] + row[5]])
        objects.setdefault(int(row[1]), {})[int(row[0])] = (box, float(row[8]))
    return objects


@cache
def _detections(root: str) -> dict[int, Detections]:
    return read_detections(Path(root) / "det" / "det.txt")


@cache
def _embeddings(embeddings: str, embedder: str, root: str) -> dict[int, FloatArray]:
    path = Path(embeddings) / embedder / f"{Path(root).name}.npy"
    return load_embeddings(path, Path(root) / "det" / "det.txt")


def find_events(root: Path, duration: int, limit: int) -> list[Event]:
    """At most one event for each object: the first gap position that satisfies the rules.

    The object must exist from PRE_FRAMES before the gap to HORIZON after it. It must be
    visible in the PRE_FRAMES before the gap and in the first PRE_FRAMES after it.
    """
    events = []
    for object_id, frames in sorted(_ground_truth(str(root)).items()):
        for start in sorted(frames):
            end = start + duration
            needed = range(start - PRE_FRAMES, end + HORIZON + 1)
            if any(f not in frames for f in needed):
                continue
            clear = [*range(start - PRE_FRAMES, start), *range(end, end + PRE_FRAMES)]
            if all(frames[f][1] >= MIN_VISIBILITY for f in clear):
                events.append(Event(root.name, object_id, start, duration))
                break
    if len(events) > limit:
        keep = np.linspace(0, len(events) - 1, limit).round().astype(int)
        events = [events[i] for i in keep]
    return events


def run_event(data: Path, embeddings: Path, event: Event, variant: str) -> Outcome:
    root = str(data / event.sequence)
    truth = _ground_truth(root)[event.object_id]
    by_frame = _detections(root)
    config, embedder = VARIANTS[variant]
    tracker = Tracker(config)
    vectors_by_frame = _embeddings(str(embeddings), embedder, root) if embedder else None
    gap = range(event.start, event.start + event.duration)
    last = event.start + event.duration + HORIZON

    followers: dict[int, int | None] = {}
    for frame in range(1, last + 1):
        detections = by_frame.get(frame, Detections.empty())
        vectors = vectors_by_frame.get(frame) if vectors_by_frame else None
        if frame in gap and len(detections):
            hidden = iou_matrix(truth[frame][0][None], detections.boxes)[0] > MATCH_IOU
            detections = detections.select(~hidden)
            vectors = None if vectors is None else vectors[~hidden]
        embed = None if vectors is None else vectors.__getitem__
        objects = tracker.update(detections, embed=embed)
        if frame in truth and frame >= event.start - PRE_FRAMES:
            followers[frame] = _follower(truth[frame][0], objects)

    before = Counter(
        followers[f] for f in range(event.start - PRE_FRAMES, event.start) if followers[f]
    )
    if not before or before.most_common(1)[0][1] <= PRE_FRAMES // 2:
        return Outcome("not_tracked")
    track_id = before.most_common(1)[0][0]

    after = range(event.start + event.duration, last + 1)
    for frame in after:
        if followers[frame] == track_id:
            return Outcome("recovered", frame - after.start)
    if any(followers[frame] is not None for frame in after):
        return Outcome("switched")
    return Outcome("missed")


def _follower(box: FloatArray, objects: list[TrackedObject]) -> int | None:
    """Id of the reported track with the largest IoU with `box`, if that IoU is sufficient."""
    if not objects:
        return None
    iou = iou_matrix(box[None], np.array([o.box for o in objects]))[0]
    best = int(iou.argmax())
    return int(objects[best].track_id) if iou[best] >= MATCH_IOU else None


def _job(args: tuple[Path, Path, Event, str]) -> Outcome:
    return run_event(*args)


def _plot(summary: dict[str, dict[str, float]], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[int, float]]] = {}
    for name, row in summary.items():
        variant, _, duration = name.partition(", D=")
        series.setdefault(variant, []).append((int(duration), row["recovered_pct"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for variant, points in series.items():
        ax.plot(*zip(*sorted(points), strict=True), marker="o", label=variant)
    ax.set_xlabel("Gap without detections of the object (frames)")
    ax.set_ylabel("Tracks that recover their ID (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--detector", choices=["DPM", "FRCNN", "SDP"], default="SDP")
    p.add_argument("--embeddings", type=Path, default=Path("runs/embeddings"))
    p.add_argument("--split", choices=sorted(SPLITS), default="test")
    p.add_argument("--out", type=Path, default=Path("runs/occlusion"))
    p.add_argument("--durations", type=int, nargs="+", default=[0, 5, 15, 30, 60])
    p.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    p.add_argument("--events", type=int, default=40, help="maximum events for each sequence")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--plot", type=Path, help="write a chart to this file")
    args = p.parse_args()

    roots = [args.data / f"MOT17-{n}-{args.detector}" for n in SPLITS[args.split]]
    for root in roots:
        MotSequence.load(root)

    summary: dict[str, dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for duration in args.durations:
            events = [e for root in roots for e in find_events(root, duration, args.events)]
            for variant in args.variants:
                jobs = [(args.data, args.embeddings, event, variant) for event in events]
                outcomes = list(pool.map(_job, jobs, chunksize=8))
                tracked = [o for o in outcomes if o.result != "not_tracked"]
                waits = [o.frames_to_recover for o in tracked if o.frames_to_recover is not None]
                counts = Counter(o.result for o in tracked)
                summary[f"{variant}, D={duration}"] = {
                    "events": float(len(tracked)),
                    "recovered_pct": 100 * counts["recovered"] / len(tracked),
                    "switched_pct": 100 * counts["switched"] / len(tracked),
                    "missed_pct": 100 * counts["missed"] / len(tracked),
                    "median_frames_to_recover": float(np.median(waits)) if waits else float("nan"),
                }

    title = f"MOT17 train, split {args.split}, {args.detector} detections"
    print(f"{title}, detector on each frame")
    columns = [
        "events",
        "recovered_pct",
        "switched_pct",
        "missed_pct",
        "median_frames_to_recover",
    ]
    print(markdown_table(summary, columns, "variant"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"summary-{args.detector}-{args.split}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if args.plot:
        _plot(summary, title, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
