from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from leantrack._types import Frame
from leantrack.detect.base import Detector
from leantrack.detect.mot_file import MotFileDetector
from leantrack.detect.yolox import YoloxDetector
from leantrack.io.mot import MotSequence, MotWriter
from leantrack.io.sources import image_dir_frames, index_frames, video_frames
from leantrack.pipeline import run
from leantrack.propagate.flow import FlowPropagator
from leantrack.reid.base import Embedder
from leantrack.reid.histogram import HistogramEmbedder
from leantrack.reid.onnx import OnnxEmbedder
from leantrack.schedule.policy import FixedInterval
from leantrack.tracks.tracker import Tracker, TrackerConfig

_DEFAULTS = TrackerConfig()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leantrack", description="Detection-budgeted tracking.")
    sub = p.add_subparsers(dest="command", required=True)

    track = sub.add_parser("track", help="track the objects of a video or an image sequence")
    track.add_argument(
        "source", help="video file, image directory, or MOTChallenge sequence directory"
    )
    track.add_argument("--out", required=True, help="result file in MOTChallenge format")
    track.add_argument(
        "--model",
        help="YOLOX model in ONNX format. Without it, the source must be a MOTChallenge "
        "sequence, and its detection file supplies the detections",
    )
    track.add_argument("--detections", help="detection file (default: <sequence>/det/det.txt)")
    track.add_argument(
        "--interval", type=int, default=1, help="run the detector on each N-th frame"
    )
    track.add_argument(
        "--flow", action="store_true", help="correct the tracks with optical flow between runs"
    )
    track.add_argument(
        "--reid",
        default="none",
        help="recovery of lost tracks by appearance: none, histogram, or a ReID model in "
        "ONNX format",
    )
    track.add_argument("--high-score", type=float, default=_DEFAULTS.high_score)
    track.add_argument("--low-score", type=float, default=_DEFAULTS.low_score)
    track.add_argument("--new-track-score", type=float, default=_DEFAULTS.new_track_score)
    track.add_argument("--max-lost-frames", type=int, default=_DEFAULTS.max_lost_frames)
    return p


def _embedder(name: str) -> Embedder | None:
    if name == "none":
        return None
    if name == "histogram":
        return HistogramEmbedder()
    if not Path(name).is_file():
        raise ValueError(f"--reid must be none, histogram, or a model file: {name} not found")
    return OnnxEmbedder(name)


def _frames(source: Path, sequence: MotSequence | None, pixels: bool) -> Iterator[Frame]:
    if sequence is not None:
        return image_dir_frames(sequence.image_dir) if pixels else index_frames(sequence.length)
    return image_dir_frames(source) if source.is_dir() else video_frames(source)


def _track(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    sequence = MotSequence.load(source) if (source / "seqinfo.ini").is_file() else None

    detector: Detector
    if args.model:
        detector = YoloxDetector(args.model)
    elif sequence is not None:
        detector = MotFileDetector(args.detections or sequence.detections_path)
    else:
        raise ValueError("--model is necessary unless the source is a MOTChallenge sequence")

    embedder = _embedder(args.reid)
    propagator = FlowPropagator() if args.flow else None
    pixels = bool(args.model) or embedder is not None or propagator is not None
    tracker = Tracker(
        TrackerConfig(
            high_score=args.high_score,
            low_score=args.low_score,
            new_track_score=args.new_track_score,
            max_lost_frames=args.max_lost_frames,
        )
    )

    frame_ms: list[float] = []
    detector_runs = 0
    ids: set[int] = set()
    results = run(
        _frames(source, sequence, pixels),
        detector,
        tracker,
        FixedInterval(args.interval),
        propagator,
        embedder,
    )
    with MotWriter(args.out) as writer:
        for result in results:
            writer.write(result.frame_index, result.objects)
            frame_ms.append(result.total_ms)
            detector_runs += result.detected
            ids.update(obj.track_id for obj in result.objects)
    if not frame_ms:
        raise ValueError(f"{source} has no frames")

    p50, p99 = np.percentile(frame_ms, [50, 99])
    print(
        f"{source.name}: {len(frame_ms)} frames, {len(ids)} tracks, "
        f"{detector_runs} detector runs, "
        f"frame time mean {np.mean(frame_ms):.2f} ms, p50 {p50:.2f} ms, p99 {p99:.2f} ms "
        f"-> {args.out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _track(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
