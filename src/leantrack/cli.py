from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from leantrack.detect.mot_file import MotFileDetector
from leantrack.io.mot import MotSequence, MotWriter
from leantrack.io.sources import index_frames
from leantrack.pipeline import run
from leantrack.tracks.tracker import Tracker, TrackerConfig

_DEFAULTS = TrackerConfig()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leantrack", description="Detection-budgeted tracking.")
    sub = p.add_subparsers(dest="command", required=True)

    track = sub.add_parser("track", help="track a MOTChallenge sequence from its detections")
    track.add_argument("sequence", help="sequence directory that contains seqinfo.ini")
    track.add_argument("--detections", help="detection file (default: <sequence>/det/det.txt)")
    track.add_argument("--out", required=True, help="result file in MOTChallenge format")
    track.add_argument("--high-score", type=float, default=_DEFAULTS.high_score)
    track.add_argument("--low-score", type=float, default=_DEFAULTS.low_score)
    track.add_argument("--new-track-score", type=float, default=_DEFAULTS.new_track_score)
    track.add_argument("--max-lost-frames", type=int, default=_DEFAULTS.max_lost_frames)
    return p


def _track(args: argparse.Namespace) -> int:
    sequence = MotSequence.load(args.sequence)
    detector = MotFileDetector(args.detections or sequence.detections_path)
    tracker = Tracker(
        TrackerConfig(
            high_score=args.high_score,
            low_score=args.low_score,
            new_track_score=args.new_track_score,
            max_lost_frames=args.max_lost_frames,
        )
    )
    track_ms: list[float] = []
    ids: set[int] = set()
    with MotWriter(args.out) as writer:
        for result in run(index_frames(sequence.length), detector, tracker):
            writer.write(result.frame_index, result.objects)
            track_ms.append(result.track_ms)
            ids.update(obj.track_id for obj in result.objects)

    p50, p99 = np.percentile(track_ms, [50, 99])
    print(
        f"{sequence.name}: {sequence.length} frames, {len(ids)} tracks, "
        f"association p50 {p50:.2f} ms, p99 {p99:.2f} ms -> {args.out}"
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
