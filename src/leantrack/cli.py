from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leantrack._types import Frame
from leantrack.detect.base import Detector
from leantrack.detect.mot_file import MotFileDetector
from leantrack.detect.yolox import YoloxDetector
from leantrack.io.mot import MotSequence, MotWriter
from leantrack.io.sources import image_dir_frames, index_frames, paced, video_fps, video_frames
from leantrack.pipeline import FrameResult, run
from leantrack.propagate.flow import FlowPropagator
from leantrack.realtime import ThreadedExecutor, run_realtime
from leantrack.reid.base import Embedder
from leantrack.reid.histogram import HistogramEmbedder
from leantrack.reid.onnx import OnnxEmbedder
from leantrack.schedule.policy import FixedInterval
from leantrack.tracks.tracker import Tracker, TrackerConfig

_DEFAULTS = TrackerConfig()


def _add_pipeline_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "source",
        help="video file, image directory, MOTChallenge sequence directory, stream address "
        "(for example rtsp://...), or camera index",
    )
    p.add_argument(
        "--model",
        help="YOLOX model in ONNX format. Without it, the source must be a MOTChallenge "
        "sequence, and its detection file supplies the detections",
    )
    p.add_argument("--detections", help="detection file (default: <sequence>/det/det.txt)")
    p.add_argument("--interval", type=int, default=1, help="run the detector on each N-th frame")
    p.add_argument(
        "--flow", action="store_true", help="correct the tracks with optical flow between runs"
    )
    p.add_argument(
        "--background",
        action="store_true",
        help="run the detector in a background thread, for a live stream. Optical flow is "
        "always on, and --interval has no effect",
    )
    p.add_argument(
        "--frame-budget-ms",
        type=float,
        default=0.0,
        help="with --background: wait for the detector if the result fits in this time",
    )
    p.add_argument(
        "--reid",
        default="none",
        help="recovery of lost tracks by appearance: none, histogram, or a ReID model in "
        "ONNX format",
    )
    p.add_argument("--high-score", type=float, default=_DEFAULTS.high_score)
    p.add_argument("--low-score", type=float, default=_DEFAULTS.low_score)
    p.add_argument("--new-track-score", type=float, default=_DEFAULTS.new_track_score)
    p.add_argument("--max-lost-frames", type=int, default=_DEFAULTS.max_lost_frames)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leantrack", description="Detection-budgeted tracking.")
    sub = p.add_subparsers(dest="command", required=True)

    track = sub.add_parser("track", help="track the objects of a video and write a result file")
    _add_pipeline_arguments(track)
    track.add_argument("--out", required=True, help="result file in MOTChallenge format")
    track.add_argument(
        "--pace",
        action="store_true",
        help="give the frames at the frame rate of the source, as a camera does",
    )

    serve = sub.add_parser("serve", help="track a video and give the results through HTTP")
    _add_pipeline_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--loop", action="store_true", help="start the source again at its end")
    return p


def _embedder(name: str) -> Embedder | None:
    if name == "none":
        return None
    if name == "histogram":
        return HistogramEmbedder()
    if not Path(name).is_file():
        raise ValueError(f"--reid must be none, histogram, or a model file: {name} not found")
    return OnnxEmbedder(name)


@dataclass(slots=True)
class _Pipeline:
    name: str
    frames: Iterator[Frame]
    results: Iterator[FrameResult]
    executor: ThreadedExecutor | None

    def close(self) -> None:
        if self.executor is not None:
            self.executor.close()


def _is_stream(source: str) -> bool:
    return source.isdigit() or "://" in source


def _source_frames(
    source: str, sequence: MotSequence | None, pixels: bool, pace: bool
) -> Iterator[Frame]:
    if _is_stream(source):
        # A camera or a network stream gives the frames at its own rate.
        return video_frames(int(source) if source.isdigit() else source)
    path = Path(source)
    if sequence is not None:
        frames = image_dir_frames(sequence.image_dir) if pixels else index_frames(sequence.length)
        fps = sequence.frame_rate
    elif path.is_dir():
        frames, fps = image_dir_frames(path), 30.0
    else:
        frames, fps = video_frames(path), video_fps(path)
    return paced(frames, fps) if pace else frames


def _build(
    args: argparse.Namespace,
    *,
    pace: bool,
    pixels: bool = False,
    watch: Callable[[Iterator[Frame]], Iterator[Frame]] | None = None,
) -> _Pipeline:
    """Make the frame source and the pipeline that the arguments describe.

    `watch` gets the frames before the pipeline, for example to keep the newest frame.
    """
    source = str(args.source)
    sequence = None
    if not _is_stream(source):
        if not Path(source).exists():
            raise FileNotFoundError(f"{source} not found")
        if (Path(source) / "seqinfo.ini").is_file():
            sequence = MotSequence.load(source)
    if args.background and not args.model:
        raise ValueError("--background needs --model")

    detector: Detector
    if args.model:
        detector = YoloxDetector(args.model)
    elif sequence is not None:
        detector = MotFileDetector(args.detections or sequence.detections_path)
    else:
        raise ValueError("--model is necessary unless the source is a MOTChallenge sequence")

    embedder = _embedder(args.reid)
    flow = args.flow or args.background
    pixels = pixels or bool(args.model) or embedder is not None or flow
    tracker = Tracker(
        TrackerConfig(
            high_score=args.high_score,
            low_score=args.low_score,
            new_track_score=args.new_track_score,
            max_lost_frames=args.max_lost_frames,
        )
    )
    # A background detector is for a live stream, so a file source is always paced.
    frames = _source_frames(source, sequence, pixels, pace or args.background)
    if watch is not None:
        frames = watch(frames)
    name = source if _is_stream(source) else Path(source).name
    if args.background:
        executor = ThreadedExecutor(detector)
        results = run_realtime(
            frames,
            executor,
            tracker,
            FlowPropagator(),
            embedder,
            frame_budget_ms=args.frame_budget_ms,
        )
        return _Pipeline(name, frames, results, executor)
    results = run(
        frames,
        detector,
        tracker,
        FixedInterval(args.interval),
        FlowPropagator() if flow else None,
        embedder,
    )
    return _Pipeline(name, frames, results, None)


def _track(args: argparse.Namespace) -> int:
    pipeline = _build(args, pace=args.pace)
    frame_ms: list[float] = []
    detector_runs = 0
    ids: set[int] = set()
    try:
        with MotWriter(args.out) as writer:
            for result in pipeline.results:
                writer.write(result.frame_index, result.objects)
                frame_ms.append(result.total_ms)
                detector_runs += result.detected
                ids.update(obj.track_id for obj in result.objects)
    finally:
        pipeline.close()
    if not frame_ms:
        raise ValueError(f"{args.source} has no frames")

    p50, p99 = np.percentile(frame_ms, [50, 99])
    print(
        f"{pipeline.name}: {len(frame_ms)} frames, {len(ids)} tracks, "
        f"{detector_runs} detector runs, "
        f"frame time mean {np.mean(frame_ms):.2f} ms, p50 {p50:.2f} ms, p99 {p99:.2f} ms "
        f"-> {args.out}"
    )
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from leantrack.serve.app import Metrics, create_app
        from leantrack.serve.service import FrameTap, TrackingService
    except ImportError as exc:
        raise ValueError(
            'the stream service is not installed: pip install "leantrack[serve]"'
        ) from exc

    # Build one time before the server starts, so a wrong argument stops the command.
    _build(args, pace=True, pixels=True).close()
    tap = FrameTap()

    def results() -> Iterator[FrameResult]:
        while True:
            pipeline = _build(args, pace=True, pixels=True, watch=tap.watch)
            try:
                yield from pipeline.results
            finally:
                pipeline.close()
            if not args.loop:
                return

    metrics = Metrics()
    service = TrackingService(results, tap, on_result=metrics.observe)
    uvicorn.run(create_app(service, metrics), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _serve(args) if args.command == "serve" else _track(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
