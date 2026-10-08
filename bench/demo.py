"""Make the demo GIFs of the README from MOT17 sequences.

- live: a blocking loop beside a background detector, on a live stream.
- occlusion: recovery of an ID after a gap, without and with appearance.

    python -m bench.demo live assets/demo_live.gif
    python -m bench.demo occlusion assets/demo_occlusion.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PilImage

from bench import occlusion
from bench.realtime import _background, _blocking
from leantrack._types import Detections, Image, TrackedObject
from leantrack.boxes import iou_matrix
from leantrack.detect.mot_file import MotFileDetector
from leantrack.io.mot import MotSequence
from leantrack.tracks.tracker import Tracker

_PANEL_WIDTH = 420
_BAR = 26
_COLORS = [(66, 135, 245), (60, 180, 75), (245, 130, 48), (200, 60, 200), (0, 190, 190)]
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TARGET = (0, 255, 255)


def _read(path: Path) -> Image:
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f"cannot read {path}")
    return image


def _panel(image: Image, title: str) -> tuple[Image, float]:
    """A small copy of the frame with a title bar. Returns the panel and its scale."""
    scale = _PANEL_WIDTH / image.shape[1]
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bar = np.full((_BAR, small.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(bar, title, (6, 18), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, small]), scale


def _draw(panel: Image, scale: float, obj: TrackedObject, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (round(v * scale) for v in obj.box)
    y1, y2 = y1 + _BAR, y2 + _BAR
    cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
    cv2.putText(panel, str(obj.track_id), (x1, y1 - 3), _FONT, 0.45, color, 1, cv2.LINE_AA)


def _save(frames: list[Image], path: Path, fps: int) -> None:
    images = [PilImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    # One palette for all frames keeps the file small and the colors stable. It comes from
    # the first, the middle and the last frame, so it has the colors of each overlay.
    samples = [images[0], images[len(images) // 2], images[-1]]
    sheet = PilImage.new("RGB", (samples[0].width, samples[0].height * len(samples)))
    for row, sample in enumerate(samples):
        sheet.paste(sample, (0, row * sample.height))
    palette = sheet.quantize(colors=160, method=PilImage.Quantize.MEDIANCUT)
    quantized = [i.quantize(palette=palette, dither=PilImage.Dither.NONE) for i in images]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:], duration=round(1000 / fps), loop=0
    )
    print(f"{path}: {len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB")


def live(args: argparse.Namespace) -> None:
    sequence = MotSequence.load(args.data / f"MOT17-{args.sequence}-FRCNN")
    cache = args.cache / args.model
    latency = np.load(cache / f"{sequence.name}.latency.npy")

    def detector() -> MotFileDetector:
        return MotFileDetector(cache / f"{sequence.name}.txt")

    blocking, _ = _blocking(sequence, detector(), latency, offline=False)
    background, stats = _background(sequence, detector(), latency, True, 0.0)
    detected = dict(zip(range(1, sequence.length + 1), stats["detected"], strict=True))

    frames = []
    for index in range(args.start, args.start + args.frames * args.step, args.step):
        image = _read(sequence.image_dir / f"{index:06d}.jpg")
        left, scale = _panel(image, "Blocking loop: the boxes are late")
        right, _ = _panel(image, "Background detector + optical flow")
        for obj in blocking[index - 1][1]:
            _draw(left, scale, obj, _COLORS[obj.track_id % len(_COLORS)])
        for obj in background[index - 1][1]:
            _draw(right, scale, obj, _COLORS[obj.track_id % len(_COLORS)])
        if detected[index]:
            # A mark on the frames that use a new detector result.
            cv2.circle(right, (right.shape[1] - 14, 13), 6, (80, 220, 80), -1, cv2.LINE_AA)
        frames.append(np.hstack([left, np.full((left.shape[0], 4, 3), 255, np.uint8), right]))
    _save(frames, args.out, args.fps)


def _tracks(
    data: Path, embeddings: Path, event: occlusion.Event, variant: str, last: int
) -> dict[int, list[TrackedObject]]:
    """The reported objects of each frame, for one occlusion event and one tracker variant."""
    root = str(data / event.sequence)
    truth = occlusion._ground_truth(root)[event.object_id]
    by_frame = occlusion._detections(root)
    config, embedder = occlusion.VARIANTS[variant]
    vectors_by_frame = occlusion._embeddings(str(embeddings), embedder, root) if embedder else None
    tracker = Tracker(config)
    gap = range(event.start, event.start + event.duration)
    outputs = {}
    for frame in range(1, last + 1):
        detections = by_frame.get(frame, Detections.empty())
        vectors = vectors_by_frame.get(frame) if vectors_by_frame else None
        if frame in gap and len(detections):
            hidden = iou_matrix(truth[frame][0][None], detections.boxes)[0] > occlusion.MATCH_IOU
            detections = detections.select(~hidden)
            vectors = None if vectors is None else vectors[~hidden]
        embed = None if vectors is None else vectors.__getitem__
        outputs[frame] = tracker.update(detections, embed=embed)
    return outputs


def occlusion_demo(args: argparse.Namespace) -> None:
    root = args.data / f"MOT17-{args.sequence}-SDP"
    sequence = MotSequence.load(root)
    # Of the events that the default tracker loses and the appearance recovers, take the
    # one with the largest object. It is the easiest to see.
    truth_all = occlusion._ground_truth(str(root))
    candidates = []
    for candidate in occlusion.find_events(root, args.gap, 200):
        base = occlusion.run_event(args.data, args.embeddings, candidate, "lost-90").result
        ours = occlusion.run_event(args.data, args.embeddings, candidate, "histogram").result
        if (base, ours) == ("switched", "recovered"):
            box = truth_all[candidate.object_id][candidate.start][0]
            candidates.append(((box[2] - box[0]) * (box[3] - box[1]), candidate))
    if not candidates:
        raise SystemExit("no event with a switch and a recovery in this sequence")
    event = max(candidates, key=lambda item: item[0])[1]

    first = event.start - occlusion.PRE_FRAMES
    last = event.start + event.duration + 24
    truth = occlusion._ground_truth(str(root))[event.object_id]
    variants = [("lost-90", "Overlap only: new ID"), ("histogram", "With appearance: same ID")]
    outputs = [_tracks(args.data, args.embeddings, event, name, last) for name, _ in variants]

    frames = []
    for index in range(first, last + 1, args.step):
        image = _read(sequence.image_dir / f"{index:06d}.jpg")
        hidden = event.start <= index < event.start + event.duration
        panels = []
        for (_, title), output in zip(variants, outputs, strict=True):
            panel, scale = _panel(image, title)
            follower = occlusion._follower(truth[index][0], output[index])
            for obj in output[index]:
                if obj.track_id == follower:
                    _draw(panel, scale, obj, _TARGET)
                    x1, y1 = round(obj.box[0] * scale), round(obj.box[1] * scale) + _BAR
                    # Keep the label below the title bar when the box is at the top.
                    cv2.putText(
                        panel,
                        f"ID {obj.track_id}",
                        (max(x1, 4), max(y1 - 16, _BAR + 22)),
                        _FONT,
                        0.7,
                        _TARGET,
                        2,
                        cv2.LINE_AA,
                    )
            if hidden:
                x1, y1, x2, y2 = (round(v * scale) for v in truth[index][0])
                cv2.rectangle(panel, (x1, y1 + _BAR), (x2, y2 + _BAR), (60, 60, 230), 1)
                cv2.putText(
                    panel, "no detections", (x1, y1 + _BAR - 4), _FONT, 0.4, (60, 60, 230), 1
                )
            panels.append(panel)
        divider = np.full((panels[0].shape[0], 4, 3), 255, np.uint8)
        frames.append(np.hstack([panels[0], divider, panels[1]]))
    print(f"event: object {event.object_id}, gap {event.start}..{event.start + event.duration - 1}")
    _save(frames, args.out, args.fps)


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("demo", choices=["live", "occlusion"])
    p.add_argument("out", type=Path)
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--cache", type=Path, default=Path("runs/detections"))
    p.add_argument("--embeddings", type=Path, default=Path("runs/embeddings"))
    p.add_argument("--model", default="yolox_s")
    p.add_argument("--sequence", default="10", help="two-digit MOT17 sequence number")
    p.add_argument("--start", type=int, default=1, help="first frame of the live demo")
    p.add_argument("--frames", type=int, default=70, help="number of frames in the live demo")
    p.add_argument("--step", type=int, default=2, help="use each N-th frame of the sequence")
    p.add_argument("--gap", type=int, default=30, help="gap length of the occlusion demo")
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()
    if args.demo == "live":
        live(args)
    else:
        occlusion_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
