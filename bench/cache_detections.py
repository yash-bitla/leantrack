"""Run a YOLOX model on each MOT17 train sequence and store the detections and latencies.

The schedule experiments replay these files. Thus each experiment uses identical
detections, and the detector runs one time for each model.

    python bench/cache_detections.py models/yolox_s.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from leantrack.detect.yolox import YoloxDetector
from leantrack.io.mot import MotSequence
from leantrack.io.sources import image_dir_frames


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model", type=Path)
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--out", type=Path, default=Path("runs/detections"))
    p.add_argument("--threads", type=int, help="ONNX Runtime threads (default: all cores)")
    args = p.parse_args()

    detector = YoloxDetector(args.model, threads=args.threads)
    out = args.out / args.model.stem
    out.mkdir(parents=True, exist_ok=True)

    # The three detector variants of a MOT17 sequence share the images. One is sufficient.
    for root in sorted(args.data.glob("MOT17-*-FRCNN")):
        sequence = MotSequence.load(root)
        lines: list[str] = []
        latency_ms: list[float] = []
        for frame in image_dir_frames(sequence.image_dir):
            start = time.perf_counter()
            detections = detector.detect(frame)
            latency_ms.append((time.perf_counter() - start) * 1000.0)
            for (x1, y1, x2, y2), score in zip(detections.boxes, detections.scores, strict=True):
                lines.append(
                    f"{frame.index},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{score:.4f}"
                )
        (out / f"{sequence.name}.txt").write_text("\n".join(lines) + "\n")
        np.save(out / f"{sequence.name}.latency.npy", np.array(latency_ms))
        print(f"{sequence.name}: {len(lines)} detections, p50 {np.median(latency_ms):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
