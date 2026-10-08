"""Compute an appearance vector for each row of each MOT17 detection file.

The occlusion experiment replays these vectors, so it does not decode images.

    python -m bench.cache_embeddings histogram --detector SDP
    python -m bench.cache_embeddings osnet --model models/osnet_x0_25_msmt17.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from leantrack.io.mot import MotSequence
from leantrack.io.sources import image_dir_frames
from leantrack.reid.base import Embedder
from leantrack.reid.histogram import HistogramEmbedder
from leantrack.reid.onnx import OnnxEmbedder


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("embedder", choices=["histogram", "osnet"])
    p.add_argument("--model", type=Path, default=Path("models/osnet_x0_25_msmt17.onnx"))
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--detector", choices=["DPM", "FRCNN", "SDP"], default="SDP")
    p.add_argument("--out", type=Path, default=Path("runs/embeddings"))
    args = p.parse_args()

    embedder: Embedder = (
        HistogramEmbedder() if args.embedder == "histogram" else OnnxEmbedder(args.model)
    )
    out = args.out / args.embedder
    out.mkdir(parents=True, exist_ok=True)

    for root in sorted(args.data.glob(f"MOT17-*-{args.detector}")):
        sequence = MotSequence.load(root)
        rows = np.loadtxt(sequence.detections_path, delimiter=",", ndmin=2)
        boxes = rows[:, 2:6].copy()
        boxes[:, 2:] += boxes[:, :2]
        frames = rows[:, 0].astype(np.int64)

        vectors: np.ndarray | None = None
        seconds = 0.0
        for frame in image_dir_frames(sequence.image_dir):
            index = np.flatnonzero(frames == frame.index)
            if len(index) == 0 or frame.image is None:
                continue
            start = time.perf_counter()
            embedded = embedder.embed(frame.image, boxes[index])
            seconds += time.perf_counter() - start
            if vectors is None:
                vectors = np.zeros((len(rows), embedded.shape[1]), dtype=np.float32)
            vectors[index] = embedded
        assert vectors is not None
        np.save(out / f"{sequence.name}.npy", vectors)
        print(f"{sequence.name}: {len(rows)} boxes, {1000 * seconds / len(rows):.2f} ms for each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
