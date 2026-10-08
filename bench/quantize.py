"""Make an INT8 version of a YOLOX ONNX model with static quantization.

The calibration images are frames of the MOT17 sequences 02, 04 and 09.

    python -m bench.quantize models/yolox_s.onnx models/yolox_s_int8.onnx
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from leantrack.detect.yolox import letterbox

CALIBRATION_SEQUENCES = ("02", "04", "09")
MIN_OPSET = 13


class _Reader(CalibrationDataReader):  # type: ignore[misc]
    def __init__(self, model: Path, images: list[Path]) -> None:
        model_input = ort.InferenceSession(str(model)).get_inputs()[0]
        self._name = model_input.name
        self._size = (int(model_input.shape[2]), int(model_input.shape[3]))
        self._images: Iterator[Path] = iter(images)

    def get_next(self) -> dict[str, np.ndarray] | None:
        path = next(self._images, None)
        if path is None:
            return None
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"cannot decode {path}")
        blob, _ = letterbox(image, *self._size)
        return {self._name: blob}


def head_nodes(model: Path) -> list[str]:
    """Names of the nodes after the last convolution of each output branch.

    These nodes join the box, objectness and class outputs. The three have different
    value ranges, so one INT8 scale for the joined tensor loses the small values.
    """
    graph = onnx.load(str(model)).graph
    names = []
    for node in reversed(graph.node):
        if node.op_type == "Conv":
            break
        names.append(node.name)
    return names


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("model", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--data", type=Path, default=Path("data/MOT17/train"))
    p.add_argument("--images", type=int, default=120, help="number of calibration images")
    p.add_argument(
        "--keep-head-float",
        action="store_true",
        help="do not quantize the nodes after the last convolution",
    )
    args = p.parse_args()

    paths: list[Path] = []
    for number in CALIBRATION_SEQUENCES:
        frames = sorted((args.data / f"MOT17-{number}-FRCNN" / "img1").glob("*.jpg"))
        step = max(len(frames) // (args.images // len(CALIBRATION_SEQUENCES)), 1)
        paths.extend(frames[::step])

    # Quantization for each channel needs opset 13. The YOLOX release files have opset 11.
    prepared = args.out.with_suffix(".prepared.onnx")
    model = onnx.load(str(args.model))
    if model.opset_import[0].version < MIN_OPSET:
        model = onnx.version_converter.convert_version(model, MIN_OPSET)
    onnx.save(model, str(prepared))
    quant_pre_process(str(prepared), str(prepared))
    excluded = head_nodes(prepared) if args.keep_head_float else []
    quantize_static(
        str(prepared),
        str(args.out),
        _Reader(prepared, paths),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=excluded,
    )
    prepared.unlink()
    size_in, size_out = args.model.stat().st_size / 1e6, args.out.stat().st_size / 1e6
    print(
        f"{args.out}: {len(paths)} calibration images, {len(excluded)} float nodes, "
        f"{size_in:.1f} MB -> {size_out:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
