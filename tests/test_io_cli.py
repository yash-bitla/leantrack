from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from leantrack._types import TrackedObject
from leantrack.cli import main
from leantrack.io.mot import MotSequence, MotWriter, read_detections
from leantrack.io.sources import image_dir_frames, index_frames, video_frames
from leantrack.synthetic import SyntheticObject, SyntheticScene


def _write_sequence(root: Path, scene: SyntheticScene) -> None:
    (root / "det").mkdir(parents=True)
    (root / "seqinfo.ini").write_text(
        f"[Sequence]\nname=synthetic\nimDir=img1\nframeRate=30\nseqLength={scene.length}\n"
    )
    lines = []
    for frame in range(1, scene.length + 1):
        det = scene.detections(frame)
        for (x1, y1, x2, y2), score in zip(det.boxes, det.scores, strict=True):
            lines.append(f"{frame},-1,{x1},{y1},{x2 - x1},{y2 - y1},{score},-1,-1,-1")
    (root / "det" / "det.txt").write_text("\n".join(lines) + "\n")


def test_writer_output_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "out" / "result.txt"
    with MotWriter(path) as writer:
        writer.write(1, [TrackedObject(7, (10.0, 20.0, 50.0, 100.0), 0.8, 0)])
        writer.write(3, [TrackedObject(7, (12.0, 20.0, 52.0, 100.0), 0.9, 0)])
    by_frame = read_detections(path)
    assert sorted(by_frame) == [1, 3]
    assert by_frame[1].boxes[0] == pytest.approx([10.0, 20.0, 50.0, 100.0])
    assert by_frame[3].scores[0] == pytest.approx(0.9)


def test_writer_requires_context_manager(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        MotWriter(tmp_path / "x.txt").write(1, [])


def test_cli_tracks_a_sequence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scene = SyntheticScene(
        (
            SyntheticObject((100, 100, 140, 180), (3.0, 0.0)),
            SyntheticObject((100, 300, 140, 380), (2.0, 0.0), first_frame=10),
        ),
        length=50,
        jitter=0.5,
    )
    _write_sequence(tmp_path / "seq", scene)
    out = tmp_path / "result.txt"

    assert main(["track", str(tmp_path / "seq"), "--out", str(out)]) == 0
    assert "50 frames, 2 tracks" in capsys.readouterr().out

    ids = np.loadtxt(out, delimiter=",")[:, 1]
    assert set(ids) == {1.0, 2.0}


def test_cli_needs_a_model_for_plain_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["track", str(tmp_path), "--out", str(tmp_path / "o.txt")]) == 2
    assert "--model is necessary" in capsys.readouterr().err


def test_cli_reports_a_missing_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["track", str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "o.txt")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_reports_an_unknown_reid_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene = SyntheticScene((SyntheticObject((100, 100, 140, 180), (3.0, 0.0)),), length=5)
    _write_sequence(tmp_path / "seq", scene)
    arguments = ["track", str(tmp_path / "seq"), "--out", str(tmp_path / "o.txt")]
    assert main([*arguments, "--reid", "missing.onnx"]) == 2
    assert "--reid must be" in capsys.readouterr().err


def _render(root: Path, scene: SyntheticScene, colors: list[tuple[int, int, int]]) -> None:
    """Write the frames of a scene as images: a textured background and one color per object."""
    (root / "img1").mkdir()
    noise = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(noise, (9, 9), 0) // 3
    for frame in range(1, scene.length + 1):
        image = background.copy()
        for object_id, box in scene.ground_truth(frame).items():
            x1, y1, x2, y2 = (round(float(v)) for v in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[object_id - 1], thickness=-1)
        cv2.imwrite(str(root / "img1" / f"{frame:06d}.png"), image)


def test_cli_with_flow_and_reid_on_a_rendered_sequence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene = SyntheticScene(
        (
            SyntheticObject((100, 100, 140, 180), (3.0, 0.0)),
            SyntheticObject((400, 200, 440, 280), (-2.0, 0.0)),
        ),
        length=30,
    )
    root = tmp_path / "seq"
    _write_sequence(root, scene)
    _render(root, scene, [(0, 0, 255), (255, 0, 0)])
    out = tmp_path / "result.txt"

    arguments = ["track", str(root), "--out", str(out), "--interval", "5", "--flow"]
    assert main([*arguments, "--reid", "histogram"]) == 0
    assert "30 frames, 2 tracks, 6 detector runs" in capsys.readouterr().out

    rows = np.loadtxt(out, delimiter=",")
    assert set(rows[:, 1]) == {1.0, 2.0}
    # Each track has a row on each frame, also between detector runs.
    assert len(rows) == 60


def test_sequence_metadata(tmp_path: Path) -> None:
    _write_sequence(tmp_path, SyntheticScene((), length=12))
    sequence = MotSequence.load(tmp_path)
    assert (sequence.name, sequence.length, sequence.frame_rate) == ("synthetic", 12, 30.0)
    assert sequence.image_dir == tmp_path / "img1"


def test_frame_sources(tmp_path: Path) -> None:
    assert [f.index for f in index_frames(3)] == [1, 2, 3]

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:06d}.jpg"), np.full((48, 64, 3), i * 40, np.uint8))
    frames = list(image_dir_frames(tmp_path))
    assert [f.index for f in frames] == [1, 2, 3]
    assert frames[0].image is not None and frames[0].image.shape == (48, 64, 3)

    with pytest.raises(OSError):
        next(video_frames(tmp_path / "missing.mp4"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        next(image_dir_frames(empty))
