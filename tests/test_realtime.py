from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from leantrack._types import Detections, Frame
from leantrack.cli import main
from leantrack.io.sources import paced
from leantrack.propagate.flow import FlowPropagator
from leantrack.realtime import SimulatedExecutor, ThreadedExecutor, run_realtime
from leantrack.tracks.tracker import Tracker

_MODEL = Path(__file__).parent.parent / "models" / "yolox_nano.onnx"
_START = np.array([100.0, 80.0, 160.0, 160.0])
_SPEED = 4


def _texture() -> np.ndarray:
    noise = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (7, 7), 0)


def _moving_frames(count: int) -> list[Frame]:
    """The full image moves `_SPEED` px to the right in each frame."""
    image = _texture()
    return [Frame(i + 1, np.roll(image, _SPEED * i, axis=1)) for i in range(count)]


def _true_box(frame_index: int) -> np.ndarray:
    shift = _SPEED * (frame_index - 1)
    return _START + np.array([shift, 0.0, shift, 0.0])


class _MovingDetector:
    """Detects one object that moves with the image."""

    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds

    def detect(self, frame: Frame) -> Detections:
        time.sleep(self.seconds)
        return Detections(_true_box(frame.index)[None], np.array([0.9]), np.array([0]))


def test_simulated_executor_follows_the_clock() -> None:
    executor = SimulatedExecutor(_MovingDetector(), lambda index: 80.0, frame_period_ms=100 / 3)
    assert executor.poll(1, 0.0) == (None, 0.0)
    executor.submit(Frame(1))
    assert executor.busy
    with pytest.raises(RuntimeError):
        executor.submit(Frame(2))
    # The run starts at 33.3 ms and is ready at 113.3 ms. Frame 3 is at 100 ms.
    assert executor.poll(3, 0.0) == (None, 0.0)
    # A wait of 10 ms is too short: the caller loses the 10 ms and gets no result.
    assert executor.poll(3, 10.0) == (None, 10.0)
    late, waited_ms = executor.poll(3, 20.0)
    assert late is not None
    assert late.frame_index == 1
    assert late.latency_ms == 80.0
    assert waited_ms == pytest.approx(13.33, abs=0.01)
    assert not executor.busy


def test_simulated_executor_gives_a_ready_result_without_a_wait() -> None:
    executor = SimulatedExecutor(_MovingDetector(), lambda index: 80.0, frame_period_ms=100 / 3)
    executor.submit(Frame(1))
    late, waited_ms = executor.poll(4, 0.0)
    assert late is not None and waited_ms == 0.0


def test_threaded_executor_does_not_block() -> None:
    executor = ThreadedExecutor(_MovingDetector(seconds=0.2))
    try:
        start = time.perf_counter()
        executor.submit(Frame(1))
        assert executor.poll(1, 0.0)[0] is None
        assert time.perf_counter() - start < 0.1
        assert executor.busy
        with pytest.raises(RuntimeError):
            executor.submit(Frame(2))

        late, _ = executor.poll(2, 2000.0)
        assert late is not None
        assert late.frame_index == 1
        assert late.detections.boxes[0] == pytest.approx(_true_box(1))
        assert not executor.busy
    finally:
        executor.close()


def test_catch_up_moves_boxes_through_stored_frames() -> None:
    propagator = FlowPropagator()
    for frame in _moving_frames(6):
        assert frame.image is not None
        propagator.observe(frame.image)
    assert propagator.history == 5

    moved = propagator.catch_up(_true_box(2)[None], frames_back=4)
    assert moved[0] == pytest.approx(_true_box(6), abs=1.0)
    assert propagator.catch_up(_true_box(6)[None], 0)[0] == pytest.approx(_true_box(6))
    with pytest.raises(ValueError, match="cannot go back"):
        propagator.catch_up(_true_box(1)[None], 6)


def _errors(compensate: bool) -> tuple[list[float], list[int]]:
    """Left-edge error of the reported box on frames with late detections, and their ages."""
    executor = SimulatedExecutor(_MovingDetector(), lambda index: 80.0, frame_period_ms=100 / 3)
    results = run_realtime(
        _moving_frames(20), executor, Tracker(), FlowPropagator(), compensate=compensate
    )
    errors, ages = [], []
    for result in results:
        if result.detected and result.objects:
            errors.append(abs(result.objects[0].box[0] - _true_box(result.frame_index)[0]))
            ages.append(result.detection_age)
    return errors, ages


def test_late_detections_are_moved_to_the_current_frame() -> None:
    errors, ages = _errors(compensate=True)
    assert set(ages) == {3}
    assert max(errors) < 2.0

    raw_errors, _ = _errors(compensate=False)
    # Without the correction, the box is behind by a part of 3 frames * 4 px.
    assert min(raw_errors[1:]) > 3.0


def test_realtime_reports_the_track_on_each_frame() -> None:
    executor = SimulatedExecutor(_MovingDetector(), lambda index: 80.0, frame_period_ms=100 / 3)
    results = list(run_realtime(_moving_frames(20), executor, Tracker(), FlowPropagator()))
    # The first result arrives on frame 4. After that, each frame has the track.
    assert [len(r.objects) for r in results[:3]] == [0, 0, 0]
    assert all(len(r.objects) == 1 for r in results[3:])
    assert sum(r.detected for r in results) < 8


def test_realtime_needs_pixels() -> None:
    executor = SimulatedExecutor(_MovingDetector(), lambda index: 0.0, frame_period_ms=33.0)
    with pytest.raises(ValueError, match="pixels"):
        list(run_realtime([Frame(1)], executor, Tracker(), FlowPropagator()))


def test_paced_frames_arrive_at_the_frame_rate() -> None:
    start = time.perf_counter()
    frames = list(paced([Frame(i) for i in range(1, 6)], fps=100.0))
    assert len(frames) == 5
    assert time.perf_counter() - start >= 0.04
    with pytest.raises(ValueError):
        list(paced([Frame(1)], fps=0.0))


def test_cli_background_needs_a_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "seq").mkdir()
    (tmp_path / "seq" / "seqinfo.ini").write_text(
        "[Sequence]\nname=s\nimDir=img1\nframeRate=30\nseqLength=1\n"
    )
    arguments = ["track", str(tmp_path / "seq"), "--out", str(tmp_path / "o.txt"), "--background"]
    assert main(arguments) == 2
    assert "--background needs --model" in capsys.readouterr().err


@pytest.mark.skipif(not _MODEL.is_file(), reason="models/yolox_nano.onnx is not present")
def test_cli_background_on_an_image_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for frame in _moving_frames(12):
        assert frame.image is not None
        cv2.imwrite(str(tmp_path / f"{frame.index:06d}.png"), frame.image)
    out = tmp_path / "out" / "tracks.txt"
    arguments = ["track", str(tmp_path), "--out", str(out), "--model", str(_MODEL), "--background"]
    assert main(arguments) == 0
    assert "12 frames" in capsys.readouterr().out
    assert out.is_file()


def _budget_run(latency_ms: float, budget_ms: float) -> list[tuple[bool, int, float]]:
    """(detected, detection age, time waited) for each frame, at a 33.3 ms frame period."""
    executor = SimulatedExecutor(
        _MovingDetector(), lambda index: latency_ms, frame_period_ms=100 / 3
    )
    results = run_realtime(
        _moving_frames(16), executor, Tracker(), FlowPropagator(), frame_budget_ms=budget_ms
    )
    return [(r.detected, r.detection_age, r.detect_ms) for r in results]


def test_without_a_budget_the_loop_does_not_wait() -> None:
    frames = _budget_run(latency_ms=20.0, budget_ms=0.0)
    assert all(waited == 0.0 for _, _, waited in frames)
    # A 20 ms result is ready one frame later.
    assert {age for detected, age, _ in frames if detected} == {1}


def test_a_budget_above_the_latency_gives_a_new_result_on_each_frame() -> None:
    frames = _budget_run(latency_ms=20.0, budget_ms=30.0)
    # The loop does not know the latency before the first result. After that, it waits.
    settled = frames[3:]
    assert all(detected and age == 0 for detected, age, _ in settled)
    assert all(waited == pytest.approx(20.0) for _, _, waited in settled)


def test_the_loop_does_not_wait_for_a_result_that_cannot_fit_the_budget() -> None:
    frames = _budget_run(latency_ms=80.0, budget_ms=10.0)
    # 80 ms is ready 13.3 ms after the start of the third frame: more than the budget.
    assert all(waited == 0.0 for _, _, waited in frames)
    assert {age for detected, age, _ in frames if detected} == {3}


def test_a_budget_decreases_the_age_of_a_slow_result() -> None:
    frames = _budget_run(latency_ms=80.0, budget_ms=25.0)
    settled = [(age, waited) for detected, age, waited in frames[4:] if detected]
    # The loop waits 13.3 ms in the third frame and gets the result one frame sooner.
    assert {age for age, _ in settled} == {2}
    assert all(waited == pytest.approx(13.33, abs=0.01) for _, waited in settled)
    assert all(waited == 0.0 for detected, _, waited in frames[4:] if not detected)


def test_executors_report_the_time_of_the_run_in_progress() -> None:
    simulated = SimulatedExecutor(_MovingDetector(), lambda index: 80.0, frame_period_ms=40.0)
    assert simulated.pending_ms(5) == 0.0
    simulated.submit(Frame(2))
    assert simulated.pending_ms(5) == pytest.approx(120.0)

    threaded = ThreadedExecutor(_MovingDetector(seconds=0.05))
    try:
        assert threaded.pending_ms(1) == 0.0
        threaded.submit(Frame(1))
        time.sleep(0.02)
        assert threaded.pending_ms(1) >= 20.0
        late, _ = threaded.poll(1, 2000.0)
        assert late is not None and late.latency_ms >= 50.0
    finally:
        threaded.close()
