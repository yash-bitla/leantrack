from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from leantrack._types import Detections, Frame
from leantrack.detect.base import Detector
from leantrack.pipeline import FrameResult, _TimedEmbed
from leantrack.propagate.flow import FlowPropagator
from leantrack.reid.base import Embedder
from leantrack.tracks.track import TrackState
from leantrack.tracks.tracker import Tracker

_LATENCY_GAIN = 0.2
"""Weight of a new detector duration in the moving average."""
_LATENCY_MARGIN = 1.1
"""The loop expects a run to take this factor of the average, so it waits a little less often."""


@dataclass(frozen=True, slots=True)
class LateDetections:
    frame_index: int
    """Index of the frame that the detector saw."""
    detections: Detections
    latency_ms: float
    """Duration of the detector run."""


class DetectionExecutor(Protocol):
    """Runs the detector without a block of the frame loop. One run at a time."""

    @property
    def busy(self) -> bool: ...

    def submit(self, frame: Frame) -> None: ...

    def pending_ms(self, frame_index: int) -> float:
        """Time since the submitted run started. Zero if no run is in progress."""
        ...

    def poll(self, frame_index: int, wait_ms: float) -> tuple[LateDetections | None, float]:
        """Wait at most `wait_ms` for the submitted run.

        Returns the result, or None if it is not ready, and the time that the wait took.
        """
        ...


class ThreadedExecutor:
    """Runs the detector in one worker thread.

    ONNX Runtime releases the global interpreter lock during inference, so the frame
    loop continues while the detector runs.
    """

    def __init__(self, detector: Detector) -> None:
        self._detector = detector
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")
        # (frame index, start time, future of (detections, duration in ms))
        self._pending: tuple[int, float, Future[tuple[Detections, float]]] | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def submit(self, frame: Frame) -> None:
        if self._pending is not None:
            raise RuntimeError("a detector run is already in progress")
        self._pending = (frame.index, time.perf_counter(), self._pool.submit(self._run, frame))

    def _run(self, frame: Frame) -> tuple[Detections, float]:
        start = time.perf_counter()
        detections = self._detector.detect(frame)
        return detections, (time.perf_counter() - start) * 1000.0

    def pending_ms(self, frame_index: int) -> float:
        if self._pending is None:
            return 0.0
        return (time.perf_counter() - self._pending[1]) * 1000.0

    def poll(self, frame_index: int, wait_ms: float) -> tuple[LateDetections | None, float]:
        if self._pending is None:
            return None, 0.0
        source, _, future = self._pending
        start = time.perf_counter()
        try:
            detections, latency_ms = future.result(timeout=max(wait_ms, 0.0) / 1000.0)
        except FutureTimeoutError:
            return None, (time.perf_counter() - start) * 1000.0
        self._pending = None
        waited_ms = (time.perf_counter() - start) * 1000.0
        return LateDetections(source, detections, latency_ms), waited_ms

    def close(self) -> None:
        self._pool.shutdown(wait=True)


class SimulatedExecutor:
    """A detector with a known latency on a simulated clock, for repeatable experiments.

    Frame `i` arrives at `i * frame_period_ms`. A run that starts on frame `i` is ready at
    `i * frame_period_ms + latency_ms(i)`. The detections themselves come from `detector`
    at once, for example from a stored file.
    """

    def __init__(
        self, detector: Detector, latency_ms: Callable[[int], float], frame_period_ms: float
    ) -> None:
        self._detector = detector
        self._latency_ms = latency_ms
        self._period = frame_period_ms
        # (frame index, detections, latency in ms)
        self._pending: tuple[int, Detections, float] | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def submit(self, frame: Frame) -> None:
        if self._pending is not None:
            raise RuntimeError("a detector run is already in progress")
        self._pending = (frame.index, self._detector.detect(frame), self._latency_ms(frame.index))

    def pending_ms(self, frame_index: int) -> float:
        if self._pending is None:
            return 0.0
        return (frame_index - self._pending[0]) * self._period

    def poll(self, frame_index: int, wait_ms: float) -> tuple[LateDetections | None, float]:
        if self._pending is None:
            return None, 0.0
        source, detections, latency_ms = self._pending
        remaining = latency_ms - self.pending_ms(frame_index)
        if remaining > wait_ms:
            # The caller waited for the full time and got no result.
            return None, max(wait_ms, 0.0)
        self._pending = None
        return LateDetections(source, detections, latency_ms), max(remaining, 0.0)


def run_realtime(
    frames: Iterable[Frame],
    executor: DetectionExecutor,
    tracker: Tracker,
    propagator: FlowPropagator,
    embedder: Embedder | None = None,
    *,
    frame_budget_ms: float = 0.0,
    compensate: bool = True,
) -> Iterator[FrameResult]:
    """Track a live sequence with the detector in the background.

    A new detector run starts when the executor is free. Each frame reports the tracks
    from the Kalman prediction and the optical flow.

    A detector result is late by some frames. With `compensate`, optical flow moves the
    detected boxes from the frame that the detector saw to the current frame, before the
    association. The stored history of the propagator limits the permitted delay.

    `frame_budget_ms` is the time that one frame can use. The loop waits for the detector
    only if it expects the result inside that budget. The estimate is a moving average of
    the durations of the earlier runs. With a budget of 0, the loop does not wait. With a
    budget above the detector latency, each frame gets a new result, as in a blocking loop.
    """
    expected_ms: float | None = None
    for frame in frames:
        if frame.image is None:
            raise ValueError("the real-time pipeline needs the frame pixels")
        start = time.perf_counter()
        propagator.observe(frame.image)
        moved = None
        active = [t for t in tracker.tracks if t.state is not TrackState.LOST]
        if active:
            results = propagator.propagate(np.stack([t.box for t in active]))
            moved = {t.track_id: r for t, r in zip(active, results, strict=True)}
        tracker.advance(moved)

        if not executor.busy:
            executor.submit(frame)
        polled = time.perf_counter()
        wait_ms = 0.0
        if expected_ms is not None:
            available = frame_budget_ms - (polled - start) * 1000.0
            remaining = expected_ms * _LATENCY_MARGIN - executor.pending_ms(frame.index)
            if remaining <= available:
                wait_ms = max(available, 0.0)
        late, waited_ms = executor.poll(frame.index, wait_ms)
        resumed = time.perf_counter()

        embed_ms = 0.0
        age = 0
        if late is None:
            objects = tracker.coast()
        else:
            if expected_ms is None:
                expected_ms = late.latency_ms
            else:
                expected_ms += _LATENCY_GAIN * (late.latency_ms - expected_ms)
            age = frame.index - late.frame_index
            detections = late.detections
            if compensate and age > 0:
                # A result that is older than the stored history gets a partial correction.
                boxes = propagator.catch_up(detections.boxes, min(age, propagator.history))
                detections = Detections(boxes, detections.scores, detections.classes)
            embed = None
            if embedder is not None:
                embed = _TimedEmbed(embedder, frame.image, detections.boxes)
            objects = tracker.associate(detections, embed)
            if embed is not None:
                embed_ms = embed.seconds * 1000.0
        end = time.perf_counter()
        yield FrameResult(
            frame.index,
            objects,
            late is not None,
            detect_ms=waited_ms,
            propagate_ms=(polled - start) * 1000.0,
            embed_ms=embed_ms,
            track_ms=(end - resumed) * 1000.0 - embed_ms,
            detection_age=age,
        )
