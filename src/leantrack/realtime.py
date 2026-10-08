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


@dataclass(frozen=True, slots=True)
class LateDetections:
    frame_index: int
    """Index of the frame that the detector saw."""
    detections: Detections
    waited_ms: float
    """Time that the caller waited for this result in `poll`."""


class DetectionExecutor(Protocol):
    """Runs the detector without a block of the frame loop. One run at a time."""

    @property
    def busy(self) -> bool: ...

    def submit(self, frame: Frame) -> None: ...

    def poll(self, frame_index: int, wait_ms: float) -> LateDetections | None:
        """The result of the submitted run, if it is ready after at most `wait_ms`."""
        ...


class ThreadedExecutor:
    """Runs the detector in one worker thread.

    ONNX Runtime releases the global interpreter lock during inference, so the frame
    loop continues while the detector runs.
    """

    def __init__(self, detector: Detector) -> None:
        self._detector = detector
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")
        self._pending: tuple[int, Future[Detections]] | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def submit(self, frame: Frame) -> None:
        if self._pending is not None:
            raise RuntimeError("a detector run is already in progress")
        self._pending = (frame.index, self._pool.submit(self._detector.detect, frame))

    def poll(self, frame_index: int, wait_ms: float) -> LateDetections | None:
        if self._pending is None:
            return None
        source, future = self._pending
        start = time.perf_counter()
        try:
            detections = future.result(timeout=max(wait_ms, 0.0) / 1000.0)
        except FutureTimeoutError:
            return None
        self._pending = None
        return LateDetections(source, detections, (time.perf_counter() - start) * 1000.0)

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
        self._pending: tuple[int, Detections, float] | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def submit(self, frame: Frame) -> None:
        if self._pending is not None:
            raise RuntimeError("a detector run is already in progress")
        ready = frame.index * self._period + self._latency_ms(frame.index)
        self._pending = (frame.index, self._detector.detect(frame), ready)

    def poll(self, frame_index: int, wait_ms: float) -> LateDetections | None:
        if self._pending is None:
            return None
        source, detections, ready = self._pending
        now = frame_index * self._period
        if ready > now + wait_ms:
            return None
        self._pending = None
        return LateDetections(source, detections, max(ready - now, 0.0))


def run_realtime(
    frames: Iterable[Frame],
    executor: DetectionExecutor,
    tracker: Tracker,
    propagator: FlowPropagator,
    embedder: Embedder | None = None,
    *,
    wait_ms: float = 0.0,
    compensate: bool = True,
) -> Iterator[FrameResult]:
    """Track a live sequence with the detector in the background.

    The frame loop does not wait for the detector, apart from `wait_ms` in each frame. A
    new detector run starts when the executor is free. Each frame reports the tracks from
    the Kalman prediction and the optical flow.

    A detector result is late by some frames. With `compensate`, optical flow moves the
    detected boxes from the frame that the detector saw to the current frame, before the
    association. The stored history of the propagator limits the permitted delay.
    """
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
        late = executor.poll(frame.index, wait_ms)
        waited_ms = (time.perf_counter() - polled) * 1000.0 if late is None else late.waited_ms
        resumed = time.perf_counter()

        embed_ms = 0.0
        age = 0
        if late is None:
            objects = tracker.coast()
        else:
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
