from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from leantrack._types import Frame, Image
from leantrack.pipeline import FrameResult

_VIDEO_WIDTH = 960
_COLORS = [(66, 135, 245), (60, 180, 75), (245, 130, 48), (145, 30, 180), (0, 190, 190)]


class FrameTap:
    """Keeps the newest frame that goes into the pipeline, for the video output."""

    def __init__(self) -> None:
        self.current: Frame | None = None

    def watch(self, frames: Iterable[Frame]) -> Iterator[Frame]:
        for frame in frames:
            self.current = frame
            yield frame


@dataclass(frozen=True, slots=True)
class Stats:
    state: str
    """One of: idle, running, finished, failed."""
    frames: int
    detector_runs: int
    tracks: int
    """Number of different track ids until now."""
    frames_per_second: float
    latency_ms_p50: float
    latency_ms_p99: float
    dropped_events: int
    error: str | None


def event(result: FrameResult) -> dict[str, Any]:
    """The JSON form of one frame result."""
    return {
        "frame": result.frame_index,
        "detected": result.detected,
        "detection_age": result.detection_age,
        "latency_ms": round(result.total_ms, 3),
        "objects": [
            {
                "id": o.track_id,
                "box": [round(v, 1) for v in o.box],
                "score": round(o.score, 3),
                "class": o.class_id,
            }
            for o in result.objects
        ],
    }


class _Subscriber:
    def __init__(self, loop: asyncio.AbstractEventLoop, size: int) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=size)
        self.dropped = 0

    def offer(self, item: dict[str, Any]) -> None:
        # Runs in the event loop. A slow client loses its oldest event, not the newest.
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait(item)


class TrackingService:
    """Runs one tracking pipeline in a thread and gives its results to many clients.

    `results` makes the iterator of frame results. `tap` gives the newest frame for the
    video output, and can be None for a source without pixels. A client that reads too
    slowly loses old events. The pipeline never waits for a client.
    """

    def __init__(
        self,
        results: Callable[[], Iterator[FrameResult]],
        tap: FrameTap | None = None,
        *,
        queue_size: int = 32,
        on_result: Callable[[FrameResult], None] | None = None,
    ) -> None:
        self._results = results
        self._tap = tap
        self._queue_size = queue_size
        self._on_result = on_result
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._subscribers: list[_Subscriber] = []
        self._viewers = 0
        self._jpeg: bytes | None = None
        self._jpeg_ready = threading.Condition()
        self._state = "idle"
        self._error: str | None = None
        self._frames = 0
        self._detector_runs = 0
        self._ids: set[int] = set()
        self._dropped = 0
        self._latency: deque[float] = deque(maxlen=1000)
        self._times: deque[float] = deque(maxlen=120)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the service was already started")
        self._state = "running"
        self._thread = threading.Thread(target=self._run, name="tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            for result in self._results():
                if self._stop.is_set():
                    break
                self._publish(result)
            self._state = "finished"
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._state = "failed"
        finally:
            with self._jpeg_ready:
                self._jpeg_ready.notify_all()

    def _publish(self, result: FrameResult) -> None:
        with self._lock:
            self._frames += 1
            self._detector_runs += result.detected
            self._ids.update(o.track_id for o in result.objects)
            self._latency.append(result.total_ms)
            self._times.append(time.perf_counter())
            subscribers = list(self._subscribers)
            viewers = self._viewers
        if self._on_result is not None:
            self._on_result(result)
        if subscribers:
            item = event(result)
            for subscriber in subscribers:
                subscriber.loop.call_soon_threadsafe(subscriber.offer, item)
        # The image work is necessary only while a client reads the video.
        frame = self._tap.current if self._tap is not None else None
        if viewers and frame is not None and frame.image is not None:
            jpeg = render(frame.image, result)
            with self._jpeg_ready:
                self._jpeg = jpeg
                self._jpeg_ready.notify_all()

    def subscribe(self) -> _Subscriber:
        subscriber = _Subscriber(asyncio.get_running_loop(), self._queue_size)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        with self._lock:
            self._subscribers.remove(subscriber)
            self._dropped += subscriber.dropped

    @property
    def running(self) -> bool:
        return self._state == "running"

    @property
    def has_video(self) -> bool:
        return self._tap is not None

    def video_frames(self) -> Iterator[bytes]:
        """JPEG images with the tracks drawn, one for each new frame. Blocks between frames."""
        with self._lock:
            self._viewers += 1
        try:
            last: bytes | None = None
            while True:
                with self._jpeg_ready:
                    self._jpeg_ready.wait_for(
                        lambda seen=last: self._jpeg is not seen or not self.running,  # type: ignore[misc]
                        timeout=1.0,
                    )
                    jpeg = self._jpeg
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield jpeg
                elif not self.running:
                    return
        finally:
            with self._lock:
                self._viewers -= 1

    def stats(self) -> Stats:
        with self._lock:
            latency = np.array(self._latency) if self._latency else np.zeros(1)
            times = list(self._times)
            dropped = self._dropped + sum(s.dropped for s in self._subscribers)
            span = times[-1] - times[0] if len(times) > 1 else 0.0
            return Stats(
                state=self._state,
                frames=self._frames,
                detector_runs=self._detector_runs,
                tracks=len(self._ids),
                frames_per_second=(len(times) - 1) / span if span > 0 else 0.0,
                latency_ms_p50=float(np.percentile(latency, 50)),
                latency_ms_p99=float(np.percentile(latency, 99)),
                dropped_events=dropped,
                error=self._error,
            )


def render(image: Image, result: FrameResult) -> bytes:
    """A JPEG of the frame with a box and an id for each track."""
    scale = min(1.0, _VIDEO_WIDTH / image.shape[1])
    canvas = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image.copy()
    for obj in result.objects:
        x1, y1, x2, y2 = (round(v * scale) for v in obj.box)
        color = _COLORS[obj.track_id % len(_COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            str(obj.track_id),
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    label = f"frame {result.frame_index}  {'DETECT' if result.detected else 'track'}"
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("cannot encode the frame")
    return encoded.tobytes()
