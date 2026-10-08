from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from leantrack.pipeline import FrameResult
from leantrack.serve.service import TrackingService

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>leantrack</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }
  img { max-width: 100%; border: 1px solid #333; }
  pre { background: #1b1b1b; padding: 8px; overflow-x: auto; }
</style>
</head>
<body>
<h1>leantrack</h1>
<img src="video" alt="Video with tracks">
<pre id="stats">waiting for data</pre>
<script>
  const out = document.getElementById("stats");
  async function refresh() {
    try {
      const response = await fetch("stats");
      out.textContent = JSON.stringify(await response.json(), null, 2);
    } catch (error) {
      out.textContent = "no connection to the service";
    }
  }
  refresh();
  setInterval(refresh, 1000);
</script>
</body>
</html>
"""

_LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.033, 0.05, 0.1, 0.2, 0.5)


class Metrics:
    """Prometheus metrics of one service. Each instance has its own registry."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.frames = Counter(
            "leantrack_frames_total", "Frames that the pipeline processed.", registry=self.registry
        )
        self.detector_runs = Counter(
            "leantrack_detector_runs_total",
            "Frames that used a detector result.",
            registry=self.registry,
        )
        self.latency = Histogram(
            "leantrack_frame_latency_seconds",
            "Time from the arrival of a frame to its result.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.detection_age = Histogram(
            "leantrack_detection_age_frames",
            "Frames between the image of a detector result and the frame that used it.",
            buckets=(0, 1, 2, 3, 5, 8, 13),
            registry=self.registry,
        )
        self.objects = Gauge(
            "leantrack_reported_objects",
            "Objects in the newest frame result.",
            registry=self.registry,
        )

    def observe(self, result: FrameResult) -> None:
        self.frames.inc()
        self.latency.observe(result.total_ms / 1000.0)
        self.objects.set(len(result.objects))
        if result.detected:
            self.detector_runs.inc()
            self.detection_age.observe(result.detection_age)


def create_app(service: TrackingService, metrics: Metrics | None = None) -> FastAPI:
    """The HTTP interface of a tracking service. The service starts with the application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service.start()
        yield
        service.stop()

    app = FastAPI(title="leantrack", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/health")
    def health() -> dict[str, str]:
        stats = service.stats()
        if stats.state == "failed":
            raise HTTPException(status_code=503, detail=stats.error)
        return {"status": stats.state}

    @app.get("/stats")
    def stats() -> dict[str, object]:
        return asdict(service.stats())

    @app.get("/metrics")
    def prometheus() -> Response:
        if metrics is None:
            raise HTTPException(status_code=404, detail="metrics are not enabled")
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/video")
    def video() -> StreamingResponse:
        if not service.has_video:
            raise HTTPException(status_code=404, detail="the source has no pixels")

        def parts() -> Iterator[bytes]:
            for jpeg in service.video_frames():
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return StreamingResponse(parts(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def events(socket: WebSocket) -> None:
        await socket.accept()
        subscriber = service.subscribe()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(subscriber.queue.get(), timeout=0.5)
                except TimeoutError:
                    if not service.running and subscriber.queue.empty():
                        break
                    continue
                await socket.send_json(item)
        except WebSocketDisconnect:
            pass
        finally:
            service.unsubscribe(subscriber)
            with suppress(RuntimeError):
                await socket.close()

    return app
