from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from leantrack._types import Frame, TrackedObject
from leantrack.cli import main
from leantrack.pipeline import FrameResult
from leantrack.serve.app import Metrics, create_app
from leantrack.serve.service import FrameTap, TrackingService, _Subscriber, event, render


def _result(index: int, detected: bool = True) -> FrameResult:
    objects = [TrackedObject(1, (10.0, 20.0, 50.0, 100.0), 0.9, 0)]
    if index > 2:
        objects.append(TrackedObject(2, (60.0, 20.0, 90.0, 100.0), 0.8, 0))
    return FrameResult(
        index, objects, detected, detect_ms=4.0, propagate_ms=1.0, embed_ms=0.0, track_ms=0.5
    )


def _results(count: int, delay: float = 0.0, tap: FrameTap | None = None) -> Iterator[FrameResult]:
    frames = (Frame(i, np.full((120, 160, 3), 40 * i % 255, np.uint8)) for i in range(1, count + 1))
    for frame in tap.watch(frames) if tap else frames:
        time.sleep(delay)
        yield _result(frame.index, detected=frame.index % 2 == 1)


def test_event_format() -> None:
    assert event(_result(3)) == {
        "frame": 3,
        "detected": True,
        "detection_age": 0,
        "latency_ms": 5.5,
        "objects": [
            {"id": 1, "box": [10.0, 20.0, 50.0, 100.0], "score": 0.9, "class": 0},
            {"id": 2, "box": [60.0, 20.0, 90.0, 100.0], "score": 0.8, "class": 0},
        ],
    }


def test_service_counts_results_and_finishes() -> None:
    service = TrackingService(lambda: _results(10))
    assert service.stats().state == "idle"
    service.start()
    service.wait(5.0)
    stats = service.stats()
    assert (stats.state, stats.frames, stats.detector_runs, stats.tracks) == ("finished", 10, 5, 2)
    assert stats.latency_ms_p50 == pytest.approx(5.5)
    assert stats.error is None
    with pytest.raises(RuntimeError):
        service.start()


def test_service_reports_a_pipeline_failure() -> None:
    def broken() -> Iterator[FrameResult]:
        yield _result(1)
        raise OSError("camera lost")

    service = TrackingService(broken)
    with TestClient(create_app(service)) as client:
        service.wait(5.0)
        assert service.stats().error == "OSError: camera lost"
        response = client.get("/health")
        assert response.status_code == 503
        assert "camera lost" in response.json()["detail"]


def test_a_slow_client_loses_its_oldest_events() -> None:
    async def scenario() -> tuple[list[int], int]:
        subscriber = _Subscriber(asyncio.get_running_loop(), size=3)
        for index in range(1, 6):
            subscriber.offer({"frame": index})
        frames = [subscriber.queue.get_nowait()["frame"] for _ in range(3)]
        return frames, subscriber.dropped

    assert asyncio.run(scenario()) == ([3, 4, 5], 2)


def test_http_endpoints() -> None:
    metrics = Metrics()
    service = TrackingService(lambda: _results(8), on_result=metrics.observe)
    with TestClient(create_app(service, metrics)) as client:
        service.wait(5.0)
        assert client.get("/health").json() == {"status": "finished"}
        assert "<title>leantrack</title>" in client.get("/").text

        stats = client.get("/stats").json()
        assert stats["frames"] == 8
        assert stats["tracks"] == 2

        text = client.get("/metrics").text
        assert "leantrack_frames_total 8.0" in text
        assert "leantrack_detector_runs_total 4.0" in text
        assert 'leantrack_frame_latency_seconds_bucket{le="0.01"} 8.0' in text
        assert "leantrack_reported_objects 2.0" in text

        # This source has no tap, so it has no video.
        assert client.get("/video").status_code == 404


def test_metrics_endpoint_is_absent_without_metrics() -> None:
    with TestClient(create_app(TrackingService(lambda: _results(1)))) as client:
        assert client.get("/metrics").status_code == 404


def test_websocket_gives_the_events_in_order() -> None:
    service = TrackingService(lambda: _results(200, delay=0.005), queue_size=500)
    with TestClient(create_app(service)) as client, client.websocket_connect("/ws") as socket:
        frames = [socket.receive_json()["frame"] for _ in range(5)]
    assert frames == sorted(frames)
    assert frames[1] == frames[0] + 1


def test_video_stream_gives_jpeg_images() -> None:
    tap = FrameTap()
    service = TrackingService(lambda: _results(60, delay=0.01, tap=tap), tap)
    with TestClient(create_app(service)) as client, client.stream("GET", "/video") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
        body = b"".join(response.iter_bytes())
    assert body.count(b"--frame\r\nContent-Type: image/jpeg") >= 2
    assert b"\xff\xd8" in body


def test_render_makes_a_jpeg() -> None:
    jpeg = render(np.zeros((1080, 1920, 3), np.uint8), _result(3))
    assert jpeg[:2] == b"\xff\xd8"


def test_cli_serve_checks_its_arguments_before_the_server_starts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["serve", str(tmp_path)]) == 2
    assert "--model is necessary" in capsys.readouterr().err
    assert main(["serve", str(tmp_path / "missing.mp4"), "--model", "m.onnx"]) == 2
    assert "not found" in capsys.readouterr().err
