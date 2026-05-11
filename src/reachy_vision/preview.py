"""MJPEG debug preview with caption + detection overlays."""

from __future__ import annotations

import asyncio
import io
import textwrap
import threading
import time
from dataclasses import dataclass

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image, ImageDraw

from reachy_voice._log import log

from .camera import Camera


_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Reachy vision preview</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:1rem;text-align:center}
img{max-width:100%;height:auto;border:1px solid #333}
footer{margin-top:1rem;font-size:0.85rem;color:#888}</style>
</head><body>
<h1>Reachy vision preview</h1>
<img src="/stream.mjpeg" alt="live">
<footer>Set <code>VISION_PREVIEW=0</code> to disable.</footer>
</body></html>"""


@dataclass
class _Overlay:
    timestamp: float
    payload: dict


class Preview:
    def __init__(
        self,
        camera: Camera,
        host: str = "0.0.0.0",
        port: int = 5050,
        overlay_ttl_s: float = 5.0,
        stream_fps: float = 5.0,
        jpeg_quality: int = 80,
    ) -> None:
        self.camera = camera
        self.host = host
        self.port = port
        self.overlay_ttl_s = overlay_ttl_s
        self.stream_fps = stream_fps
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._last_caption: _Overlay | None = None
        self._last_detection: _Overlay | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        app = self._build_app()
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._server.serve()),
            daemon=True,
        )
        self._thread.start()
        log(f"[preview] started on :{self.port}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._server = None
        log("[preview] stopped")

    def register_caption(self, question: str, answer: str) -> None:
        with self._lock:
            self._last_caption = _Overlay(
                time.monotonic(), {"question": question, "answer": answer}
            )

    def register_detection(self, target: str, points: list[tuple[float, float]]) -> None:
        with self._lock:
            self._last_detection = _Overlay(
                time.monotonic(), {"target": target, "points": points}
            )

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return _INDEX_HTML

        @app.get("/stream.mjpeg")
        def stream() -> StreamingResponse:
            return StreamingResponse(
                self._mjpeg(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.get("/state.json")
        def state() -> dict:
            with self._lock:
                caption = self._last_caption.payload if self._last_caption else None
                detection = self._last_detection.payload if self._last_detection else None
            return {
                "camera_running": self.camera.is_running(),
                "last_caption": caption,
                "last_detection": detection,
            }

        return app

    def _mjpeg(self):
        period = 1.0 / max(self.stream_fps, 0.1)
        while True:
            jpeg = self._compose_jpeg()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(period)

    def _compose_jpeg(self) -> bytes:
        frame = self.camera.get_latest()
        if frame is None:
            image = Image.new("RGB", (640, 480), (32, 32, 32))
            draw = ImageDraw.Draw(image)
            draw.text((20, 220), "no camera frame yet", fill=(220, 220, 220))
        else:
            image = Image.fromarray(np.ascontiguousarray(frame[..., ::-1]))
            self._draw_overlays(image)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    def _draw_overlays(self, image: Image.Image) -> None:
        now = time.monotonic()
        with self._lock:
            caption = (
                self._last_caption
                if self._last_caption and now - self._last_caption.timestamp < self.overlay_ttl_s
                else None
            )
            detection = (
                self._last_detection
                if self._last_detection and now - self._last_detection.timestamp < self.overlay_ttl_s
                else None
            )
        draw = ImageDraw.Draw(image, "RGBA")
        w, h = image.size
        if detection is not None:
            for cx_norm, cy_norm in detection.payload["points"]:
                cx, cy = int(cx_norm * w), int(cy_norm * h)
                draw.line([(cx - 20, cy), (cx + 20, cy)], fill=(255, 80, 80, 255), width=2)
                draw.line([(cx, cy - 20), (cx, cy + 20)], fill=(255, 80, 80, 255), width=2)
                draw.text(
                    (cx + 24, cy - 8),
                    f"-> {detection.payload['target']}",
                    fill=(255, 200, 200, 255),
                )
        if caption is not None:
            q = caption.payload["question"]
            a = caption.payload["answer"]
            lines = textwrap.wrap(f"Q: {q}", width=70) + textwrap.wrap(f"A: {a}", width=70)
            line_h = 16
            band_h = line_h * len(lines) + 12
            draw.rectangle([(0, h - band_h), (w, h)], fill=(0, 0, 0, 180))
            for i, line in enumerate(lines):
                draw.text(
                    (10, h - band_h + 6 + i * line_h),
                    line,
                    fill=(255, 255, 255, 255),
                )
