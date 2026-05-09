"""Abstract base class for voice API bridges (OpenAI Realtime / Grok Voice).

Concrete subclasses implement provider-specific session config, auth,
URL, transcript parsing, and cost computation. The transport layer
(WebSocket pump, mic loop, tool dispatch, barge-in) lives here.
"""

from __future__ import annotations

import abc
import json
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import websocket
from scipy.signal import resample_poly

from .._log import log
from ..emotions import EmotionPlayer
from ..tools import LOOK_POSES, _make_head_pose, build_tools
from ..vision import CameraWorker, VisionBackend, build_vision_backend

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


# Audio rate used over the realtime websockets (PCM16 mono).
REALTIME_RATE = 24_000


class VoiceBridge(abc.ABC):
    """Abstract base class for voice API WebSocket bridges.
    
    Subclasses implement provider-specific logic (OpenAI vs xAI/Grok).
    """
    
    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        self.media = mini.media
        self.in_rate = self.media.get_input_audio_samplerate()
        self.out_rate = self.media.get_output_audio_samplerate()

        self._ws = None  # type: websocket.WebSocketApp | None
        self._send_q = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._sender_thread = None  # type: threading.Thread | None
        self._mic_thread = None  # type: threading.Thread | None
        self._move_thread = None  # type: threading.Thread | None

        # Response state — the robot has no TTS, so we only track
        # whether a response is in flight (to allow barge-in cancel)
        # and dedup the response.done event.
        self._response_active = False
        self._last_done_response_id = None
        self._needs_followup_response = False

        # Cost tracking
        self._cost_total = 0.0
        self._turns = 0

        # Single EmotionPlayer reused across the session: preloads
        # every WAV once at startup so each play_emotion call is just a
        # dict lookup + push_audio_sample.
        from reachy_mini.motion.recorded_move import RecordedMoves
        self._emotions = EmotionPlayer(
            self.mini, RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        )

        # Optional vision backend (Moondream cloud / FastVLM HTTP / off).
        # The camera worker is only started when a backend is configured
        # to avoid wasting CPU/bandwidth on the Pi when vision is unused.
        self._vision: VisionBackend | None = None
        self._camera: CameraWorker | None = None
        try:
            self._vision = build_vision_backend()
        except Exception as e:
            log(f"[vision] backend init failed, vision disabled: {e}")
            self._vision = None
        if self._vision is not None:
            self._camera = CameraWorker(self.mini, fps=5.0)

        # Build the tools list with the play_emotion enum populated
        # from the library actually loaded — keeps the API in sync with
        # the dataset (no more hardcoded list to maintain by hand).
        vision_enabled = self._vision is not None
        vision_grounding = bool(self._vision and self._vision.supports_grounding)
        self.tools = build_tools(
            sorted(self._emotions.library.list_moves()),
            vision_enabled=vision_enabled,
            vision_grounding=vision_grounding,
        )
    
    # ---- Abstract methods (provider-specific) ----
    @abc.abstractmethod
    def get_ws_url(self) -> str:
        """Return WebSocket URL for the provider."""
        pass
    
    @abc.abstractmethod
    def get_auth_header(self) -> str:
        """Return authorization header for the provider."""
        pass
    
    @abc.abstractmethod
    def get_session_config(self) -> dict:
        """Return session configuration for the provider."""
        pass
    
    @abc.abstractmethod
    def handle_transcript_delta(self, evt: dict) -> None:
        """Handle transcript delta from provider."""
        pass
    
    @abc.abstractmethod
    def handle_input_transcription(self, evt: dict) -> None:
        """Handle input audio transcription completion."""
        pass
    
    @abc.abstractmethod
    def compute_cost(self, usage: dict) -> tuple[float, dict]:
        """Compute cost from usage block. Returns (cost_usd, breakdown)."""
        pass
    
    @abc.abstractmethod
    def _print_config(self) -> None:
        """Print configuration banner."""
        pass
    
    # ---- Common helpers ----
    def _send(self, payload: dict) -> None:
        try:
            self._send_q.put_nowait(json.dumps(payload))
        except queue.Full:
            pass
    
    def _reset_response_state(self) -> None:
        self._response_active = False

    # ---- Audio helpers ----
    @staticmethod
    def f32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype("<i2").tobytes()

    @staticmethod
    def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return samples
        g = np.gcd(src_rate, dst_rate)
        return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)

    # ---- barge-in ----
    def _barge_in(self) -> None:
        """User started speaking → cancel any in-flight response.

        With TTS off, there is no audio to truncate; we just cancel so
        the model stops generating tool calls / text for the previous
        turn and listens to the new utterance.
        """
        if not self._response_active:
            return
        self._send({"type": "response.cancel"})
        log("[barge-in] canceled in-flight response")

    # ---- WebSocket lifecycle ----
    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        log("[ws] connected")
        ws.send(json.dumps({"type": "session.update", "session": self.get_session_config()}))
        self.media.start_playing()
        self.media.start_recording()
        if self._camera is not None:
            self._camera.start()
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._mic_thread.start()
    
    def _mic_loop(self) -> None:
        """Stream microphone audio to the provider."""
        import base64
        
        while not self._stop.is_set():
            samples = self.media.get_audio_sample()
            if samples is None or len(samples) == 0:
                time.sleep(0.01)
                continue
            mono = samples if samples.ndim == 1 else samples.mean(axis=1)
            resampled = self.resample(mono.astype(np.float32), self.in_rate, REALTIME_RATE)
            b64 = base64.b64encode(self.f32_to_pcm16_bytes(resampled)).decode("ascii")
            self._send({"type": "input_audio_buffer.append", "audio": b64})
    
    def _sender_loop(self) -> None:
        """Send queued messages to WebSocket."""
        while not self._stop.is_set():
            try:
                msg = self._send_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._ws.send(msg)  # type: ignore[union-attr]
            except Exception as e:
                log(f"[ws] send failed: {e}")
                return

    def _on_error(self, ws: websocket.WebSocketApp, err: Exception) -> None:
        log(f"[ws] error: {err}")

    def _on_close(self, ws, code, msg) -> None:
        log(f"[ws] closed code={code} msg={msg}")
        log(f"[cost] session total: ${self._cost_total:.4f} over {self._turns} turn(s)")
        self._stop.set()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as e:
                log(f"[camera] stop failed: {e}")
        if self._vision is not None:
            try:
                self._vision.close()
            except Exception:
                pass
    
    # ---- Main message handler ----
    def _on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        evt = json.loads(raw)
        t = evt.get("type", "")

        # TTS audio deltas are silently dropped: the robot has no voice.
        # If a provider ignores our text-only request and ships audio,
        # we just discard it.
        if t in ("response.audio.delta", "response.output_audio.delta",
                 "response.audio_transcript.delta", "response.output_audio_transcript.delta",
                 "response.audio_transcript.done", "response.output_audio_transcript.done"):
            return

        # Speech detection
        if t == "input_audio_buffer.speech_started":
            self._barge_in()
        elif t == "input_audio_buffer.speech_stopped":
            pass

        # Response lifecycle
        elif t == "response.created":
            self._reset_response_state()
            self._response_active = True

        # Text output (model's internal monologue — printed for debug only)
        elif t == "response.output_text.delta":
            self.handle_transcript_delta(evt)

        elif t == "conversation.item.input_audio_transcription.completed":
            self.handle_input_transcription(evt)

        # Tool calls
        elif t == "response.output_item.done":
            item = evt.get("item") or {}
            if item.get("type") == "function_call":
                self._handle_tool_call({
                    "name": item.get("name"),
                    "call_id": item.get("call_id"),
                    "arguments": item.get("arguments") or "{}",
                })

        # Response done
        elif t == "response.done":
            self._handle_response_done(evt)

        # Errors
        elif t == "error":
            err = evt.get("error") or {}
            code = err.get("code")
            if code not in ("response_cancel_not_active", "conversation_already_has_active_response"):
                log(f"[ws] error: {err}")
    
    def _handle_response_done(self, evt: dict) -> None:
        """Handle response.done event."""
        rsp = evt.get("response") or {}
        rsp_id = rsp.get("id")
        if rsp_id and rsp_id == self._last_done_response_id:
            return
        self._last_done_response_id = rsp_id

        # OpenAI nests `usage` inside `response`; xAI puts it at the
        # event top level (with an empty `response.usage: {}`).
        usage = evt.get("usage") or rsp.get("usage") or {}
        cost, br = self.compute_cost(usage)
        self._cost_total += cost
        self._turns += 1
        log(
            f"[bot] response.done cost=${cost:.4f} cumul=${self._cost_total:.4f} "
            f"in=txt:{br['text_in']} cached:{br['text_in_cached']} "
            f"audio:{br['audio_in']} audio_cached:{br['audio_in_cached']} "
            f"out=txt:{br['text_out']} audio:{br['audio_out']} "
            f"(turn #{self._turns})"
        )

        needs_followup = self._needs_followup_response
        self._needs_followup_response = False
        self._reset_response_state()

        if needs_followup:
            self._send({"type": "response.create"})
    
    # ---- Tool handling (shared) ----
    def _run_move_async(self, fn) -> None:
        """Run a movement off the WS thread."""
        prev = self._move_thread
        
        def _wrapper() -> None:
            if prev is not None and prev.is_alive():
                prev.join(timeout=10.0)
            t0 = time.monotonic()
            try:
                fn()
            except Exception as e:
                log(f"[move] failed: {e}")
                return
            log(f"[move] done in {time.monotonic() - t0:.2f}s")
        
        t = threading.Thread(target=_wrapper, daemon=True)
        self._move_thread = t
        t.start()
    
    def _handle_tool_call(self, evt: dict) -> None:
        """Handle function call from the provider."""
        name = evt.get("name")
        call_id = evt.get("call_id")
        args_raw = evt.get("arguments") or "{}"
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
        log(f"[tool] {name}({args})")
        
        if name == "play_emotion":
            emo = args.get("name", "")
            played = self._emotions.play(emo, min_interval=0.0)
            result = f"played:{emo}" if played else f"skipped:{emo}"
        
        elif name == "look":
            direction = args.get("direction", "center")
            pose = LOOK_POSES.get(direction)
            if pose is None:
                result = f"unknown_direction:{direction}"
            else:
                self._run_move_async(lambda: self.mini.goto_target(pose, duration=0.6))
                result = f"looking:{direction}"
        
        elif name == "move_sequence":
            steps = args.get("steps") or []
            if not steps:
                result = "empty_sequence"
            else:
                self._run_move_async(lambda: self._play_sequence(steps))
                result = f"playing:{len(steps)}_steps"

        elif name == "look_and_describe":
            result = self._handle_look_and_describe(args.get("question", ""))

        elif name == "find_object":
            result = self._handle_find_object(args.get("target", ""))

        else:
            result = f"unknown_tool:{name}"
        
        self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"status": result}),
            },
        })
        self._needs_followup_response = True
    
    # ---- Vision tool handlers ----
    def _handle_look_and_describe(self, question: str) -> str:
        if self._vision is None or self._camera is None:
            return "vision_not_enabled"
        question = (question or "").strip()
        if not question:
            return "empty_question"
        frame = self._camera.get_latest()
        if frame is None:
            return "no_frame_available"
        try:
            res = self._vision.query(frame, question)
        except Exception as e:
            log(f"[vision] query failed: {e}")
            return f"error:{e}"
        return res.text or "no_answer"

    def _handle_find_object(self, target: str) -> str:
        if self._vision is None or self._camera is None:
            return "vision_not_enabled"
        if not self._vision.supports_grounding:
            return "grounding_not_supported"
        target = (target or "").strip()
        if not target:
            return "empty_target"
        frame = self._camera.get_latest()
        if frame is None:
            return "no_frame_available"

        # Prefer point() (single coord) over detect() — closer to what
        # we actually need (where to aim the head).
        try:
            res = self._vision.point(frame, target)
        except Exception as e:
            log(f"[vision] point failed: {e}")
            return f"error:{e}"

        if not res.points:
            # Fallback: try detect() and use bbox center.
            try:
                det = self._vision.detect(frame, target)
            except Exception as e:
                log(f"[vision] detect failed: {e}")
                return f"not_found:{target}"
            if not det.boxes:
                return f"not_found:{target}"
            cx, cy = det.boxes[0].center
        else:
            cx, cy = res.points[0].x, res.points[0].y

        # Normalized [0,1] → pixels for look_at_image.
        h, w, _ = frame.shape
        px, py = cx * w, cy * h

        def _aim() -> None:
            try:
                pose = self.mini.look_at_image(px, py, duration=0.0, perform_movement=False)
                self.mini.goto_target(pose, duration=0.6)
            except Exception as e:
                log(f"[vision] look_at_image failed: {e}")

        self._run_move_async(_aim)
        return f"found:{target}@({cx:.2f},{cy:.2f})"

    def _play_sequence(self, steps: list) -> None:
        """Execute a planned head choreography."""
        import numpy as np
        from reachy_mini.reachy_mini import INIT_HEAD_POSE, INIT_ANTENNAS_JOINT_POSITIONS
        
        for step in steps:
            try:
                roll = max(-30.0, min(30.0, float(step.get("roll", 0.0))))
                pitch = max(-30.0, min(30.0, float(step.get("pitch", 0.0))))
                yaw = max(-60.0, min(60.0, float(step.get("yaw", 0.0))))
                x_mm = max(-30.0, min(30.0, float(step.get("x", 0.0))))
                y_mm = max(-30.0, min(30.0, float(step.get("y", 0.0))))
                z_mm = max(-30.0, min(30.0, float(step.get("z", 0.0))))
                duration = max(0.1, min(3.0, float(step.get("duration", 0.4))))
                pose = _make_head_pose(roll, pitch, yaw, x_mm, y_mm, z_mm)
                
                antennas = None
                al = step.get("antenna_left")
                ar = step.get("antenna_right")
                if al is not None or ar is not None:
                    al_rad = np.deg2rad(max(-90.0, min(90.0, float(al or 0.0))))
                    ar_rad = np.deg2rad(max(-90.0, min(90.0, float(ar or 0.0))))
                    antennas = [al_rad, ar_rad]
                
                self.mini.goto_target(pose, antennas=antennas, duration=duration)
            except Exception as e:
                log(f"[seq] step failed: {e}")
                return

        # back to neutral
        try:
            self.mini.goto_target(
                INIT_HEAD_POSE,
                antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                duration=0.5,
            )
        except Exception as e:
            log(f"[seq] return to neutral failed: {e}")
    
    # ---- Run method ----
    def run(self) -> None:
        """Main run loop."""
        import signal
        
        self._print_config()
        
        headers = [self.get_auth_header()]
        self._ws = websocket.WebSocketApp(
            self.get_ws_url(),
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()
        
        def _sigint(_sig, _frm):
            self._stop.set()
            try:
                self._ws.close()
            except Exception:
                pass
        
        signal.signal(signal.SIGINT, _sigint)
        
        try:
            self._ws.run_forever()
        finally:
            try:
                self.media.stop_recording()
            except Exception:
                pass
            try:
                self.media.stop_playing()
            except Exception:
                pass
