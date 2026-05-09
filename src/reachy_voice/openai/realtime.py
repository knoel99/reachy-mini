"""OpenAI Realtime WebSocket bridge (text-only output, tool dispatch only)."""

from __future__ import annotations

import base64
import json
import os
import queue
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import websocket
from scipy.signal import resample_poly

from .._actions import RobotActions
from .._log import log
from ..tools import INSTRUCTIONS

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


# Audio rate over the OpenAI realtime websocket (PCM16 mono).
REALTIME_RATE = 24_000

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PRICING = {
    "gpt-realtime-mini": {
        "text_input":         0.60 / 1_000_000,
        "text_input_cached":  0.06 / 1_000_000,
        "text_output":        2.40 / 1_000_000,
        "audio_input":       10.00 / 1_000_000,
        "audio_input_cached": 0.30 / 1_000_000,
        "audio_output":      20.00 / 1_000_000,
    },
    "gpt-realtime": {
        "text_input":         4.00 / 1_000_000,
        "text_input_cached":  0.40 / 1_000_000,
        "text_output":       16.00 / 1_000_000,
        "audio_input":       32.00 / 1_000_000,
        "audio_input_cached": 0.40 / 1_000_000,
        "audio_output":      64.00 / 1_000_000,
    },
    "gpt-realtime-2": {
        "text_input":         4.00 / 1_000_000,
        "text_input_cached":  0.40 / 1_000_000,
        "text_output":       24.00 / 1_000_000,
        "audio_input":       32.00 / 1_000_000,
        "audio_input_cached": 0.40 / 1_000_000,
        "audio_output":      64.00 / 1_000_000,
    },
}

REASONING_MODELS = {"gpt-realtime-2"}
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "medium")

# Drop mic samples for this long after WS open. The audio pipeline
# usually has a short burst of buffered noise around the connect
# moment, which VAD would otherwise commit as a phantom user turn.
_MIC_GRACE_S = 1.2


class OpenAIRealtimeBridge:
    """Bidirectional realtime bridge: mic → OpenAI Realtime → tool calls."""

    def __init__(self, mini: ReachyMini, model: str = "gpt-realtime-mini") -> None:
        if model not in PRICING:
            raise ValueError(
                f"Unsupported OpenAI model: {model}. Available: {list(PRICING)}"
            )
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model
        self.mini = mini
        self.media = mini.media
        self.in_rate = self.media.get_input_audio_samplerate()

        self.actions = RobotActions(mini)

        self._ws: websocket.WebSocketApp | None = None
        self._send_q: queue.Queue[str] = queue.Queue(maxsize=100)
        self._stop = threading.Event()

        self._response_active = False
        self._last_done_response_id: str | None = None
        self._mic_open_at: float = 0.0

        self._cost_total = 0.0
        self._turns = 0

    # ---- helpers ----
    def _send(self, payload: dict) -> None:
        try:
            self._send_q.put_nowait(json.dumps(payload))
        except queue.Full:
            pass

    @staticmethod
    def _f32_to_pcm16(samples: np.ndarray) -> bytes:
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype("<i2").tobytes()

    @staticmethod
    def _resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
        if src == dst:
            return samples
        g = np.gcd(src, dst)
        return resample_poly(samples, dst // g, src // g).astype(np.float32)

    # ---- session config ----
    def _session_config(self) -> dict:
        session = {
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": INSTRUCTIONS,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                    "transcription": {
                        "model": "gpt-4o-mini-transcribe",
                        "language": "fr",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.6,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                    },
                },
            },
            "tools": self.actions.tools,
            "tool_choice": "auto",
        }
        if self.model in REASONING_MODELS:
            session["reasoning"] = {"effort": REASONING_EFFORT}
        return session

    def _print_config(self) -> None:
        p = PRICING[self.model]
        reasoning_str = (
            f" reasoning.effort={REASONING_EFFORT}"
            if self.model in REASONING_MODELS else " (no reasoning support)"
        )
        log(
            f"[config] provider=openai model={self.model}{reasoning_str} text-only output  "
            f"prices /1M tok: text in ${p['text_input']*1e6:.2f} "
            f"(cached ${p['text_input_cached']*1e6:.2f}) "
            f"text out ${p['text_output']*1e6:.2f}  "
            f"audio in ${p['audio_input']*1e6:.2f} "
            f"(cached ${p['audio_input_cached']*1e6:.2f})"
        )

    # ---- cost ----
    def _compute_cost(self, usage: dict) -> tuple[float, dict]:
        p = PRICING[self.model]
        inp = usage.get("input_token_details") or {}
        out = usage.get("output_token_details") or {}
        cached = inp.get("cached_tokens_details") or {}

        text_in_total = inp.get("text_tokens", 0)
        audio_in_total = inp.get("audio_tokens", 0)
        text_in_cached = cached.get("text_tokens", 0)
        audio_in_cached = cached.get("audio_tokens", 0)
        text_in = max(0, text_in_total - text_in_cached)
        audio_in = max(0, audio_in_total - audio_in_cached)

        text_out = out.get("text_tokens", 0)
        audio_out = out.get("audio_tokens", 0)

        cost = (
            text_in        * p["text_input"]
          + text_in_cached * p["text_input_cached"]
          + audio_in       * p["audio_input"]
          + audio_in_cached * p["audio_input_cached"]
          + text_out       * p["text_output"]
          + audio_out      * p["audio_output"]
        )
        return cost, {
            "text_in": text_in, "text_in_cached": text_in_cached,
            "audio_in": audio_in, "audio_in_cached": audio_in_cached,
            "text_out": text_out, "audio_out": audio_out,
        }

    # ---- WebSocket lifecycle ----
    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        log("[ws] connected")
        ws.send(json.dumps({"type": "session.update", "session": self._session_config()}))
        self.media.start_playing()
        self.media.start_recording()
        self._mic_open_at = time.monotonic()
        threading.Thread(target=self._mic_loop, daemon=True).start()

    def _mic_loop(self) -> None:
        grace_cleared = False
        while not self._stop.is_set():
            # Discard buffered audio captured before/during the
            # connect handshake; otherwise VAD commits a phantom turn
            # at startup and the model "responds" before the user has
            # said anything.
            if time.monotonic() - self._mic_open_at < _MIC_GRACE_S:
                self.media.get_audio_sample()
                time.sleep(0.01)
                continue
            if not grace_cleared:
                self._send({"type": "input_audio_buffer.clear"})
                grace_cleared = True
            samples = self.media.get_audio_sample()
            if samples is None or len(samples) == 0:
                time.sleep(0.01)
                continue
            # Drop frames while the robot's own audio is playing — the
            # speaker can otherwise loop back as user input.
            if self.actions.is_speaking():
                continue
            mono = samples if samples.ndim == 1 else samples.mean(axis=1)
            resampled = self._resample(
                mono.astype(np.float32), self.in_rate, REALTIME_RATE
            )
            b64 = base64.b64encode(self._f32_to_pcm16(resampled)).decode("ascii")
            self._send({"type": "input_audio_buffer.append", "audio": b64})

    def _sender_loop(self) -> None:
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

    def _on_error(self, ws, err: Exception) -> None:
        log(f"[ws] error: {err}")

    def _on_close(self, ws, code, msg) -> None:
        log(f"[ws] closed code={code} msg={msg}")
        log(f"[cost] session total: ${self._cost_total:.4f} over {self._turns} turn(s)")
        self._stop.set()

    # ---- barge-in ----
    def _barge_in(self) -> None:
        if not self._response_active:
            return
        self._send({"type": "response.cancel"})
        log("[barge-in] canceled in-flight response")

    # ---- message handler ----
    def _on_message(self, ws, raw: str) -> None:
        evt = json.loads(raw)
        t = evt.get("type", "")

        # Drop TTS audio: we never request it in text-only mode.
        if t in ("response.audio.delta", "response.output_audio.delta",
                 "response.audio_transcript.delta", "response.output_audio_transcript.delta",
                 "response.audio_transcript.done", "response.output_audio_transcript.done"):
            return

        if t == "input_audio_buffer.speech_started":
            self._barge_in()
        elif t == "response.created":
            self._response_active = True
        elif t == "response.output_text.delta":
            sys.stdout.write(evt.get("delta", ""))
            sys.stdout.flush()
        elif t == "conversation.item.input_audio_transcription.completed":
            log(f"[user] {evt.get('transcript', '').strip()}", lead="\n")
        elif t == "response.output_item.done":
            item = evt.get("item") or {}
            if item.get("type") == "function_call":
                self._handle_tool_call(
                    name=item.get("name", ""),
                    call_id=item.get("call_id", ""),
                    args_raw=item.get("arguments") or "{}",
                )
        elif t == "response.done":
            self._handle_response_done(evt)
        elif t == "error":
            err = evt.get("error") or {}
            code = err.get("code")
            if code not in ("response_cancel_not_active",
                            "conversation_already_has_active_response"):
                log(f"[ws] error: {err}")

    def _handle_tool_call(self, name: str, call_id: str, args_raw: str) -> None:
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
        log(f"[tool] {name}({args})")
        result = self.actions.execute(name, args)
        # Provide the function result so the conversation history is
        # consistent, but do NOT trigger a follow-up `response.create`.
        # The robot has no voice and no user-facing text, so a follow-up
        # response only causes the model to chain extra tool calls (e.g.
        # play_emotion → look → move_sequence) and emit stray prose.
        self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"status": result}),
            },
        })

    def _handle_response_done(self, evt: dict) -> None:
        rsp = evt.get("response") or {}
        rsp_id = rsp.get("id")
        if rsp_id and rsp_id == self._last_done_response_id:
            return
        self._last_done_response_id = rsp_id

        usage = evt.get("usage") or rsp.get("usage") or {}
        cost, br = self._compute_cost(usage)
        self._cost_total += cost
        self._turns += 1
        log(
            f"[bot] response.done cost=${cost:.4f} cumul=${self._cost_total:.4f} "
            f"in=txt:{br['text_in']} cached:{br['text_in_cached']} "
            f"audio:{br['audio_in']} audio_cached:{br['audio_in_cached']} "
            f"out=txt:{br['text_out']} audio:{br['audio_out']} "
            f"(turn #{self._turns})"
        )

        self._response_active = False

    # ---- run ----
    def run(self) -> None:
        self._print_config()
        self._ws = websocket.WebSocketApp(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            header=[f"Authorization: Bearer {OPENAI_API_KEY}"],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        threading.Thread(target=self._sender_loop, daemon=True).start()

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
