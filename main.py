"""Reachy Mini <-> OpenAI Realtime (gpt-realtime-mini).

Stays close to the OpenAI Realtime WebSocket example:
  https://developers.openai.com/api/docs/guides/realtime-websocket

Audio I/O goes through the Reachy Mini media manager, and emotions
from `pollen-robotics/reachy-mini-emotions-library` are triggered
on demand by the model via the `play_emotion` tool.

Conversation flow tuned for naturalness:
  - barge-in: when the user starts speaking, cancel the in-flight
    response and truncate the assistant item to what was actually heard
  - no emotion plays while the bot is speaking; tool-call emotions are
    queued and the latest one fires on response.done
  - per-turn and cumulative cost printed from the `usage` block
"""

from __future__ import annotations

import base64
import json
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import websocket
from scipy.signal import resample_poly

from reachy_mini import ReachyMini
from reachy_mini.io.ws_client import WSClient
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.reachy_mini import INIT_HEAD_POSE, INIT_ANTENNAS_JOINT_POSITIONS


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")
VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "alloy")

REACHY_HOST = os.environ.get("REACHY_HOST")
REACHY_PORT = int(os.environ.get("REACHY_PORT", "8000"))

if REACHY_HOST:
    _orig_get_status = WSClient.get_status

    def _patched_get_status(self):  # type: ignore[no-redef]
        status = _orig_get_status(self)
        try:
            status.wlan_ip = REACHY_HOST
        except Exception:
            pass
        return status

    WSClient.get_status = _patched_get_status

REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
REALTIME_RATE = 24_000

# USD per token. Verify against https://openai.com/api/pricing/
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
}


def compute_cost(usage: dict, model: str) -> tuple[float, dict]:
    """Returns (cost_usd, breakdown) from a Realtime usage block."""
    p = PRICING.get(model) or PRICING["gpt-realtime-mini"]
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


INSTRUCTIONS = (
    "Tu es la voix d'un petit robot Reachy Mini. "
    "Tu DOIS toujours répondre EN FRANÇAIS, jamais dans une autre langue, "
    "même si l'utilisateur parle dans une autre langue. "
    "Réponds vocalement à chaque tour, de manière brève (1 à 2 phrases), "
    "chaleureuse et expressive. "
    "Quand une émotion est naturelle (joie, surprise, curiosité, doute…), "
    "appelle l'outil `play_emotion` avec le nom approprié. L'émotion sera "
    "jouée à la fin de ton tour pour ne pas couvrir ta voix."
)

EMOTION_NAMES = [
    "amazed1", "anxiety1", "attentive1", "attentive2", "calming1",
    "cheerful1", "confused1", "curious1", "displeased1", "enthusiastic1",
    "exhausted1", "frustrated1", "grateful1", "helpful1", "inquiring1",
    "irritated1", "laughing1", "loving1", "no1", "oops1", "proud1",
    "relief1", "sad1", "scared1", "serenity1", "shy1", "success1",
    "surprised1", "thoughtful1", "tired1", "uncertain1",
    "understanding1", "welcoming1", "yes1",
]

TOOLS = [{
    "type": "function",
    "name": "play_emotion",
    "description": (
        "Joue une émotion physique sur le robot Reachy Mini "
        "(mouvements de tête + antennes). À utiliser quand une émotion "
        "renforce naturellement la réponse. L'émotion sera jouée à la "
        "fin du tour de parole."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Nom de l'émotion à jouer.",
                "enum": EMOTION_NAMES,
            },
        },
        "required": ["name"],
    },
}]


def f32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def pcm16_bytes_to_f32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    g = np.gcd(src_rate, dst_rate)
    return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)


class EmotionPlayer:
    """Serializes emotion playback. Skips silently if one is already playing
    or if called too soon after the previous one."""

    def __init__(self, mini: ReachyMini, library: RecordedMoves) -> None:
        self.mini = mini
        self.library = library
        self._busy = threading.Lock()
        self._last_t = 0.0

    def play(self, name: str, min_interval: float = 2.5) -> bool:
        now = time.monotonic()
        if (now - self._last_t) < min_interval:
            return False
        if not self._busy.acquire(blocking=False):
            return False  # already playing — skip rather than queue

        move = self.library.get(name)
        if move is None:
            self._busy.release()
            print(f"[emotion] unknown: {name}", flush=True)
            return False

        self._last_t = now

        def _run() -> None:
            try:
                self.mini.play_move(move, initial_goto_duration=0.5)
            except Exception as e:
                print(f"[emotion] play '{name}' failed: {e}", flush=True)
            finally:
                self._busy.release()

        threading.Thread(target=_run, daemon=True).start()
        return True


class RealtimeBridge:
    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        self.media = mini.media
        self.in_rate = self.media.get_input_audio_samplerate()
        self.out_rate = self.media.get_output_audio_samplerate()
        print(f"[audio] mic={self.in_rate} Hz, speaker={self.out_rate} Hz", flush=True)

        self.emotions = EmotionPlayer(
            mini, RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        )

        self._ws: websocket.WebSocketApp | None = None
        self._stop = threading.Event()
        self._send_q: queue.Queue[str] = queue.Queue(maxsize=256)
        self._sender_thread: threading.Thread | None = None
        self._mic_thread: threading.Thread | None = None

        # Per-response state (reset on response.created)
        self._response_active = False
        self._speaking = False  # True between first audio delta and response.done
        self._current_item_id: str | None = None
        self._current_content_index: int = 0
        self._audio_bytes_in_response: int = 0
        self._first_delta_t: float | None = None
        self._audio_chunks: int = 0
        self._pending_emotion: str | None = None  # tool-called during speech, played at end
        self._needs_followup_response = False  # tool call → reply, sent at response.done

        # Cost
        self._cost_total = 0.0
        self._turns = 0

    # ---- helpers ----
    def _send(self, payload: dict) -> None:
        try:
            self._send_q.put_nowait(json.dumps(payload))
        except queue.Full:
            pass

    def _reset_response_state(self) -> None:
        self._response_active = False
        self._speaking = False
        self._current_item_id = None
        self._current_content_index = 0
        self._audio_bytes_in_response = 0
        self._first_delta_t = None
        self._audio_chunks = 0

    # ---- session ----
    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        print("[ws] connected", flush=True)
        ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
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
                    "output": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                        "voice": VOICE,
                    },
                },
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        }))
        self.media.start_playing()
        self.media.start_recording()
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._mic_thread.start()

    # ---- mic -> openai ----
    def _mic_loop(self) -> None:
        while not self._stop.is_set():
            samples = self.media.get_audio_sample()
            if samples is None or len(samples) == 0:
                time.sleep(0.01)
                continue
            mono = samples if samples.ndim == 1 else samples.mean(axis=1)
            resampled = resample(mono.astype(np.float32), self.in_rate, REALTIME_RATE)
            b64 = base64.b64encode(f32_to_pcm16_bytes(resampled)).decode("ascii")
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
                print(f"[ws] send failed: {e}", flush=True)
                return

    # ---- barge-in ----
    def _barge_in(self) -> None:
        """User started speaking. Cancel in-flight response, truncate assistant
        item to what was actually played, drop any pending emotion."""
        if not self._response_active:
            return
        # estimate played ms: PCM16 mono @ 24kHz -> 48 bytes/ms
        recv_ms = self._audio_bytes_in_response // 48
        wall_ms = 0
        if self._first_delta_t is not None:
            wall_ms = int((time.monotonic() - self._first_delta_t) * 1000)
        played_ms = min(recv_ms, wall_ms) if wall_ms > 0 else recv_ms

        if self._current_item_id and played_ms > 0:
            self._send({
                "type": "conversation.item.truncate",
                "item_id": self._current_item_id,
                "content_index": self._current_content_index,
                "audio_end_ms": played_ms,
            })
        self._send({"type": "response.cancel"})
        self._pending_emotion = None
        print(f"[barge-in] cut at {played_ms} ms", flush=True)

    # ---- openai -> speaker / emotions ----
    def _on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        evt = json.loads(raw)
        t = evt.get("type", "")

        if t in ("response.audio.delta", "response.output_audio.delta"):
            self._audio_chunks += 1
            if self._first_delta_t is None:
                self._first_delta_t = time.monotonic()
                self._speaking = True
            if not self._current_item_id:
                self._current_item_id = evt.get("item_id")
                self._current_content_index = evt.get("content_index", 0)
            try:
                pcm_b64 = evt["delta"]
                self._audio_bytes_in_response += (len(pcm_b64) * 3) // 4  # base64 -> raw bytes approx
                self._play_audio_delta(pcm_b64)
            except Exception as e:
                print(f"[speaker] push failed: {e}", flush=True)

        elif t == "input_audio_buffer.speech_started":
            self._barge_in()

        elif t == "input_audio_buffer.speech_stopped":
            pass

        elif t == "response.created":
            self._reset_response_state()
            self._response_active = True

        elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
            sys.stdout.write(evt.get("delta", ""))
            sys.stdout.flush()

        elif t == "conversation.item.input_audio_transcription.completed":
            print(f"\n[user] {evt.get('transcript', '').strip()}", flush=True)

        elif t in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
            print(flush=True)

        elif t in ("response.function_call_arguments.done", "response.output_item.done"):
            if t == "response.output_item.done":
                item = evt.get("item") or {}
                if item.get("type") != "function_call":
                    return
                self._handle_tool_call({
                    "name": item.get("name"),
                    "call_id": item.get("call_id"),
                    "arguments": item.get("arguments") or "{}",
                })
            else:
                self._handle_tool_call(evt)

        elif t == "response.done":
            usage = (evt.get("response") or {}).get("usage") or {}
            cost, br = compute_cost(usage, MODEL)
            self._cost_total += cost
            self._turns += 1
            print(
                f"[bot] response.done audio_chunks={self._audio_chunks} "
                f"in:{br['text_in']}t/{br['text_in_cached']}c/{br['audio_in']}a/{br['audio_in_cached']}ac "
                f"out:{br['text_out']}t/{br['audio_out']}a "
                f"cost=${cost:.4f} cumul=${self._cost_total:.4f} (turn #{self._turns})",
                flush=True,
            )
            # Emotion deferred from a tool call during speech
            if self._pending_emotion:
                self.emotions.play(self._pending_emotion, min_interval=0.0)
                self._pending_emotion = None
            needs_followup = self._needs_followup_response
            self._needs_followup_response = False
            self._reset_response_state()
            if needs_followup:
                # Now that the previous response is fully done, ask the model
                # to react to the tool output(s).
                self._send({"type": "response.create"})

        elif t == "error":
            err = evt.get("error") or {}
            code = err.get("code")
            # Benign races: cancel arrives after the response naturally ends,
            # or follow-up response.create races with response.done.
            if code in ("response_cancel_not_active",
                        "conversation_already_has_active_response"):
                return
            print(f"[ws] error: {err}", flush=True)

    def _play_audio_delta(self, b64_audio: str) -> None:
        pcm = pcm16_bytes_to_f32(base64.b64decode(b64_audio))
        out = resample(pcm, REALTIME_RATE, self.out_rate)
        self.media.push_audio_sample(out)

    def _handle_tool_call(self, evt: dict) -> None:
        name = evt.get("name")
        call_id = evt.get("call_id")
        args_raw = evt.get("arguments") or "{}"
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}

        if name == "play_emotion":
            emo = args.get("name", "")
            if self._speaking:
                # defer to end of turn so it doesn't talk over the voice
                self._pending_emotion = emo
                result = f"queued:{emo}"
            else:
                played = self.emotions.play(emo, min_interval=0.0)
                result = f"played:{emo}" if played else f"skipped:{emo}"
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
        # Defer response.create to response.done — sending it now while the
        # current response is still active triggers
        # `conversation_already_has_active_response`.
        self._needs_followup_response = True

    # ---- lifecycle ----
    def _on_error(self, ws: websocket.WebSocketApp, err: Exception) -> None:
        print(f"[ws] error: {err}", flush=True)

    def _on_close(self, ws, code, msg) -> None:
        print(f"[ws] closed code={code} msg={msg}", flush=True)
        print(f"[cost] session total: ${self._cost_total:.4f} over {self._turns} turn(s)", flush=True)
        self._stop.set()

    def run(self) -> None:
        if not OPENAI_API_KEY:
            sys.exit("OPENAI_API_KEY is not set")

        headers = [f"Authorization: Bearer {OPENAI_API_KEY}"]
        self._ws = websocket.WebSocketApp(
            REALTIME_URL,
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
            self._stop.set()
            try:
                self.media.stop_recording()
            except Exception:
                pass
            try:
                self.media.stop_playing()
            except Exception:
                pass


def main() -> None:
    kwargs = {}
    if REACHY_HOST:
        kwargs["host"] = REACHY_HOST
        kwargs["port"] = REACHY_PORT
    with ReachyMini(**kwargs) as mini:
        try:
            mini.wake_up()
        except Exception as e:
            print(f"[robot] wake_up skipped: {e}", flush=True)
        try:
            RealtimeBridge(mini).run()
        finally:
            # Return to neutral pose (do NOT goto_sleep — user wants the
            # robot to stay awake / powered on).
            try:
                mini.goto_target(
                    INIT_HEAD_POSE,
                    antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                    duration=1.0,
                )
            except Exception as e:
                print(f"[robot] goto neutral failed: {e}", flush=True)


if __name__ == "__main__":
    main()
