"""Reachy Mini <-> OpenAI Realtime (gpt-realtime-mini).

Stays close to the OpenAI Realtime WebSocket example:
  https://developers.openai.com/api/docs/guides/realtime-websocket

Audio I/O goes through the Reachy Mini media manager, and emotions
from `pollen-robotics/reachy-mini-emotions-library` are triggered:
  - automatically on listening / response events
  - on demand by the model via the `play_emotion` tool
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


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")
VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "alloy")

# Reachy daemon: leave empty for SDK auto-detection (USB / reachy-mini.local).
# On WSL2 mDNS rarely resolves, so set REACHY_HOST to the robot's IP.
REACHY_HOST = os.environ.get("REACHY_HOST")
REACHY_PORT = int(os.environ.get("REACHY_PORT", "8000"))

if REACHY_HOST:
    # The daemon's reported wlan_ip can refer to a different interface
    # (e.g. its own WLAN) that's not reachable from this machine. Force the
    # WebRTC signalling host to the same IP we're using for the SDK channel.
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
REALTIME_RATE = 24_000  # PCM16 mono @ 24 kHz, per OpenAI doc

INSTRUCTIONS = (
    "Tu es la voix d'un petit robot Reachy Mini. "
    "Tu DOIS toujours répondre EN FRANÇAIS, jamais dans une autre langue, "
    "même si l'utilisateur parle dans une autre langue. "
    "Réponds vocalement à chaque tour, de manière brève (1 à 2 phrases), "
    "chaleureuse et expressive. "
    "Quand une émotion est naturelle (joie, surprise, curiosité, doute…), "
    "appelle l'outil `play_emotion` avec le nom approprié pendant ta "
    "réponse pour que ton corps exprime ce que tu ressens."
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
EMOTION_LISTEN = "attentive1"
EMOTION_THINK = "thoughtful1"
EMOTION_REST = "serenity1"

TOOLS = [{
    "type": "function",
    "name": "play_emotion",
    "description": (
        "Joue une émotion physique sur le robot Reachy Mini "
        "(mouvements de tête + antennes). À utiliser quand une émotion "
        "renforce naturellement la réponse."
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
    """Serializes emotion playback so they don't overlap on the robot."""

    def __init__(self, mini: ReachyMini, library: RecordedMoves) -> None:
        self.mini = mini
        self.library = library
        self._lock = threading.Lock()
        self._last_name: str | None = None
        self._last_t = 0.0

    def play(self, name: str, min_interval: float = 0.0) -> None:
        now = time.monotonic()
        if name == self._last_name and (now - self._last_t) < min_interval:
            return
        move = self.library.get(name)
        if move is None:
            print(f"[emotion] unknown: {name}", flush=True)
            return

        def _run() -> None:
            with self._lock:
                try:
                    self.mini.play_move(move, initial_goto_duration=0.5)
                except Exception as e:  # robot may be busy / disconnected
                    print(f"[emotion] play '{name}' failed: {e}", flush=True)

        self._last_name = name
        self._last_t = now
        threading.Thread(target=_run, daemon=True).start()


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
        self._mic_thread: threading.Thread | None = None
        self._send_q: queue.Queue[str] = queue.Queue(maxsize=256)
        self._sender_thread: threading.Thread | None = None
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
        self.emotions.play(EMOTION_REST)

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
            try:
                self._send_q.put_nowait(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                }))
            except queue.Full:
                pass  # drop on overload rather than blocking the mic

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

    # ---- openai -> speaker / emotions ----
    def _on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        evt = json.loads(raw)
        t = evt.get("type", "")

        if t in ("response.audio.delta", "response.output_audio.delta"):
            self._audio_chunks += 1
            self._play_audio_delta(evt["delta"])
        elif t == "input_audio_buffer.speech_started":
            self.emotions.play(EMOTION_LISTEN, min_interval=2.0)
        elif t == "input_audio_buffer.speech_stopped":
            self.emotions.play(EMOTION_THINK, min_interval=2.0)
        elif t == "response.created":
            pass  # already covered by speech_stopped
        elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
            sys.stdout.write(evt.get("delta", ""))
            sys.stdout.flush()
        elif t == "conversation.item.input_audio_transcription.completed":
            print(f"\n[user] {evt.get('transcript', '').strip()}", flush=True)
        elif t in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
            print(flush=True)
        elif t == "response.created":
            self._audio_chunks = 0
            print("[bot] thinking...", flush=True)
        elif t in ("response.function_call_arguments.done",
                   "response.output_item.done"):
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
            print(f"[bot] response.done (audio chunks: {self._audio_chunks})", flush=True)
            self.emotions.play(EMOTION_REST, min_interval=4.0)
        elif t == "error":
            print(f"[ws] error: {evt.get('error')}", flush=True)

    def _play_audio_delta(self, b64_audio: str) -> None:
        try:
            pcm = pcm16_bytes_to_f32(base64.b64decode(b64_audio))
            out = resample(pcm, REALTIME_RATE, self.out_rate)
            self.media.push_audio_sample(out)
        except Exception as e:
            print(f"[speaker] push failed: {e}", flush=True)

    def _handle_tool_call(self, evt: dict) -> None:
        name = evt.get("name")
        call_id = evt.get("call_id")
        args_raw = evt.get("arguments") or "{}"
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
        result = "ok"
        if name == "play_emotion":
            emo = args.get("name", "")
            self.emotions.play(emo)
            result = f"played:{emo}"
        else:
            result = f"unknown_tool:{name}"

        self._send_q.put(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"status": result}),
            },
        }))
        self._send_q.put(json.dumps({"type": "response.create"}))

    # ---- lifecycle ----
    def _on_error(self, ws: websocket.WebSocketApp, err: Exception) -> None:
        print(f"[ws] error: {err}", flush=True)

    def _on_close(self, ws, code, msg) -> None:
        print(f"[ws] closed code={code} msg={msg}", flush=True)
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
            try:
                mini.goto_sleep()
            except Exception:
                pass


if __name__ == "__main__":
    main()
