"""Grok chat-with-tools bridge.

Pipeline: mic → VAD (webrtcvad) → STT (/v1/stt) → chat completion
with tools (/v1/chat/completions) → tool dispatch → loop.

This bypasses the realtime audio API entirely: we only pay for STT
audio time + text-only chat tokens, and the model never produces
TTS we'd have to discard. Roughly 100× cheaper per turn than the
realtime path for a tool-only robot.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from typing import TYPE_CHECKING

import httpx
import numpy as np
from scipy.signal import resample_poly

from .._actions import RobotActions
from .._log import log
from ..tools import INSTRUCTIONS, to_chat_tools
from .stt import XaiSTT
from .vad import SAMPLE_RATE, UtteranceDetector

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
XAI_API_KEY = os.environ.get("XAI_API_KEY")

# Per-1M-token rates from https://docs.x.ai/developers/models (2026-05).
# Both fast variants share the same pricing; user picks reasoning vs
# non-reasoning on latency, not cost.
PRICING = {
    "grok-4-1-fast-non-reasoning": {"input": 0.20 / 1e6, "output": 0.50 / 1e6},
    "grok-4-1-fast-reasoning":     {"input": 0.20 / 1e6, "output": 0.50 / 1e6},
}
STT_HOURLY_USD = 0.10  # /v1/stt REST


class GrokChatBridge:
    """Mic-driven chat loop with tool calling on the robot."""

    def __init__(self,
                 mini: ReachyMini,
                 model: str = "grok-4-1-fast-non-reasoning",
                 language: str = "fr") -> None:
        if model not in PRICING:
            raise ValueError(
                f"Unsupported Grok chat model: {model}. Available: {list(PRICING)}"
            )
        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY is not set")

        self.mini = mini
        self.media = mini.media
        self.in_rate = self.media.get_input_audio_samplerate()
        self.model = model

        self.actions = RobotActions(mini)
        self.tools_chat = to_chat_tools(self.actions.tools)

        self.vad = UtteranceDetector()
        self.stt = XaiSTT(api_key=XAI_API_KEY, language=language)

        # Native context: keep the full conversation. grok-4-1-fast has
        # ample headroom for a robot session; we don't truncate.
        self.messages: list[dict] = [
            {"role": "system", "content": INSTRUCTIONS},
        ]

        self._stop = threading.Event()
        self._chat_lock = threading.Lock()  # serialise chat turns vs barge-in
        self._cost_total = 0.0
        self._stt_seconds = 0.0
        self._chat_turns = 0

        self._client = httpx.Client(timeout=60.0)

    # ---- audio path ----
    def _resample_to_16k(self, mono: np.ndarray) -> np.ndarray:
        if self.in_rate == SAMPLE_RATE:
            return mono.astype(np.float32, copy=False)
        g = np.gcd(self.in_rate, SAMPLE_RATE)
        return resample_poly(
            mono, SAMPLE_RATE // g, self.in_rate // g
        ).astype(np.float32)

    def _mic_loop(self) -> None:
        while not self._stop.is_set():
            samples = self.media.get_audio_sample()
            if samples is None or len(samples) == 0:
                time.sleep(0.01)
                continue
            # Drop frames while the robot's own audio is playing so the
            # mic doesn't pick up the speaker as a "user" utterance.
            if self.actions.is_speaking():
                continue
            mono = samples if samples.ndim == 1 else samples.mean(axis=1)
            mono = self._resample_to_16k(mono.astype(np.float32))
            for utt in self.vad.feed(mono):
                # Run STT + chat off the mic thread so we can keep
                # capturing audio (and detect barge-in via VAD).
                threading.Thread(
                    target=self._handle_utterance, args=(utt,), daemon=True,
                ).start()

    # ---- utterance pipeline ----
    def _handle_utterance(self, utt) -> None:
        with self._chat_lock:
            try:
                t0 = time.monotonic()
                text = self.stt.transcribe_pcm16(utt.pcm16, SAMPLE_RATE)
                stt_dt = time.monotonic() - t0
            except Exception as e:
                log(f"[stt] failed: {e}")
                return

            if not text:
                return
            self._stt_seconds += utt.duration_s
            log(
                f"[user] {text}  (utterance {utt.duration_s:.1f}s, "
                f"STT {stt_dt:.2f}s)",
                lead="\n",
            )
            self.messages.append({"role": "user", "content": text})
            self._chat_until_tools_settled()

    def _chat_until_tools_settled(self) -> None:
        """Loop chat → tools → chat until the model stops calling tools."""
        while not self._stop.is_set():
            try:
                t0 = time.monotonic()
                resp = self._post_chat()
                chat_dt = time.monotonic() - t0
            except Exception as e:
                log(f"[chat] failed: {e}")
                return

            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            usage = resp.get("usage") or {}
            cost = self._account_chat(usage)
            self._chat_turns += 1

            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""

            log(
                f"[bot] chat.done cost=${cost:.4f} cumul=${self._cost_total:.4f} "
                f"in={usage.get('prompt_tokens', 0)} "
                f"out={usage.get('completion_tokens', 0)} "
                f"tools={len(tool_calls)} ({chat_dt:.2f}s, turn #{self._chat_turns})"
            )

            assistant_msg: dict = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            if not tool_calls:
                if content:
                    log(f"[assistant] {content}")
                return

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args_raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
                log(f"[tool] {name}({args})")
                result = self.actions.execute(name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps({"status": result}),
                })

    def _post_chat(self) -> dict:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "tools": self.tools_chat,
            "tool_choice": "auto",
        }
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        r = self._client.post(XAI_CHAT_URL, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

    def _account_chat(self, usage: dict) -> float:
        p = PRICING[self.model]
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = in_tok * p["input"] + out_tok * p["output"]
        self._cost_total += cost
        return cost

    # ---- run ----
    def _print_config(self) -> None:
        p = PRICING[self.model]
        log(
            f"[config] provider=grok-chat model={self.model}  "
            f"prices /1M tok: in ${p['input']*1e6:.2f} out ${p['output']*1e6:.2f}  "
            f"+ STT REST ${STT_HOURLY_USD:.2f}/hr"
        )

    def run(self) -> None:
        self._print_config()
        self.media.start_playing()
        self.media.start_recording()
        threading.Thread(target=self._mic_loop, daemon=True).start()

        def _sigint(_sig, _frm):
            self._stop.set()
        signal.signal(signal.SIGINT, _sigint)

        try:
            while not self._stop.is_set():
                time.sleep(0.2)
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
            try:
                self._client.close()
                self.stt.close()
            except Exception:
                pass
            stt_cost = (self._stt_seconds / 3600.0) * STT_HOURLY_USD
            log(
                f"[cost] session total: chat ${self._cost_total:.4f} "
                f"+ STT ~${stt_cost:.4f} = ${self._cost_total + stt_cost:.4f} "
                f"({self._chat_turns} chat turn(s), {self._stt_seconds:.1f}s audio)"
            )
