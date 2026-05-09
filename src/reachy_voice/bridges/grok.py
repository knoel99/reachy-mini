"""xAI Grok Voice Think Fast bridge implementation. Text-only output (no TTS)."""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

from .base import VoiceBridge, REALTIME_RATE
from .._log import log
from ..tools import build_instructions

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


XAI_API_KEY = os.environ.get("XAI_API_KEY")
DEBUG_USAGE = bool(os.environ.get("GROK_DEBUG_USAGE"))

# Pricing placeholders for Grok Voice Think Fast — update with real values.
# audio_output kept at 0 since we never request audio output.
PRICING = {
    "grok-voice-think-fast-1.0": {
        "text_input":         4.00 / 1_000_000,
        "text_input_cached":  0.40 / 1_000_000,
        "text_output":       24.00 / 1_000_000,
        "audio_input":       32.00 / 1_000_000,
        "audio_input_cached": 0.40 / 1_000_000,
        "audio_output":       0.00 / 1_000_000,
    },
}


class GrokVoiceBridge(VoiceBridge):
    """Grok Voice Think Fast implementation. Text-only output (no TTS).

    The robot reacts via tool calls (movements + emotion sounds) only;
    no audio output is requested or played.
    """

    def __init__(self, mini: ReachyMini,
                 model: str = "grok-voice-think-fast-1.0") -> None:
        if model not in PRICING:
            raise ValueError(f"Unsupported Grok model: {model}")
        self.model = model
        self._debug_seen_types: set[str] = set()
        super().__init__(mini)

    def _on_message(self, ws, raw: str) -> None:
        # Surface event types and any "usage" payload — xAI's
        # response.done has no usage, so we need to find where (if
        # anywhere) it lives.
        if DEBUG_USAGE:
            try:
                t = json.loads(raw).get("type", "?")
            except Exception:
                t = "?"
            if t not in self._debug_seen_types:
                self._debug_seen_types.add(t)
                log(f"[debug] new event type: {t}")
            if "usage" in raw:
                log(f"[debug] evt with 'usage': {raw[:800]}")
        super()._on_message(ws, raw)

    def get_ws_url(self) -> str:
        return f"wss://api.x.ai/v1/realtime?model={self.model}"

    def get_auth_header(self) -> str:
        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY is not set")
        return f"Authorization: Bearer {XAI_API_KEY}"

    def get_session_config(self) -> dict:
        # NOTE: tried `output_modalities: ["text"]` (session) and
        # `modalities: ["text"]` (response.create) per OpenAI Realtime
        # / liteLLM conventions — both silently ignored by xAI (audio
        # output still streamed). Reverted; xAI has no documented
        # text-only switch as of 2026-05.
        return {
            "instructions": build_instructions(
                vision_enabled=self._vision is not None,
                vision_grounding=bool(self._vision and self._vision.supports_grounding),
            ),
            "turn_detection": {"type": "server_vad"},
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                },
            },
            "tools": self.tools,
        }

    def handle_transcript_delta(self, evt: dict) -> None:
        delta = evt.get("delta", "")
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()

    def handle_input_transcription(self, evt: dict) -> None:
        transcript = evt.get("transcript", "")
        if transcript:
            log(f"[user] {transcript.strip()}", lead="\n")

    def compute_cost(self, usage: dict) -> tuple[float, dict]:
        p = PRICING[self.model]

        # Set GROK_DEBUG_USAGE=1 to dump the raw usage block — useful
        # the first time, to confirm which shape xAI returns.
        if DEBUG_USAGE:
            payload = json.dumps(usage, separators=(",", ":")) if usage else "(empty/missing)"
            log(f"[debug] grok usage raw: {payload}")

        # Handle three known shapes:
        #   (1) OpenAI Realtime style: input_token_details / output_token_details
        #   (2) xAI Chat Completions style: prompt_tokens / completion_tokens
        #       (+ prompt_tokens_details with cached_tokens / audio_tokens)
        #   (3) Flat: input_tokens / output_tokens
        inp = usage.get("input_token_details") or {}
        out = usage.get("output_token_details") or {}
        cached = inp.get("cached_tokens_details") or {}

        text_in_total = inp.get("text_tokens", 0)
        audio_in_total = inp.get("audio_tokens", 0)
        grok_in = inp.get("grok_tokens", 0)
        text_in_cached = cached.get("text_tokens", 0)
        audio_in_cached = cached.get("audio_tokens", 0)
        text_in = max(0, text_in_total - text_in_cached)
        audio_in = max(0, audio_in_total - audio_in_cached)

        text_out = out.get("text_tokens", 0)
        audio_out = out.get("audio_tokens", 0)
        grok_out = out.get("grok_tokens", 0)

        # (2) prompt_tokens / completion_tokens
        if text_in_total == 0 and audio_in_total == 0:
            ptd = usage.get("prompt_tokens_details") or {}
            ctd = usage.get("completion_tokens_details") or {}
            audio_in_total = ptd.get("audio_tokens", 0)
            text_in_cached = ptd.get("cached_tokens", 0)
            prompt_total = usage.get("prompt_tokens", 0)
            text_in_total = max(0, prompt_total - audio_in_total)
            text_in = max(0, text_in_total - text_in_cached)
            audio_in = audio_in_total

            audio_out = ctd.get("audio_tokens", 0)
            completion_total = usage.get("completion_tokens", 0)
            text_out = max(0, completion_total - audio_out)

        # (3) flat input_tokens / output_tokens
        if text_in_total == 0 and audio_in_total == 0:
            text_in = usage.get("input_tokens", 0)
            text_out = usage.get("output_tokens", 0)

        # `grok_tokens` (reasoning) priced as text until xAI publishes a
        # separate rate. Observed at 0 in current responses — guard kept
        # for the day they show up.
        cost = (
            text_in        * p["text_input"]
          + text_in_cached * p["text_input_cached"]
          + audio_in       * p["audio_input"]
          + audio_in_cached * p["audio_input_cached"]
          + text_out       * p["text_output"]
          + audio_out      * p["audio_output"]
          + grok_in        * p["text_input"]
          + grok_out       * p["text_output"]
        )
        return cost, {
            "text_in": text_in, "text_in_cached": text_in_cached,
            "audio_in": audio_in, "audio_in_cached": audio_in_cached,
            "text_out": text_out, "audio_out": audio_out,
            "grok_in": grok_in, "grok_out": grok_out,
        }

    def _print_config(self) -> None:
        p = PRICING[self.model]
        log(
            f"[config] provider=xai model={self.model} text-only output  "
            f"prices /1M tok: text in ${p['text_input']*1e6:.2f} "
            f"(cached ${p['text_input_cached']*1e6:.2f}) "
            f"text out ${p['text_output']*1e6:.2f}  "
            f"audio in ${p['audio_input']*1e6:.2f} "
            f"(cached ${p['audio_input_cached']*1e6:.2f})"
        )
