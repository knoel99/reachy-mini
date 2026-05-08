"""xAI Grok Voice Think Fast bridge implementation. Text-only output (no TTS)."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from .voice_bridge import (
    VoiceBridge,
    REALTIME_RATE,
    INSTRUCTIONS,
)

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


XAI_API_KEY = os.environ.get("XAI_API_KEY")

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
        super().__init__(mini)

    def get_ws_url(self) -> str:
        return f"wss://api.x.ai/v1/realtime?model={self.model}"

    def get_auth_header(self) -> str:
        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY is not set")
        return f"Authorization: Bearer {XAI_API_KEY}"

    def get_session_config(self) -> dict:
        # Text-only: no `voice`, no `audio.output` block. Mic stays
        # active for STT but the model never produces TTS audio.
        return {
            "instructions": INSTRUCTIONS,
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
            print(f"\n[user] {transcript.strip()}", flush=True)

    def compute_cost(self, usage: dict) -> tuple[float, dict]:
        p = PRICING[self.model]

        # OpenAI-style nested structure
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

        # Fallback: flat structure
        if text_in_total == 0 and audio_in_total == 0:
            text_in = usage.get("input_tokens", 0)
            text_out = usage.get("output_tokens", 0)

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

    def _print_config(self) -> None:
        p = PRICING[self.model]
        print(
            f"[config] provider=xai model={self.model} text-only output  "
            f"prices /1M tok: text in ${p['text_input']*1e6:.2f} "
            f"(cached ${p['text_input_cached']*1e6:.2f}) "
            f"text out ${p['text_output']*1e6:.2f}  "
            f"audio in ${p['audio_input']*1e6:.2f} "
            f"(cached ${p['audio_input_cached']*1e6:.2f})",
            flush=True,
        )
