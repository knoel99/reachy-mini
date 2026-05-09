"""OpenAI Realtime API bridge implementation."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from .base import VoiceBridge, REALTIME_RATE
from .._log import log
from ..tools import INSTRUCTIONS

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


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

REASONING_MODELS = {"gpt-realtime", "gpt-realtime-2"}
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "medium")

class OpenAIRealtimeBridge(VoiceBridge):
    """OpenAI Realtime API implementation. Text-only output (no TTS)."""

    def __init__(self, mini: ReachyMini, model: str = "gpt-realtime-mini") -> None:
        if model not in PRICING:
            raise ValueError(f"Unsupported OpenAI model: {model}. Available: {list(PRICING.keys())}")
        self.model = model
        super().__init__(mini)
    
    def get_ws_url(self) -> str:
        return f"wss://api.openai.com/v1/realtime?model={self.model}"
    
    def get_auth_header(self) -> str:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        return f"Authorization: Bearer {OPENAI_API_KEY}"
    
    def get_session_config(self) -> dict:
        # Text-only output: the robot reacts via tool calls (movements +
        # emotion sounds) and never speaks through the API. This avoids
        # paying for audio output tokens and removes the TTS latency.
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
            "tools": self.tools,
            "tool_choice": "auto",
        }
        if self.model in REASONING_MODELS:
            session["reasoning"] = {"effort": REASONING_EFFORT}
        return session

    def handle_transcript_delta(self, evt: dict) -> None:
        sys.stdout.write(evt.get("delta", ""))
        sys.stdout.flush()
    
    def handle_input_transcription(self, evt: dict) -> None:
        log(f"[user] {evt.get('transcript', '').strip()}", lead="\n")

    def compute_cost(self, usage: dict) -> tuple[float, dict]:
        p = PRICING.get(self.model) or PRICING["gpt-realtime-mini"]
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
    
    def _print_config(self) -> None:
        p = PRICING.get(self.model)
        reasoning_str = (
            f" reasoning.effort={REASONING_EFFORT}"
            if self.model in REASONING_MODELS else " (no reasoning support)"
        )
        if p is None:
            log(
                f"[config] provider=openai model={self.model}{reasoning_str} "
                "text-only output (pricing UNKNOWN — cost will be 0)"
            )
        else:
            log(
                f"[config] provider=openai model={self.model}{reasoning_str} text-only output  "
                f"prices /1M tok: text in ${p['text_input']*1e6:.2f} "
                f"(cached ${p['text_input_cached']*1e6:.2f}) "
                f"text out ${p['text_output']*1e6:.2f}  "
                f"audio in ${p['audio_input']*1e6:.2f} "
                f"(cached ${p['audio_input_cached']*1e6:.2f})"
            )
