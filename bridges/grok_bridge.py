"""Grok Voice Think Fast API bridge implementation."""

import base64
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from .voice_bridge import (
    VoiceBridge,
    REALTIME_RATE,
    INSTRUCTIONS,
    EMOTION_NAMES,
    LOOK_POSES,
    TOOLS,
    _make_head_pose,
)

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


XAI_API_KEY = os.environ.get("XAI_API_KEY")

# Pricing for Grok Voice Think Fast (to be updated based on actual pricing)
# Note: xAI pricing may differ from OpenAI
PRICING = {
    "grok-voice-think-fast-1.0": {
        "text_input":         4.00 / 1_000_000,  # Placeholder - update with actual pricing
        "text_input_cached":  0.40 / 1_000_000,
        "text_output":       24.00 / 1_000_000,
        "audio_input":       32.00 / 1_000_000,
        "audio_input_cached": 0.40 / 1_000_000,
                },
            },
            "required": ["steps"],
        },
    },
]


class GrokVoiceBridge(VoiceBridge):
    """Grok Voice Think Fast API implementation."""
    
    def __init__(self, mini: ReachyMini, model: str = "grok-voice-think-fast-1.0",
                 voice: str = "eve") -> None:
        if model not in PRICING:
            raise ValueError(f"Unsupported Grok model: {model}")
        if voice not in GROK_VOICES:
            raise ValueError(f"Unsupported Grok voice: {voice}. Available: {GROK_VOICES}")
        
        self.model = model
        self.voice = voice
        super().__init__(mini)
    
    def get_ws_url(self) -> str:
        return f"wss://api.x.ai/v1/realtime?model={self.model}"
    
    def get_auth_header(self) -> str:
        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY is not set")
        return f"Authorization: Bearer {XAI_API_KEY}"
    
    def get_session_config(self) -> dict:
        # Grok Voice uses a simpler session structure
        # Note: Grok does not support reasoning.effort parameter
        session = {
            "instructions": INSTRUCTIONS,
            "voice": self.voice,
            "turn_detection": {"type": "server_vad"},
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                },
            },
            "tools": TOOLS,
        }
        return session
    
    def handle_audio_delta(self, evt: dict) -> None:
        delta = evt.get("delta")
        if not delta:
            return
        pcm = self.pcm16_bytes_to_f32(base64.b64decode(delta))
        out = self.resample(pcm, REALTIME_RATE, self.out_rate)
        self.media.push_audio_sample(out)
    
    def handle_transcript_delta(self, evt: dict) -> None:
        # Grok uses response.output_text.delta instead of response.audio_transcript.delta
        delta = evt.get("delta", "")
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
    
    def handle_input_transcription(self, evt: dict) -> None:
        # Grok may not support input transcription in the same way
        # We'll try to handle it if available
        transcript = evt.get("transcript", "")
        if transcript:
            print(f"\n[user] {transcript.strip()}", flush=True)
    
    def supports_truncate(self) -> bool:
        # Grok does NOT support conversation.item.truncate
        # This means barge-in will only cancel, not truncate
        return False
    
    def compute_cost(self, usage: dict) -> tuple[float, dict]:
        # Grok usage structure may differ from OpenAI
        # We'll try to map it to our structure
        p = PRICING.get(self.model, PRICING["grok-voice-think-fast-1.0"])
        
        # Try OpenAI-style structure first
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
        
        # If no detailed breakdown, try simple structure
        if text_in_total == 0 and audio_in_total == 0:
            text_in = usage.get("input_tokens", 0)
            audio_in = 0
            text_out = usage.get("output_tokens", 0)
            audio_out = 0
        
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
    
    def get_tools(self) -> list[dict]:
        return TOOLS
    
    def _print_config(self) -> None:
        p = PRICING.get(self.model)
        print(
            f"[config] provider=xai model={self.model} voice={self.voice}  "
            f"prices /1M tok: text in ${p['text_input']*1e6:.2f} "
            f"(cached ${p['text_input_cached']*1e6:.2f}) "
            f"text out ${p['text_output']*1e6:.2f}  "
            f"audio in ${p['audio_input']*1e6:.2f} "
            f"(cached ${p['audio_input_cached']*1e6:.2f}) "
            f"audio out ${p['audio_output']*1e6:.2f}",
            flush=True,
        )
