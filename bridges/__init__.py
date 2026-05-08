"""Voice API bridges for Reachy Mini.

This package contains implementations for different voice API providers:
- VoiceBridge: Abstract base class
- OpenAIRealtimeBridge: OpenAI Realtime API implementation
- GrokVoiceBridge: xAI Grok Voice Think Fast implementation
"""

from .voice_bridge import (
    VoiceBridge,
    EmotionPlayer,
    LOOK_POSES,
    _make_head_pose,
    REALTIME_RATE,
    INSTRUCTIONS,
    EMOTION_NAMES,
    TOOLS,
)
from .openai_bridge import OpenAIRealtimeBridge, OPENAI_VOICES
from .grok_bridge import GrokVoiceBridge, GROK_VOICES

__all__ = [
    "VoiceBridge",
    "EmotionPlayer",
    "LOOK_POSES",
    "_make_head_pose",
    "REALTIME_RATE",
    "INSTRUCTIONS",
    "EMOTION_NAMES",
    "TOOLS",
    "OpenAIRealtimeBridge",
    "OPENAI_VOICES",
    "GrokVoiceBridge",
    "GROK_VOICES",
]
