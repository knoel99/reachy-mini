"""Voice API bridges for Reachy Mini.

Implementations for different voice API providers, all configured for
text-only output (no TTS — the robot reacts via tool calls only):
- VoiceBridge: abstract base class
- OpenAIRealtimeBridge: OpenAI Realtime API
- GrokVoiceBridge: xAI Grok Voice Think Fast
"""

from .voice_bridge import (
    VoiceBridge,
    EmotionPlayer,
    LOOK_POSES,
    _make_head_pose,
    REALTIME_RATE,
    INSTRUCTIONS,
    build_tools,
)
from .openai_bridge import OpenAIRealtimeBridge
from .grok_bridge import GrokVoiceBridge

__all__ = [
    "VoiceBridge",
    "EmotionPlayer",
    "LOOK_POSES",
    "_make_head_pose",
    "REALTIME_RATE",
    "INSTRUCTIONS",
    "build_tools",
    "OpenAIRealtimeBridge",
    "GrokVoiceBridge",
]
