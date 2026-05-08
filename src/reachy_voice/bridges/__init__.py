"""Voice API bridge implementations for Reachy Mini.

All bridges are configured for text-only output (no TTS — the robot
reacts via tool calls only):

- :class:`VoiceBridge`: abstract base class (transport, mic loop, tool
  dispatch, barge-in).
- :class:`OpenAIRealtimeBridge`: OpenAI Realtime API.
- :class:`GrokVoiceBridge`: xAI Grok Voice Think Fast.
"""

from .base import REALTIME_RATE, VoiceBridge
from .grok import GrokVoiceBridge
from .openai import OpenAIRealtimeBridge

__all__ = [
    "GrokVoiceBridge",
    "OpenAIRealtimeBridge",
    "REALTIME_RATE",
    "VoiceBridge",
]
