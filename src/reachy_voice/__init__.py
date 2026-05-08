"""Reachy Mini voice bridge package.

Connects a Reachy Mini Wireless to a realtime voice API (OpenAI
Realtime / Grok Voice Think Fast). The model has no TTS: it analyses
the user's speech and reacts purely via tool calls (head movements +
preloaded emotion sounds).

Public API:

- :class:`bridges.VoiceBridge`, :class:`bridges.OpenAIRealtimeBridge`,
  :class:`bridges.GrokVoiceBridge`
- :class:`emotions.EmotionPlayer`
- :func:`tools.build_tools`, :data:`tools.INSTRUCTIONS`,
  :data:`tools.LOOK_POSES`
"""

from .bridges import GrokVoiceBridge, OpenAIRealtimeBridge, VoiceBridge
from .emotions import EmotionPlayer

__all__ = [
    "EmotionPlayer",
    "GrokVoiceBridge",
    "OpenAIRealtimeBridge",
    "VoiceBridge",
]
