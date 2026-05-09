"""Reachy Mini voice bridge package.

The robot has no TTS: each provider parses the user's speech (server
VAD on the realtime path, local VAD + STT on the chat path) and the
LLM reacts purely via tool calls (head movements + preloaded emotion
sounds).

Public API:

- :class:`openai.OpenAIRealtimeBridge` — OpenAI Realtime WebSocket.
- :class:`grok.GrokChatBridge` — Grok chat-with-tools via xAI STT.
- :class:`emotions.EmotionPlayer`
- :func:`tools.build_tools`, :data:`tools.INSTRUCTIONS`
"""

from .emotions import EmotionPlayer
from .grok import GrokChatBridge
from .openai import OpenAIRealtimeBridge

__all__ = [
    "EmotionPlayer",
    "GrokChatBridge",
    "OpenAIRealtimeBridge",
]
