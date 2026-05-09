"""Grok chat-with-tools bridge (mic → VAD → STT → chat → tools).

Stack: webrtcvad (local), xAI `/v1/stt` (REST), xAI
`/v1/chat/completions` with tool calling. Replaces the realtime
WebSocket bridge — much cheaper, predictable text-token billing,
and no wasted TTS audio.
"""

from .chat import GrokChatBridge

__all__ = ["GrokChatBridge"]
