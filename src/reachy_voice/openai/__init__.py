"""OpenAI Realtime bridge.

Streams microphone audio over a WebSocket to the OpenAI Realtime API,
parses tool calls out of `response.output_item.done`, and dispatches
them to the robot via :class:`reachy_voice._actions.RobotActions`.
"""

from .realtime import OpenAIRealtimeBridge

__all__ = ["OpenAIRealtimeBridge"]
