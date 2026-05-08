"""Reachy Mini <-> Voice API Bridge (OpenAI Realtime / Grok Voice Think Fast).

Supports multiple voice API providers through a modular architecture:
  - OpenAI Realtime (gpt-realtime-mini, gpt-realtime, gpt-realtime-2)
  - xAI Grok Voice Think Fast (grok-voice-think-fast-1.0)

Audio I/O goes through the Reachy Mini media manager, and emotions
from `pollen-robotics/reachy-mini-emotions-library` are triggered
on demand by the model via the `play_emotion` tool.

Conversation flow tuned for naturalness:
  - barge-in: when the user starts speaking, cancel the in-flight
    response (and truncate on OpenAI, cancel-only on Grok)
  - no emotion plays while the bot is speaking; tool-call emotions are
    queued and the latest one fires on response.done
  - per-turn and cumulative cost printed from the `usage` block
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from reachy_mini import ReachyMini
from reachy_mini.io.ws_client import WSClient
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.reachy_mini import INIT_HEAD_POSE, INIT_ANTENNAS_JOINT_POSITIONS

from bridges import VoiceBridge, OpenAIRealtimeBridge, GrokVoiceBridge


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reachy Mini <-> Voice API bridge (OpenAI / Grok)")
    p.add_argument(
        "--provider",
        default=None,
        choices=["openai", "xai"],
        help="API provider: openai (default) or xai (Grok Voice).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override model. Examples: "
             "OpenAI: gpt-realtime-mini, gpt-realtime, gpt-realtime-2; "
             "xAI: grok-voice-think-fast-1.0.",
    )
    return p.parse_args()


_ARGS = _parse_args()
PROVIDER = _ARGS.provider or os.environ.get("VOICE_PROVIDER", "openai")

# Provider-specific defaults
if PROVIDER == "openai":
    MODEL = _ARGS.model or os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")
elif PROVIDER == "xai":
    MODEL = _ARGS.model or os.environ.get("GROK_MODEL", "grok-voice-think-fast-1.0")
else:
    sys.exit(f"Unknown provider: {PROVIDER}")

REACHY_HOST = os.environ.get("REACHY_HOST")
REACHY_PORT = int(os.environ.get("REACHY_PORT", "8000"))

if REACHY_HOST:
    _orig_get_status = WSClient.get_status

    def _patched_get_status(self):  # type: ignore[no-redef]
        status = _orig_get_status(self)
        try:
            status.wlan_ip = REACHY_HOST
        except Exception:
            pass
        return status

    WSClient.get_status = _patched_get_status

# Re-export constants for backward compatibility
REALTIME_RATE = 24_000

# Helper to create bridge based on provider
def create_bridge(mini: ReachyMini, provider: str = PROVIDER,
                  model: str = MODEL) -> VoiceBridge:
    """Factory function to create the appropriate bridge based on provider."""
    if provider == "openai":
        return OpenAIRealtimeBridge(mini, model=model)
    elif provider == "xai":
        return GrokVoiceBridge(mini, model=model)
    else:
        sys.exit(f"Unknown provider: {provider}")


def main() -> None:
    kwargs = {}
    if REACHY_HOST:
        kwargs["host"] = REACHY_HOST
        kwargs["port"] = REACHY_PORT

    with ReachyMini(**kwargs) as mini:
        try:
            mini.wake_up()
        except Exception as e:
            print(f"[robot] wake_up skipped: {e}", flush=True)
        try:
            bridge = create_bridge(mini, provider=PROVIDER, model=MODEL)
            bridge.run()
        finally:
            # Return to neutral pose (do NOT goto_sleep — user wants the
            # robot to stay awake / powered on).
            try:
                mini.goto_target(
                    INIT_HEAD_POSE,
                    antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                    duration=1.0,
                )
            except Exception as e:
                print(f"[robot] goto neutral failed: {e}", flush=True)


if __name__ == "__main__":
    main()
