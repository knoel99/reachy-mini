"""Entrypoint for the Reachy Mini voice bridge.

Run with:

    python -m reachy_voice [--provider openai|grok] [--model NAME]

The robot has no TTS: the model parses the user's speech and reacts
via tool calls only (head movements + preloaded emotion sounds).

Provider stacks:
- ``openai`` → realtime WebSocket (audio-in, text-out).
- ``grok``   → mic → VAD → STT → /v1/chat/completions with tools.
"""

from __future__ import annotations

import argparse
import os
import sys

from reachy_mini import ReachyMini
from reachy_mini.io.ws_client import WSClient
from reachy_mini.reachy_mini import (
    INIT_ANTENNAS_JOINT_POSITIONS,
    INIT_HEAD_POSE,
)

from ._log import log


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="reachy-voice",
        description="Reachy Mini <-> Voice API bridge (OpenAI Realtime / Grok chat).",
    )
    p.add_argument(
        "--provider",
        default=None,
        choices=["openai", "grok"],
        help="API provider: openai (realtime) or grok (chat with STT).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override model. "
             "OpenAI: gpt-realtime-mini, gpt-realtime, gpt-realtime-2. "
             "Grok: grok-4-1-fast-non-reasoning, grok-4-1-fast-reasoning.",
    )
    return p.parse_args()


def _resolve_provider_and_model(args: argparse.Namespace) -> tuple[str, str]:
    provider = args.provider or os.environ.get("VOICE_PROVIDER", "openai")
    if provider == "openai":
        model = args.model or os.environ.get(
            "OPENAI_REALTIME_MODEL", "gpt-realtime-mini"
        )
    elif provider == "grok":
        model = args.model or os.environ.get(
            "GROK_CHAT_MODEL", "grok-4-1-fast-non-reasoning"
        )
    else:
        sys.exit(f"Unknown provider: {provider}")
    return provider, model


def _patch_wlan_ip_if_needed(reachy_host: str | None) -> None:
    """Override the daemon-reported `wlan_ip` with `REACHY_HOST` when set.

    The daemon may report a non-routable IP (typical with WSL2 / multi-NIC
    setups). When the user has explicitly set `REACHY_HOST`, trust that
    value instead.
    """
    if not reachy_host:
        return
    _orig_get_status = WSClient.get_status

    def _patched(self):  # type: ignore[no-redef]
        status = _orig_get_status(self)
        try:
            status.wlan_ip = reachy_host
        except Exception:
            pass
        return status

    WSClient.get_status = _patched


def _create_bridge(mini: ReachyMini, provider: str, model: str):
    if provider == "openai":
        from .openai import OpenAIRealtimeBridge
        return OpenAIRealtimeBridge(mini, model=model)
    if provider == "grok":
        from .grok import GrokChatBridge
        return GrokChatBridge(mini, model=model)
    sys.exit(f"Unknown provider: {provider}")


def main() -> None:
    args = _parse_args()
    provider, model = _resolve_provider_and_model(args)

    reachy_host = os.environ.get("REACHY_HOST")
    reachy_port = int(os.environ.get("REACHY_PORT", "8000"))
    _patch_wlan_ip_if_needed(reachy_host)

    kwargs: dict = {}
    if reachy_host:
        kwargs["host"] = reachy_host
        kwargs["port"] = reachy_port

    with ReachyMini(**kwargs) as mini:
        try:
            mini.wake_up()
        except Exception as e:
            log(f"[robot] wake_up skipped: {e}")
        try:
            bridge = _create_bridge(mini, provider, model)
            bridge.run()
        finally:
            try:
                mini.goto_target(
                    INIT_HEAD_POSE,
                    antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                    duration=1.0,
                )
            except Exception as e:
                log(f"[robot] goto neutral failed: {e}")


if __name__ == "__main__":
    main()
