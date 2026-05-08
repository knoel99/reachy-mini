"""Entrypoint for the Reachy Mini voice bridge.

Run with:

    python -m reachy_voice [--provider openai|xai] [--model NAME]

Connects a Reachy Mini Wireless to a realtime voice API (OpenAI
Realtime / Grok Voice Think Fast). The robot has no TTS: the model
analyses the user's speech and reacts via tool calls only (head
movements + preloaded emotion sounds).
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

from .bridges import GrokVoiceBridge, OpenAIRealtimeBridge, VoiceBridge


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="reachy-voice",
        description="Reachy Mini <-> Voice API bridge (OpenAI / Grok).",
    )
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


def _resolve_provider_and_model(args: argparse.Namespace) -> tuple[str, str]:
    provider = args.provider or os.environ.get("VOICE_PROVIDER", "openai")
    if provider == "openai":
        model = args.model or os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")
    elif provider == "xai":
        model = args.model or os.environ.get("GROK_MODEL", "grok-voice-think-fast-1.0")
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


def _create_bridge(mini: ReachyMini, provider: str, model: str) -> VoiceBridge:
    if provider == "openai":
        return OpenAIRealtimeBridge(mini, model=model)
    if provider == "xai":
        return GrokVoiceBridge(mini, model=model)
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
            print(f"[robot] wake_up skipped: {e}", flush=True)
        try:
            bridge = _create_bridge(mini, provider, model)
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
