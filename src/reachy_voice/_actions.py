"""Robot action dispatcher shared by every provider bridge.

A provider only has to translate its incoming tool-call event into a
`(name, args)` pair and call `RobotActions.execute(name, args)`. The
dispatcher owns the EmotionPlayer (preloaded WAVs), the tool registry
sent to the LLM, and serialises head movements through a single
async worker so two `move_sequence` calls never collide.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from ._log import log
from .emotions import EmotionPlayer
from .tools import LOOK_POSES, _make_head_pose, build_tools

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


class RobotActions:
    """Maps LLM tool calls to robot motion + emotion playback."""

    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        from reachy_mini.motion.recorded_move import RecordedMoves
        self._emotions = EmotionPlayer(
            self.mini,
            RecordedMoves("pollen-robotics/reachy-mini-emotions-library"),
        )
        self.tools = build_tools(sorted(self._emotions.library.list_moves()))
        self._move_thread: threading.Thread | None = None

    def is_speaking(self) -> bool:
        """True while an emotion's bundled audio is still playing."""
        return self._emotions.is_speaking()

    def execute(self, name: str, args: dict) -> str:
        """Schedule a tool call. Returns a short status string for the LLM.

        Every action goes through the same single-worker queue so head
        movements and emotion sounds never compete on the motors or
        speaker. Calls return immediately; the worker runs each action
        to completion before starting the next.
        """
        if name == "play_emotion":
            emo = args.get("name", "")
            self._run_async(lambda: self._emotions.play(emo))
            return f"queued:{emo}"

        if name == "look":
            direction = args.get("direction", "center")
            pose = LOOK_POSES.get(direction)
            if pose is None:
                return f"unknown_direction:{direction}"
            self._run_async(lambda: self.mini.goto_target(pose, duration=0.6))
            return f"looking:{direction}"

        if name == "move_sequence":
            steps = args.get("steps") or []
            if not steps:
                return "empty_sequence"
            self._run_async(lambda: self._play_sequence(steps))
            return f"queued:{len(steps)}_steps"

        return f"unknown_tool:{name}"

    def _run_async(self, fn) -> None:
        """Run fn off the caller thread, serialised after the previous move."""
        prev = self._move_thread

        def _wrapper() -> None:
            if prev is not None and prev.is_alive():
                prev.join(timeout=10.0)
            t0 = time.monotonic()
            try:
                fn()
            except Exception as e:
                log(f"[move] failed: {e}")
                return
            log(f"[move] done in {time.monotonic() - t0:.2f}s")

        t = threading.Thread(target=_wrapper, daemon=True)
        self._move_thread = t
        t.start()

    def _play_sequence(self, steps: list) -> None:
        """Execute a planned head choreography then return to neutral."""
        import numpy as np
        from reachy_mini.reachy_mini import (
            INIT_ANTENNAS_JOINT_POSITIONS,
            INIT_HEAD_POSE,
        )

        for step in steps:
            try:
                roll = max(-30.0, min(30.0, float(step.get("roll", 0.0))))
                pitch = max(-30.0, min(30.0, float(step.get("pitch", 0.0))))
                yaw = max(-60.0, min(60.0, float(step.get("yaw", 0.0))))
                x_mm = max(-30.0, min(30.0, float(step.get("x", 0.0))))
                y_mm = max(-30.0, min(30.0, float(step.get("y", 0.0))))
                z_mm = max(-30.0, min(30.0, float(step.get("z", 0.0))))
                duration = max(0.1, min(3.0, float(step.get("duration", 0.4))))
                pose = _make_head_pose(roll, pitch, yaw, x_mm, y_mm, z_mm)

                antennas = None
                al = step.get("antenna_left")
                ar = step.get("antenna_right")
                if al is not None or ar is not None:
                    al_rad = np.deg2rad(max(-90.0, min(90.0, float(al or 0.0))))
                    ar_rad = np.deg2rad(max(-90.0, min(90.0, float(ar or 0.0))))
                    antennas = [al_rad, ar_rad]

                self.mini.goto_target(pose, antennas=antennas, duration=duration)
            except Exception as e:
                log(f"[seq] step failed: {e}")
                return

        try:
            self.mini.goto_target(
                INIT_HEAD_POSE,
                antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                duration=0.5,
            )
        except Exception as e:
            log(f"[seq] return to neutral failed: {e}")
