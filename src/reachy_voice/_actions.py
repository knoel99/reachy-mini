"""Robot action dispatcher shared by every provider bridge.

A provider only has to translate its incoming tool-call event into a
`(name, args)` pair and call `RobotActions.execute(name, args)`. The
dispatcher owns the EmotionPlayer (preloaded WAVs), the tool registry
sent to the LLM, and serialises head movements through a single
async worker so two `move_sequence` calls never collide.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

from ._log import log
from .emotions import EmotionPlayer
from .melody import MelodyPlayer
from .tools import LOOK_POSES, _make_head_pose, build_tools

if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_vision import Camera, FastVLM, Moondream, Preview


class RobotActions:
    """Maps LLM tool calls to robot motion + emotion playback.

    Optional vision backend (Moondream / FastVLM) is wired in here so
    every provider bridge gets vision tools for free — bridges only
    need to call :meth:`start_camera` / :meth:`stop_camera` around
    their session lifecycle and propagate :attr:`vision_enabled` /
    :attr:`vision_grounding` to their system prompt builder.
    """

    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        from reachy_mini.motion.recorded_move import RecordedMoves
        self._emotions = EmotionPlayer(
            self.mini,
            RecordedMoves("pollen-robotics/reachy-mini-emotions-library"),
        )
        self._melody = MelodyPlayer(self.mini)

        self._vision: Moondream | FastVLM | None = None
        self._camera: Camera | None = None
        self._preview: Preview | None = None
        self._init_vision()

        self.vision_enabled = self._vision is not None
        self.vision_grounding = bool(self._vision and self._vision.supports_grounding)

        self.tools = build_tools(
            sorted(self._emotions.library.list_moves()),
            vision_enabled=self.vision_enabled,
            vision_grounding=self.vision_grounding,
        )
        self._move_thread: threading.Thread | None = None

    def _init_vision(self) -> None:
        backend = os.environ.get("VISION_BACKEND", "").strip().lower()
        if not backend or backend == "none":
            return
        try:
            from reachy_vision import Camera, FastVLM, Moondream, Preview
        except ImportError as e:
            log(f"[vision] reachy_vision import failed, vision disabled: {e}")
            return
        try:
            if backend == "moondream":
                api_key = os.environ.get("MOONDREAM_API_KEY", "").strip() or None
                local = os.environ.get("MOONDREAM_LOCAL", "").lower() in ("1", "true", "yes")
                self._vision = Moondream(api_key=api_key, local=local)
            elif backend == "fastvlm":
                url = os.environ.get("FASTVLM_URL", "").strip()
                timeout = float(os.environ.get("FASTVLM_TIMEOUT", "30"))
                self._vision = FastVLM(base_url=url, timeout=timeout)
            else:
                log(f"[vision] unknown VISION_BACKEND={backend!r}, vision disabled")
                return
        except Exception as e:
            log(f"[vision] backend init failed, vision disabled: {e}")
            self._vision = None
            return
        self._camera = Camera(self.mini, fps=5.0)
        if os.environ.get("VISION_PREVIEW", "1") != "0":
            try:
                self._preview = Preview(
                    self._camera,
                    host=os.environ.get("VISION_PREVIEW_HOST", "127.0.0.1"),
                    port=int(os.environ.get("VISION_PREVIEW_PORT", "5050")),
                )
            except Exception as e:
                log(f"[preview] init failed, preview disabled: {e}")
                self._preview = None

    def start_camera(self) -> None:
        if self._camera is not None:
            self._camera.start()
        if self._preview is not None:
            try:
                self._preview.start()
            except Exception as e:
                log(f"[preview] start failed: {e}")
                self._preview = None

    def stop_camera(self) -> None:
        if self._preview is not None:
            try:
                self._preview.stop()
            except Exception as e:
                log(f"[preview] stop failed: {e}")
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as e:
                log(f"[camera] stop failed: {e}")
        if self._vision is not None and hasattr(self._vision, "close"):
            try:
                self._vision.close()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        """True while an emotion or melody is still playing."""
        return self._emotions.is_speaking() or self._melody.is_speaking()

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

        if name == "play_melody":
            notes = args.get("notes") or []
            if not notes:
                return "empty_melody"
            tempo = args.get("tempo_bpm")
            self._run_async(lambda: self._melody.play(notes, tempo))
            return f"queued:{len(notes)}_notes"

        if name == "look_and_describe":
            return self._handle_look_and_describe(args.get("question", ""))

        if name == "find_object":
            return self._handle_find_object(args.get("target", ""))

        return f"unknown_tool:{name}"

    def _handle_look_and_describe(self, question: str) -> str:
        if self._vision is None or self._camera is None:
            return "vision_not_enabled"
        question = (question or "").strip()
        if not question:
            return "empty_question"
        frame = self._camera.get_latest()
        if frame is None:
            return "no_frame_available"
        try:
            text = self._vision.caption(frame, question)
        except Exception as e:
            log(f"[vision] caption failed: {e}")
            return f"error:{e}"
        if not text:
            return "no_answer"
        if len(text) > 300:
            text = text[:297] + "..."
        if self._preview is not None:
            self._preview.register_caption(question, text)
        return text

    def _handle_find_object(self, target: str) -> str:
        if self._vision is None or self._camera is None:
            return "vision_not_enabled"
        if not self._vision.supports_grounding:
            return "grounding_not_supported"
        target = (target or "").strip()
        if not target:
            return "empty_target"
        frame = self._camera.get_latest()
        if frame is None:
            return "no_frame_available"
        try:
            points = self._vision.point(frame, target)
        except Exception as e:
            log(f"[vision] point failed: {e}")
            return f"error:{e}"
        if not points:
            return f"not_found:{target}"
        cx, cy = points[0]
        if self._preview is not None:
            self._preview.register_detection(target, [(cx, cy)])
        h, w, _ = frame.shape
        px, py = cx * w, cy * h

        def _aim() -> None:
            try:
                pose = self.mini.look_at_image(px, py, duration=0.0, perform_movement=False)
                self.mini.goto_target(pose, duration=0.6)
            except Exception as e:
                log(f"[vision] look_at_image failed: {e}")

        self._run_async(_aim)
        return f"found:{target}@({cx:.2f},{cy:.2f})"

    def _run_async(self, fn) -> None:
        """Run fn off the caller thread, serialised after the previous move."""
        prev = self._move_thread

        def _wrapper() -> None:
            if prev is not None and prev.is_alive():
                # Outlast a realistic play_melody buffer (~30s) so
                # back-to-back audio actions don't overlap on the
                # speaker. Theoretical max is much higher (_MAX_NOTES
                # × _MAX_DUR_S), but 120s keeps a deadlock ceiling.
                prev.join(timeout=120.0)
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

                body_yaw_rad = None
                by = step.get("body_yaw")
                if by is not None:
                    by_deg = max(-160.0, min(160.0, float(by)))
                    body_yaw_rad = np.deg2rad(by_deg)

                self.mini.goto_target(
                    pose, antennas=antennas, body_yaw=body_yaw_rad,
                    duration=duration,
                )
            except Exception as e:
                log(f"[seq] step failed: {e}")
                return

        try:
            self.mini.goto_target(
                INIT_HEAD_POSE,
                antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                body_yaw=0.0,
                duration=0.5,
            )
        except Exception as e:
            log(f"[seq] return to neutral failed: {e}")
