"""Structural sanity checks for every registered MelodyBundle.

Iterates `melody_tools.BUNDLES`, so a new bundle dropped into the
registry is automatically covered with no test edits.

Guards against typos that would silently shorten the loop, blow past
joint clamps, or desync the 1:1 audio↔motion mapping.
"""

from __future__ import annotations

import unittest

from reachy_voice.melody import _pitch_to_midi
from reachy_voice.melody_tools import BUNDLES


_ALLOWED_KEYFRAME_FIELDS = {
    "roll", "pitch", "yaw", "x", "y", "z",
    "body_yaw", "antenna_left", "antenna_right",
}
_AXIS_LIMITS = {
    "roll": 30.0, "pitch": 30.0, "yaw": 60.0,
    "x": 30.0, "y": 30.0, "z": 30.0,
    "body_yaw": 160.0,
    "antenna_left": 90.0, "antenna_right": 90.0,
}


class BundleStructure(unittest.TestCase):

    def test_registry_non_empty(self) -> None:
        self.assertGreater(len(BUNDLES), 0)

    def test_tool_names_unique(self) -> None:
        names = [b.tool_name for b in BUNDLES]
        self.assertEqual(len(names), len(set(names)))

    def test_each_bundle(self) -> None:
        for b in BUNDLES:
            with self.subTest(bundle=b.name):
                self.assertEqual(len(b.notes), len(b.keyframes))
                self.assertGreaterEqual(len(b.notes), 8)
                self.assertGreater(b.bpm, 0.0)

                # Every non-rest pitch must parse to a valid MIDI number.
                midis = [_pitch_to_midi(n["pitch"]) for n in b.notes]
                pitched = [m for m in midis if m is not None]
                self.assertGreater(len(pitched), 0)
                for n, m in zip(b.notes, midis):
                    if n["pitch"].strip().lower() not in ("r", "rest", ""):
                        self.assertIsNotNone(
                            m, msg=f"{b.name}: unparseable pitch {n['pitch']!r}"
                        )

                for i, kf in enumerate(b.keyframes):
                    with self.subTest(step=i):
                        self.assertLessEqual(
                            set(kf.keys()), _ALLOWED_KEYFRAME_FIELDS
                        )
                        for axis, limit in _AXIS_LIMITS.items():
                            self.assertLessEqual(
                                abs(kf.get(axis, 0.0)), limit,
                                msg=f"{b.name}[{i}].{axis} out of range",
                            )


if __name__ == "__main__":
    unittest.main()
