"""Sanity checks for the preprogrammed `melody_id='macarena'` data.

These are structural assertions on the constants. They guard against
typos that would silently shorten the loop, blow past joint clamps,
or desync the 1:1 audio↔motion mapping.
"""

from __future__ import annotations

import unittest

from reachy_voice.macarena import (
    MACARENA_BPM,
    MACARENA_KEYFRAMES,
    MACARENA_MELODY,
)
from reachy_voice.melody import _pitch_to_midi


class MacarenaConstants(unittest.TestCase):

    def test_lengths_match(self) -> None:
        self.assertEqual(len(MACARENA_MELODY), 16)
        self.assertEqual(len(MACARENA_KEYFRAMES), len(MACARENA_MELODY))

    def test_tempo(self) -> None:
        # Standard Latin-pop bracket, ~9.3s for 16 quarter notes.
        self.assertAlmostEqual(MACARENA_BPM, 103.0)

    def test_tessitura_within_one_octave(self) -> None:
        midis = [_pitch_to_midi(n["pitch"]) for n in MACARENA_MELODY]
        self.assertTrue(all(m is not None for m in midis))
        self.assertLessEqual(max(midis) - min(midis), 12)

    def test_keyframe_fields_within_clamps(self) -> None:
        # Matches `_play_sequence` and `_dance` clamps. Out-of-range
        # values would be silently squashed at runtime.
        for i, kf in enumerate(MACARENA_KEYFRAMES):
            with self.subTest(step=i):
                self.assertLessEqual(abs(kf.get("roll", 0.0)), 30.0)
                self.assertLessEqual(abs(kf.get("pitch", 0.0)), 30.0)
                self.assertLessEqual(abs(kf.get("yaw", 0.0)), 60.0)
                self.assertLessEqual(abs(kf.get("x", 0.0)), 30.0)
                self.assertLessEqual(abs(kf.get("y", 0.0)), 30.0)
                self.assertLessEqual(abs(kf.get("z", 0.0)), 30.0)
                self.assertLessEqual(abs(kf.get("body_yaw", 0.0)), 160.0)
                self.assertLessEqual(
                    abs(kf.get("antenna_left", 0.0)), 90.0
                )
                self.assertLessEqual(
                    abs(kf.get("antenna_right", 0.0)), 90.0
                )

    def test_keyframe_known_fields_only(self) -> None:
        # Typo guard: a misspelled key would silently default to neutral.
        allowed = {
            "roll", "pitch", "yaw", "x", "y", "z",
            "body_yaw", "antenna_left", "antenna_right",
        }
        for i, kf in enumerate(MACARENA_KEYFRAMES):
            with self.subTest(step=i):
                self.assertLessEqual(set(kf.keys()), allowed)


if __name__ == "__main__":
    unittest.main()
