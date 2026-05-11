"""Unit tests for the pure pieces of reachy_voice.melody.

Uses unittest (stdlib) so the project doesn't need a new dev dep.
`MelodyPlayer.__init__` touches the robot; where an instance is
needed, we bypass __init__ via __new__ and set `_target_rate`
manually so the audio + motion timelines can be tested without
hardware.

Run from the repo root:
    python -m unittest tests.test_melody
"""

from __future__ import annotations

import unittest

import numpy as np

from reachy_voice.melody import (
    MelodyPlayer,
    _MAX_DUR_S,
    _MAX_NOTES,
    _MAX_TOTAL_DUR_S,
    _MIN_DUR_S,
    _pitch_to_midi,
)


def _player(rate: int = 16_000) -> MelodyPlayer:
    p = MelodyPlayer.__new__(MelodyPlayer)
    p._target_rate = rate
    p._speaking_until = 0.0
    return p


class PitchToMidi(unittest.TestCase):
    """Pitch-string parser. Gate at the entry of the synth pipeline:
    if `C4` doesn't map to MIDI 60 the whole melody is transposed,
    and bad inputs from the LLM must degrade to a rest, not crash."""

    def test_basic(self) -> None:
        # Universal references: A4 = 440 Hz = MIDI 69, C4 = middle C = 60.
        self.assertEqual(_pitch_to_midi("C4"), 60)
        self.assertEqual(_pitch_to_midi("A4"), 69)

    def test_accidentals(self) -> None:
        # Both '#' and 'b' must shift the semitone in the right direction.
        self.assertEqual(_pitch_to_midi("F#5"), 78)
        self.assertEqual(_pitch_to_midi("Bb3"), 58)

    def test_case_insensitive_letter(self) -> None:
        # The LLM sometimes lowercases note letters; we accept both.
        self.assertEqual(_pitch_to_midi("c4"), 60)

    def test_rest(self) -> None:
        # All rest spellings collapse to None (= silent gap, not a note).
        for r in ("R", "r", "rest", "REST", ""):
            self.assertIsNone(_pitch_to_midi(r))

    def test_bad_input(self) -> None:
        # Garbage in → None out (never raises). The LLM can emit weird
        # values; we want a silent rest, not a crash mid-melody.
        self.assertIsNone(_pitch_to_midi("H4"))   # invalid letter
        self.assertIsNone(_pitch_to_midi("C"))    # missing octave
        self.assertIsNone(_pitch_to_midi(None))   # type: ignore[arg-type]
        self.assertIsNone(_pitch_to_midi(60))     # type: ignore[arg-type]

    def test_out_of_range(self) -> None:
        # Below A1 (MIDI 33) and above C7 (MIDI 96) are rejected to keep
        # synthesis in the speaker's usable range (~55 Hz–2 kHz).
        self.assertIsNone(_pitch_to_midi("C0"))
        self.assertIsNone(_pitch_to_midi("C8"))


class Envelope(unittest.TestCase):
    """ADSR shaping. Guards the no-click contract: every note must
    ramp up from 0 and decay back to 0 so concatenated notes don't
    pop at the boundary, and the envelope length must equal `n` so
    the multiply-in-place against the waveform stays aligned."""

    def test_length_matches_n(self) -> None:
        # Envelope must match the waveform length exactly across realistic
        # note sizes — a mismatch silently truncates or pads the audio.
        for n in (800, 4_000, 64_000):
            env = MelodyPlayer._envelope(n, 16_000)
            self.assertEqual(env.shape, (n,))

    def test_starts_and_ends_at_zero(self) -> None:
        # First/last sample must be 0 — otherwise we get an audible click
        # at the start of the note and at the join with the next one.
        env = MelodyPlayer._envelope(16_000, 16_000)  # 1 s
        self.assertAlmostEqual(float(env[0]), 0.0, places=5)
        self.assertAlmostEqual(float(env[-1]), 0.0, places=5)

    def test_short_note_no_nan(self) -> None:
        # On a 50 ms note (the minimum allowed) attack/decay/release all
        # get clamped hard. Make sure that clamping never produces NaN,
        # which would poison the whole concatenated buffer.
        env = MelodyPlayer._envelope(800, 16_000)
        self.assertEqual(env.shape, (800,))
        self.assertFalse(np.isnan(env).any())


class ResolveNotes(unittest.TestCase):
    """LLM-input normaliser. This is where every safety bound lives:
    duration clamps, note-count cap, total-duration cap (matched to
    the `_run_async` 120 s join timeout), tempo→seconds conversion,
    and tolerance to garbage entries. Bugs here either crash the
    pipeline or let a runaway melody overlap the next queued action."""

    def test_seconds_mode(self) -> None:
        # No tempo → durations are in seconds directly.
        # 0.5 s at 16 kHz = 8000 samples, pitch C4 = MIDI 60.
        p = _player()
        out = p._resolve_notes([{"pitch": "C4", "duration": 0.5}], None)
        self.assertEqual(len(out), 1)
        midi, dur_s, n = out[0]
        self.assertEqual(midi, 60)
        self.assertAlmostEqual(dur_s, 0.5)
        self.assertEqual(n, 8_000)

    def test_tempo_mode_quarter_at_120bpm(self) -> None:
        # With a tempo, `duration` is in beats. A quarter note (1.0 beat)
        # at 120 BPM = 60 / 120 = 0.5 s. Validates the units conversion.
        p = _player()
        out = p._resolve_notes([{"pitch": "C4", "duration": 1.0}], 120.0)
        _, dur_s, _ = out[0]
        self.assertAlmostEqual(dur_s, 0.5)

    def test_clamp_short_duration(self) -> None:
        # If the LLM emits an absurdly short note (1 ms) we clamp up
        # to _MIN_DUR_S (50 ms) — below that the envelope can't fit.
        p = _player()
        out = p._resolve_notes([{"pitch": "C4", "duration": 0.001}], None)
        _, dur_s, _ = out[0]
        self.assertAlmostEqual(dur_s, _MIN_DUR_S)

    def test_clamp_long_duration(self) -> None:
        # And we clamp down for absurdly long notes (99 s → 4 s) so a
        # single note can't monopolise the speaker.
        p = _player()
        out = p._resolve_notes([{"pitch": "C4", "duration": 99.0}], None)
        _, dur_s, _ = out[0]
        self.assertAlmostEqual(dur_s, _MAX_DUR_S)

    def test_truncates_at_max_notes(self) -> None:
        # Hard cap on note count (memory + planning sanity). Anything
        # beyond _MAX_NOTES is silently dropped with a log warning.
        p = _player()
        notes = [{"pitch": "C4", "duration": 0.1}] * (_MAX_NOTES + 10)
        out = p._resolve_notes(notes, None)
        self.assertEqual(len(out), _MAX_NOTES)

    def test_caps_total_duration(self) -> None:
        # The per-note caps still allow a theoretical 64 × 4 s = 256 s
        # melody, which would outlast the 120 s _run_async join window
        # and let the next queued action overlap on the speaker.
        # _MAX_TOTAL_DUR_S exists exactly to prevent that, so we expect
        # truncation well before _MAX_NOTES is reached.
        p = _player()
        notes = [{"pitch": "C4", "duration": _MAX_DUR_S}] * _MAX_NOTES
        out = p._resolve_notes(notes, None)
        total = sum(d for _, d, _ in out)
        self.assertLessEqual(total, _MAX_TOTAL_DUR_S + 1e-6)
        self.assertLess(len(out), _MAX_NOTES)

    def test_rest_in_sequence(self) -> None:
        # A rest must survive the round-trip as midi=None (silence),
        # not be dropped — the synth needs it to keep the rhythm.
        p = _player()
        out = p._resolve_notes(
            [{"pitch": "C4", "duration": 0.2}, {"pitch": "R", "duration": 0.2}],
            None,
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], 60)
        self.assertIsNone(out[1][0])

    def test_skips_non_dict(self) -> None:
        # The LLM occasionally injects garbage items into the array.
        # We silently drop them rather than raising, keeping the rest
        # of the melody playable.
        p = _player()
        out = p._resolve_notes(
            [{"pitch": "C4", "duration": 0.2}, "garbage", 42],  # type: ignore[list-item]
            None,
        )
        self.assertEqual(len(out), 1)

    def test_bad_duration_defaults(self) -> None:
        # Non-numeric duration ('abc') falls back to the 0.25 s default
        # instead of crashing — single bad note shouldn't kill the melody.
        p = _player()
        out = p._resolve_notes([{"pitch": "C4", "duration": "abc"}], None)
        self.assertEqual(len(out), 1)
        _, dur_s, _ = out[0]
        self.assertAlmostEqual(dur_s, 0.25)


class Synthesize(unittest.TestCase):
    """End-to-end audio buffer. Guards the audio↔motion sync contract
    (sample count must equal the sum of per-note `n`, since `_dance`
    iterates the same `resolved` list) and the speaker-safety contract
    (no NaN, no clipping > ±1.0)."""

    def test_length_matches_sum_of_n(self) -> None:
        # Audio and motion share the same per-note `n` budget. If the
        # synthesised buffer length drifts from sum(n), the dance step
        # durations will desync from the audio.
        p = _player()
        notes = [
            {"pitch": "C4", "duration": 0.2},
            {"pitch": "R",  "duration": 0.1},
            {"pitch": "G4", "duration": 0.3},
        ]
        resolved = p._resolve_notes(notes, None)
        samples = p._synthesize(resolved)
        self.assertEqual(samples.size, sum(n for _, _, n in resolved))

    def test_rest_is_silence(self) -> None:
        # A rest must render to actual zeros — not a low-amplitude tone,
        # not random noise — so silence in the score is silence on the
        # speaker.
        p = _player()
        resolved = p._resolve_notes([{"pitch": "R", "duration": 0.1}], None)
        samples = p._synthesize(resolved)
        self.assertTrue(np.all(samples == 0.0))

    def test_no_nan_no_clipping(self) -> None:
        # NaN would propagate through the audio pipeline and could
        # damage the speaker. Clipping > ±1.0 would be audibly nasty.
        # Both are non-negotiable safety bounds on the output buffer.
        p = _player()
        resolved = p._resolve_notes(
            [{"pitch": "C5", "duration": 0.3}], None
        )
        samples = p._synthesize(resolved)
        self.assertFalse(np.isnan(samples).any())
        self.assertLessEqual(float(np.max(np.abs(samples))), 1.0)


if __name__ == "__main__":
    unittest.main()
