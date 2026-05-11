"""Melody synthesis: LLM-planned notes rendered as a sine wave (sync).

`play(notes, tempo_bpm=None)` synthesises the whole buffer in one
shot, pushes it via `mini.media.push_audio_sample` (the same channel
used by `EmotionPlayer`), then drives a head + antenna dance whose
per-step durations match the note timeline so motion stays locked to
the rhythm.

Like `EmotionPlayer`, this module never spawns threads — concurrency
is the caller's job (`RobotActions._run_async`). A monotonic
`_speaking_until` timestamp is exposed via `is_speaking()` so the mic
loop gates capture and avoids re-injecting the robot's own audio.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import numpy as np

from ._log import log
from .tools import _make_head_pose

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


_SPEAKING_PAD_S = 0.5

_MAX_NOTES = 64
_MIN_DUR_S = 0.05
_MAX_DUR_S = 4.0
# Total audio length cap. Kept ≤ the `_run_async` join timeout in
# `_actions.py` so a max-length melody can't be still playing when the
# next queued action starts pushing audio onto the same speaker.
_MAX_TOTAL_DUR_S = 120.0
_MIN_BPM = 30.0
_MAX_BPM = 300.0
_MIN_MIDI = 33   # A1  (~55 Hz)
_MAX_MIDI = 96   # C7  (~2093 Hz)
_AMPLITUDE = 0.5

# Additive timbre: fundamental + 2nd + 3rd harmonics. Gives a flute-ish
# tone that's far less sterile than a pure sine without ever needing a
# soundfont.
_HARMONICS = ((1, 1.00), (2, 0.30), (3, 0.15))
_HARMONIC_SUM = sum(a for _, a in _HARMONICS)

# Vibrato: ±5 cents at 5 Hz. 5 cents = 2^(5/1200) - 1 ≈ 0.00289.
_VIBRATO_DEPTH = 0.00289
_VIBRATO_HZ = 5.0
# Don't apply vibrato to very short notes (it sounds like detuning).
_VIBRATO_MIN_DUR_S = 0.20

# ADSR envelope (in seconds, before per-note clamping).
_ATTACK_S = 0.015
_DECAY_S = 0.050
_SUSTAIN_LVL = 0.7
_RELEASE_S = 0.080

# Dance choreography. Antennas flap on every note (alternating sign);
# head rolls slightly side to side with amplitude scaled by pitch.
_DANCE_ANTENNA_BASE_DEG = 12.0
_DANCE_ANTENNA_GAIN_DEG = 35.0
_DANCE_HEAD_ROLL_DEG = 6.0
_DANCE_HEAD_PITCH_DEG = 3.0
# Amplitude floor so silent / rest steps still wiggle a little.
_DANCE_MIN_AMP = 0.25
# Pitch range mapped to amplitude 0..1 (MIDI 55=G3 through 84=C6).
_DANCE_PITCH_LOW_MIDI = 55
_DANCE_PITCH_HIGH_MIDI = 84

_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _pitch_to_midi(pitch: str) -> int | None:
    """Parse scientific pitch notation ('C4', 'F#5', 'Bb3') to a MIDI number.

    Returns None for rests ('R'/'rest') or out-of-range pitches.
    """
    if not isinstance(pitch, str):
        return None
    p = pitch.strip()
    if p.lower() in ("r", "rest", ""):
        return None
    m = _PITCH_RE.match(p)
    if not m:
        log(f"[melody] bad pitch: {pitch!r}")
        return None
    letter, accidental, octave_s = m.groups()
    semitone = _NOTE_OFFSETS[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi = 12 * (int(octave_s) + 1) + semitone
    if midi < _MIN_MIDI or midi > _MAX_MIDI:
        log(f"[melody] pitch {pitch!r} out of range (midi={midi})")
        return None
    return midi


class MelodyPlayer:
    """Renders LLM-supplied note sequences as a sine wave (sync)."""

    DEFAULT_TARGET_RATE = 16_000  # ReSpeaker XVF3800 default

    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        self._speaking_until: float = 0.0

        rate = -1
        try:
            rate = mini.media.get_output_audio_samplerate()
        except Exception as e:
            log(f"[melody] get_output_audio_samplerate failed: {e}")
        self._target_rate = rate if rate and rate > 0 else self.DEFAULT_TARGET_RATE

    def is_speaking(self) -> bool:
        """True while a previously pushed melody is still playing."""
        return time.monotonic() < self._speaking_until

    @staticmethod
    def _envelope(n: int, rate: int) -> np.ndarray:
        """Linear ADSR envelope, clamped to fit very short notes."""
        a = min(int(rate * _ATTACK_S), n // 4)
        d = min(int(rate * _DECAY_S), max(0, (n - a) // 3))
        r = min(int(rate * _RELEASE_S), max(0, n - a - d))
        s_len = max(0, n - a - d - r)
        env = np.empty(n, dtype=np.float32)
        if a > 0:
            env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
        if d > 0:
            env[a:a + d] = np.linspace(
                1.0, _SUSTAIN_LVL, d, dtype=np.float32
            )
        if s_len > 0:
            env[a + d:a + d + s_len] = _SUSTAIN_LVL
        if r > 0:
            env[a + d + s_len:] = np.linspace(
                _SUSTAIN_LVL if d > 0 or s_len > 0 else 1.0,
                0.0, r, dtype=np.float32,
            )
        return env

    @staticmethod
    def _render_note(
        freq: float, dur_s: float, n: int, rate: int
    ) -> np.ndarray:
        """Sum-of-harmonics tone with optional vibrato and ADSR."""
        t = np.arange(n, dtype=np.float32) / float(rate)
        if dur_s >= _VIBRATO_MIN_DUR_S:
            vib = 1.0 + _VIBRATO_DEPTH * np.sin(
                2.0 * np.pi * _VIBRATO_HZ * t
            )
            phase = 2.0 * np.pi * freq * np.cumsum(vib) / float(rate)
        else:
            phase = 2.0 * np.pi * freq * t

        wave = np.zeros(n, dtype=np.float32)
        for k, amp in _HARMONICS:
            wave += amp * np.sin(k * phase)
        wave *= _AMPLITUDE / _HARMONIC_SUM
        wave *= MelodyPlayer._envelope(n, rate)
        return wave.astype(np.float32, copy=False)

    def _resolve_notes(
        self, notes: list[dict], tempo_bpm: float | None
    ) -> list[tuple[int | None, float, int]]:
        """Normalise the LLM-supplied notes into ``(midi, dur_s, n_samples)``.

        Shared by audio synthesis and motion choreography so the two
        timelines stay sample-accurate.
        """
        rate = self._target_rate
        beat_s: float | None = None
        if tempo_bpm is not None:
            try:
                bpm = float(tempo_bpm)
            except (TypeError, ValueError):
                bpm = 120.0
            bpm = max(_MIN_BPM, min(_MAX_BPM, bpm))
            beat_s = 60.0 / bpm

        if len(notes) > _MAX_NOTES:
            log(f"[melody] {len(notes)} notes > {_MAX_NOTES}, truncating")
            notes = notes[:_MAX_NOTES]

        out: list[tuple[int | None, float, int]] = []
        total_dur = 0.0
        for note in notes:
            if not isinstance(note, dict):
                continue
            pitch = note.get("pitch", "R")
            try:
                raw_dur = float(note.get("duration", 0.25))
            except (TypeError, ValueError):
                raw_dur = 0.25
            dur_s = raw_dur * beat_s if beat_s is not None else raw_dur
            dur_s = max(_MIN_DUR_S, min(_MAX_DUR_S, dur_s))
            n = int(rate * dur_s)
            if n <= 0:
                continue
            if total_dur + dur_s > _MAX_TOTAL_DUR_S:
                log(
                    f"[melody] total duration would exceed "
                    f"{_MAX_TOTAL_DUR_S}s, truncating at {len(out)} notes"
                )
                break
            total_dur += dur_s
            out.append((_pitch_to_midi(pitch), dur_s, n))
        return out

    def _synthesize(
        self, resolved: list[tuple[int | None, float, int]]
    ) -> np.ndarray:
        rate = self._target_rate
        chunks: list[np.ndarray] = []
        for midi, dur_s, n in resolved:
            if midi is None:
                chunks.append(np.zeros(n, dtype=np.float32))
                continue
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            chunks.append(self._render_note(freq, dur_s, n, rate))

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _dance(
        self, resolved: list[tuple[int | None, float, int]]
    ) -> None:
        """Drive head + antennas in lockstep with the synthesised audio.

        One ``goto_target`` step per note; antennas alternate flap
        direction; pitch height drives amplitude. The sum of step
        durations equals the audio length, so motion stays locked to
        the rhythm.
        """
        if not resolved:
            return

        from reachy_mini.reachy_mini import (
            INIT_ANTENNAS_JOINT_POSITIONS,
            INIT_HEAD_POSE,
        )

        pitch_span = _DANCE_PITCH_HIGH_MIDI - _DANCE_PITCH_LOW_MIDI
        try:
            for i, (midi, dur_s, _n) in enumerate(resolved):
                sign = 1.0 if i % 2 == 0 else -1.0
                if midi is None:
                    amp = _DANCE_MIN_AMP
                else:
                    norm = (midi - _DANCE_PITCH_LOW_MIDI) / pitch_span
                    amp = max(_DANCE_MIN_AMP, min(1.0, norm))

                antenna_l_deg = sign * (
                    _DANCE_ANTENNA_BASE_DEG + amp * _DANCE_ANTENNA_GAIN_DEG
                )
                antenna_r_deg = -antenna_l_deg
                roll_deg = sign * _DANCE_HEAD_ROLL_DEG * amp
                pitch_deg = -_DANCE_HEAD_PITCH_DEG * amp

                pose = _make_head_pose(roll_deg=roll_deg, pitch_deg=pitch_deg)
                antennas = [
                    np.deg2rad(antenna_l_deg),
                    np.deg2rad(antenna_r_deg),
                ]
                try:
                    self.mini.goto_target(
                        pose, antennas=antennas, duration=dur_s
                    )
                except Exception as e:
                    log(f"[melody] dance step failed: {e}")
                    return
        finally:
            try:
                self.mini.goto_target(
                    INIT_HEAD_POSE,
                    antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                    duration=0.4,
                )
            except Exception as e:
                log(f"[melody] return to neutral failed: {e}")

    def play(
        self, notes: list[dict], tempo_bpm: float | None = None
    ) -> bool:
        """Synthesise + push audio, dance to the rhythm, then block
        until the audio has played out."""
        if not notes:
            return False

        resolved = self._resolve_notes(notes, tempo_bpm)
        samples = self._synthesize(resolved)
        if samples.size == 0:
            log("[melody] empty buffer after synthesis")
            return False

        sound_dur = samples.size / self._target_rate
        t0 = time.monotonic()
        self._speaking_until = t0 + sound_dur + _SPEAKING_PAD_S
        try:
            try:
                self.mini.media.push_audio_sample(samples)
            except Exception as e:
                log(f"[melody] push_audio_sample failed: {e}")
                return False
            self._dance(resolved)
            rem = self._speaking_until - time.monotonic()
            if rem > 0:
                time.sleep(rem)
        finally:
            self._speaking_until = 0.0
        log(
            f"[melody] {len(notes)} notes, {sound_dur:.2f}s done in "
            f"{time.monotonic() - t0:.2f}s"
        )
        return True
