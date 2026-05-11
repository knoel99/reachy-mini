"""Scientific pitch notation parsing.

Single source of truth for the runtime (`melody.py`) and the offline
curation pipeline under `tools/humming/`.
"""
from __future__ import annotations

import re

_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Useful range for both the WORLD analyzer and physical singing.
MIN_MIDI = 33   # A1  (~55 Hz)
MAX_MIDI = 96   # C7  (~2093 Hz)


def pitch_to_midi(pitch: str) -> int | None:
    """Parse scientific pitch notation ('C4', 'F#5', 'Bb3') to MIDI.

    Returns None for rests ('R'/'rest') or out-of-range pitches.
    """
    if not isinstance(pitch, str):
        return None
    p = pitch.strip()
    if p.lower() in ("r", "rest", ""):
        return None
    m = _PITCH_RE.match(p)
    if not m:
        return None
    letter, accidental, octave_s = m.groups()
    semitone = _NOTE_OFFSETS[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi = 12 * (int(octave_s) + 1) + semitone
    if midi < MIN_MIDI or midi > MAX_MIDI:
        return None
    return midi


def pitch_to_hz(pitch: str) -> float | None:
    """Parse scientific pitch notation to Hz. None for rests / invalid."""
    midi = pitch_to_midi(pitch)
    if midi is None:
        return None
    return midi_to_hz(midi)


def midi_to_hz(midi: int) -> float:
    """MIDI number → frequency in Hz (A4=69, 440 Hz)."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))
