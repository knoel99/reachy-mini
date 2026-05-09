"""Scientific pitch <-> Hz, plus per-note picking helpers."""
from __future__ import annotations

import re

_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_to_hz(pitch: str) -> float | None:
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
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def cents(target_hz: float, source_hz: float) -> float:
    """Pitch distance in cents (1200 cents = 1 octave)."""
    import math
    return 1200.0 * math.log2(target_hz / source_hz)


def pick_nearest_sample(target_hz: float, samples: list[dict]) -> int:
    """Return index of the sample whose mean f0 is closest to target."""
    import math
    best_i, best_d = 0, float("inf")
    for i, s in enumerate(samples):
        d = abs(math.log2(target_hz / float(s["mean_f0_hz"])))
        if d < best_d:
            best_d, best_i = d, i
    return best_i
