"""Test melodies — public-domain only, encoded as note dicts.

Schema (matches src/reachy_voice/melody.py):
  {"pitch": "C4", "duration": 0.5}    # duration in beats if tempo set, else seconds
"""
from __future__ import annotations


def _n(pitch: str, beats: float) -> dict:
    return {"pitch": pitch, "duration": beats}


# Joyeux Anniversaire / Happy Birthday (US public domain since 2016).
HAPPY_BIRTHDAY = dict(
    title="Joyeux Anniversaire",
    tempo_bpm=110,
    notes=[
        _n("C4", 0.75), _n("C4", 0.25), _n("D4", 1.0), _n("C4", 1.0),
        _n("F4", 1.0), _n("E4", 2.0),
        _n("C4", 0.75), _n("C4", 0.25), _n("D4", 1.0), _n("C4", 1.0),
        _n("G4", 1.0), _n("F4", 2.0),
        _n("C4", 0.75), _n("C4", 0.25), _n("C5", 1.0), _n("A4", 1.0),
        _n("F4", 1.0), _n("E4", 1.0), _n("D4", 2.0),
        _n("Bb4", 0.75), _n("Bb4", 0.25), _n("A4", 1.0), _n("F4", 1.0),
        _n("G4", 1.0), _n("F4", 2.0),
    ],
)

# Vive le vent / Jingle Bells (J. Pierpont, 1857 — public domain).
JINGLE_BELLS = dict(
    title="Vive le vent (Jingle Bells)",
    tempo_bpm=130,
    notes=[
        _n("E4", 1.0), _n("E4", 1.0), _n("E4", 2.0),
        _n("E4", 1.0), _n("E4", 1.0), _n("E4", 2.0),
        _n("E4", 1.0), _n("G4", 1.0), _n("C4", 1.5), _n("D4", 0.5),
        _n("E4", 4.0),
        _n("F4", 1.0), _n("F4", 1.0), _n("F4", 1.5), _n("F4", 0.5),
        _n("F4", 1.0), _n("E4", 1.0), _n("E4", 1.0), _n("E4", 0.5), _n("E4", 0.5),
        _n("E4", 1.0), _n("D4", 1.0), _n("D4", 1.0), _n("E4", 1.0),
        _n("D4", 2.0), _n("G4", 2.0),
    ],
)

# Frère Jacques (French folk, traditional — public domain).
FRERE_JACQUES = dict(
    title="Frère Jacques",
    tempo_bpm=110,
    notes=[
        _n("C4", 1.0), _n("D4", 1.0), _n("E4", 1.0), _n("C4", 1.0),
        _n("C4", 1.0), _n("D4", 1.0), _n("E4", 1.0), _n("C4", 1.0),
        _n("E4", 1.0), _n("F4", 1.0), _n("G4", 2.0),
        _n("E4", 1.0), _n("F4", 1.0), _n("G4", 2.0),
        _n("G4", 0.5), _n("A4", 0.5), _n("G4", 0.5), _n("F4", 0.5),
        _n("E4", 1.0), _n("C4", 1.0),
        _n("G4", 0.5), _n("A4", 0.5), _n("G4", 0.5), _n("F4", 0.5),
        _n("E4", 1.0), _n("C4", 1.0),
        _n("C4", 1.0), _n("G3", 1.0), _n("C4", 2.0),
        _n("C4", 1.0), _n("G3", 1.0), _n("C4", 2.0),
    ],
)

# Au clair de la lune (French folk, traditional — public domain).
AU_CLAIR_DE_LA_LUNE = dict(
    title="Au clair de la lune",
    tempo_bpm=120,
    notes=[
        _n("C4", 1.0), _n("C4", 1.0), _n("C4", 1.0), _n("D4", 1.0),
        _n("E4", 2.0), _n("D4", 2.0),
        _n("C4", 1.0), _n("E4", 1.0), _n("D4", 1.0), _n("D4", 1.0),
        _n("C4", 4.0),
        _n("C4", 1.0), _n("C4", 1.0), _n("C4", 1.0), _n("D4", 1.0),
        _n("E4", 2.0), _n("D4", 2.0),
        _n("C4", 1.0), _n("E4", 1.0), _n("D4", 1.0), _n("D4", 1.0),
        _n("C4", 4.0),
    ],
)

# Brahms Wiegenlied / Berceuse (Op. 49 No. 4, 1868 — public domain).
BRAHMS_LULLABY = dict(
    title="Berceuse de Brahms",
    tempo_bpm=80,
    notes=[
        _n("F4", 1.0), _n("F4", 1.0), _n("A4", 2.0),
        _n("F4", 1.0), _n("F4", 1.0), _n("A4", 2.0),
        _n("F4", 1.0), _n("A4", 1.0), _n("C5", 1.0), _n("Bb4", 1.0),
        _n("A4", 1.0), _n("G4", 1.0), _n("F4", 2.0),
        _n("E4", 1.0), _n("F4", 1.0), _n("G4", 2.0),
        _n("E4", 1.0), _n("F4", 1.0), _n("G4", 2.0),
    ],
)

# Ode à la joie / Ode to Joy (Beethoven 9th, 1824 — public domain).
ODE_TO_JOY = dict(
    title="Ode à la joie",
    tempo_bpm=120,
    notes=[
        _n("E4", 1.0), _n("E4", 1.0), _n("F4", 1.0), _n("G4", 1.0),
        _n("G4", 1.0), _n("F4", 1.0), _n("E4", 1.0), _n("D4", 1.0),
        _n("C4", 1.0), _n("C4", 1.0), _n("D4", 1.0), _n("E4", 1.0),
        _n("E4", 1.5), _n("D4", 0.5), _n("D4", 2.0),
        _n("E4", 1.0), _n("E4", 1.0), _n("F4", 1.0), _n("G4", 1.0),
        _n("G4", 1.0), _n("F4", 1.0), _n("E4", 1.0), _n("D4", 1.0),
        _n("C4", 1.0), _n("C4", 1.0), _n("D4", 1.0), _n("E4", 1.0),
        _n("D4", 1.5), _n("C4", 0.5), _n("C4", 2.0),
    ],
)

# Greensleeves (English folk, 16th century — public domain).
GREENSLEEVES = dict(
    title="Greensleeves",
    tempo_bpm=90,
    notes=[
        _n("A3", 1.0),
        _n("C4", 2.0), _n("D4", 1.0), _n("E4", 1.5), _n("F4", 0.5), _n("E4", 1.0),
        _n("D4", 2.0), _n("B3", 1.0), _n("G3", 1.5), _n("A3", 0.5), _n("B3", 1.0),
        _n("C4", 2.0), _n("A3", 1.0), _n("A3", 1.5), _n("Ab3", 0.5), _n("A3", 1.0),
        _n("B3", 2.0), _n("Ab3", 1.0), _n("E3", 3.0),
        _n("A3", 1.0),
        _n("C4", 2.0), _n("D4", 1.0), _n("E4", 1.5), _n("F4", 0.5), _n("E4", 1.0),
        _n("D4", 2.0), _n("B3", 1.0), _n("G3", 1.5), _n("A3", 0.5), _n("B3", 1.0),
        _n("C4", 1.0), _n("B3", 0.5), _n("A3", 0.5), _n("Ab3", 1.0), _n("F#3", 0.5), _n("Ab3", 0.5),
        _n("A3", 3.0),
    ],
)

# Twinkle, Twinkle / Ah vous dirai-je maman (French folk, public domain).
TWINKLE = dict(
    title="Ah vous dirai-je maman",
    tempo_bpm=110,
    notes=[
        _n("C4", 1.0), _n("C4", 1.0), _n("G4", 1.0), _n("G4", 1.0),
        _n("A4", 1.0), _n("A4", 1.0), _n("G4", 2.0),
        _n("F4", 1.0), _n("F4", 1.0), _n("E4", 1.0), _n("E4", 1.0),
        _n("D4", 1.0), _n("D4", 1.0), _n("C4", 2.0),
        _n("G4", 1.0), _n("G4", 1.0), _n("F4", 1.0), _n("F4", 1.0),
        _n("E4", 1.0), _n("E4", 1.0), _n("D4", 2.0),
        _n("G4", 1.0), _n("G4", 1.0), _n("F4", 1.0), _n("F4", 1.0),
        _n("E4", 1.0), _n("E4", 1.0), _n("D4", 2.0),
    ],
)


ALL_MELODIES = {
    "happy_birthday": HAPPY_BIRTHDAY,
    "jingle_bells": JINGLE_BELLS,
    "frere_jacques": FRERE_JACQUES,
    "au_clair_de_la_lune": AU_CLAIR_DE_LA_LUNE,
    "brahms_lullaby": BRAHMS_LULLABY,
    "ode_to_joy": ODE_TO_JOY,
    "greensleeves": GREENSLEEVES,
    "twinkle": TWINKLE,
}
