"""Original Latin-pop hook + 16-keyframe robot choreography.

Exposed to the LLM as the standalone tool `play_macarena()`.
"""

from __future__ import annotations

from ._spec import MelodyBundle


_DESCRIPTION = (
    "Joue UN PASTICHE original façon Macarena (hook latino-pop des "
    "années 90), accompagné d'une chorégraphie scriptée corps + "
    "antennes (16 keyframes verrouillées sur le rythme). À utiliser "
    "dès que l'intention de l'utilisateur renvoie à cette chanson, "
    "même de façon détournée : « Macarena », « Heeeey Macarena », "
    "« danse la Macarena », « la danse des bras », « Los del Río », "
    "un hook latino des années 90. Aucun paramètre. N'émets PAS "
    "`move_sequence`, `play_emotion` ni un autre outil audio en "
    "parallèle — la chorégraphie est déjà incluse."
)

# 16 quarter notes at 103 BPM ⇒ ~9.3 s total. Em arpeggio ascending then
# descending (bars 1–2), then a B7 chord walk resolving to Em (bars 3–4).
# Tessitura E4–D5 (MIDI 64–74), within a single octave.
_NOTES: list[dict] = [
    {"pitch": "E4",  "duration": 1.0},
    {"pitch": "G4",  "duration": 1.0},
    {"pitch": "B4",  "duration": 1.0},
    {"pitch": "D5",  "duration": 1.0},
    {"pitch": "D5",  "duration": 1.0},
    {"pitch": "B4",  "duration": 1.0},
    {"pitch": "G4",  "duration": 1.0},
    {"pitch": "E4",  "duration": 1.0},
    {"pitch": "F#4", "duration": 1.0},
    {"pitch": "A4",  "duration": 1.0},
    {"pitch": "B4",  "duration": 1.0},
    {"pitch": "D5",  "duration": 1.0},
    {"pitch": "D5",  "duration": 1.0},
    {"pitch": "A4",  "duration": 1.0},
    {"pitch": "F#4", "duration": 1.0},
    {"pitch": "E4",  "duration": 1.0},
]

# One keyframe per note. Same field set as `move_sequence` steps
# (roll/pitch/yaw/x/y/z/body_yaw/antenna_left/right, degrees / mm).
# Missing fields default to 0 → neutral.
_KEYFRAMES: list[dict] = [
    {"yaw":  20.0, "antenna_right":  60.0},
    {"yaw": -20.0, "antenna_left":   60.0},
    {"yaw":  20.0, "antenna_right": -60.0},
    {"yaw": -20.0, "antenna_left":  -60.0},
    {"roll":  15.0, "antenna_right": -45.0},
    {"roll": -15.0, "antenna_left":  -45.0},
    {"pitch": -15.0, "antenna_left":  45.0, "antenna_right":  45.0},
    {"pitch":  15.0, "antenna_left": -45.0, "antenna_right": -45.0},
    {"body_yaw":  25.0},
    {"body_yaw": -25.0},
    {"body_yaw":  25.0, "z": -10.0},
    {"body_yaw": -25.0, "z": -10.0},
    {"body_yaw":  20.0, "roll": -8.0},
    {"body_yaw": -20.0, "roll":  8.0},
    {"body_yaw":  20.0, "roll": -8.0},
    {"body_yaw":  90.0, "antenna_left": 90.0, "antenna_right": 90.0,
     "pitch": -10.0},
]


BUNDLE = MelodyBundle(
    name="macarena",
    description=_DESCRIPTION,
    bpm=103.0,
    notes=_NOTES,
    keyframes=_KEYFRAMES,
)
