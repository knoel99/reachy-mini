"""Disney-style ascending ballad hook + 16-keyframe robot choreography.

Exposed to the LLM as the standalone tool `play_let_it_go()`.
"""

from __future__ import annotations

from ._spec import MelodyBundle


_DESCRIPTION = (
    "Joue UN PASTICHE original façon Let It Go (ballade Disney en "
    "fa mineur, climax sur saut d'octave), accompagné d'une "
    "chorégraphie scriptée tête + corps + antennes (16 keyframes "
    "verrouillées sur le rythme — les deux climax « go ! » lèvent "
    "la tête et déploient les antennes à 90°). À utiliser dès que "
    "l'intention de l'utilisateur renvoie à cette chanson ou au "
    "film, même de façon détournée : « Let It Go », « Libérée "
    "délivrée », « la chanson d'Elsa », « la musique d'Elsa », "
    "« la chanson de la Reine des Neiges », « Frozen », « La Reine "
    "des Neiges », ou toute référence à Elsa qui appelle clairement "
    "SA chanson-phare. Aucun paramètre. N'émets PAS `move_sequence`, "
    "`play_emotion` ni un autre outil audio en parallèle — la "
    "chorégraphie est déjà incluse."
)

# Two chorus phrases (16 notes). Tessitura Eb4–Ab5 (MIDI 63–80). The
# defining gesture is the octave leap Eb4 → Eb5 on the first "go!" and
# the further reach to Ab5 on the second — both held over a beat-and-a-
# half so the climax registers before the descending tail releases the
# tension back down to Ab4.
_NOTES: list[dict] = [
    {"pitch": "Eb4", "duration": 0.5},   # "Let"
    {"pitch": "Eb4", "duration": 0.5},   # "it"
    {"pitch": "Eb5", "duration": 1.5},   # "go!"  (octave leap)
    {"pitch": "R",   "duration": 0.5},   # (held silence)
    {"pitch": "Eb4", "duration": 0.5},   # "let"
    {"pitch": "Eb4", "duration": 0.5},   # "it"
    {"pitch": "Ab5", "duration": 1.5},   # "go!"  (climax, perfect 4th higher)
    {"pitch": "R",   "duration": 0.5},   # (held silence)
    {"pitch": "Bb4", "duration": 0.5},   # "Can't"
    {"pitch": "C5",  "duration": 0.5},   # "hold"
    {"pitch": "Db5", "duration": 0.5},   # "it"
    {"pitch": "C5",  "duration": 0.5},   # "back"
    {"pitch": "Bb4", "duration": 0.5},   # "an-"
    {"pitch": "Ab4", "duration": 0.5},   # "y-"
    {"pitch": "G4",  "duration": 0.5},   # "more"
    {"pitch": "Ab4", "duration": 1.5},   # (resolution)
]

# Two climax frames (indices 2 and 6) tilt the head right back, lift on
# z, and flare both antennas to ~90° so the octave leap registers
# visually; the two rests after each climax (indices 3 and 7) hold the
# pose so the audience reads it.
_KEYFRAMES: list[dict] = [
    {"pitch":  12.0, "antenna_left": -25.0, "antenna_right": -25.0},
    {"pitch":  10.0, "antenna_left": -15.0, "antenna_right": -15.0},
    {"pitch": -28.0, "z":  25.0, "antenna_left": 85.0, "antenna_right": 85.0},
    {"pitch": -25.0, "z":  22.0, "antenna_left": 80.0, "antenna_right": 80.0},
    {"pitch":  10.0, "body_yaw":  20.0,
     "antenna_left": -15.0, "antenna_right": -15.0},
    {"pitch":   5.0, "body_yaw":  25.0,
     "antenna_left":   0.0, "antenna_right":   0.0},
    {"pitch": -28.0, "z":  28.0, "body_yaw": -30.0,
     "antenna_left":  90.0, "antenna_right":  90.0},
    {"pitch": -25.0, "z":  25.0, "body_yaw": -25.0,
     "antenna_left":  85.0, "antenna_right":  85.0},
    {"roll":  10.0, "body_yaw":  15.0, "antenna_right": 40.0},
    {"roll": -10.0, "body_yaw": -15.0, "antenna_left":  40.0},
    {"roll":  10.0, "body_yaw":  15.0, "antenna_right": 30.0},
    {"roll": -10.0, "body_yaw": -15.0, "antenna_left":  30.0},
    {"roll":   5.0, "body_yaw":  10.0, "antenna_right": 20.0},
    {"roll":  -5.0, "body_yaw": -10.0, "antenna_left":  20.0},
    {"pitch":  15.0, "antenna_left": -20.0, "antenna_right": -20.0},
    {"pitch":  20.0, "z": -15.0,
     "antenna_left": -40.0, "antenna_right": -40.0},
]


BUNDLE = MelodyBundle(
    name="let_it_go",
    description=_DESCRIPTION,
    bpm=137.0,
    notes=_NOTES,
    keyframes=_KEYFRAMES,
)
