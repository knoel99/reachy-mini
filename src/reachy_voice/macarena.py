"""Original Latin-pop hook + 16-keyframe robot choreography.

This module is the implementation behind `play_melody(melody_id='macarena')`.
The label refers to the *user request* the feature is designed to serve
(prompt: "danse la Macarena"). The audio that actually plays is a brand-new
musical phrase composed for this project — a generic Em-chord arpeggio
followed by a B7→Em cadence walk, i.e. a first-year-music-theory exercise.
It is not derived from, and bears no melodic resemblance to, the copyrighted
song by Los del Río (1993, Sony BMG); only the tempo bracket (~103 BPM, 4/4)
and minor-key Latin-pop *feeling* are evoked, neither of which are
protectable in isolation. The choreography is similarly a generic 16-beat
Latin-pop dance mapped onto Reachy's 9 DoFs (head 6 + body_yaw + antennas
×2) — there are no arms, so iconic arm gestures are mimed via antenna
flicks, head turns and body sway.
"""

from __future__ import annotations

MACARENA_BPM = 103.0

# 16 quarter notes at 103 BPM ⇒ ~9.3 s total. Em arpeggio ascending then
# descending (bars 1–2), then a B7 chord walk resolving to Em (bars 3–4).
# Tessitura E4–D5 (MIDI 64–74), within a single octave.
MACARENA_MELODY: list[dict] = [
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

# One keyframe per note. Each dict uses the same field names accepted by
# `_play_sequence` (roll/pitch/yaw/x/y/z/body_yaw/antenna_left/right, all
# in degrees / mm). Missing fields default to 0 → neutral.
MACARENA_KEYFRAMES: list[dict] = [
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

assert len(MACARENA_MELODY) == len(MACARENA_KEYFRAMES), (
    "keyframes and notes must be 1:1 aligned"
)
