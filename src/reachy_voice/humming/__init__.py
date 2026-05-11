"""WORLD-vocoder humming synthesis for Reachy melodies.

Public API used by the runtime (`reachy_voice.melody`) and the offline
curation pipeline (`tools/humming/`).
"""
from __future__ import annotations

from .pitch import midi_to_hz, pitch_to_hz, pitch_to_midi
from .synth import HummingSynth
from .world_vocoder import analyze, render_note

__all__ = [
    "HummingSynth",
    "analyze",
    "midi_to_hz",
    "pitch_to_hz",
    "pitch_to_midi",
    "render_note",
]
