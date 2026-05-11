"""Runtime humming synthesizer.

Loads the packaged voiced sample + manifest once, runs WORLD analysis
once, and serves per-note renders to `MelodyPlayer`.
"""
from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .pitch import midi_to_hz
from .world_vocoder import analyze, render_note

_DATA_PKG = "reachy_voice.data.humming"


class HummingSynth:
    """Per-note WORLD renderer with a cached analysis of one sample."""

    def __init__(self, target_rate: int) -> None:
        self.target_rate = int(target_rate)
        self.sample, self.sample_f0 = self._load_sample(self.target_rate)
        self.pre_analyzed = analyze(self.sample, self.target_rate)

    @staticmethod
    def _load_sample(target_rate: int) -> tuple[np.ndarray, float]:
        manifest_text = (
            files(_DATA_PKG).joinpath("manifest.json").read_text()
        )
        manifest = json.loads(manifest_text)
        wav_bytes = (
            files(_DATA_PKG).joinpath(manifest["sample_file"]).read_bytes()
        )
        import io
        src_rate, data = wavfile.read(io.BytesIO(wav_bytes))

        if data.dtype == np.int16:
            samples = data.astype(np.float32) / 32768.0
        elif data.dtype == np.float32:
            samples = data
        else:
            samples = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        if samples.ndim == 2:
            samples = samples.mean(axis=1).astype(np.float32, copy=False)

        if src_rate != target_rate:
            g = np.gcd(src_rate, target_rate)
            samples = resample_poly(
                samples, target_rate // g, src_rate // g
            ).astype(np.float32, copy=False)
        return samples, float(manifest["mean_f0_hz"])

    def render(
        self, midi: int | None, dur_s: float, n: int
    ) -> np.ndarray:
        """Render one note. Rests (midi is None) → silence of n samples."""
        if midi is None:
            return np.zeros(n, dtype=np.float32)
        return render_note(
            self.sample,
            self.sample_f0,
            midi_to_hz(midi),
            dur_s,
            self.target_rate,
            pre_analyzed=self.pre_analyzed,
        )
