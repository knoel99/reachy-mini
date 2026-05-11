"""Robust WAV loader.

scipy.io.wavfile.read() rejects WAVs whose nAvgBytesPerSec field is not the
canonical product (some of the Reachy emotion clips trip this). We fall back
to soundfile, which is permissive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: str | Path, target_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV as float32 mono. Optionally resample to target_rate."""
    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32, copy=False)
    if target_rate is not None and sr != target_rate:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_rate)
        sr = target_rate
    return data.astype(np.float32, copy=False), sr


def save_wav(path: str | Path, samples: np.ndarray, rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
    sf.write(str(path), samples, rate, subtype="PCM_16")
