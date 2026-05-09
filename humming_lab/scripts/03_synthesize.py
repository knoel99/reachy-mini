"""Render every melody in `melodies.py` with WORLD vocoder, mono mode.

For each melody:
  - load the LLM-style note list,
  - pick the default mono sample (sample #0),
  - render each note with WORLD (formant-preserving pitch shift),
  - concatenate notes with a 30 ms gap, normalise peak, save WAV.

Output: data/outputs/<melody_key>.wav
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audio_io import load_mono, save_wav  # noqa: E402
from config import OUTPUTS_DIR, SAMPLES_DIR  # noqa: E402
from melodies import ALL_MELODIES  # noqa: E402
from methods import world_vocoder  # noqa: E402
from pitch import pitch_to_hz  # noqa: E402


DEFAULT_MONO_SAMPLE = 0
SILENCE_GAP_S = 0.030


def load_curation() -> tuple[list[dict], list[np.ndarray], int]:
    with open(SAMPLES_DIR / "curation.json") as f:
        cur = json.load(f)
    samples_meta = cur["samples"]
    rate = cur["target_rate"]
    samples_audio = [
        load_mono(SAMPLES_DIR / s["sample_file"], target_rate=rate)[0]
        for s in samples_meta
    ]
    return samples_meta, samples_audio, rate


def melody_to_seconds(melody: dict) -> list[tuple[str, float, float | None]]:
    bpm = float(melody.get("tempo_bpm", 120))
    beat_s = 60.0 / max(30.0, min(300.0, bpm))
    out = []
    for note in melody["notes"]:
        dur_s = max(0.05, min(2.0, float(note["duration"]) * beat_s))
        freq = pitch_to_hz(note["pitch"])
        out.append((note["pitch"], dur_s, freq))
    return out


def render_melody(
    melody: dict,
    sample: np.ndarray,
    sample_f0: float,
    rate: int,
    pre_analyzed: tuple,
) -> np.ndarray:
    seq = melody_to_seconds(melody)
    chunks: list[np.ndarray] = []
    silence_gap_n = int(SILENCE_GAP_S * rate)
    for _, dur_s, freq in seq:
        n = int(dur_s * rate)
        if freq is None:
            chunks.append(np.zeros(n, dtype=np.float32))
            continue
        chunk = world_vocoder.render_note(
            sample, sample_f0, freq, dur_s, rate,
            pre_analyzed=pre_analyzed,
        )
        chunks.append(chunk)
        if silence_gap_n > 0:
            chunks.append(np.zeros(silence_gap_n, dtype=np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def main() -> int:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    samples_meta, samples_audio, rate = load_curation()
    sample = samples_audio[DEFAULT_MONO_SAMPLE]
    sample_meta = samples_meta[DEFAULT_MONO_SAMPLE]
    sample_f0 = float(sample_meta["mean_f0_hz"])
    print(
        f"Source: #{DEFAULT_MONO_SAMPLE} {sample_meta['source']} "
        f"({sample_meta['duration_s']}s @ f0≈{sample_f0:.0f} Hz)"
    )
    pre = world_vocoder.analyze(sample, rate)

    t_global = time.monotonic()
    for melody_key, melody in ALL_MELODIES.items():
        t0 = time.monotonic()
        wav = render_melody(melody, sample, sample_f0, rate, pre)
        peak = float(np.max(np.abs(wav)) + 1e-9)
        wav = wav * (0.85 / peak)
        out_path = OUTPUTS_DIR / f"{melody_key}.wav"
        save_wav(out_path, wav, rate)
        print(
            f"  {melody_key:<22s}  {wav.size / rate:5.1f}s  "
            f"({time.monotonic() - t0:.2f}s render)"
        )
    print(f"\nDone in {time.monotonic() - t_global:.1f}s. "
          f"Wrote {len(ALL_MELODIES)} WAVs to {OUTPUTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
