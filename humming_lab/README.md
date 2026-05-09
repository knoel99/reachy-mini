# Reachy Humming Lab

Render melodies as Reachy hums them — using the WORLD vocoder over a curated
Reachy emotion sample. Single deterministic path: WORLD pitch-shift in mono
mode (one fixed source sample for the whole melody, formants preserved).

## Pipeline

```bash
# 1. (one-off) Extract the 81 raw emotion WAVs from the HuggingFace dataset.
python scripts/export_emotion_sounds.py humming_lab/data/sources

# 2. Auto-curate ~8 voiced base samples (harvest f0, vowel-family allowlist).
python humming_lab/scripts/02_curate_samples.py

# 3. Render every melody in src/melodies.py to data/outputs/<melody>.wav.
python humming_lab/scripts/03_synthesize.py
```

## Layout

```
humming_lab/
  scripts/
    02_curate_samples.py    # picks N voiced segments + writes curation.json
    03_synthesize.py        # WORLD × mono × all melodies → data/outputs/
  src/
    audio_io.py
    config.py
    melodies.py             # 8 public-domain melodies as note JSON
    pitch.py                # scientific pitch ↔ Hz
    methods/
      world_vocoder.py      # analyze() + render_note() via pyworld
      _envelope.py          # shared ADSR
  data/
    sources/                # 81 raw WAVs
    samples/                # 8 curated voiced excerpts + curation.json
    outputs/                # one WAV per melody
```

## Notes

- Sample rate is 22 050 Hz internally. The robot's media pipeline resamples
  on its own when the buffer is pushed via `mini.media.push_audio_sample`.
- The mono pick (sample #0 in `curation.json`) is the most consistent
  source across the whole melodic range. Multi-sample picking was tried
  and discarded — it produced timbre breaks at every note that switched
  source, defeating the WORLD-preserved-formant property.
- The melody schema matches `src/reachy_voice/melody.py` so plugging the
  rendered audio into the live Reachy stack only requires swapping the
  sine synthesiser for `world_vocoder.render_note(...)`.
