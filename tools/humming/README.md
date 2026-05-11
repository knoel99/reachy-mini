# Reachy Humming — offline tooling

This folder produces the voiced sample that the live `reachy_voice` stack
uses to hum melodies. WORLD vocoder pitch-shift in mono mode (one fixed
source sample for the whole melody, formants preserved).

The runtime synthesizer (`reachy_voice.humming.HummingSynth`,
`src/reachy_voice/humming/`) is the single source of truth for analyze /
render_note / pitch parsing — this tooling imports from there.

## Install (lab-only deps)

```bash
pip install -r tools/humming/requirements.txt
```

`reachy_voice` itself already pulls `pyworld`; this only adds `librosa`
and `soundfile`, needed to load arbitrarily-formatted source WAVs at
curation time.

## Pipeline

```bash
# 1. (one-off) Extract the 81 raw emotion WAVs from the HuggingFace dataset.
python scripts/export_emotion_sounds.py tools/humming/data/sources

# 2. Auto-curate ~8 voiced base samples (harvest f0, vowel-family allowlist).
python tools/humming/curate.py

# 3. Render every melody in melodies.py to data/outputs/<melody>.wav
#    (sanity-check the hum quality before promoting a new sample).
python tools/humming/synthesize.py
```

## Promoting a new sample to the runtime

The live `MelodyPlayer` reads `src/reachy_voice/data/humming/sample.wav`
plus `manifest.json`. To replace it:

1. Run the pipeline above.
2. Pick the sample whose hum sounds best across the melody range (the
   "mono" pick — historically index 0).
3. Copy it to the runtime asset path and update the manifest:
   ```bash
   cp tools/humming/data/samples/sample_00_*.wav \
      src/reachy_voice/data/humming/sample.wav
   ```
   Then edit `src/reachy_voice/data/humming/manifest.json` to match the
   new sample's `mean_f0_hz` and `sample_rate` (from `curation.json`).

## Layout

```
tools/humming/
  curate.py            # picks N voiced segments + writes curation.json
  synthesize.py        # WORLD × mono × all melodies → data/outputs/
  audio_io.py          # librosa-based loader (lab-only)
  config.py            # paths + lab sample rate
  melodies.py          # 8 public-domain melodies as note JSON
  data/
    sources/           # 81 raw WAVs (gitignored)
    samples/           # 8 curated voiced excerpts + curation.json (committed)
    outputs/           # one WAV per melody (gitignored)
```

## Notes

- Lab sample rate is 22 050 Hz (`config.TARGET_RATE`). The runtime
  resamples on its own when loading `sample.wav`.
- The mono pick (sample #0 in `curation.json`) is the most consistent
  source across the whole melodic range. Multi-sample picking was tried
  and discarded — it produced timbre breaks at every note that switched
  source, defeating the WORLD-preserved-formant property.
