"""Auto-curate base voiced samples from the 81 extracted WAVs.

Strategy
--------
1. Use pyworld.harvest (not dio) — significantly more accurate on cartoon-y
   vocalisations and far less prone to octave-doubling errors.
2. Constrain f0 search to [80, 350] Hz so a robotic squeak that DIO would
   have called 700 Hz is now correctly detected at 350 Hz (its likely true
   f0) or rejected.
3. Allow segments up to 3 seconds (was 0.9 s). Long stable kernels are what
   WORLD needs to sound clean.
4. Bias the picker toward emotion families likely to contain held vowels
   (cheerful, proud, loving, calming, …) and away from rage / frustration /
   disgust (growls, distortion).
5. Pick at most one segment per family and spread picks across the f0 range
   so the multi-sample rendering has good per-note coverage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyworld as pw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audio_io import load_mono, save_wav  # noqa: E402
from config import SOURCES_DIR, SAMPLES_DIR, TARGET_RATE  # noqa: E402


FRAME_MS = 5.0
F0_FLOOR = 80.0
F0_CEIL = 350.0          # cap to avoid octave-doubling on robotic timbres
MIN_SEG_S = 0.50
# Lift the cap: WORLD benefits from longer voiced material (more natural
# micro-variation in the spectral envelope). Methods A/B don't suffer either.
MAX_SEG_S = 5.00
N_TARGET = 8

# Emotion families biased toward held-vowel / sustained-tone content.
# Rough hand-pick from the dataset's emotion names.
PREFERRED_FAMILIES = {
    "cheerful": 1.5,
    "proud": 1.4,
    "amazed": 1.3,
    "loving": 1.4,
    "calming": 1.3,
    "grateful": 1.3,
    "welcoming": 1.2,
    "attentive": 1.2,
    "helpful": 1.2,
    "thoughtful": 1.2,
    "understanding": 1.2,
    "curious": 1.1,
    "enthusiastic": 1.2,
    "relief": 1.1,
    "serenity": 1.2,
    "shy": 1.1,
    "surprised": 1.0,
    "yes": 1.1,
    "come": 1.0,
    "lonely": 1.05,
    "sad": 0.95,
    "boredom": 0.9,
    "tired": 0.9,
}
# Penalised families (likely growl, distortion, electronic).
PENALISED_FAMILIES = {
    "rage": 0.4, "furious": 0.4, "disgusted": 0.4, "frustrated": 0.5,
    "irritated": 0.5, "displeased": 0.6, "contempt": 0.5, "scared": 0.7,
    "fear": 0.7, "dying": 0.5, "electric": 0.4, "dance": 0.6,
    "incomprehensible": 0.5, "exhausted": 0.7, "downcast": 0.7,
    "anxiety": 0.6, "uncomfortable": 0.7,
}


def family_of(filename: str) -> str:
    """`amazed1.wav` -> `amazed`."""
    stem = Path(filename).stem
    return "".join(ch for ch in stem if not ch.isdigit() and ch != "_")


def family_weight(family: str) -> float:
    if family in PENALISED_FAMILIES:
        return PENALISED_FAMILIES[family]
    return PREFERRED_FAMILIES.get(family, 1.0)


def find_best_voiced_segment(
    wav: np.ndarray, sr: int
) -> tuple[float, float, float, float, float] | None:
    """Return (start_s, end_s, mean_f0_hz, stability_score, raw_score)."""
    x = wav.astype(np.float64, copy=False)
    f0, t = pw.harvest(
        x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_MS
    )
    f0 = pw.stonemask(x, f0, t, sr)

    voiced = (f0 > F0_FLOOR) & (f0 < F0_CEIL)
    if voiced.sum() < int(MIN_SEG_S * 1000.0 / FRAME_MS):
        return None

    # Find contiguous voiced runs (allowing 30 ms gaps to bridge tiny dropouts).
    bridge = int(30.0 / FRAME_MS)
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    gap = 0
    for i, v in enumerate(voiced):
        if v:
            if not in_run:
                start, in_run, gap = i, True, 0
            else:
                gap = 0
        else:
            if in_run:
                gap += 1
                if gap > bridge:
                    runs.append((start, i - gap + 1))
                    in_run = False
    if in_run:
        runs.append((start, len(voiced)))

    best = None
    for a, b in runs:
        seg_dur = (b - a) * FRAME_MS / 1000.0
        if seg_dur < MIN_SEG_S:
            continue
        # Trim to MAX_SEG_S, centered on the most stable window.
        if seg_dur > MAX_SEG_S:
            n_keep = int(MAX_SEG_S * 1000.0 / FRAME_MS)
            seg_f0 = f0[a:b]
            # Sliding window of constant length, pick the window with min std.
            best_off = 0
            best_std = float("inf")
            for off in range(0, len(seg_f0) - n_keep + 1, 4):
                w = seg_f0[off:off + n_keep]
                w = w[w > F0_FLOOR]
                if len(w) < n_keep // 2:
                    continue
                s = float(np.std(1200.0 * np.log2(w / w.mean())))
                if s < best_std:
                    best_std, best_off = s, off
            a = a + best_off
            b = a + n_keep

        seg_f0 = f0[a:b]
        seg_f0 = seg_f0[seg_f0 > F0_FLOOR]
        if len(seg_f0) < 8:
            continue
        mean_f0 = float(np.mean(seg_f0))
        cents = 1200.0 * np.log2(seg_f0 / mean_f0)
        f0_std_cents = float(np.std(cents))
        # Stability: 0¢ → 1.0 ; 60¢ → 0.4
        stab = 1.0 / (1.0 + f0_std_cents / 40.0)

        i0 = int(a * FRAME_MS / 1000.0 * sr)
        i1 = int(b * FRAME_MS / 1000.0 * sr)
        rms = float(np.sqrt(np.mean(wav[i0:i1] ** 2) + 1e-12))
        if rms < 0.005:
            continue

        seg_dur_actual = (b - a) * FRAME_MS / 1000.0
        score = seg_dur_actual * stab * (rms ** 0.5)
        if best is None or score > best[4]:
            best = (i0 / sr, i1 / sr, mean_f0, stab, score)
    return best


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale samples from previous runs.
    for f in SAMPLES_DIR.glob("sample_*.wav"):
        f.unlink()

    sources = sorted(SOURCES_DIR.glob("*.wav"))
    print(f"Scanning {len(sources)} sources @ {TARGET_RATE} Hz with harvest…")

    candidates: list[dict] = []
    for path in sources:
        try:
            wav, sr = load_mono(path, target_rate=TARGET_RATE)
        except Exception as e:
            print(f"  skip {path.name}: load failed ({e})")
            continue
        if len(wav) < int(MIN_SEG_S * sr):
            continue
        best = find_best_voiced_segment(wav, sr)
        if best is None:
            continue
        a_s, b_s, mean_f0, stab, raw_score = best
        family = family_of(path.name)
        weighted = raw_score * family_weight(family)
        candidates.append(
            dict(
                source=path.name,
                family=family,
                start_s=round(a_s, 3),
                end_s=round(b_s, 3),
                duration_s=round(b_s - a_s, 3),
                mean_f0_hz=round(mean_f0, 2),
                stability=round(stab, 3),
                raw_score=round(raw_score, 4),
                family_weight=round(family_weight(family), 2),
                weighted_score=round(weighted, 4),
            )
        )

    candidates.sort(key=lambda c: -c["weighted_score"])
    print(f"\nTop 12 candidates after family-weighting:")
    for c in candidates[:12]:
        print(
            f"  {c['source']:<24s} fam={c['family']:<14s} "
            f"dur={c['duration_s']:4.2f} f0={c['mean_f0_hz']:6.1f} "
            f"stab={c['stability']:.2f} score={c['weighted_score']:.3f}"
        )

    # Diversify by family + by f0 (>=120 cents apart).
    picked: list[dict] = []
    used_families: set[str] = set()
    for c in candidates:
        if c["family"] in used_families:
            continue
        too_close = any(
            abs(1200.0 * np.log2(c["mean_f0_hz"] / p["mean_f0_hz"])) < 120.0
            for p in picked
        )
        if too_close:
            continue
        used_families.add(c["family"])
        picked.append(c)
        if len(picked) >= N_TARGET:
            break

    print(f"\nFinal pick ({len(picked)}):")
    manifest = []
    for i, c in enumerate(picked):
        src_path = SOURCES_DIR / c["source"]
        wav, sr = load_mono(src_path, target_rate=TARGET_RATE)
        i0 = int(c["start_s"] * sr)
        i1 = int(c["end_s"] * sr)
        seg = wav[i0:i1]
        fade = int(0.020 * sr)
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        peak = float(np.max(np.abs(seg)) + 1e-9)
        seg = seg * (0.9 / peak)
        out_name = f"sample_{i:02d}_{Path(c['source']).stem}.wav"
        out_path = SAMPLES_DIR / out_name
        save_wav(out_path, seg, sr)
        c["sample_file"] = out_name
        c["sample_rate"] = sr
        manifest.append(c)
        print(
            f"  #{i}: {c['source']:<24s} dur={c['duration_s']:4.2f}s "
            f"f0={c['mean_f0_hz']:6.1f} stab={c['stability']:.2f} "
            f"score={c['weighted_score']:.3f}"
        )

    out_json = SAMPLES_DIR / "curation.json"
    with open(out_json, "w") as f:
        json.dump(
            dict(target_rate=TARGET_RATE, samples=manifest), f, indent=2
        )
    print(f"\nWrote {len(manifest)} samples + curation.json to {SAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
