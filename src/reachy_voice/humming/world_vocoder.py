"""WORLD vocoder analysis + resynthesis.

WORLD decomposes a voice signal into three independent streams:
  - f0:  pitch contour (Hz, per frame)
  - sp:  spectral envelope (formants), one row per frame
  - ap:  aperiodicity (breathiness), one row per frame

Replacing f0 wholesale retunes the note. Duplicating/dropping sp/ap
frames stretches duration. The spectral envelope stays fixed under
pitch change, so the timbre is preserved — that's the win versus a raw
resampler.
"""
from __future__ import annotations

import numpy as np
import pyworld as pw

from .envelope import adsr

FRAME_MS = 5.0  # WORLD default
# Wide range for analyzing arbitrary source voice clips at runtime. The
# offline curation step (tools/humming/curate.py) deliberately narrows
# this to [80, 350] Hz to suppress octave-doubling on cartoon-y robot
# timbres; on a sample that already passed curation, that asymmetry is
# harmless because the analysis only re-runs once for caching.
F0_FLOOR = 60.0
F0_CEIL = 800.0


def analyze(
    sample: np.ndarray, rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (f0, sp, ap, mean_voiced_f0). Uses harvest (more accurate)."""
    x = sample.astype(np.float64, copy=False)
    f0, t = pw.harvest(
        x, rate, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_MS
    )
    f0 = pw.stonemask(x, f0, t, rate)
    sp = pw.cheaptrick(x, f0, t, rate)
    ap = pw.d4c(x, f0, t, rate)
    voiced = f0[f0 > 0]
    mean_f0 = float(voiced.mean()) if voiced.size else 0.0
    return f0.astype(np.float64), sp, ap, mean_f0


def _resize_frames(arr: np.ndarray, target_n: int) -> np.ndarray:
    """Linearly resample a frame-axis array to target_n frames (axis 0)."""
    n_in = arr.shape[0]
    if n_in == target_n:
        return arr
    if n_in <= 1:
        return np.repeat(arr, target_n, axis=0)
    src = np.linspace(0.0, n_in - 1, target_n, dtype=np.float64)
    lo = np.clip(np.floor(src).astype(int), 0, n_in - 1)
    hi = np.clip(lo + 1, 0, n_in - 1)
    a = (src - lo)[:, None]
    return ((1.0 - a) * arr[lo] + a * arr[hi]).astype(arr.dtype, copy=False)


def render_note(
    sample: np.ndarray,
    sample_f0: float,
    target_f0: float,
    duration_s: float,
    rate: int,
    *,
    pre_analyzed: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None,
) -> np.ndarray:
    """Resynthesise the sample at target_f0 with the requested duration.

    pre_analyzed: skips re-running WORLD analysis when rendering many
    notes from the same source sample.
    """
    n_target = int(duration_s * rate)
    if n_target <= 0:
        return np.zeros(0, dtype=np.float32)
    if sample.size == 0:
        return np.zeros(n_target, dtype=np.float32)

    if pre_analyzed is None:
        f0, sp, ap, mean_f0 = analyze(sample, rate)
    else:
        f0, sp, ap, mean_f0 = pre_analyzed
        f0 = f0.copy()
    if mean_f0 <= 0.0:
        return np.zeros(n_target, dtype=np.float32)

    target_frames = max(1, int(round(duration_s * 1000.0 / FRAME_MS)))
    f0_s = _resize_frames(f0[:, None], target_frames).reshape(-1)
    sp_s = _resize_frames(sp, target_frames)
    ap_s = _resize_frames(ap, target_frames)

    # Scale voiced frames toward target_f0 (preserves the micro-vibrato
    # pattern rather than going dead-flat). Then fill unvoiced frames
    # with target_f0 too — otherwise a sustained note inherits the
    # source's breath gaps and sounds full of holes.
    voiced_mask = f0_s > 0
    if voiced_mask.any():
        cur_voiced_mean = float(f0_s[voiced_mask].mean())
        if cur_voiced_mean > 0:
            f0_s[voiced_mask] *= (target_f0 / cur_voiced_mean)
    f0_s[~voiced_mask] = target_f0
    # Clamp to the analyzer's range so pyworld.synthesize doesn't clip
    # or distort on octave outliers.
    np.clip(f0_s, F0_FLOOR, F0_CEIL, out=f0_s)

    y = pw.synthesize(f0_s, sp_s, ap_s, rate, frame_period=FRAME_MS)
    out = y.astype(np.float32)
    if out.size >= n_target:
        out = out[:n_target]
    else:
        out = np.pad(out, (0, n_target - out.size), mode="constant")
    out = out * adsr(out.size, rate)
    return out
