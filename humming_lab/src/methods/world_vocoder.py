"""Method C — WORLD vocoder resynthesis.

WORLD decomposes a voice signal into three independent streams:
  - f0:  pitch contour (Hz, per frame)
  - sp:  spectral envelope (formants), one row per frame
  - ap:  aperiodicity (breathiness), one row per frame

To retune to a new pitch we replace f0 wholesale. To change duration we
duplicate or drop frames of sp/ap. The key win: the *spectral envelope*
stays fixed during the pitch change — formants no longer move with pitch,
so the timbre is preserved across the entire range.

Pros: best timbre preservation across multi-octave shifts.
Cons: native dep (pyworld), heavier code path.
"""
from __future__ import annotations

import numpy as np
import pyworld as pw

from ._envelope import adsr

FRAME_MS = 5.0  # WORLD default
F0_FLOOR = 60.0
F0_CEIL = 800.0


def analyze(sample: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (f0, sp, ap, mean_voiced_f0). Uses harvest (more accurate)."""
    x = sample.astype(np.float64, copy=False)
    f0, t = pw.harvest(x, rate, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL,
                       frame_period=FRAME_MS)
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

    pre_analyzed: avoids re-running WORLD analysis when rendering many notes
    from the same source sample.
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
        # Source had no voicing — fall back to silence.
        return np.zeros(n_target, dtype=np.float32)

    # Stretch the frame sequence to match the target duration.
    target_frames = max(1, int(round(duration_s * 1000.0 / FRAME_MS)))
    f0_s = _resize_frames(f0[:, None], target_frames).reshape(-1)
    sp_s = _resize_frames(sp, target_frames)
    ap_s = _resize_frames(ap, target_frames)

    # Replace pitch: scale voiced frames so they hit target_f0 (preserving the
    # micro-vibrato pattern of the source instead of going dead-flat). Then
    # *fill unvoiced frames* with target_f0 too — otherwise a long sustained
    # note inherits the source's silence/breath gaps and sounds full of holes.
    voiced_mask = f0_s > 0
    if voiced_mask.any():
        cur_voiced_mean = float(f0_s[voiced_mask].mean())
        if cur_voiced_mean > 0:
            f0_s[voiced_mask] *= (target_f0 / cur_voiced_mean)
    f0_s[~voiced_mask] = target_f0

    y = pw.synthesize(f0_s, sp_s, ap_s, rate, frame_period=FRAME_MS)
    out = y.astype(np.float32)
    # Trim/pad to exact n_target.
    if out.size >= n_target:
        out = out[:n_target]
    else:
        out = np.pad(out, (0, n_target - out.size), mode="constant")
    out = out * adsr(out.size, rate)
    return out
