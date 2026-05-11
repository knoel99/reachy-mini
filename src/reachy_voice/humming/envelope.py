"""Per-note ADSR envelope, applied to the WORLD-synthesised buffer.

Aligned with the values the sine path used to use, so notes have the
same perceived attack/release shape regardless of the underlying
synthesis.
"""
from __future__ import annotations

import numpy as np


ATTACK_S = 0.015
DECAY_S = 0.050
SUSTAIN_LVL = 0.70
RELEASE_S = 0.080


def adsr(
    n: int,
    rate: int,
    attack_s: float = ATTACK_S,
    decay_s: float = DECAY_S,
    sustain_lvl: float = SUSTAIN_LVL,
    release_s: float = RELEASE_S,
) -> np.ndarray:
    a = min(int(rate * attack_s), n // 4)
    d = min(int(rate * decay_s), max(0, (n - a) // 3))
    r = min(int(rate * release_s), max(0, n - a - d))
    s_len = max(0, n - a - d - r)
    env = np.empty(n, dtype=np.float32)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if d > 0:
        env[a:a + d] = np.linspace(1.0, sustain_lvl, d, dtype=np.float32)
    if s_len > 0:
        env[a + d:a + d + s_len] = sustain_lvl
    if r > 0:
        env[a + d + s_len:] = np.linspace(
            sustain_lvl if d > 0 or s_len > 0 else 1.0,
            0.0, r, dtype=np.float32,
        )
    return env
