"""Shared per-note ADSR envelope."""
from __future__ import annotations

import numpy as np


def adsr(n: int, rate: int,
         attack_s: float = 0.020,
         decay_s: float = 0.050,
         sustain_lvl: float = 0.75,
         release_s: float = 0.100) -> np.ndarray:
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
