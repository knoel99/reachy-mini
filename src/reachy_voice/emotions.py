"""Emotion playback: preloaded WAVs + parallel motion via the SDK."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


class EmotionPlayer:
    """Plays emotion movements with their preloaded sounds.

    Sounds are loaded into RAM at construction (float32 mono @ target rate)
    so playback adds only a dict lookup + push_audio_sample call.
    """

    DEFAULT_TARGET_RATE = 16_000  # ReSpeaker XVF3800 default

    def __init__(self, mini: ReachyMini, library) -> None:
        self.mini = mini
        self.library = library
        self._busy = threading.Lock()
        self._last_t = 0.0

        rate = -1
        try:
            rate = mini.media.get_output_audio_samplerate()
        except Exception:
            pass
        self._target_rate = rate if rate and rate > 0 else self.DEFAULT_TARGET_RATE

        self._sounds: dict[str, np.ndarray] = self._preload_sounds()

    def _preload_sounds(self) -> dict[str, np.ndarray]:
        sounds: dict[str, np.ndarray] = {}
        skipped: list[str] = []
        for name in self.library.list_moves():
            try:
                move = self.library.get(name)
            except Exception as e:
                print(f"[emotion] preload '{name}': get failed: {e}", flush=True)
                continue

            path = getattr(move, "sound_path", None)
            if path is None:
                skipped.append(name)
                continue

            try:
                src_rate, data = wavfile.read(str(path))
            except Exception as e:
                print(f"[emotion] preload '{name}': read failed: {e}", flush=True)
                continue

            samples = self._to_float32_mono(data)
            if samples is None:
                print(f"[emotion] preload '{name}': unsupported dtype {data.dtype}", flush=True)
                continue

            if src_rate != self._target_rate:
                g = np.gcd(src_rate, self._target_rate)
                samples = resample_poly(
                    samples, self._target_rate // g, src_rate // g
                ).astype(np.float32, copy=False)

            sounds[name] = samples

        total_mb = sum(s.nbytes for s in sounds.values()) / 1e6
        print(
            f"[emotion] preloaded {len(sounds)} sounds "
            f"({total_mb:.1f} MB @ {self._target_rate} Hz), "
            f"{len(skipped)} silent",
            flush=True,
        )
        return sounds

    @staticmethod
    def _to_float32_mono(data: np.ndarray) -> np.ndarray | None:
        if data.dtype == np.int16:
            samples = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            samples = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            samples = (data.astype(np.float32) - 128.0) / 128.0
        elif data.dtype == np.float32:
            samples = data
        elif data.dtype == np.float64:
            samples = data.astype(np.float32)
        else:
            return None
        if samples.ndim == 2:
            samples = samples.mean(axis=1).astype(np.float32, copy=False)
        return samples

    def play(self, name: str, min_interval: float = 2.5) -> bool:
        now = time.monotonic()
        if (now - self._last_t) < min_interval:
            return False
        if not self._busy.acquire(blocking=False):
            return False

        try:
            move = self.library.get(name)
        except Exception as e:
            self._busy.release()
            print(f"[emotion] unknown: {name} ({e})", flush=True)
            return False
        if move is None:
            self._busy.release()
            print(f"[emotion] unknown: {name}", flush=True)
            return False

        self._last_t = now
        samples = self._sounds.get(name)

        def _run() -> None:
            try:
                if samples is not None:
                    try:
                        self.mini.media.push_audio_sample(samples)
                    except Exception as e:
                        print(f"[emotion] push_audio_sample '{name}' failed: {e}",
                              flush=True)
                self.mini.play_move(move, initial_goto_duration=0.5)
            except Exception as e:
                print(f"[emotion] play '{name}' failed: {e}", flush=True)
            finally:
                self._busy.release()

        threading.Thread(target=_run, daemon=True).start()
        return True
