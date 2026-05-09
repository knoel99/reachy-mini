"""Emotion playback: preloaded WAVs + bundled motion via the SDK.

`play(name)` runs **synchronously** — it pushes the audio sample,
blocks on the move, then sleeps until the sound has finished
playing. Callers should drive concurrency through their own queue
(e.g. `RobotActions._run_async`); the player itself never spawns
threads.

A monotonic `_speaking_until` timestamp is exposed via
`is_speaking()` so the mic loop can gate its capture and avoid
re-injecting the robot's own audio as a user utterance.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from ._log import log

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


# Extra padding kept on `_speaking_until` after the nominal sound
# duration, to absorb media-pipeline latency between push and audio
# actually leaving the speaker.
_SPEAKING_PAD_S = 0.5


class EmotionPlayer:
    """Plays emotion movements with their preloaded sounds (sync)."""

    DEFAULT_TARGET_RATE = 16_000  # ReSpeaker XVF3800 default

    def __init__(self, mini: ReachyMini, library) -> None:
        self.mini = mini
        self.library = library
        self._speaking_until: float = 0.0

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
                log(f"[emotion] preload '{name}': get failed: {e}")
                continue

            path = getattr(move, "sound_path", None)
            if path is None:
                skipped.append(name)
                continue

            try:
                src_rate, data = wavfile.read(str(path))
            except Exception as e:
                log(f"[emotion] preload '{name}': read failed: {e}")
                continue

            samples = self._to_float32_mono(data)
            if samples is None:
                log(f"[emotion] preload '{name}': unsupported dtype {data.dtype}")
                continue

            if src_rate != self._target_rate:
                g = np.gcd(src_rate, self._target_rate)
                samples = resample_poly(
                    samples, self._target_rate // g, src_rate // g
                ).astype(np.float32, copy=False)

            sounds[name] = samples

        total_mb = sum(s.nbytes for s in sounds.values()) / 1e6
        log(
            f"[emotion] preloaded {len(sounds)} sounds "
            f"({total_mb:.1f} MB @ {self._target_rate} Hz), "
            f"{len(skipped)} silent"
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

    def is_speaking(self) -> bool:
        """True while a previously pushed sound is still playing."""
        return time.monotonic() < self._speaking_until

    def play(self, name: str) -> bool:
        """Push the sound, run the move, then block until the sound has
        finished. Returns False only if the emotion is unknown.
        """
        try:
            move = self.library.get(name)
        except Exception as e:
            log(f"[emotion] unknown: {name} ({e})")
            return False
        if move is None:
            log(f"[emotion] unknown: {name}")
            return False

        samples = self._sounds.get(name)
        sound_dur = (
            len(samples) / self._target_rate if samples is not None else 0.0
        )

        t0 = time.monotonic()
        # Mark the speaker as busy *before* pushing so the mic loop
        # doesn't capture the very first samples back as user input.
        if sound_dur > 0:
            self._speaking_until = t0 + sound_dur + _SPEAKING_PAD_S
        try:
            if samples is not None:
                try:
                    self.mini.media.push_audio_sample(samples)
                except Exception as e:
                    log(f"[emotion] push_audio_sample '{name}' failed: {e}")
            self.mini.play_move(move, initial_goto_duration=0.5)
            # Hold until the sound has actually played out — even if the
            # bundled move was shorter.
            rem = self._speaking_until - time.monotonic()
            if rem > 0:
                time.sleep(rem)
        except Exception as e:
            log(f"[emotion] play '{name}' failed: {e}")
            return False
        finally:
            self._speaking_until = 0.0
        log(f"[emotion] '{name}' done in {time.monotonic() - t0:.2f}s")
        return True
