"""WebRTC VAD wrapper that segments mic audio into utterances.

Feeds mono float32 samples at 16 kHz frame-by-frame to webrtcvad and
emits one `Utterance` (raw PCM16 bytes) each time a stretch of speech
is followed by `silence_ms` of consecutive non-speech.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import webrtcvad


SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480
FRAME_BYTES = FRAME_SAMPLES * 2                 # 960 (PCM16)


@dataclass
class Utterance:
    pcm16: bytes
    duration_s: float


class UtteranceDetector:
    """Streaming WebRTC VAD: feed PCM samples, get back utterances."""

    def __init__(self,
                 aggressiveness: int = 2,
                 min_speech_ms: int = 200,
                 silence_ms: int = 700) -> None:
        self._vad = webrtcvad.Vad(aggressiveness)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._silence_target = max(1, silence_ms // FRAME_MS)
        self._buffer = bytearray()
        self._frames: list[bytes] = []
        self._silence_count = 0
        self._in_speech = False

    def feed(self, samples_f32: np.ndarray) -> list[Utterance]:
        """Append `samples_f32` (mono float32 @ 16 kHz) and yield ready utterances."""
        pcm16 = (np.clip(samples_f32, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        self._buffer.extend(pcm16)

        out: list[Utterance] = []
        while len(self._buffer) >= FRAME_BYTES:
            frame = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]
            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                self._frames.append(frame)
                self._silence_count = 0
                self._in_speech = True
            elif self._in_speech:
                # Trailing silence inside an active utterance — keep
                # the frame so whoever consumes it has a natural pad.
                self._frames.append(frame)
                self._silence_count += 1
                if self._silence_count >= self._silence_target:
                    speech_count = len(self._frames) - self._silence_count
                    if speech_count >= self._min_speech_frames:
                        keep = len(self._frames) - self._silence_count
                        committed = b"".join(self._frames[:keep])
                        out.append(Utterance(
                            pcm16=committed,
                            duration_s=len(committed) / 2 / SAMPLE_RATE,
                        ))
                    self._frames = []
                    self._silence_count = 0
                    self._in_speech = False
        return out
