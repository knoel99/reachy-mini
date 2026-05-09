"""xAI Speech-to-Text REST client (POST /v1/stt).

Pricing: $0.10/hr REST (vs $0.20/hr streaming). Validated against
https://docs.x.ai/developers/rest-api-reference/inference/voice .
"""

from __future__ import annotations

import os

import httpx


XAI_STT_URL = "https://api.x.ai/v1/stt"


class XaiSTT:
    """Stateless POST client for xAI's STT REST endpoint."""

    def __init__(self,
                 api_key: str | None = None,
                 language: str = "fr",
                 timeout: float = 15.0) -> None:
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY is not set")
        self.language = language
        self._client = httpx.Client(timeout=timeout)

    def transcribe_pcm16(self, pcm16: bytes, sample_rate: int = 16_000) -> str:
        """Transcribe raw PCM16 mono bytes → text (empty string on no speech)."""
        files = {"file": ("audio.pcm", pcm16, "application/octet-stream")}
        data = {
            "language": self.language,
            "audio_format": "pcm",
            "sample_rate": str(sample_rate),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = self._client.post(XAI_STT_URL, data=data, files=files, headers=headers)
        r.raise_for_status()
        return (r.json().get("text") or "").strip()

    def close(self) -> None:
        self._client.close()
