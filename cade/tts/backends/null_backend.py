"""Test/dry-run TTS backend."""

from __future__ import annotations

import time

import numpy as np

from cade.tts.backends.base import TTSResult


class NullBackend:
    """Small backend used in tests or when audio synthesis is disabled."""

    def __init__(self, name: str = "null", sample_rate: int = 16000):
        self.name = name
        self.sample_rate = sample_rate
        self.model_id = "null"
        self.cache_identity = f"{name}:null:{sample_rate}"

    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        sid: int | None = None,
        profile: str = "dialogue",
    ) -> TTSResult:
        start = time.monotonic()
        duration_s = max(0.12, min(1.0, len(text.strip()) / 40.0))
        samples = np.zeros(int(self.sample_rate * duration_s), dtype=np.float32)
        elapsed = time.monotonic() - start
        return TTSResult(
            samples=samples,
            sample_rate=self.sample_rate,
            audio_duration_s=duration_s,
            synth_latency_s=elapsed,
            backend=self.name,
            model_id=self.model_id,
            profile=profile,
            normalized_text=text,
            rtf=elapsed / duration_s if duration_s else 0.0,
        )
