"""Shared TTS backend types."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np


@dataclass
class TTSResult:
    """Audio plus synthesis metadata returned by all TTS backends."""

    samples: np.ndarray
    sample_rate: int
    audio_duration_s: float
    synth_latency_s: float
    backend: str
    model_id: str = ""
    profile: str = "dialogue"
    normalized_text: str = ""
    cache_hit: bool = False
    fallback: bool = False
    rtf: float = 0.0
    playback_duration_s: float | None = None

    def with_updates(self, **kwargs) -> "TTSResult":
        return replace(self, **kwargs)


class TTSBackend(Protocol):
    """Protocol for text-to-audio TTS backends."""

    name: str
    model_id: str
    cache_identity: str

    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        sid: int | None = None,
        profile: str = "dialogue",
    ) -> TTSResult:
        ...
