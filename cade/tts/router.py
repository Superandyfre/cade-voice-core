"""Routing and fallback logic for CPU-only TTS."""

from __future__ import annotations

import logging
import os
from typing import Optional

from cade.tts.backends.base import TTSBackend, TTSResult
from cade.tts.cache import TTSCache
from cade.tts.normalizer import TextNormalizer


logger = logging.getLogger(__name__)


class TTSRouter:
    """Selects a backend, applies normalization/cache, and handles fallback."""

    def __init__(
        self,
        *,
        default_backend: TTSBackend,
        fast_backend: Optional[TTSBackend] = None,
        fallback_backend: Optional[TTSBackend] = None,
        cache: Optional[TTSCache] = None,
        normalizer: Optional[TextNormalizer] = None,
        short_text_chars: int = 25,
        load_threshold: float = 0.85,
    ):
        self.default_backend = default_backend
        self.fast_backend = fast_backend or default_backend
        self.fallback_backend = fallback_backend or self.fast_backend
        self.cache = cache or TTSCache("", enabled=False)
        self.normalizer = normalizer or TextNormalizer(enabled=False)
        self.short_text_chars = short_text_chars
        self.load_threshold = load_threshold
        self.fallback_count = 0

    def synthesize(self, text: str, *, profile: str = "dialogue") -> TTSResult:
        normalized = self.normalizer.normalize(text)
        backend = self.select_backend(normalized, profile=profile)
        try:
            return self._synthesize_with_cache(backend, normalized, profile=profile)
        except Exception as exc:
            if backend is self.fallback_backend:
                raise
            logger.warning(
                "TTS backend %s failed, using fallback %s: %s",
                backend.name,
                self.fallback_backend.name,
                exc,
            )
            self.fallback_count += 1
            result = self._synthesize_with_cache(self.fallback_backend, normalized, profile=profile)
            return result.with_updates(fallback=True)

    def select_backend(self, normalized_text: str, *, profile: str = "dialogue") -> TTSBackend:
        if self._system_load() > self.load_threshold:
            return self.fallback_backend
        if profile in {"fallback"}:
            return self.fallback_backend
        if profile in {"fast", "error"}:
            return self.fast_backend
        if len(normalized_text.strip()) <= self.short_text_chars:
            return self.fast_backend
        return self.default_backend

    def _synthesize_with_cache(self, backend: TTSBackend, text: str, *, profile: str) -> TTSResult:
        cached = self.cache.get(backend, text, profile=profile)
        if cached is not None:
            return cached.with_updates(profile=profile, normalized_text=text)

        result = backend.synthesize(text, profile=profile)
        result = result.with_updates(profile=profile, normalized_text=text)
        self.cache.put(backend, text, result)
        return result

    @staticmethod
    def _system_load() -> float:
        try:
            import psutil
            return psutil.cpu_percent(interval=None) / 100.0
        except Exception:
            pass
        try:
            load1, _, _ = os.getloadavg()
            cpus = os.cpu_count() or 1
            return min(1.0, load1 / cpus)
        except Exception:
            return 0.0
