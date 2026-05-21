"""Compatibility TTS engine facade.

The public ``TTSEngine`` API remains compatible with the original blocking
implementation while routing synthesis through CPU-only backends, cache, text
normalization, and a playback manager.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from cade.config import Config
from cade.tts.backends.base import TTSBackend, TTSResult
from cade.tts.backends.sherpa_kokoro import SherpaKokoroBackend
from cade.tts.backends.sherpa_vits import SherpaVitsBackend
from cade.tts.cache import TTSCache
from cade.tts.chunker import SentenceChunker
from cade.tts.normalizer import TextNormalizer
from cade.tts.playback import PlaybackManager
from cade.tts.router import TTSRouter

try:
    import soundfile as sf
except ImportError:
    sf = None


logger = logging.getLogger(__name__)


_PREWARM_PHRASES = [
    "Hello, how can I help you?",
    "What would you like to order?",
    "Could you repeat that?",
    "Anything else?",
    "Your order is confirmed.",
    "Sorry, I didn't catch that.",
    "Please wait a moment.",
    "OK.",
    "Sure.",
]


class TTSEngine:
    """Backward-compatible TTS engine facade."""

    def __init__(
        self,
        model_dir: str,
        provider: str = "cpu",
        num_threads: Optional[int] = None,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ):
        if sf is None:
            raise ImportError("soundfile is required for TTS. Install with: pip install soundfile")

        self.model_dir = Path(model_dir)
        self.provider = provider or Config.TTS_PROVIDER
        self.num_threads = int(os.getenv("CADE_TTS_NUM_THREADS", str(num_threads if num_threads is not None else Config.TTS_NUM_THREADS)))
        self.sid = int(os.getenv("CADE_TTS_SID", str(sid if sid is not None else Config.TTS_SID)))
        self.speed = float(os.getenv("CADE_TTS_SPEED", str(speed if speed is not None else Config.TTS_SPEED)))
        self._event_callback = event_callback
        self._chunking_enabled = Config.TTS_CHUNKING_ENABLED
        self._chunk_min = Config.TTS_MIN_CHUNK_CHARS
        self._chunk_max = Config.TTS_MAX_CHUNK_CHARS

        if Config.TTS_ROUTER_ENABLED:
            fallback_backend = self._build_fallback_backend()
            default_backend = self._build_backend_or_fallback(
                Config.TTS_DEFAULT_BACKEND,
                name=Config.TTS_DEFAULT_BACKEND,
                fallback=fallback_backend,
            )
            fast_backend = self._build_backend_or_fallback(
                Config.TTS_FAST_BACKEND,
                name=Config.TTS_FAST_BACKEND,
                fallback=fallback_backend,
            )
        else:
            fallback_backend = self._build_backend("auto", str(self.model_dir), name="legacy")
            default_backend = fallback_backend
            fast_backend = fallback_backend

        cache = TTSCache(Config.TTS_CACHE_DIR, enabled=Config.TTS_CACHE_ENABLED)
        normalizer = TextNormalizer(enabled=Config.TTS_TEXT_NORMALIZE)
        self.router = TTSRouter(
            default_backend=default_backend,
            fast_backend=fast_backend,
            fallback_backend=fallback_backend,
            cache=cache,
            normalizer=normalizer,
        )
        self.playback = PlaybackManager(
            Config.TTS_PLAYBACK_BACKEND,
            use_paplay_fallback=Config.TTS_USE_PAPLAY_FALLBACK,
        )

        logger.info(
            "TTS engine initialized: default=%s fast=%s fallback=%s provider=%s threads=%s",
            self.router.default_backend.name,
            self.router.fast_backend.name,
            self.router.fallback_backend.name,
            self.provider,
            self.num_threads,
        )
        self._start_prewarm_if_enabled()

    def synthesize(self, text: str, profile: str = "dialogue") -> Tuple[np.ndarray, int]:
        """Synthesize speech and return ``(audio_samples, sample_rate)``."""
        result = self.router.synthesize(text, profile=profile)
        self._log_synthesis(result)
        return result.samples, result.sample_rate

    def synthesize_detailed(self, text: str, profile: str = "dialogue") -> TTSResult:
        """Synthesize speech and return detailed backend/cache metadata."""
        result = self.router.synthesize(text, profile=profile)
        self._log_synthesis(result)
        return result

    def speak(
        self,
        text: str,
        device: Optional[int] = None,
        save_path: Optional[str] = None,
        pulse_sink: Optional[str] = None,
        profile: str = "dialogue",
    ) -> Tuple[float, float]:
        """Synthesize and play speech, preserving the legacy return tuple."""
        result = self.speak_detailed(
            text,
            device=device,
            save_path=save_path,
            pulse_sink=pulse_sink,
            profile=profile,
        )
        return float(result.playback_duration_s or 0.0), float(result.audio_duration_s)

    def speak_detailed(
        self,
        text: str,
        device: Optional[int] = None,
        save_path: Optional[str] = None,
        pulse_sink: Optional[str] = None,
        profile: str = "dialogue",
    ) -> TTSResult:
        """Synthesize and play speech, returning a rich result object."""
        chunks = self._chunks_for_text(text)
        if len(chunks) <= 1:
            result = self.router.synthesize(text, profile=profile)
            self._log_synthesis(result)
            self._write_save_path(save_path, result)
            playback_s = self._play_result(result, device=device, save_path=save_path, pulse_sink=pulse_sink)
            return result.with_updates(playback_duration_s=playback_s)

        results: list[TTSResult] = []
        total_playback = 0.0
        for chunk in chunks:
            result = self.router.synthesize(chunk, profile=profile)
            self._log_synthesis(result)
            total_playback += self._play_result(result, device=device, save_path=None, pulse_sink=pulse_sink)
            results.append(result)

        aggregate = self._aggregate_chunk_results(results, text)
        self._write_save_path(save_path, aggregate)
        return aggregate.with_updates(playback_duration_s=total_playback)

    def stop(self) -> None:
        self.playback.stop()
        self._emit("tts.interrupted", {})

    def close(self) -> None:
        self.playback.close()

    def _build_fallback_backend(self) -> TTSBackend:
        fallback_kind = Config.TTS_FALLBACK_BACKEND or "vits"
        fallback_dir = self._model_dir_for(fallback_kind)
        if not Path(fallback_dir).is_dir():
            fallback_dir = str(self.model_dir)
        return self._build_backend(fallback_kind, fallback_dir, name=fallback_kind)

    def _build_backend_or_fallback(self, kind: str, *, name: str, fallback: TTSBackend) -> TTSBackend:
        model_dir = self._model_dir_for(kind)
        if not Path(model_dir).is_dir():
            logger.warning("TTS %s backend model dir not found: %s; using fallback %s", name, model_dir, fallback.name)
            return fallback
        try:
            return self._build_backend(kind, model_dir, name=name)
        except Exception as exc:
            logger.warning("Failed to initialize TTS %s backend from %s: %s; using fallback", name, model_dir, exc)
            return fallback

    def _build_backend(self, kind: str, model_dir: str, *, name: str) -> TTSBackend:
        normalized = str(kind or "").strip().lower()
        if normalized == "kokoro":
            return SherpaKokoroBackend(
                model_dir,
                name=name,
                provider=self.provider,
                num_threads=self.num_threads,
                sid=self.sid,
                speed=self.speed,
            )
        if normalized in {"piper", "vits", "vits_fallback"}:
            return SherpaVitsBackend(
                model_dir,
                name=name,
                provider=self.provider,
                num_threads=self.num_threads,
                sid=self.sid,
                speed=self.speed,
            )

        model_path = str(model_dir).lower()
        if "vits" in model_path or "piper" in model_path:
            return SherpaVitsBackend(
                model_dir,
                name=name,
                provider=self.provider,
                num_threads=self.num_threads,
                sid=self.sid,
                speed=self.speed,
            )
        return SherpaKokoroBackend(
            model_dir,
            name=name,
            provider=self.provider,
            num_threads=self.num_threads,
            sid=self.sid,
            speed=self.speed,
        )

    def _model_dir_for(self, kind: str) -> str:
        normalized = str(kind or "").strip().lower()
        if normalized == "kokoro":
            return Config.TTS_KOKORO_MODEL_DIR
        if normalized == "piper":
            return Config.TTS_PIPER_MODEL_DIR
        if normalized in {"vits", "vits_fallback", "fallback"}:
            return Config.TTS_VITS_MODEL_DIR or str(self.model_dir)
        return str(self.model_dir)

    def _chunks_for_text(self, text: str) -> list[str]:
        if not self._chunking_enabled or len(str(text or "")) <= self._chunk_max:
            return [text]
        chunker = SentenceChunker(min_chars=self._chunk_min, max_chars=self._chunk_max)
        chunks = chunker.feed(text)
        chunks.extend(chunker.flush())
        return chunks or [text]

    def _play_result(
        self,
        result: TTSResult,
        *,
        device: Optional[int],
        save_path: Optional[str],
        pulse_sink: Optional[str],
    ) -> float:
        self._emit(
            "tts.playback_started",
            {
                "backend": result.backend,
                "profile": result.profile,
                "audio_duration_s": result.audio_duration_s,
            },
        )
        playback_s = self.playback.play_blocking(
            result.samples,
            result.sample_rate,
            device=device,
            wav_path=save_path,
            pulse_sink=pulse_sink,
        )
        self._emit(
            "tts.playback_completed",
            {
                "backend": result.backend,
                "profile": result.profile,
                "playback_duration_s": playback_s,
                "audio_duration_s": result.audio_duration_s,
            },
        )
        return playback_s

    @staticmethod
    def _write_save_path(save_path: Optional[str], result: TTSResult) -> None:
        if not save_path:
            return
        sf.write(save_path, result.samples, samplerate=result.sample_rate, subtype="PCM_16")
        logger.info("Audio saved to %s", save_path)

    def _aggregate_chunk_results(self, results: list[TTSResult], original_text: str) -> TTSResult:
        first = results[0]
        sample_rate = first.sample_rate
        samples = []
        for result in results:
            if result.sample_rate == sample_rate:
                samples.append(result.samples)
        aggregate_samples = np.concatenate(samples) if samples else first.samples
        return TTSResult(
            samples=aggregate_samples,
            sample_rate=sample_rate,
            audio_duration_s=sum(r.audio_duration_s for r in results),
            synth_latency_s=sum(r.synth_latency_s for r in results),
            backend=first.backend if all(r.backend == first.backend for r in results) else "chunked",
            model_id=first.model_id,
            profile=first.profile,
            normalized_text=original_text,
            cache_hit=all(r.cache_hit for r in results),
            fallback=any(r.fallback for r in results),
            rtf=(
                sum(r.synth_latency_s for r in results) / sum(r.audio_duration_s for r in results)
                if sum(r.audio_duration_s for r in results) > 0
                else 0.0
            ),
        )

    def _start_prewarm_if_enabled(self) -> None:
        if not Config.TTS_CACHE_ENABLED or not Config.TTS_PREWARM_ENABLED:
            return

        def run() -> None:
            time.sleep(2.0)
            for phrase in _PREWARM_PHRASES:
                try:
                    self.router.synthesize(phrase, profile="fast" if len(phrase) <= 25 else "order_prompt")
                except Exception:
                    logger.debug("Ignoring TTS prewarm failure for %r", phrase, exc_info=True)

        thread = threading.Thread(target=run, name="cade-tts-prewarm", daemon=True)
        thread.start()

    def _log_synthesis(self, result: TTSResult) -> None:
        logger.info(
            "TTS[%s]: %.3fs generation, %.3fs audio, RTF=%.3f cache=%s fallback=%s",
            result.backend,
            result.synth_latency_s,
            result.audio_duration_s,
            result.rtf,
            result.cache_hit,
            result.fallback,
        )
        self._emit(
            "tts.synthesis_completed",
            {
                "backend": result.backend,
                "model_id": result.model_id,
                "profile": result.profile,
                "normalized_text": result.normalized_text,
                "cache_hit": result.cache_hit,
                "fallback": result.fallback,
                "synth_latency_s": result.synth_latency_s,
                "audio_duration_s": result.audio_duration_s,
                "rtf": result.rtf,
            },
        )

    def _emit(self, event: str, payload: dict) -> None:
        if not self._event_callback:
            return
        try:
            self._event_callback(event, payload)
        except Exception:
            logger.debug("Ignoring TTS event callback failure for %s", event, exc_info=True)
