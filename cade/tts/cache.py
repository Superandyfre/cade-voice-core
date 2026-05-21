"""Filesystem audio cache for CPU-only TTS."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from cade.tts.backends.base import TTSResult

try:
    import soundfile as sf
except ImportError:
    sf = None


logger = logging.getLogger(__name__)


class TTSCache:
    """Small WAV cache keyed by backend identity and normalized text."""

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir).expanduser()
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, backend: Any, normalized_text: str, *, profile: str = "dialogue") -> TTSResult | None:
        if not self.enabled or sf is None:
            return None
        key = self.key_for(backend, normalized_text)
        path = self._path_for(backend, key)
        if not path.is_file():
            return None
        start = time.monotonic()
        try:
            samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim > 1:
                samples = samples[:, 0]
            if len(samples) == 0:
                return None
            elapsed = time.monotonic() - start
            audio_s = len(samples) / int(sample_rate)
            return TTSResult(
                samples=samples,
                sample_rate=int(sample_rate),
                audio_duration_s=audio_s,
                synth_latency_s=elapsed,
                backend=backend.name,
                model_id=getattr(backend, "model_id", backend.name),
                profile=profile,
                normalized_text=normalized_text,
                cache_hit=True,
                rtf=elapsed / audio_s if audio_s else 0.0,
            )
        except Exception as exc:
            logger.warning("Ignoring corrupt TTS cache file %s: %s", path, exc)
            return None

    def put(self, backend: Any, normalized_text: str, result: TTSResult) -> None:
        if not self.enabled or sf is None or result.cache_hit:
            return
        key = self.key_for(backend, normalized_text)
        path = self._path_for(backend, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.wav")
        try:
            sf.write(str(tmp), result.samples, samplerate=result.sample_rate, subtype="PCM_16")
            tmp.replace(path)
            self._update_index(backend, key, path, normalized_text, result)
        except Exception as exc:
            logger.warning("Failed to write TTS cache file %s: %s", path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def key_for(self, backend: Any, normalized_text: str) -> str:
        identity = getattr(backend, "cache_identity", getattr(backend, "model_id", backend.name))
        sample_rate = getattr(backend, "sample_rate", "")
        raw = "\n".join([str(backend.name), str(identity), str(sample_rate), str(normalized_text)])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path_for(self, backend: Any, key: str) -> Path:
        identity = getattr(backend, "cache_identity", getattr(backend, "model_id", backend.name))
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in f"{backend.name}_{identity}")
        return self.cache_dir / safe / f"{key}.wav"

    def _update_index(self, backend: Any, key: str, path: Path, text: str, result: TTSResult) -> None:
        index = path.parent / "index.json"
        payload: dict[str, Any] = {}
        try:
            if index.is_file():
                payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        payload[key] = {
            "path": path.name,
            "text": text,
            "backend": result.backend,
            "model_id": result.model_id,
            "sample_rate": result.sample_rate,
            "audio_duration_s": result.audio_duration_s,
            "updated_at": time.time(),
        }
        try:
            index.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to update TTS cache index %s: %s", index, exc)
