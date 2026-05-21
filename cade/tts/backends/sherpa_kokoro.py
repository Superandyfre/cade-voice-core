"""sherpa-onnx Kokoro TTS backend."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from cade.tts.backends.base import TTSResult

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None


logger = logging.getLogger(__name__)


class SherpaKokoroBackend:
    """Kokoro backend backed by ``sherpa_onnx.OfflineTts``."""

    def __init__(
        self,
        model_dir: str,
        *,
        name: str = "kokoro",
        provider: str = "cpu",
        num_threads: int = 1,
        sid: int = 0,
        speed: float = 1.0,
    ):
        if sherpa_onnx is None:
            raise ImportError("sherpa-onnx is required for TTS. Install with: pip install sherpa-onnx")

        self.name = name
        self.model_dir = Path(model_dir)
        self.provider = provider
        self.num_threads = num_threads
        self.sid = sid
        self.speed = speed
        self.model_id = f"{name}:{self.model_dir.name}"
        self.cache_identity = self._build_cache_identity()
        self._tts = self._create_tts()
        logger.info("TTS backend initialized: %s from %s", self.name, self.model_dir)

    def _build_cache_identity(self) -> str:
        raw = f"{self.name}|{self.model_dir}|{self.sid}|{self.speed}|{self.provider}|{self.num_threads}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _find_file(self, *patterns: str) -> str:
        for pattern in patterns:
            match = list(self.model_dir.glob(pattern))
            if match:
                return str(match[0])
        return ""

    def _create_tts(self):
        model_file = self._find_file("model.onnx", "*.onnx")
        voices_file = self._find_file("voices.bin")
        tokens_file = self._find_file("tokens.txt")
        if not model_file or not voices_file or not tokens_file:
            raise FileNotFoundError(f"Kokoro model files not found in {self.model_dir}")

        data_dir = self._find_file("espeak-ng-data")
        if data_dir and not Path(data_dir).is_dir():
            data_dir = str(self.model_dir / "espeak-ng-data")

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            provider=self.provider,
            num_threads=self.num_threads,
            debug=False,
        )
        model_config.kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=model_file,
            voices=voices_file,
            tokens=tokens_file,
            data_dir=data_dir,
            dict_dir=str(self.model_dir) + "/",
            length_scale=1.0,
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config)
        if not tts_config.validate():
            raise ValueError(f"Invalid Kokoro TTS configuration for {self.model_dir}")
        return sherpa_onnx.OfflineTts(tts_config)

    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        sid: int | None = None,
        profile: str = "dialogue",
    ) -> TTSResult:
        start = time.monotonic()
        use_speed = self.speed if speed is None else speed
        use_sid = self.sid if sid is None else sid
        audio = self._tts.generate(text, sid=use_sid, speed=use_speed)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if len(samples) == 0:
            raise ValueError(f"{self.name} TTS generation failed: empty audio")

        elapsed = time.monotonic() - start
        duration = len(samples) / int(audio.sample_rate)
        return TTSResult(
            samples=samples,
            sample_rate=int(audio.sample_rate),
            audio_duration_s=duration,
            synth_latency_s=elapsed,
            backend=self.name,
            model_id=self.model_id,
            profile=profile,
            normalized_text=text,
            rtf=elapsed / duration if duration > 0 else float("inf"),
        )
