"""TTS backend implementations."""

from cade.tts.backends.base import TTSBackend, TTSResult
from cade.tts.backends.null_backend import NullBackend
from cade.tts.backends.sherpa_kokoro import SherpaKokoroBackend
from cade.tts.backends.sherpa_vits import SherpaVitsBackend

__all__ = [
    "NullBackend",
    "SherpaKokoroBackend",
    "SherpaVitsBackend",
    "TTSBackend",
    "TTSResult",
]
