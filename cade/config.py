"""
Configuration layer.

Supports seamless switching between cloud and local LLM backends.
All settings are env-first: set CADE_* environment variables to override defaults.
"""

from enum import Enum
from pathlib import Path
from typing import List
import os
from dotenv import load_dotenv

# Load .env; fall back to .env.example so out-of-the-box defaults work.
_env = Path(__file__).resolve().parent.parent / ".env"
if not _env.is_file():
    _env = Path(__file__).resolve().parent.parent / ".env.example"
load_dotenv(_env, override=False)

_MODELS_ROOT = Path("/home/pinggu/audio/models")


class RunMode(str, Enum):
    """Runtime mode."""
    CLOUD = "CLOUD"
    LOCAL = "LOCAL"


class Config:
    """
    Global configuration.

    Development usually uses CLOUD mode. Deployment can switch to LOCAL mode
    through environment variables without changing application code.
    """

    # -- Mode --
    MODE: RunMode = RunMode(os.getenv("CADE_MODE", "CLOUD").upper())

    # -- Cloud LLM --
    CLOUD_BASE_URL = os.getenv("CADE_CLOUD_BASE_URL", "https://api.deepseek.com")
    CLOUD_API_KEY = os.getenv("CADE_CLOUD_API_KEY", "sk-placeholder")
    CLOUD_MODEL = os.getenv("CADE_CLOUD_MODEL", "deepseek-chat")

    # -- Local LLM (llama-server / Ollama / any OpenAI-compatible) --
    LOCAL_BASE_URL = os.getenv("CADE_LOCAL_BASE_URL", "http://127.0.0.1:8080/v1")
    LOCAL_API_KEY = os.getenv("CADE_LOCAL_API_KEY", "not-needed")
    LOCAL_MODEL = os.getenv("CADE_LOCAL_MODEL", "qwen3.5-9b-q8-local")

    # -- Generation --
    TEMPERATURE = float(os.getenv("CADE_TEMPERATURE", "0.2"))
    MAX_TOKENS = int(os.getenv("CADE_MAX_TOKENS", "256"))
    TIMEOUT = int(os.getenv("CADE_TIMEOUT", "60"))

    # -- Robot identity --
    ROBOT_NAME = os.getenv("CADE_ROBOT_NAME", "LARA")

    # -- Voice / Language --
    VOICE_LANGUAGE = os.getenv("CADE_VOICE_LANGUAGE", "en")
    ECHO_SUPPRESS_MS = int(os.getenv("CADE_ECHO_SUPPRESS_MS", "300"))
    ECHO_SIMILARITY_THRESHOLD = float(os.getenv("CADE_ECHO_SIMILARITY_THRESHOLD", "0.75"))
    ECHO_SIMILARITY_WINDOW_SEC = float(os.getenv("CADE_ECHO_SIMILARITY_WINDOW_SEC", "2.5"))

    # -- Audio devices --
    INPUT_DEVICE = os.getenv("CADE_INPUT_DEVICE", "default")
    OUTPUT_DEVICE = os.getenv("CADE_OUTPUT_DEVICE", "default")
    # PipeWire/PulseAudio source name for virtual devices (e.g. nx_remapped_out).
    # When set, ASR uses parec instead of sounddevice so it can capture
    # from PipeWire virtual sources that sounddevice cannot see.
    INPUT_SOURCE = os.getenv("CADE_INPUT_SOURCE", "")

    # -- ASR --
    ASR_PROVIDER = os.getenv("CADE_ASR_PROVIDER", "cpu")
    ASR_MODEL_DIR = os.getenv(
        "CADE_ASR_MODEL_DIR",
        str(_MODELS_ROOT / "asr" / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"),
    )
    ASR_MODEL_TYPE = os.getenv("CADE_ASR_MODEL_TYPE", "streaming_nemotron")
    VAD_MODEL = os.getenv(
        "CADE_VAD_MODEL",
        str(_MODELS_ROOT / "asr" / "silero_vad.onnx"),
    )

    # -- ASR Fallback --
    ASR_FALLBACK_MODEL_TYPE = os.getenv("CADE_ASR_FALLBACK_MODEL_TYPE", "streaming_zipformer")
    ASR_FALLBACK_MODEL_DIR = os.getenv(
        "CADE_ASR_FALLBACK_MODEL_DIR",
        str(_MODELS_ROOT / "asr" / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17-mobile"),
    )

    # -- ASR Text Normalization --
    ASR_REPLACEMENTS = os.getenv("CADE_ASR_REPLACEMENTS", "")

    # -- TTS --
    TTS_PROVIDER = os.getenv("CADE_TTS_PROVIDER", "cpu")
    TTS_NUM_THREADS = int(os.getenv("CADE_TTS_NUM_THREADS", "2"))
    TTS_SPEED = float(os.getenv("CADE_TTS_SPEED", "1.05"))
    TTS_SID = int(os.getenv("CADE_TTS_SID", "0"))
    TTS_ROUTER_ENABLED = os.getenv("CADE_TTS_ROUTER_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    TTS_DEFAULT_BACKEND = os.getenv("CADE_TTS_DEFAULT_BACKEND", "kokoro")
    TTS_FAST_BACKEND = os.getenv("CADE_TTS_FAST_BACKEND", "piper")
    TTS_FALLBACK_BACKEND = os.getenv("CADE_TTS_FALLBACK_BACKEND", "vits")
    TTS_MODEL_DIR = os.getenv(
        "CADE_TTS_MODEL_DIR",
        str(_MODELS_ROOT / "tts" / "vits-piper-en_US-libritts_r-medium-int8"),
    )
    TTS_KOKORO_MODEL_DIR = os.getenv(
        "CADE_TTS_KOKORO_MODEL_DIR",
        str(_MODELS_ROOT / "tts" / "kokoro-en-v0_19"),
    )
    TTS_PIPER_MODEL_DIR = os.getenv(
        "CADE_TTS_PIPER_MODEL_DIR",
        str(_MODELS_ROOT / "tts" / "vits-piper-en_US-lessac-medium"),
    )
    TTS_VITS_MODEL_DIR = os.getenv(
        "CADE_TTS_VITS_MODEL_DIR",
        TTS_MODEL_DIR,
    )
    TTS_CACHE_ENABLED = os.getenv("CADE_TTS_CACHE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    TTS_CACHE_DIR = os.getenv("CADE_TTS_CACHE_DIR", "/home/pinggu/.cache/cade/tts")
    TTS_PREWARM_ENABLED = os.getenv("CADE_TTS_PREWARM_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    TTS_TEXT_NORMALIZE = os.getenv("CADE_TTS_TEXT_NORMALIZE", "true").lower() in ("1", "true", "yes", "on")
    TTS_PLAYBACK_BACKEND = os.getenv("CADE_TTS_PLAYBACK_BACKEND", "sounddevice_stream")
    TTS_USE_PAPLAY_FALLBACK = os.getenv("CADE_TTS_USE_PAPLAY_FALLBACK", "true").lower() in ("1", "true", "yes", "on")
    TTS_CHUNKING_ENABLED = os.getenv("CADE_TTS_CHUNKING_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    TTS_MIN_CHUNK_CHARS = int(os.getenv("CADE_TTS_MIN_CHUNK_CHARS", "20"))
    TTS_MAX_CHUNK_CHARS = int(os.getenv("CADE_TTS_MAX_CHUNK_CHARS", "160"))
    BARGE_IN_ENABLED = os.getenv("CADE_BARGE_IN_ENABLED", "false").lower() in ("1", "true", "yes", "on")
    ECHO_SUPPRESS_AFTER_MS = int(os.getenv("CADE_ECHO_SUPPRESS_AFTER_MS", str(ECHO_SUPPRESS_MS)))

    # -- Logging --
    LOG_LEVEL = os.getenv("CADE_LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("CADE_LOG_FILE", "logs/robot.log")

    # -- Structured Output --
    LLM_STRUCTURED_PROFILE = os.getenv("CADE_LLM_STRUCTURED_PROFILE", "auto")
    LLM_REQUIRE_STRUCTURED = os.getenv("CADE_LLM_REQUIRE_STRUCTURED", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_llm_config(cls) -> dict:
        """Get the LLM configuration for the active runtime mode."""
        if cls.MODE == RunMode.CLOUD:
            return {
                "base_url": cls.CLOUD_BASE_URL,
                "api_key": cls.CLOUD_API_KEY,
                "model": cls.CLOUD_MODEL,
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS,
                "timeout": cls.TIMEOUT,
            }
        else:
            return {
                "base_url": cls.LOCAL_BASE_URL,
                "api_key": cls.LOCAL_API_KEY,
                "model": cls.LOCAL_MODEL,
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS,
                "timeout": cls.TIMEOUT,
            }

    @classmethod
    def is_cloud_mode(cls) -> bool:
        return cls.MODE == RunMode.CLOUD

    @classmethod
    def is_local_mode(cls) -> bool:
        return cls.MODE == RunMode.LOCAL

    @classmethod
    def validate(cls) -> List[str]:
        """
        Validate configuration and return a list of human-readable warnings.
        Empty list means everything looks fine.
        """
        warnings: List[str] = []

        # LLM URL reachable? (skip for cloud — we can't check without a key)
        if cls.is_local_mode():
            url = cls.LOCAL_BASE_URL.rstrip("/")
            if not url.startswith("http"):
                warnings.append(f"CADE_LOCAL_BASE_URL looks invalid: {cls.LOCAL_BASE_URL}")

        # ASR model dir
        asr_dir = Path(cls.ASR_MODEL_DIR)
        if not asr_dir.is_dir():
            warnings.append(f"ASR model dir not found: {asr_dir}")

        # VAD model
        vad_path = Path(cls.VAD_MODEL)
        if not vad_path.is_file():
            warnings.append(f"VAD model not found: {vad_path}")

        # TTS model dir
        tts_dir = Path(cls.TTS_MODEL_DIR)
        if not tts_dir.is_dir():
            warnings.append(f"TTS model dir not found: {tts_dir}")
        vits_dir = Path(cls.TTS_VITS_MODEL_DIR)
        if not vits_dir.is_dir():
            warnings.append(f"TTS VITS fallback model dir not found: {vits_dir}")
        kokoro_dir = Path(cls.TTS_KOKORO_MODEL_DIR)
        if cls.TTS_DEFAULT_BACKEND.lower() == "kokoro" and not kokoro_dir.is_dir():
            warnings.append(
                f"TTS Kokoro model dir not found: {kokoro_dir}; runtime will use fallback if available"
            )
        piper_dir = Path(cls.TTS_PIPER_MODEL_DIR)
        if cls.TTS_FAST_BACKEND.lower() == "piper" and not piper_dir.is_dir():
            warnings.append(
                f"TTS Piper fast model dir not found: {piper_dir}; runtime will use VITS fallback if available"
            )

        return warnings
