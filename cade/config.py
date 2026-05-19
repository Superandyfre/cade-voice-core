"""
Configuration layer.

Supports seamless switching between cloud and local LLM backends.
"""

from enum import Enum
import os
from dotenv import load_dotenv

# Load .env as a fallback while preserving explicit system environment values.
load_dotenv(override=False)


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

    MODE: RunMode = RunMode(os.getenv("CADE_MODE", "CLOUD").upper())

    CLOUD_BASE_URL = os.getenv("CADE_CLOUD_BASE_URL", "https://api.deepseek.com")
    CLOUD_API_KEY = os.getenv("CADE_CLOUD_API_KEY", "")
    CLOUD_MODEL = os.getenv("CADE_CLOUD_MODEL", "deepseek-chat")

    LOCAL_BASE_URL = os.getenv("CADE_LOCAL_BASE_URL", "http://localhost:11434/v1")
    LOCAL_API_KEY = os.getenv("CADE_LOCAL_API_KEY", "ollama")
    LOCAL_MODEL = os.getenv("CADE_LOCAL_MODEL", "qwen2.5:3b")

    TEMPERATURE = float(os.getenv("CADE_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("CADE_MAX_TOKENS", "512"))
    TIMEOUT = int(os.getenv("CADE_TIMEOUT", "30"))

    ROBOT_NAME = os.getenv("CADE_ROBOT_NAME", "LARA")

    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/robot.log"

    @classmethod
    def get_llm_config(cls) -> dict:
        """
        Get the LLM configuration for the active runtime mode.

        Returns:
            Dictionary containing base_url, api_key, model, and generation settings.
        """
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
        """Return True when cloud mode is active."""
        return cls.MODE == RunMode.CLOUD

    @classmethod
    def is_local_mode(cls) -> bool:
        """Return True when local mode is active."""
        return cls.MODE == RunMode.LOCAL
