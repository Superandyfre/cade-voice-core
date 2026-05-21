"""Shared test fixtures.

Mocks sounddevice / soundfile at sys.modules level so tests never need
PortAudio or libsndfile installed.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# Mock sounddevice and soundfile before any cade module imports them.
# These are C-extension libraries that fail at import if PortAudio / libsndfile
# are not installed on the host — which is fine for pure-logic unit tests.
for _mod_name in ("sounddevice", "soundfile"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

import pytest


@pytest.fixture(autouse=True)
def _env_override(monkeypatch):
    """Force LOCAL mode and point to non-existent model dirs so that
    tests never accidentally hit a real LLM or load real models."""
    monkeypatch.setenv("CADE_MODE", "LOCAL")
    monkeypatch.setenv("CADE_LOCAL_BASE_URL", "http://127.0.0.1:19999/v1")
    monkeypatch.setenv("CADE_LOCAL_API_KEY", "test-key")
    monkeypatch.setenv("CADE_LOCAL_MODEL", "qwen3.5-9b-q8-local")
    monkeypatch.setenv("CADE_TEMPERATURE", "0.2")
    monkeypatch.setenv("CADE_MAX_TOKENS", "256")
    monkeypatch.setenv("CADE_TIMEOUT", "5")
    monkeypatch.setenv("CADE_ASR_PROVIDER", "cpu")
    monkeypatch.setenv("CADE_TTS_PROVIDER", "cpu")
    monkeypatch.setenv("CADE_ECHO_SUPPRESS_MS", "100")
