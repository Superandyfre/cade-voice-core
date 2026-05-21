"""Unit tests for VoiceSession — echo gate and error fallback."""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from cade.config import Config
from cade.voice.session import VoiceSession, TTSPlaybackState, _FALLBACK_REPLY


def _make_session():
    """Create a VoiceSession with all real engines mocked out."""
    session = VoiceSession.__new__(VoiceSession)
    session.controller = MagicMock()
    session.asr = MagicMock()
    session.tts = MagicMock()
    session.echo_suppress_ms = 100
    session._on_transcript = None
    session._on_decision = None
    session._is_speaking = False
    session._speaking_until = 0.0
    session._tts_state = TTSPlaybackState.IDLE
    session._lock = __import__("threading").Lock()
    session.total_turns = 0
    session.total_errors = 0
    return session


# ------------------------------------------------------------------
# Echo gate
# ------------------------------------------------------------------

class TestEchoGate:

    def test_transcript_ignored_while_speaking(self):
        session = _make_session()
        session._is_speaking = True
        session.controller.process_input = MagicMock()

        session._on_asr_callback("hello")
        session.controller.process_input.assert_not_called()

    def test_transcript_ignored_within_suppress_window(self):
        session = _make_session()
        session._is_speaking = False
        session._speaking_until = time.monotonic() + 10.0  # 10s in the future
        session.controller.process_input = MagicMock()

        session._on_asr_callback("hello")
        session.controller.process_input.assert_not_called()

    def test_transcript_processed_after_window(self):
        session = _make_session()
        session._is_speaking = False
        session._speaking_until = time.monotonic() - 1.0  # already passed
        session.tts.speak.return_value = (0.1, 0.1)
        session.controller.process_input.return_value = {
            "decision": MagicMock(),
            "action_success": True,
            "spoken_text": "hi",
            "timings": {},
        }

        session._on_asr_callback("hello")
        session.controller.process_input.assert_called_once_with("hello")

    def test_empty_transcript_ignored(self):
        session = _make_session()
        session.controller.process_input = MagicMock()

        session._on_asr_callback("")
        session.controller.process_input.assert_not_called()

        session._on_asr_callback("   ")
        session.controller.process_input.assert_not_called()


# ------------------------------------------------------------------
# Error fallback
# ------------------------------------------------------------------

class TestErrorFallback:

    def test_llm_error_triggers_fallback_tts(self):
        session = _make_session()
        session.controller.process_input.side_effect = RuntimeError("LLM down")
        session.tts.speak.return_value = (0.5, 0.5)

        result = session.process_transcript("hello")

        assert result["fallback"] is True
        assert result["spoken_text"] == _FALLBACK_REPLY
        session.tts.speak.assert_called_once_with(_FALLBACK_REPLY, device=None, profile="error")
        assert session.total_errors == 1

    def test_tts_error_still_returns_result(self):
        session = _make_session()
        session.controller.process_input.return_value = {
            "decision": MagicMock(),
            "action_success": True,
            "spoken_text": "ok",
            "timings": {},
        }
        session.tts.speak.side_effect = RuntimeError("audio device gone")

        result = session.process_transcript("hello")
        # Should not raise — TTS error is caught
        assert result["spoken_text"] == "ok"
        assert session.total_turns == 1


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

class TestCallbacks:

    def test_on_transcript_called(self):
        session = _make_session()
        cb = MagicMock()
        session._on_transcript = cb
        session._is_speaking = False
        session._speaking_until = 0.0
        session.tts.speak.return_value = (0.1, 0.1)
        session.controller.process_input.return_value = {
            "decision": MagicMock(),
            "action_success": True,
            "spoken_text": "hi",
            "timings": {},
        }

        session._on_asr_callback("hello")
        cb.assert_called_once_with("hello")

    def test_on_decision_called(self):
        session = _make_session()
        cb = MagicMock()
        session._on_decision = cb
        session.tts.speak.return_value = (0.1, 0.1)

        decision = MagicMock()
        session.controller.process_input.return_value = {
            "decision": decision,
            "action_success": True,
            "spoken_text": "hi",
            "timings": {},
        }

        session.process_transcript("hello")
        cb.assert_called_once()


class TestBargeIn:

    def test_barge_in_stops_tts_and_processes_transcript(self, monkeypatch):
        session = _make_session()
        monkeypatch.setattr(Config, "BARGE_IN_ENABLED", True)
        session._is_speaking = True
        session._tts_state = TTSPlaybackState.PLAYING
        session.tts.stop = MagicMock()
        session.controller.process_input.return_value = {
            "decision": MagicMock(),
            "action_success": True,
            "spoken_text": None,
            "timings": {},
        }

        session._on_asr_callback("new user speech")

        session.tts.stop.assert_called_once()
        session.controller.process_input.assert_called_once_with("new user speech")
