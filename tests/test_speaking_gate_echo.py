"""SpeakingGate echo suppression tests for the ordering sub-FSM."""

import time

from cade.fsm.voice_runtime import SpeakingGate


class TestSpeakingGateBasicBlocking:
    def test_blocks_while_speaking(self):
        gate = SpeakingGate(echo_suppress_ms=0)
        gate.begin("Let me confirm. You ordered one coke.")
        assert gate.is_blocked("yes") is True
        assert gate.is_blocked("i want a water") is True
        assert gate.is_blocked("") is True

    def test_blocks_during_tail_suppress_window(self):
        gate = SpeakingGate(echo_suppress_ms=500)
        gate.begin("What would you like?")
        gate.end("What would you like?")
        assert gate.is_blocked("anything") is True

    def test_allows_after_suppress_window(self):
        gate = SpeakingGate(echo_suppress_ms=50)
        gate.begin("hello")
        gate.end("hello")
        time.sleep(0.1)
        assert gate.is_blocked("i want a coke") is False


class TestSpeakingGateSimilarityBlocking:
    def test_blocks_exact_match_of_tts_text(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=5.0)
        gate.begin("Let me confirm. You ordered coke.")
        gate.end("Let me confirm. You ordered coke.")
        assert gate.is_blocked("let me confirm you ordered coke") is True

    def test_blocks_high_similarity_variant(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=5.0)
        gate.begin("Let me confirm. You ordered coke.")
        gate.end("Let me confirm. You ordered coke.")
        assert gate.is_blocked("let me confirm i ordered coke") is True

    def test_allows_dissimilar_text(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=5.0)
        gate.begin("Let me confirm. You ordered coke.")
        gate.end("Let me confirm. You ordered coke.")
        assert gate.is_blocked("i want a water") is False

    def test_blocks_keyword_echo_from_confirm_sentence(self):
        gate = SpeakingGate(0, similarity_threshold=0.5, similarity_window_sec=5.0)
        gate.begin("Is that correct?")
        gate.end("Is that correct?")
        assert gate.is_blocked("correct") is True

    def test_allows_real_yes_after_window(self):
        gate = SpeakingGate(50, similarity_threshold=0.6, similarity_window_sec=0.05)
        gate.begin("Is that correct?")
        gate.end("Is that correct?")
        time.sleep(0.1)
        assert gate.is_blocked("yes that's correct") is False


class TestSpeakingGateSentenceTypes:
    def test_blocks_echo_of_ask_prompt(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=5.0)
        gate.begin("What would you like to order?")
        gate.end("What would you like to order?")
        assert gate.is_blocked("what would you like to order") is True

    def test_blocks_echo_of_menu_summary(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=5.0)
        gate.begin("We have water, coke, juice, and burger.")
        gate.end("We have water, coke, juice, and burger.")
        assert gate.is_blocked("we have water coke juice and burger") is True

    def test_blocks_echo_of_finish_sentence(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=5.0)
        gate.begin("OK I'll get coke for you")
        gate.end("OK I'll get coke for you")
        assert gate.is_blocked("ok ill get coke for you") is True

    def test_multiple_tts_sentences_tracked(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=5.0, recent_text_limit=4)
        gate.begin("What would you like to order?")
        gate.end("What would you like to order?")
        gate.begin("Let me confirm. You ordered coke.")
        gate.end("Let me confirm. You ordered coke.")
        assert gate.is_blocked("what would you like to order") is True
        assert gate.is_blocked("let me confirm you ordered coke") is True


class TestSpeakingGateWindowExpiry:
    def test_similarity_blocks_expire_after_window(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=0.05)
        gate.begin("You ordered two cokes.")
        gate.end("You ordered two cokes.")
        time.sleep(0.1)
        assert gate.is_blocked("you ordered two cokes") is False

    def test_recent_texts_have_independent_expiry(self):
        gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=0.05, recent_text_limit=4)
        gate.begin("first sentence")
        gate.end("first sentence")
        time.sleep(0.15)
        gate.begin("second sentence")
        gate.end("second sentence")
        time.sleep(0.1)
        assert gate.is_blocked("first sentence") is False
        assert gate.is_blocked("second sentence") is False
