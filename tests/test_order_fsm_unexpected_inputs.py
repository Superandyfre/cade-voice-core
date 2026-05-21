"""
Unexpected-input test matrix for OrderSubFSM.

Tests that every possible user input has a deterministic outcome:
correct state transition, correct events, correct TTS behavior.
"""

import os
import json
import threading
import time
import pytest
from unittest.mock import MagicMock
from typing import List

from cade.brain.schemas import OrderAction, OrderItem, OrderCheckDecision, OrderSpeakDecision, SpeakAction, FixOrderAction
from cade.fsm.config import OrderFSMConfig
from cade.fsm.events import OrderStateEvent, OrderConfirmedEvent, OrderCancelledEvent
from cade.fsm.order_fsm import (
    OrderSubFSM,
    LocalOrderIdProvider,
    LocalOrderStorage,
    CallbackTTSSink,
    CallbackEventSink,
    TTSPlaybackError,
    InputResult,
)
from cade.fsm.order_parser import OrderInputKind
from cade.fsm.states import OrderState


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_config(**overrides) -> OrderFSMConfig:
    defaults = dict(
        order_base_dir="/tmp/test_orders_unexpected",
        food_aliases={
            "water": ["water", "bottle of water"],
            "coke": ["coke", "cola"],
            "fried_rice": ["fried rice"],
            "fries": ["fries", "french fries"],
            "burger": ["burger", "hamburger"],
        },
        ask_prompt="What would you like to order?",
        repeat_instruction="Repeat the order.",
        listen_retry_prompt="Sorry, please say again.",
        fix_missing_prompt="Please tell me the changes.",
        check_retry_prompt="Is the order correct?",
        finish_template="OK I'll get {foods} for you",
        input_dedup_window_sec=1.5,
        llm_max_retries=1,
        order_id_proposal_timeout_sec=2.0,
        listen_max_retries=5,
        check_max_retries=5,
        empty_input_max=3,
    )
    defaults.update(overrides)
    return OrderFSMConfig(**defaults)


def _make_fsm(config=None, llm=None, tts_sink=None, event_sink=None):
    config = config or _make_config()
    llm = llm or _make_mock_llm()
    storage = MagicMock()
    storage.load_known_ids.return_value = set()
    storage.create_order_dir.return_value = "/tmp/test_orders_unexpected/00001"

    events = event_sink or MagicMock()
    tts = tts_sink or MagicMock()

    fsm = OrderSubFSM(
        llm_client=llm,
        config=config,
        order_id_provider=LocalOrderIdProvider(),
        order_storage=storage,
        tts_sink=tts,
        event_sink=events,
    )
    return fsm


def _make_mock_llm():
    llm = MagicMock()
    return llm


def _enter_paused_ordering(fsm, serving_payload=None):
    payload = serving_payload or {"state": "PAUSED_ORDERING"}
    fsm.handle_serving_state(payload)
    time.sleep(0.3)


def _send_listen_input(fsm, text, wait=0.3):
    fsm.handle_user_text(text, source="primary")
    time.sleep(wait)


def _advance_to_check(fsm, items=None):
    """Advance FSM from NOT_PERMITTED through to CHECK state using FSM's existing LLM."""
    items = items or [OrderItem(name="coke", qty=1)]
    # Configure the FSM's existing LLM mock
    fsm._llm.get_order_action.return_value = OrderAction(type="order", items=items)
    fsm._llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
        action=SpeakAction(content="Coke. Correct?")
    )

    _enter_paused_ordering(fsm)
    _send_listen_input(fsm, "I want a coke")
    return fsm._llm


# ------------------------------------------------------------------
# LISTEN stage: non-order inputs
# ------------------------------------------------------------------

class TestListenCancel:
    def test_cancel_goes_to_not_permitted(self):
        tts_texts = []
        events = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        _enter_paused_ordering(fsm)
        assert fsm._state == OrderState.LISTEN

        _send_listen_input(fsm, "cancel")
        assert fsm._state == OrderState.NOT_PERMITTED

        cancelled_events = [e for e in events if e[0] == "order.cancelled"]
        assert len(cancelled_events) >= 1

    def test_never_mind_goes_to_not_permitted(self):
        fsm = _make_fsm()
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "never mind")
        assert fsm._state == OrderState.NOT_PERMITTED


class TestListenRepeatRequest:
    def test_say_again_replays_ask(self):
        tts_texts = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
        )
        _enter_paused_ordering(fsm)
        assert fsm._state == OrderState.LISTEN

        _send_listen_input(fsm, "say again")
        assert fsm._state == OrderState.LISTEN
        assert fsm.config.ask_prompt in tts_texts


class TestListenPauseRequest:
    def test_wait_stays_in_listen(self):
        tts_texts = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
        )
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "wait a second")
        assert fsm._state == OrderState.LISTEN
        assert any("time" in t.lower() for t in tts_texts)


class TestListenMenuQuestion:
    def test_what_do_you_have_stays_in_listen(self):
        tts_texts = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
        )
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "what do you have")
        assert fsm._state == OrderState.LISTEN
        assert any("we have" in t.lower() for t in tts_texts)


class TestListenOutOfWorkMenu:
    def test_i_want_pizza_not_in_menu(self):
        tts_texts = []
        config = _make_config(
            food_aliases={"coke": ["coke"], "water": ["water"]},
            rule_parse_enabled=True,
        )
        fsm = _make_fsm(
            config=config,
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
        )
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want pizza")
        assert fsm._state == OrderState.LISTEN
        assert any("don't have" in t.lower() or "sorry" in t.lower() for t in tts_texts)


class TestListenSmalltalk:
    def test_hello_stays_in_listen(self):
        tts_texts = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
        )
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "hello")
        assert fsm._state == OrderState.LISTEN


class TestListenNoise:
    def test_uh_stays_in_listen(self):
        fsm = _make_fsm()
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "uh")
        assert fsm._state == OrderState.LISTEN

    def test_empty_text_returns_not_accepted(self):
        fsm = _make_fsm()
        result = fsm.handle_user_text("", source="primary")
        assert isinstance(result, InputResult)
        assert result.accepted is False

    def test_exceeding_empty_input_max_cancels(self):
        config = _make_config(empty_input_max=2)
        fsm = _make_fsm(config=config)
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "uh")
        assert fsm._state == OrderState.LISTEN
        _send_listen_input(fsm, "um")
        assert fsm._state == OrderState.LISTEN
        # Third empty input should exceed limit
        _send_listen_input(fsm, "hmm")
        assert fsm._state == OrderState.NOT_PERMITTED


class TestListenRetryLimit:
    def test_exceeding_listen_max_retries_cancels(self):
        config = _make_config(
            listen_max_retries=2,
            rule_parse_enabled=False,
        )
        llm = _make_mock_llm()
        llm.get_order_action.side_effect = Exception("LLM failed")
        fsm = _make_fsm(config=config, llm=llm)
        _enter_paused_ordering(fsm)

        _send_listen_input(fsm, "something unclear")
        assert fsm._state == OrderState.LISTEN
        _send_listen_input(fsm, "still unclear")
        assert fsm._state == OrderState.LISTEN
        # Third attempt exceeds limit
        _send_listen_input(fsm, "another try")
        assert fsm._state == OrderState.NOT_PERMITTED


class TestListenOutOfWorkScope:
    def test_bathroom_stays_in_listen(self):
        tts_texts = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
        )
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "where is the bathroom")
        assert fsm._state == OrderState.LISTEN
        assert any("order" in t.lower() for t in tts_texts)


# ------------------------------------------------------------------
# CHECK stage: non-confirm inputs
# ------------------------------------------------------------------

class TestCheckCancel:
    def test_cancel_in_check_goes_to_not_permitted(self):
        events = []
        fsm = _make_fsm(
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        _send_listen_input(fsm, "cancel")
        assert fsm._state == OrderState.NOT_PERMITTED

        cancelled_events = [e for e in events if e[0] == "order.cancelled"]
        assert len(cancelled_events) >= 1


class TestCheckRepeatRequest:
    def test_repeat_request_replays_order(self):
        tts_texts = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
        )
        llm = _advance_to_check(fsm)

        _send_listen_input(fsm, "what did I order")
        # Should go back through REPEAT to CHECK
        time.sleep(0.3)
        assert fsm._state == OrderState.CHECK


class TestCheckYesButModify:
    """Critical: modification signals must override positive words."""

    def test_yes_but_make_it_two_does_not_confirm(self):
        events = []
        fsm = _make_fsm(
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        _send_listen_input(fsm, "yes but make it two")
        time.sleep(0.3)
        # Should NOT be NOT_PERMITTED (would mean confirmed)
        # Instead should be back in CHECK (after REPEAT of modified order)
        confirmed_events = [e for e in events if e[0] == "order.confirmed"]
        assert len(confirmed_events) == 0

    def test_ok_add_fries_does_not_confirm(self):
        events = []
        fsm = _make_fsm(
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        _send_listen_input(fsm, "ok add fries")
        time.sleep(0.3)
        confirmed_events = [e for e in events if e[0] == "order.confirmed"]
        assert len(confirmed_events) == 0

    def test_yeah_no_coke_does_not_confirm(self):
        events = []
        fsm = _make_fsm(
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        _send_listen_input(fsm, "yeah no coke")
        time.sleep(0.3)
        confirmed_events = [e for e in events if e[0] == "order.confirmed"]
        assert len(confirmed_events) == 0


class TestCheckPlainConfirm:
    def test_yes_confirms_order(self):
        events = []
        fsm = _make_fsm(
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        _send_listen_input(fsm, "yes")
        time.sleep(0.3)
        assert fsm._state == OrderState.NOT_PERMITTED
        confirmed_events = [e for e in events if e[0] == "order.confirmed"]
        assert len(confirmed_events) >= 1

    def test_correct_confirms_order(self):
        events = []
        fsm = _make_fsm(
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        _send_listen_input(fsm, "that's correct")
        time.sleep(0.3)
        confirmed_events = [e for e in events if e[0] == "order.confirmed"]
        assert len(confirmed_events) >= 1


class TestCheckNoWithFix:
    def test_no_two_waters_instead_updates_order(self):
        fsm = _make_fsm()
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(
            result="wrong",
            action=FixOrderAction(type="fix_order", items=[OrderItem(name="water", qty=2)]),
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Two waters. Correct?")
        )

        _send_listen_input(fsm, "no, two waters instead")
        time.sleep(0.3)
        assert fsm._state == OrderState.CHECK
        assert fsm._current_order is not None
        names = [i.name for i in fsm._current_order.items]
        assert "water" in names


class TestCheckPlainNo:
    def test_plain_no_asks_for_changes(self):
        tts_texts = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda t, **kw: tts_texts.append(t)),
        )
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(
            result="wrong", reply="What would you like instead?"
        )

        _send_listen_input(fsm, "no")
        time.sleep(0.3)
        assert fsm._state == OrderState.LISTEN


class TestCheckNoiseAndEmpty:
    def test_uh_in_check_stays_in_check(self):
        fsm = _make_fsm()
        llm = _advance_to_check(fsm)

        _send_listen_input(fsm, "uh")
        # Should still be in CHECK (noise ignored)
        assert fsm._state == OrderState.CHECK

    def test_empty_in_check_returns_not_accepted(self):
        fsm = _make_fsm()
        llm = _advance_to_check(fsm)
        result = fsm.handle_user_text("", source="primary")
        assert isinstance(result, InputResult)
        assert result.accepted is False


class TestCheckRetryLimit:
    def test_exceeding_check_max_retries_cancels(self):
        config = _make_config(check_max_retries=2, rule_parse_enabled=False)
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(type="order", items=[OrderItem(name="coke", qty=1)])
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke. Correct?")
        )
        llm.get_order_check_decision.return_value = OrderCheckDecision(
            result="wrong", reply="What would you like?"
        )
        fsm = _make_fsm(config=config, llm=llm)
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")

        # Send multiple "wrong" responses
        _send_listen_input(fsm, "hmm not sure", wait=0.3)
        assert fsm._state == OrderState.LISTEN

        _send_listen_input(fsm, "I want a coke", wait=0.3)
        _send_listen_input(fsm, "hmm not sure", wait=0.3)

        _send_listen_input(fsm, "I want a coke", wait=0.3)
        _send_listen_input(fsm, "hmm not sure", wait=0.3)

        # After enough retries, should cancel
        assert fsm._state == OrderState.NOT_PERMITTED


# ------------------------------------------------------------------
# Component failure tests
# ------------------------------------------------------------------

class TestTTSFailures:
    def test_tts_fail_at_finish_still_confirms(self):
        """FINISH TTS failure should not prevent order confirmation."""
        events = []
        call_count = [0]
        def tts_fail_on_finish(text, **kw):
            call_count[0] += 1
            if "I'll get" in text:
                raise RuntimeError("TTS device error")
            return None

        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(tts_fail_on_finish),
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        llm = _advance_to_check(fsm)
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        # Order should still be confirmed even though finish TTS failed
        confirmed = [e for e in events if e[0] == "order.confirmed"]
        assert len(confirmed) >= 1
        assert fsm._state == OrderState.NOT_PERMITTED

    def test_tts_fail_at_listen_retry_is_soft(self):
        """LISTEN retry TTS failure should not prevent further input."""
        events = []
        call_count = [0]
        def tts_fail_on_retry(text, **kw):
            call_count[0] += 1
            if "sorry" in text.lower() or "again" in text.lower():
                raise RuntimeError("TTS error on retry")
            return None

        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(tts_fail_on_retry),
            event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
        )
        _enter_paused_ordering(fsm)

        # Send unclear input that triggers retry
        _send_listen_input(fsm, "something unclear", wait=0.3)
        # FSM should still be in LISTEN despite TTS failure on retry prompt
        assert fsm._state == OrderState.LISTEN

        # Should still accept new input
        result = fsm.handle_user_text("I want a coke", source="primary")
        assert result.accepted is True


class TestLLMFailures:
    def test_llm_fail_at_listen_retries(self):
        config = _make_config(rule_parse_enabled=False)
        llm = _make_mock_llm()
        llm.get_order_action.side_effect = Exception("LLM timeout")
        fsm = _make_fsm(config=config, llm=llm)
        _enter_paused_ordering(fsm)

        _send_listen_input(fsm, "I want something")
        assert fsm._state == OrderState.LISTEN

    def test_llm_fail_at_check_retries(self):
        config = _make_config(rule_parse_enabled=False)
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(type="order", items=[OrderItem(name="coke", qty=1)])
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke. Correct?")
        )
        llm.get_order_check_decision.side_effect = Exception("LLM timeout")
        fsm = _make_fsm(config=config, llm=llm)
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")

        _send_listen_input(fsm, "yes")
        assert fsm._state == OrderState.CHECK


# ------------------------------------------------------------------
# InputResult / ACK tests
# ------------------------------------------------------------------

class TestInputResult:
    def test_accepted_returns_input_result(self):
        fsm = _make_fsm()
        _enter_paused_ordering(fsm)
        result = fsm.handle_user_text("I want a coke", source="primary")
        assert isinstance(result, InputResult)
        assert result.accepted is True
        assert result.state == "LISTEN"

    def test_invalid_state_returns_reason(self):
        fsm = _make_fsm()
        result = fsm.handle_user_text("hello", source="primary")
        assert isinstance(result, InputResult)
        assert result.accepted is False
        assert result.reason == "invalid_state"

    def test_empty_text_returns_reason(self):
        fsm = _make_fsm()
        result = fsm.handle_user_text("", source="primary")
        assert isinstance(result, InputResult)
        assert result.accepted is False
        assert result.reason == "empty_text"

    def test_duplicate_returns_reason(self):
        config = _make_config(input_dedup_window_sec=10.0)
        fsm = _make_fsm(config=config)
        _enter_paused_ordering(fsm)

        result1 = fsm.handle_user_text("hello", source="primary")
        assert result1.accepted is True

        result2 = fsm.handle_user_text("hello", source="primary")
        assert result2.accepted is False
        assert result2.reason == "duplicate_input"

    def test_processing_busy_returns_reason(self):
        config = _make_config(rule_parse_enabled=False)
        llm = _make_mock_llm()
        # Make LLM slow so processing flag stays True
        def slow_llm(*args, **kwargs):
            time.sleep(1)
            return OrderAction(type="order", items=[OrderItem(name="coke", qty=1)])
        llm.get_order_action.side_effect = slow_llm

        fsm = _make_fsm(config=config, llm=llm)
        _enter_paused_ordering(fsm)

        result1 = fsm.handle_user_text("I want a coke", source="primary")
        assert result1.accepted is True

        time.sleep(0.05)  # let thread start
        result2 = fsm.handle_user_text("I want water", source="primary")
        assert result2.accepted is False
        assert result2.reason == "processing_busy"


# ------------------------------------------------------------------
# InputClassifier unit tests
# ------------------------------------------------------------------

class TestInputClassifier:
    def test_cancel_detected(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("cancel")
        assert result.kind == OrderInputKind.CANCEL_REQUEST

    def test_repeat_request_detected(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("say again")
        assert result.kind == OrderInputKind.REPEAT_REQUEST

    def test_menu_question_detected(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("what do you have")
        assert result.kind == OrderInputKind.MENU_QUESTION

    def test_valid_order_detected(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"], "water": ["water"]})
        result = cls.classify("I want a coke")
        assert result.kind == OrderInputKind.VALID_ORDER

    def test_noise_detected(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("uh")
        assert result.kind == OrderInputKind.EMPTY_OR_NOISE

    def test_no_is_not_noise(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("no")
        assert result.kind != OrderInputKind.EMPTY_OR_NOISE

    def test_yes_is_not_noise(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("yes")
        assert result.kind != OrderInputKind.EMPTY_OR_NOISE

    def test_smalltalk_detected(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("hello")
        assert result.kind == OrderInputKind.SMALLTALK

    def test_out_of_scope_detected(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("where is the bathroom")
        assert result.kind == OrderInputKind.OUT_OF_SCOPE

    def test_pause_detected(self):
        from cade.fsm.input_classifier import OrderInputClassifier
        cls = OrderInputClassifier({"coke": ["coke"]})
        result = cls.classify("wait a second")
        assert result.kind == OrderInputKind.PAUSE_REQUEST


# ------------------------------------------------------------------
# ConfirmationParser modification-priority tests
# ------------------------------------------------------------------

class TestConfirmationParserModificationPriority:
    def test_yes_but_does_not_confirm(self):
        from cade.fsm.order_parser import ConfirmationParser
        from cade.fsm.menu_context import MenuContext, MenuItem
        parser = ConfirmationParser()
        menu = MenuContext(candidates=[
            MenuItem(canonical="coke", aliases=["coke"]),
            MenuItem(canonical="fries", aliases=["fries"]),
        ])
        result = parser.parse("yes but add fries", menu)
        assert result.result == "wrong"

    def test_ok_add_fries_does_not_confirm(self):
        from cade.fsm.order_parser import ConfirmationParser
        from cade.fsm.menu_context import MenuContext, MenuItem
        parser = ConfirmationParser()
        menu = MenuContext(candidates=[
            MenuItem(canonical="fries", aliases=["fries"]),
        ])
        result = parser.parse("ok add fries", menu)
        assert result.result == "wrong"

    def test_yes_alone_does_confirm(self):
        from cade.fsm.order_parser import ConfirmationParser
        from cade.fsm.menu_context import MenuContext, MenuItem
        parser = ConfirmationParser()
        menu = MenuContext(candidates=[])
        result = parser.parse("yes", menu)
        assert result.result == "correct"

    def test_cancel_in_check(self):
        from cade.fsm.order_parser import ConfirmationParser
        from cade.fsm.menu_context import MenuContext, MenuItem
        parser = ConfirmationParser()
        menu = MenuContext(candidates=[])
        result = parser.parse("cancel", menu)
        assert result.result == "cancel"

    def test_repeat_in_check(self):
        from cade.fsm.order_parser import ConfirmationParser
        from cade.fsm.menu_context import MenuContext, MenuItem
        parser = ConfirmationParser()
        menu = MenuContext(candidates=[])
        result = parser.parse("repeat that", menu)
        assert result.result == "repeat_request"

    def test_no_coke_is_modification(self):
        from cade.fsm.order_parser import ConfirmationParser
        from cade.fsm.menu_context import MenuContext, MenuItem
        parser = ConfirmationParser()
        menu = MenuContext(candidates=[
            MenuItem(canonical="coke", aliases=["coke"]),
        ])
        result = parser.parse("yeah no coke", menu)
        assert result.result == "wrong"


# ------------------------------------------------------------------
# Storage tests
# ------------------------------------------------------------------

class TestAtomicWriteStorage:
    def test_save_order_group_uses_atomic_write(self, tmp_path):
        storage = LocalOrderStorage(str(tmp_path / "orders"))
        order_dir = storage.create_order_dir("12345")

        storage.save_order_group(order_dir, {"stage": "confirmed", "order_id": "12345"})

        target = tmp_path / "orders" / "12345" / "order_group.json"
        assert target.exists()
        data = json.loads(target.read_text())
        assert data["stage"] == "confirmed"

        # No tmp file should remain
        assert not (tmp_path / "orders" / "12345" / "order_group.json.tmp").exists()

    def test_append_event_creates_jsonl(self, tmp_path):
        storage = LocalOrderStorage(str(tmp_path / "orders"))
        order_dir = storage.create_order_dir("12345")

        storage.append_event(order_dir, {"type": "session_started", "ts": 1.0})
        storage.append_event(order_dir, {"type": "order_parsed", "ts": 2.0})

        log_path = tmp_path / "orders" / "12345" / "events.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "session_started"
        assert json.loads(lines[1])["type"] == "order_parsed"
