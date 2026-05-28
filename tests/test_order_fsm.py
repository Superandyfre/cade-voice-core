"""Unit tests for OrderSubFSM — pure business logic, no ROS, no real LLM."""

import threading
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, List, Optional, Any

from cade.brain.schemas import OrderAction, OrderItem, OrderCheckDecision, OrderSpeakDecision, SpeakAction, FixOrderAction
from cade.fsm.config import OrderFSMConfig
from cade.fsm.events import OrderStateEvent, TtsRequestEvent, OrderConfirmedEvent
from cade.fsm.order_fsm import (
    OrderSubFSM,
    LocalOrderIdProvider,
    LocalOrderStorage,
    CallbackTTSSink,
    CallbackEventSink,
)
from cade.fsm.states import OrderState


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_config(**overrides) -> OrderFSMConfig:
    defaults = dict(
        order_base_dir="/tmp/test_orders",
        food_aliases={
            "water": ["water", "bottle of water"],
            "coke": ["coke", "cola"],
            "fried_rice": ["fried rice"],
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
    )
    defaults.update(overrides)
    return OrderFSMConfig(**defaults)


def _make_mock_llm():
    llm = MagicMock()
    return llm


def _make_fsm(config=None, llm=None, tts_sink=None, event_sink=None, order_id_provider=None, order_storage=None):
    config = config or _make_config()
    llm = llm or _make_mock_llm()
    storage = order_storage or MagicMock()
    storage.load_known_ids.return_value = set()
    storage.create_order_dir.return_value = "/tmp/test_orders/00001"

    events = event_sink or MagicMock()
    tts = tts_sink or MagicMock()

    fsm = OrderSubFSM(
        llm_client=llm,
        config=config,
        order_id_provider=order_id_provider or LocalOrderIdProvider(),
        order_storage=storage,
        tts_sink=tts,
        event_sink=events,
    )
    return fsm


def _enter_paused_ordering(fsm, serving_payload=None):
    """Helper to transition FSM into PAUSED_ORDERING and wait for ASK->LISTEN."""
    payload = serving_payload or {"state": "PAUSED_ORDERING"}
    fsm.handle_serving_state(payload)
    time.sleep(0.3)  # let ASK stage thread run


def _send_listen_input(fsm, text, wait=0.3):
    """Helper to send text while in LISTEN state."""
    fsm.handle_user_text(text, source="primary")
    time.sleep(wait)


# ------------------------------------------------------------------
# State enum
# ------------------------------------------------------------------

class TestOrderState:

    def test_states_exist(self):
        assert OrderState.NOT_PERMITTED.value == "NOT_PERMITTED"
        assert OrderState.PERMITTED.value == "PERMITTED"
        assert OrderState.ASK.value == "ASK"
        assert OrderState.LISTEN.value == "LISTEN"
        assert OrderState.REPEAT.value == "REPEAT"
        assert OrderState.CHECK.value == "CHECK"
        assert OrderState.FINISH.value == "FINISH"

    def test_states_are_string_enum(self):
        assert isinstance(OrderState.NOT_PERMITTED, str)
        assert OrderState.NOT_PERMITTED == "NOT_PERMITTED"


# ------------------------------------------------------------------
# Happy path: NOT_PERMITTED -> ... -> FINISH -> NOT_PERMITTED
# ------------------------------------------------------------------

class TestHappyPath:

    def test_full_order_flow(self):
        """Complete order: ASK -> LISTEN -> REPEAT -> CHECK(correct) -> FINISH."""
        llm = _make_mock_llm()

        # LISTEN: parse order
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        # REPEAT: generate speak
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="You ordered coke. Is that correct?")
        )
        # CHECK: confirm correct
        llm.get_order_check_decision.return_value = OrderCheckDecision(
            result="correct"
        )

        tts_texts = []
        confirmed_events = []

        def capture_tts(text):
            tts_texts.append(text)

        def capture_event(topic, payload):
            if topic == "order.confirmed":
                confirmed_events.append(payload)

        fsm = _make_fsm(
            llm=llm,
            tts_sink=CallbackTTSSink(capture_tts),
            event_sink=CallbackEventSink(capture_event),
        )

        # Enter PAUSED_ORDERING
        _enter_paused_ordering(fsm)
        assert fsm._state == OrderState.LISTEN

        # LISTEN: user orders
        _send_listen_input(fsm, "I want a coke")
        assert fsm._state == OrderState.CHECK

        # CHECK: user confirms
        _send_listen_input(fsm, "yes")

        # Should be back to NOT_PERMITTED
        time.sleep(0.3)
        assert fsm._state == OrderState.NOT_PERMITTED
        assert fsm.orders_confirmed == 1
        assert len(confirmed_events) == 1
        assert confirmed_events[0]["foods"] == ["coke"]

    def test_order_with_multiple_items(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order",
            items=[OrderItem(name="coke", qty=2), OrderItem(name="water", qty=1)],
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="2 coke and water. Correct?")
        )
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        confirmed_events = []
        fsm = _make_fsm(
            llm=llm,
            event_sink=CallbackEventSink(
                lambda t, p: confirmed_events.append(p) if t == "order.confirmed" else None
            ),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "two cokes and a water")
        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        assert fsm.orders_confirmed == 1
        assert len(confirmed_events) == 1
        foods_with_qty = confirmed_events[0]["foods_with_qty"]
        assert {"name": "coke", "qty": 2} in foods_with_qty
        assert {"name": "water", "qty": 1} in foods_with_qty


# ------------------------------------------------------------------
# Wrong + fix_order: CHECK(wrong+fix) -> REPEAT -> CHECK(correct)
# ------------------------------------------------------------------

class TestWrongWithFix:

    def test_wrong_then_fix(self):
        llm = _make_mock_llm()

        # First LISTEN: parse wrong order
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        # REPEAT
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="You ordered coke. Correct?")
        )
        # First CHECK: wrong with fix
        llm.get_order_check_decision.side_effect = [
            OrderCheckDecision(
                result="wrong",
                action=FixOrderAction(
                    type="fix_order",
                    items=[OrderItem(name="water", qty=2)],
                ),
            ),
            OrderCheckDecision(result="correct"),
        ]

        confirmed_events = []
        fsm = _make_fsm(
            llm=llm,
            event_sink=CallbackEventSink(
                lambda t, p: confirmed_events.append(p) if t == "order.confirmed" else None
            ),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")
        _send_listen_input(fsm, "no, I want two waters")
        time.sleep(0.3)
        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        assert fsm.orders_confirmed == 1
        assert len(confirmed_events) == 1
        assert confirmed_events[0]["foods"] == ["water"]


# ------------------------------------------------------------------
# Wrong without fix: CHECK(wrong, no action) -> LISTEN
# ------------------------------------------------------------------

class TestWrongWithoutFix:

    def test_wrong_no_fix_goes_to_listen(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke. Correct?")
        )
        # First CHECK: wrong, no fix, has reply
        # Second: after re-listen, correct
        llm.get_order_check_decision.side_effect = [
            OrderCheckDecision(result="wrong", reply="What would you like instead?"),
            OrderAction(type="order", items=[OrderItem(name="water", qty=1)]),
            OrderCheckDecision(result="correct"),
        ]

        tts_texts = []
        fsm = _make_fsm(
            config=_make_config(rule_parse_enabled=False),
            llm=llm,
            tts_sink=CallbackTTSSink(lambda t: tts_texts.append(t)),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")
        # CHECK -> wrong without fix -> LISTEN
        _send_listen_input(fsm, "no")
        time.sleep(0.3)
        assert fsm._state == OrderState.LISTEN
        assert any("What would you like instead?" in t for t in tts_texts)


# ------------------------------------------------------------------
# Dual-channel dedup
# ------------------------------------------------------------------

class TestDualChannelDedup:

    def test_duplicate_text_within_window_is_ignored(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke. Correct?")
        )

        fsm = _make_fsm(llm=llm, config=_make_config(input_dedup_window_sec=5.0))

        _enter_paused_ordering(fsm)

        # First input
        fsm.handle_user_text("I want a coke", source="primary")
        time.sleep(0.5)

        # Duplicate within window
        fsm.handle_user_text("I want a coke", source="secondary")
        assert fsm.ignored_inputs >= 1

    def test_same_text_outside_window_is_not_deduped(self):
        """Same text outside dedup window should not be treated as duplicate."""
        fsm = _make_fsm(config=_make_config(input_dedup_window_sec=0.01))

        # Manually test _is_duplicate_input
        assert fsm._is_duplicate_input("hello") is False  # first time
        time.sleep(0.05)  # outside 0.01s window
        assert fsm._is_duplicate_input("hello") is False  # not duplicate


# ------------------------------------------------------------------
# Ignore input when not in PAUSED_ORDERING
# ------------------------------------------------------------------

class TestIgnoreInputWhenNotPermitted:

    def test_input_ignored_in_not_permitted(self):
        llm = _make_mock_llm()
        fsm = _make_fsm(llm=llm)

        # Not in PAUSED_ORDERING
        fsm.handle_user_text("hello", source="primary")
        assert fsm.ignored_inputs == 1
        llm.get_order_action.assert_not_called()

    def test_input_ignored_when_serving_state_not_paused(self):
        llm = _make_mock_llm()
        fsm = _make_fsm(llm=llm)

        fsm.handle_serving_state({"state": "IDLE"})
        fsm.handle_user_text("hello", source="primary")
        assert fsm.ignored_inputs == 1

    def test_input_ignored_in_ask_state(self):
        llm = _make_mock_llm()
        fsm = _make_fsm(llm=llm)

        fsm.handle_serving_state({"state": "PAUSED_ORDERING"})
        # In ASK state, not yet LISTEN
        time.sleep(0.05)
        fsm.handle_user_text("hello", source="primary")
        # Should be ignored since we're in ASK, not LISTEN/CHECK


# ------------------------------------------------------------------
# Session reset on serving_state change
# ------------------------------------------------------------------

class TestSessionReset:

    def test_reset_when_serving_state_leaves_paused(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )

        fsm = _make_fsm(llm=llm)

        _enter_paused_ordering(fsm)
        assert fsm._state == OrderState.LISTEN

        # Leave PAUSED_ORDERING
        fsm.handle_serving_state({"state": "IDLE"})
        assert fsm._state == OrderState.NOT_PERMITTED
        assert fsm._current_order is None

    def test_session_id_increments_on_reset(self):
        fsm = _make_fsm()
        initial_id = fsm._session_id

        _enter_paused_ordering(fsm)
        mid_id = fsm._session_id
        assert mid_id > initial_id

        fsm.handle_serving_state({"state": "IDLE"})
        after_id = fsm._session_id
        assert after_id > mid_id


# ------------------------------------------------------------------
# LLM failure -> retry prompt
# ------------------------------------------------------------------

class TestLLMFailureRetry:

    def test_listen_llm_failure_sends_retry(self):
        llm = _make_mock_llm()
        llm.get_order_action.side_effect = RuntimeError("LLM down")

        tts_texts = []
        fsm = _make_fsm(
            config=_make_config(rule_parse_enabled=False),
            llm=llm,
            tts_sink=CallbackTTSSink(lambda t: tts_texts.append(t)),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")
        time.sleep(0.3)

        assert any("Sorry" in t or "again" in t for t in tts_texts)
        assert fsm._state == OrderState.LISTEN

    def test_repeat_llm_failure_uses_fallback(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.side_effect = RuntimeError("LLM down")

        tts_texts = []
        fsm = _make_fsm(
            llm=llm,
            tts_sink=CallbackTTSSink(lambda t: tts_texts.append(t)),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")
        time.sleep(0.3)

        # Should use deterministic fallback
        assert any("coke" in t.lower() for t in tts_texts)
        assert fsm._state == OrderState.CHECK

    def test_check_llm_failure_sends_retry(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )
        llm.get_order_check_decision.side_effect = RuntimeError("LLM down")

        tts_texts = []
        fsm = _make_fsm(
            config=_make_config(rule_parse_enabled=False),
            llm=llm,
            tts_sink=CallbackTTSSink(lambda t: tts_texts.append(t)),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")
        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        assert any("correct" in t.lower() or "updated" in t.lower() for t in tts_texts)


# ------------------------------------------------------------------
# TTS failure events
# ------------------------------------------------------------------

class TestTTSFailure:

    def test_tts_error_stops_success_path(self):
        def failing_tts(text):
            raise RuntimeError("audio device gone")

        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        topics = []
        fsm = _make_fsm(
            llm=llm,
            tts_sink=CallbackTTSSink(failing_tts),
            event_sink=CallbackEventSink(
                lambda t, p: topics.append((t, p))
            ),
        )

        fsm.handle_serving_state({"state": "PAUSED_ORDERING"})
        time.sleep(0.3)

        assert fsm._state == OrderState.LISTEN
        assert fsm.orders_confirmed == 0
        assert any(t == "tts.failed" for t, _ in topics)
        assert any(t == "order.error" for t, _ in topics)


# ------------------------------------------------------------------
# TTS completion gates state changes
# ------------------------------------------------------------------

class TestTTSCompletionGates:

    def test_tts_sink_receives_profile(self):
        calls = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda text, profile="dialogue": calls.append((text, profile))),
        )

        _enter_paused_ordering(fsm)

        assert calls
        assert calls[0][1] == "order_prompt"

    def test_tts_completed_includes_playback_durations(self):
        completed = []
        fsm = _make_fsm(
            tts_sink=CallbackTTSSink(lambda text: (1.2, 0.8)),
            event_sink=CallbackEventSink(
                lambda t, p: completed.append(p) if t == "tts.completed" else None
            ),
        )

        _enter_paused_ordering(fsm)

        assert completed
        assert completed[0]["playback_duration_s"] == 1.2
        assert completed[0]["audio_duration_s"] == 0.8

    def test_ask_transitions_to_listen_before_tts_returns(self):
        release = threading.Event()
        started = threading.Event()

        def blocking_tts(text):
            started.set()
            release.wait(timeout=5)

        fsm = _make_fsm(tts_sink=CallbackTTSSink(blocking_tts))
        fsm.handle_serving_state({"state": "PAUSED_ORDERING"})

        assert started.wait(timeout=2)
        time.sleep(0.1)
        assert fsm._state == OrderState.LISTEN

        release.set()
        time.sleep(0.3)
        assert fsm._state == OrderState.LISTEN

    def test_repeat_transitions_to_check_before_tts_returns(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )

        repeat_started = threading.Event()
        repeat_release = threading.Event()

        def tts(text):
            if "confirm" in text.lower() and "coke" in text.lower():
                repeat_started.set()
                repeat_release.wait(timeout=5)

        fsm = _make_fsm(llm=llm, tts_sink=CallbackTTSSink(tts))
        _enter_paused_ordering(fsm)
        fsm.handle_user_text("I want a coke", source="primary")

        assert repeat_started.wait(timeout=2)
        time.sleep(0.1)
        assert fsm._state == OrderState.CHECK

        repeat_release.set()
        time.sleep(0.3)
        assert fsm._state == OrderState.CHECK


# ------------------------------------------------------------------
# Duplicate order ID
# ------------------------------------------------------------------

class TestDuplicateOrderId:

    def test_local_provider_avoids_known_ids(self):
        known = {"12345"}
        provider = LocalOrderIdProvider(known_ids=known)

        # Should not return "12345"
        for _ in range(100):
            result = provider.propose()
            assert result != "12345"

    def test_external_order_id_used_when_proposed(self):
        llm = _make_mock_llm()
        storage = MagicMock()
        storage.load_known_ids.return_value = set()

        fsm = _make_fsm(llm=llm, order_storage=storage)
        fsm._serving_state = "PAUSED_ORDERING"

        # Propose an external ID
        fsm.handle_order_id("99999")
        assert fsm._external_order_id == "99999"


# ------------------------------------------------------------------
# Local order ID fallback
# ------------------------------------------------------------------

class TestLocalOrderIdFallback:

    def test_local_id_generated_when_no_external(self):
        provider = LocalOrderIdProvider()
        result = provider.propose()
        assert result is not None
        assert len(result) == 5
        assert result.isdigit()


# ------------------------------------------------------------------
# Food alias normalization
# ------------------------------------------------------------------

class TestFoodAliasNormalization:

    def test_alias_normalized_to_canonical(self):
        fsm = _make_fsm()
        assert fsm._canonicalize_food_name("cola") == "coke"
        assert fsm._canonicalize_food_name("bottle of water") == "water"
        assert fsm._canonicalize_food_name("fried rice") == "fried_rice"
        assert fsm._canonicalize_food_name("unknown_food") == "unknown_food"

    def test_case_insensitive(self):
        fsm = _make_fsm()
        assert fsm._canonicalize_food_name("COLA") == "coke"
        assert fsm._canonicalize_food_name("Water") == "water"

    def test_order_items_normalized(self):
        fsm = _make_fsm()
        items = [
            OrderItem(name="cola", qty=2),
            OrderItem(name="bottle of water", qty=1),
        ]
        normalized = fsm._normalize_order_items(items)
        names = [i.name for i in normalized]
        assert "coke" in names
        assert "water" in names

    def test_duplicate_items_merged(self):
        fsm = _make_fsm()
        items = [
            OrderItem(name="coke", qty=1),
            OrderItem(name="cola", qty=2),
        ]
        normalized = fsm._normalize_order_items(items)
        assert len(normalized) == 1
        assert normalized[0].name == "coke"
        assert normalized[0].qty == 3


# ------------------------------------------------------------------
# Snapshot
# ------------------------------------------------------------------

class TestSnapshot:

    def test_snapshot_returns_current_state(self):
        fsm = _make_fsm()
        snap = fsm.snapshot()

        assert "state_event" in snap
        assert "order_snapshot" in snap
        assert "metrics" in snap
        assert snap["state_event"]["state"] == "NOT_PERMITTED"

    def test_snapshot_after_order_has_data(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        fsm = _make_fsm(llm=llm)
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")
        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        snap = fsm.snapshot()
        assert snap["metrics"]["orders_confirmed"] == 1


# ------------------------------------------------------------------
# Cancel
# ------------------------------------------------------------------

class TestCancel:

    def test_cancel_resets_to_not_permitted(self):
        llm = _make_mock_llm()
        fsm = _make_fsm(llm=llm)

        _enter_paused_ordering(fsm)
        assert fsm._state != OrderState.NOT_PERMITTED

        fsm.cancel("test_cancel")
        assert fsm._state == OrderState.NOT_PERMITTED


# ------------------------------------------------------------------
# Empty order items -> retry
# ------------------------------------------------------------------

class TestEmptyOrderItems:

    def test_empty_items_sends_retry(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(type="order", items=[])

        tts_texts = []
        fsm = _make_fsm(
            llm=llm,
            tts_sink=CallbackTTSSink(lambda t: tts_texts.append(t)),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "blah blah")
        time.sleep(0.3)

        assert any("Sorry" in t or "again" in t for t in tts_texts)


# ------------------------------------------------------------------
# Concurrent input ignored
# ------------------------------------------------------------------

class TestConcurrentInputIgnored:

    def test_second_input_while_processing_is_ignored(self):
        llm = _make_mock_llm()
        processing_started = threading.Event()
        processing_continue = threading.Event()

        def slow_listen(*args, **kwargs):
            processing_started.set()
            processing_continue.wait(timeout=5)
            return OrderAction(type="order", items=[OrderItem(name="coke", qty=1)])

        llm.get_order_action.side_effect = slow_listen
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )

        fsm = _make_fsm(llm=llm)
        _enter_paused_ordering(fsm)

        # First input starts processing
        fsm.handle_user_text("first input", source="primary")
        processing_started.wait(timeout=2)

        # Second input should be ignored
        fsm.handle_user_text("second input", source="secondary")

        processing_continue.set()
        time.sleep(0.5)

        assert fsm.ignored_inputs >= 1


# ------------------------------------------------------------------
# Order group JSON saved
# ------------------------------------------------------------------

class TestOrderGroupSaved:

    def test_order_group_saved_on_listen(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )

        storage = MagicMock()
        storage.load_known_ids.return_value = set()

        fsm = _make_fsm(llm=llm, order_storage=storage)
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")
        time.sleep(0.3)

        storage.save_order_group.assert_called()

    def test_confirmed_order_group_contains_check_text_before_success(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        storage = MagicMock()
        storage.load_known_ids.return_value = set()
        storage.create_order_dir.return_value = "/tmp/test_orders/00001"

        fsm = _make_fsm(llm=llm, order_storage=storage)
        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "I want a coke")
        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        _, payload = storage.save_order_group.call_args.args
        assert payload["stage"] == "confirmed"
        assert payload["recognized_text"] == "I want a coke"
        assert payload["check_text"] == "yes"


# ------------------------------------------------------------------
# Sustance payload pass-through
# ------------------------------------------------------------------

class TestServingPayloadPassthrough:

    def test_customer_fields_passed_to_confirm(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        confirmed = []
        fsm = _make_fsm(
            llm=llm,
            event_sink=CallbackEventSink(
                lambda t, p: confirmed.append(p) if t == "order.confirmed" else None
            ),
        )

        serving_payload = {
            "state": "PAUSED_ORDERING",
            "customer_id": "cust_42",
            "customer_no": "7",
            "folder": "/data/cust42",
            "customer_folder": "/data/cust42/orders",
        }
        _enter_paused_ordering(fsm, serving_payload=serving_payload)
        _send_listen_input(fsm, "I want a coke")
        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        assert len(confirmed) == 1
        assert confirmed[0].get("customer_id") == "cust_42"
        assert confirmed[0].get("customer_no") == "7"


# ------------------------------------------------------------------
# Input channel mode filtering
# ------------------------------------------------------------------

class TestInputChannelMode:

    def test_mode_primary_ignores_secondary(self):
        llm = _make_mock_llm()
        config = _make_config(input_channel_mode="primary")
        fsm = _make_fsm(llm=llm, config=config)
        _enter_paused_ordering(fsm)
        assert fsm._state == OrderState.LISTEN

        fsm.handle_user_text("I want a coke", source="secondary")
        assert fsm.ignored_inputs == 1
        llm.get_order_action.assert_not_called()

    def test_mode_secondary_ignores_primary(self):
        llm = _make_mock_llm()
        config = _make_config(input_channel_mode="secondary")
        fsm = _make_fsm(llm=llm, config=config)
        _enter_paused_ordering(fsm)
        assert fsm._state == OrderState.LISTEN

        fsm.handle_user_text("I want a coke", source="primary")
        assert fsm.ignored_inputs == 1
        llm.get_order_action.assert_not_called()

    def test_mode_both_accepts_primary_and_secondary(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )
        config = _make_config(input_channel_mode="both")
        fsm = _make_fsm(llm=llm, config=config)
        _enter_paused_ordering(fsm)
        assert fsm._state == OrderState.LISTEN

        fsm.handle_user_text("I want a coke", source="primary")
        time.sleep(0.5)

        assert llm.get_order_action.called
        assert fsm._state == OrderState.CHECK

    def test_invalid_mode_raises_validation_error(self):
        with pytest.raises(Exception):
            _make_config(input_channel_mode="invalid_mode")

    def test_mode_primary_accepts_asr_microphone(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )
        config = _make_config(input_channel_mode="primary")
        fsm = _make_fsm(llm=llm, config=config)
        _enter_paused_ordering(fsm)
        assert fsm._state == OrderState.LISTEN

        fsm.handle_user_text("I want a coke", source="asr_microphone")
        time.sleep(0.5)

        assert llm.get_order_action.called


# ------------------------------------------------------------------
# Rule parser path
# ------------------------------------------------------------------

class TestRuleParserPath:

    def test_simple_order_uses_rule_parser(self):
        llm = _make_mock_llm()
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke. Correct?")
        )
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        tts_texts = []
        fsm = _make_fsm(
            llm=llm,
            config=_make_config(rule_parse_enabled=True),
            tts_sink=CallbackTTSSink(lambda t: tts_texts.append(t)),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "coke")
        time.sleep(0.5)

        # Rule parser should have handled "coke" without calling LLM
        llm.get_order_action.assert_not_called()
        assert fsm._current_order is not None

    def test_yes_confirmation_uses_rule_parser(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )

        confirmed = []
        fsm = _make_fsm(
            llm=llm,
            config=_make_config(rule_parse_enabled=True),
            event_sink=CallbackEventSink(
                lambda t, p: confirmed.append(p) if t == "order.confirmed" else None
            ),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "coke")
        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        # Rule parser should handle "yes" without calling LLM check
        llm.get_order_check_decision.assert_not_called()
        assert fsm.orders_confirmed == 1

    def test_rule_parser_disabled_uses_llm(self):
        llm = _make_mock_llm()
        llm.get_order_action.return_value = OrderAction(
            type="order", items=[OrderItem(name="coke", qty=1)]
        )
        llm.get_order_repeat_speak.return_value = OrderSpeakDecision(
            action=SpeakAction(content="Coke?")
        )
        llm.get_order_check_decision.return_value = OrderCheckDecision(result="correct")

        fsm = _make_fsm(
            llm=llm,
            config=_make_config(rule_parse_enabled=False),
        )

        _enter_paused_ordering(fsm)
        _send_listen_input(fsm, "coke")
        _send_listen_input(fsm, "yes")
        time.sleep(0.3)

        llm.get_order_action.assert_called()
        llm.get_order_check_decision.assert_called()
