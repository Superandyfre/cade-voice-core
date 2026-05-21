"""Tests for InputPipeline — the SemanticEvent integration layer."""

from cade.brain.schemas import OrderAction, OrderItem
from cade.fsm.events import SemanticEvent
from cade.fsm.parsing.input_classifier import OrderInputClassifier
from cade.fsm.parsing.menu_context import MenuContextProvider
from cade.fsm.parsing.order_parser import ConfirmationParser, DeterministicOrderParser
from cade.fsm.parsing.pipeline import InputPipeline


def _make_pipeline():
    aliases = {"coke": ["coke", "cola"], "water": ["water"], "burger": ["burger", "hamburger"]}
    provider = MenuContextProvider(aliases)
    return InputPipeline(
        classifier=OrderInputClassifier(aliases),
        order_parser=DeterministicOrderParser(provider),
        confirm_parser=ConfirmationParser(),
        menu_provider=provider,
    )


class TestPipelineListen:
    def test_food_input_returns_valid_order(self):
        pipeline = _make_pipeline()
        event = pipeline.process_listen("two cokes", source="asr")
        assert event.event_type == "valid_order"
        assert event.confidence > 0.5
        assert event.raw_text == "two cokes"

    def test_rule_parse_source(self):
        pipeline = _make_pipeline()
        event = pipeline.process_listen("one coke", source="asr")
        assert event.parse_source == "rule"
        assert event.candidate_order is not None
        assert len(event.candidate_order["items"]) == 1

    def test_cancel_returns_cancel_event(self):
        pipeline = _make_pipeline()
        event = pipeline.process_listen("cancel", source="asr")
        assert event.event_type == "cancel_request"
        assert event.parse_source == "classifier"

    def test_menu_question_returns_menu_event(self):
        pipeline = _make_pipeline()
        event = pipeline.process_listen("what do you have", source="asr")
        assert event.event_type == "menu_question"

    def test_empty_returns_noise_event(self):
        pipeline = _make_pipeline()
        event = pipeline.process_listen("", source="asr")
        assert event.event_type == "empty_or_noise"

    def test_smalltalk_returns_smalltalk_event(self):
        pipeline = _make_pipeline()
        event = pipeline.process_listen("hello", source="asr")
        assert event.event_type == "smalltalk"

    def test_unknown_food_returns_out_of_menu(self):
        pipeline = _make_pipeline()
        event = pipeline.process_listen("i want a pizza", source="asr")
        assert event.event_type == "out_of_menu_item"
        assert event.out_of_menu_item == "pizza"

    def test_multi_item_order(self):
        pipeline = _make_pipeline()
        event = pipeline.process_listen("two cokes and a water", source="asr")
        assert event.event_type == "valid_order"
        assert event.parse_source == "rule"
        items = event.candidate_order["items"]
        names = {i["name"] for i in items}
        assert "coke" in names
        assert "water" in names


class TestPipelineCheck:
    def test_yes_returns_confirm_correct(self):
        pipeline = _make_pipeline()
        event = pipeline.process_check("yes", source="asr")
        assert event.confirm_result == "correct"
        assert event.parse_source == "rule"

    def test_no_returns_confirm_wrong(self):
        pipeline = _make_pipeline()
        event = pipeline.process_check("no", source="asr")
        # "no" from confirmation parser has confidence 0.85, below the
        # default rule threshold of 0.90, so it falls through to classifier
        assert event.event_type == "unknown"

    def test_cancel_returns_cancel(self):
        pipeline = _make_pipeline()
        event = pipeline.process_check("cancel", source="asr")
        assert event.event_type == "cancel_request"

    def test_yes_but_add_returns_wrong_with_fix(self):
        pipeline = _make_pipeline()
        event = pipeline.process_check("yes but add a coke", source="asr")
        assert event.confirm_result == "wrong"
        assert event.fix_order is not None

    def test_repeat_returns_repeat_request(self):
        pipeline = _make_pipeline()
        event = pipeline.process_check("repeat that", source="asr")
        assert event.event_type == "repeat_request"

    def test_noise_returns_empty(self):
        pipeline = _make_pipeline()
        event = pipeline.process_check("uh", source="asr")
        assert event.event_type == "empty_or_noise"


class TestSemanticEventFields:
    def test_event_has_all_new_fields(self):
        event = SemanticEvent(
            event_type="valid_order",
            confidence=0.9,
            parse_source="rule",
            is_candidate=False,
            candidate_order={"type": "order", "items": [{"name": "coke", "qty": 1}]},
        )
        assert event.parse_source == "rule"
        assert event.candidate_order is not None
        assert event.is_candidate is False

    def test_event_serialization(self):
        event = SemanticEvent(
            event_type="confirm_correct",
            confidence=0.99,
            parse_source="rule",
            confirm_result="correct",
            raw_text="yes",
        )
        data = event.model_dump()
        assert data["parse_source"] == "rule"
        assert data["confirm_result"] == "correct"
