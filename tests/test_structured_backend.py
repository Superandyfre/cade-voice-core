"""Tests for Phase 1: structured backend, schema export, schema consistency."""

import pytest
from cade.brain.schemas import (
    OrderAction, OrderItem, OrderCheckDecision, OrderSpeakDecision,
    RobotDecision, PickAction, PlaceAction, SearchAction, SpeakAction, WaitAction,
    parse_action, parse_order_action, parse_order_check_decision,
)
from cade.brain.response_formats import (
    build_listen_response_format,
    build_repeat_response_format,
    build_check_response_format,
    build_robot_decision_response_format,
)
from cade.brain.schema_export import export_llm_json_schema


class TestSchemaConsistency:
    """Verify hand-written response_format schemas match Pydantic models."""

    def test_listen_schema_matches_order_action(self):
        rf = build_listen_response_format({"coke": ["coke", "cola"], "water": ["water"]})
        schema = rf["json_schema"]["schema"]
        # schema has type=const and items array
        assert schema["properties"]["type"]["const"] == "order"
        item_schema = schema["properties"]["items"]["items"]
        assert "name" in item_schema["properties"]
        assert "qty" in item_schema["properties"]

    def test_repeat_schema_matches_speak_decision(self):
        rf = build_repeat_response_format()
        schema = rf["json_schema"]["schema"]
        action_schema = schema["properties"]["action"]
        assert action_schema["properties"]["type"]["const"] == "speak"

    def test_check_schema_matches_check_decision(self):
        rf = build_check_response_format({"coke": ["coke"]})
        schema = rf["json_schema"]["schema"]
        assert "result" in schema["properties"]
        assert schema["properties"]["result"]["enum"] == ["correct", "wrong"]
        assert "action" in schema["properties"]
        assert "reply" in schema["properties"]

    def test_robot_decision_schema_has_all_action_types(self):
        rf = build_robot_decision_response_format()
        schema = rf["json_schema"]["schema"]
        action_oneof = schema["properties"]["action"]["oneOf"]
        # First element is the action object, second is null
        action_obj = action_oneof[0]
        assert set(action_obj["properties"]["type"]["enum"]) == {
            "search", "pick", "place", "speak", "wait"
        }

    def test_all_schemas_have_additional_properties_false(self):
        schemas = [
            build_listen_response_format({"coke": ["coke"]}),
            build_repeat_response_format(),
            build_check_response_format({"coke": ["coke"]}),
            build_robot_decision_response_format(),
        ]
        for rf in schemas:
            _assert_no_additional_properties(rf["json_schema"]["schema"])


def _assert_no_additional_properties(schema: dict):
    """Recursively verify all objects have additionalProperties: false."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            assert schema.get("additionalProperties") is False, \
                f"Missing additionalProperties: false in {schema}"
        for v in schema.values():
            if isinstance(v, dict):
                _assert_no_additional_properties(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _assert_no_additional_properties(item)


class TestSchemaExport:

    def test_export_order_action(self):
        schema = export_llm_json_schema(OrderAction)
        assert schema["type"] == "object"
        assert "items" in schema["properties"]
        assert schema.get("additionalProperties") is False

    def test_export_robot_decision(self):
        schema = export_llm_json_schema(RobotDecision)
        assert "action" in schema["properties"]
        assert schema.get("additionalProperties") is False

    def test_export_order_check_decision(self):
        schema = export_llm_json_schema(OrderCheckDecision)
        assert "result" in schema["properties"]
        assert schema.get("additionalProperties") is False

    def test_export_llama_cpp_profile_strips_defaults(self):
        schema = export_llm_json_schema(OrderAction, profile="llama_cpp")
        assert "default" not in str(schema)

    def test_nested_objects_have_no_additional_properties(self):
        schema = export_llm_json_schema(RobotDecision, profile="openai")
        _assert_no_additional_properties(schema)


class TestPydanticParsers:

    def test_parse_action_search(self):
        action = parse_action({"type": "search", "object_name": "apple"})
        assert isinstance(action, SearchAction)
        assert action.object_name == "apple"

    def test_parse_action_pick(self):
        action = parse_action({"type": "pick", "object_name": "cup"})
        assert isinstance(action, PickAction)

    def test_parse_action_place(self):
        action = parse_action({"type": "place", "location": "table"})
        assert isinstance(action, PlaceAction)

    def test_parse_action_speak(self):
        action = parse_action({"type": "speak", "content": "hello"})
        assert isinstance(action, SpeakAction)

    def test_parse_action_wait(self):
        action = parse_action({"type": "wait", "reason": "idle"})
        assert isinstance(action, WaitAction)

    def test_parse_action_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown action type"):
            parse_action({"type": "fly", "destination": "mars"})

    def test_parse_order_action_requires_order_type(self):
        with pytest.raises(ValueError, match="Expected type='order'"):
            parse_order_action({"type": "wrong", "items": []})

    def test_parse_order_check_correct_no_action(self):
        result = parse_order_check_decision({"result": "correct", "action": None, "reply": None})
        assert result.result == "correct"
        assert result.action is None

    def test_parse_order_check_wrong_with_fix(self):
        result = parse_order_check_decision({
            "result": "wrong",
            "action": {"type": "fix_order", "items": [{"name": "water", "qty": 2}]},
            "reply": None,
        })
        assert result.result == "wrong"
        assert result.action is not None

    def test_parse_order_check_correct_with_action_rejected(self):
        with pytest.raises(Exception):
            parse_order_check_decision({
                "result": "correct",
                "action": {"type": "fix_order", "items": [{"name": "coke", "qty": 1}]},
                "reply": None,
            })

    def test_robot_decision_with_action(self):
        d = RobotDecision(
            thought="test",
            reply="ok",
            action=SearchAction(object_name="book"),
        )
        assert d.action.type == "search"

    def test_robot_decision_no_action(self):
        d = RobotDecision(reply="hello", action=None)
        assert d.action is None

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            OrderAction(type="order", items=[], unexpected=True)
        with pytest.raises(Exception):
            OrderCheckDecision(result="correct", action=None, reply=None, extra=True)
        with pytest.raises(Exception):
            PickAction(type="pick", object_name="x", extra_field=True)
