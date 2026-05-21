"""Tests for structured output stability: schemas, prompts, parsers, JSON extractor.

These tests verify the ordering sub-FSM produces stable, parseable JSON
under noisy/off-domain/repair inputs.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from cade.brain.llm_client import LLMClient
from cade.brain.schemas import (
    OrderAction,
    OrderCheckDecision,
    OrderItem,
    OrderSpeakDecision,
    FixOrderAction,
    parse_order_action,
    parse_order_check_decision,
    parse_order_speak_decision,
)
from cade.brain.prompts import (
    ORDER_LISTEN_PROMPT_TEMPLATE,
    ORDER_REPEAT_PROMPT_TEMPLATE,
    ORDER_CHECK_PROMPT_TEMPLATE,
    get_order_listen_prompt,
    get_order_repeat_prompt,
    get_order_check_prompt,
    _canonical_food_names_from_aliases,
)


# ======================================================================
# Prompt quality tests
# ======================================================================

class TestPromptQuality:
    """Prompts must not contain issues that mislead Qwen."""

    def test_repeat_prompt_no_double_braces(self):
        """REPEAT prompt must NOT contain {{ or }} as Jinja-like escaped braces.
        We check for '{{ ' and ' }}' patterns which indicate template-escaped braces,
        not the harmless '}}' that appears at end of JSON strings."""
        assert "{{ " not in ORDER_REPEAT_PROMPT_TEMPLATE
        assert " }}" not in ORDER_REPEAT_PROMPT_TEMPLATE

    def test_listen_prompt_no_double_braces(self):
        assert "{{ " not in ORDER_LISTEN_PROMPT_TEMPLATE
        assert " }}" not in ORDER_LISTEN_PROMPT_TEMPLATE

    def test_check_prompt_no_double_braces(self):
        assert "{{ " not in ORDER_CHECK_PROMPT_TEMPLATE
        assert " }}" not in ORDER_CHECK_PROMPT_TEMPLATE

    def test_listen_prompt_no_markdown_fence(self):
        """LISTEN prompt must not contain ```json — prevents model from
        outputting markdown code fences."""
        assert "```" not in ORDER_LISTEN_PROMPT_TEMPLATE

    def test_repeat_prompt_no_markdown_fence(self):
        assert "```" not in ORDER_REPEAT_PROMPT_TEMPLATE

    def test_check_prompt_no_markdown_fence(self):
        assert "```" not in ORDER_CHECK_PROMPT_TEMPLATE

    def test_listen_prompt_has_canonical_placeholder(self):
        assert "{canonical_foods}" in ORDER_LISTEN_PROMPT_TEMPLATE

    def test_listen_prompt_explicit_json_start_rule(self):
        """LISTEN prompt must instruct output starts with { and ends with }."""
        prompt = get_order_listen_prompt({"coke": ["coke"]})
        assert "{" in prompt

    def test_listen_prompt_off_input_rule(self):
        """LISTEN prompt must specify how to handle off-domain inputs."""
        prompt = get_order_listen_prompt({"coke": ["coke"]})
        assert "items" in prompt
        assert "[]" in prompt

    def test_check_prompt_has_correct_and_wrong(self):
        """CHECK prompt must show both 'correct' and 'wrong' outcomes."""
        prompt = get_order_check_prompt({"coke": ["coke"]})
        assert "correct" in prompt
        assert "wrong" in prompt

    def test_canonical_food_names_returns_default_when_empty(self):
        result = _canonical_food_names_from_aliases({})
        assert len(result) > 0

    def test_canonical_food_names_returns_sorted_keys(self):
        aliases = {"zebra": ["z"], "apple": ["a"], "mango": ["m"]}
        result = _canonical_food_names_from_aliases(aliases)
        names = [n.strip() for n in result.split(",")]
        assert names == ["apple", "mango", "zebra"]


# ======================================================================
# Parser / schema tests
# ======================================================================

class TestParserStrictness:
    """Parsers must reject extra fields and enforce required structure."""

    def test_order_action_rejects_extra_field(self):
        with pytest.raises(Exception):
            parse_order_action({
                "type": "order",
                "items": [{"name": "coke", "qty": 1}],
                "confidence": 0.95,
            })

    def test_order_action_requires_type_order(self):
        with pytest.raises(Exception):
            parse_order_action({
                "items": [{"name": "coke", "qty": 1}],
            })

    def test_order_action_rejects_wrong_type(self):
        with pytest.raises(Exception):
            parse_order_action({
                "type": "speak",
                "items": [],
            })

    def test_order_item_rejects_extra_field(self):
        with pytest.raises(Exception):
            OrderItem(name="coke", qty=1, confidence=0.9)

    def test_order_item_qty_minimum_1(self):
        with pytest.raises(Exception):
            OrderItem(name="coke", qty=0)

    def test_order_item_qty_negative(self):
        with pytest.raises(Exception):
            OrderItem(name="coke", qty=-1)

    def test_order_item_name_empty_rejected(self):
        with pytest.raises(Exception):
            OrderItem(name="  ", qty=1)

    def test_order_speak_rejects_extra_field(self):
        with pytest.raises(Exception):
            parse_order_speak_decision({
                "action": {"type": "speak", "content": "hello"},
                "extra": "bad",
            })

    def test_order_speak_requires_speak_action_type(self):
        with pytest.raises(Exception):
            parse_order_speak_decision({
                "action": {"type": "pick", "object_name": "cup"},
            })

    def test_order_check_rejects_extra_field(self):
        with pytest.raises(Exception):
            parse_order_check_decision({
                "result": "correct",
                "action": None,
                "reply": None,
                "explanation": "user confirmed",
            })

    def test_order_check_correct_with_action_rejected(self):
        """result='correct' must NOT allow a non-null action."""
        with pytest.raises(Exception):
            parse_order_check_decision({
                "result": "correct",
                "action": {
                    "type": "fix_order",
                    "items": [{"name": "coke", "qty": 1}],
                },
                "reply": None,
            })

    def test_order_check_wrong_with_empty_fix_rejected(self):
        """fix_order items must not be empty."""
        with pytest.raises(Exception):
            parse_order_check_decision({
                "result": "wrong",
                "action": {
                    "type": "fix_order",
                    "items": [],
                },
                "reply": None,
            })

    def test_order_check_wrong_without_action_is_valid(self):
        result = parse_order_check_decision({
            "result": "wrong",
            "action": None,
            "reply": "What would you like?",
        })
        assert result.result == "wrong"
        assert result.action is None

    def test_order_check_wrong_with_valid_fix(self):
        result = parse_order_check_decision({
            "result": "wrong",
            "action": {
                "type": "fix_order",
                "items": [{"name": "water", "qty": 2}],
            },
            "reply": None,
        })
        assert result.result == "wrong"
        assert result.action is not None
        assert result.action.type == "fix_order"


# ======================================================================
# JSON extractor tests
# ======================================================================

class TestExtractJsonAdvanced:
    """_extract_json must handle adversarial LLM outputs."""

    def test_multiple_json_objects_extracts_last(self):
        """When text contains two JSON objects, the last one should be
        extracted — the model's actual output typically comes last,
        especially in reasoning_content where earlier JSON may be echoed
        from the prompt."""
        text = '{"type":"order","items":[]} Some explanation {"type":"order","items":[{"name":"coke","qty":1}]}'
        result = LLMClient._extract_json(text)
        assert result == {"type": "order", "items": [{"name": "coke", "qty": 1}]}

    def test_thinking_block_stripped(self):
        """Qwen with /no_think sometimes still emits <think/> blocks."""
        text = '<think className="foo">reasoning here</thinkClass>{"type":"order","items":[]}'
        # Should still extract JSON after stripping
        result = LLMClient._extract_json(text)
        assert result["type"] == "order"

    def test_json_after_explanation(self):
        text = 'Here is the result:\n{"result": "correct"}\nDone.'
        result = LLMClient._extract_json(text)
        assert result["result"] == "correct"

    def test_nested_braces_balanced(self):
        """Must extract each balanced JSON object separately (not greedy span).
        With multiple top-level objects, the last one is returned."""
        inner = {"action": {"type": "speak", "content": "hi"}}
        text = json.dumps(inner) + " extra " + json.dumps({"other": True})
        result = LLMClient._extract_json(text)
        assert result == {"other": True}

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            LLMClient._extract_json("")

    def test_no_json_at_all_raises(self):
        with pytest.raises(json.JSONDecodeError):
            LLMClient._extract_json("just some random text without any json")

    def test_pure_json_object(self):
        data = {"type": "order", "items": [{"name": "coke", "qty": 1}]}
        result = LLMClient._extract_json(json.dumps(data))
        assert result == data

    def test_fenced_json_block(self):
        text = '```json\n{"type": "order", "items": []}\n```'
        result = LLMClient._extract_json(text)
        assert result["type"] == "order"

    def test_unclosed_brace_does_not_match(self):
        """Must NOT match when there is no complete balanced JSON object."""
        with pytest.raises(json.JSONDecodeError):
            LLMClient._extract_json('{"type": "order", "items":')

    def test_whitespace_json(self):
        data = {"type": "order", "items": []}
        result = LLMClient._extract_json("   \n  " + json.dumps(data) + "  \n  ")
        assert result == data


# ======================================================================
# response_format forwarding tests
# ======================================================================

class TestResponseFormatForwarding:
    """Structured request helpers must pass response_format to chat()."""

    @patch("cade.brain.llm_client.OpenAI")
    @patch("cade.brain.llm_client.Config")
    def test_get_order_action_passes_response_format(self, mock_config, _openai):
        mock_config.is_cloud_mode.return_value = False
        mock_config.get_llm_config.return_value = {
            "base_url": "http://localhost:1/v1",
            "api_key": "k",
            "model": "test",
            "temperature": 0.2,
            "max_tokens": 10,
            "timeout": 5,
        }

        client = LLMClient.__new__(LLMClient)
        client.model = "test"
        client.temperature = 0.2
        client.max_tokens = 10
        client.timeout = 5

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"type":"order","items":[{"name":"coke","qty":1}]}'))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client.client = mock_client

        client.get_order_action("I want a coke", food_aliases={"coke": ["coke"]})

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        rf = call_kwargs.get("response_format")
        assert rf is not None
        assert rf["type"] == "json_schema"
        assert "json_schema" in rf
        schema = rf["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "type" in schema["properties"]
        assert "items" in schema["properties"]

    @patch("cade.brain.llm_client.OpenAI")
    @patch("cade.brain.llm_client.Config")
    def test_get_order_repeat_speak_passes_response_format(self, mock_config, _openai):
        mock_config.is_cloud_mode.return_value = False
        mock_config.get_llm_config.return_value = {
            "base_url": "http://localhost:1/v1",
            "api_key": "k",
            "model": "test",
            "temperature": 0.2,
            "max_tokens": 10,
            "timeout": 5,
        }

        client = LLMClient.__new__(LLMClient)
        client.model = "test"
        client.temperature = 0.2
        client.max_tokens = 10
        client.timeout = 5

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action":{"type":"speak","content":"ok"}}'))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client.client = mock_client

        order = OrderAction(type="order", items=[OrderItem(name="coke", qty=1)])
        client.get_order_repeat_speak("Repeat the order.", order)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        rf = call_kwargs.get("response_format")
        assert rf is not None
        assert rf["type"] == "json_schema"

    @patch("cade.brain.llm_client.OpenAI")
    @patch("cade.brain.llm_client.Config")
    def test_get_order_check_decision_passes_response_format(self, mock_config, _openai):
        mock_config.is_cloud_mode.return_value = False
        mock_config.get_llm_config.return_value = {
            "base_url": "http://localhost:1/v1",
            "api_key": "k",
            "model": "test",
            "temperature": 0.2,
            "max_tokens": 10,
            "timeout": 5,
        }

        client = LLMClient.__new__(LLMClient)
        client.model = "test"
        client.temperature = 0.2
        client.max_tokens = 10
        client.timeout = 5

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"result":"correct","action":null,"reply":null}'))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client.client = mock_client

        order = OrderAction(type="order", items=[OrderItem(name="coke", qty=1)])
        client.get_order_check_decision("yes", order, food_aliases={"coke": ["coke"]})

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        rf = call_kwargs.get("response_format")
        assert rf is not None
        assert rf["type"] == "json_schema"
