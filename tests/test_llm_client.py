"""Unit tests for LLMClient — pure-logic tests that do NOT call any LLM."""

import json
import pytest
from unittest.mock import MagicMock, patch

from cade.brain.llm_client import LLMClient


# ------------------------------------------------------------------
# _extract_json
# ------------------------------------------------------------------

class TestExtractJson:
    """LLMClient._extract_json must handle all common LLM output formats."""

    def test_pure_json(self):
        text = '{"reply": "hello", "action": {"type": "wait"}}'
        result = LLMClient._extract_json(text)
        assert result["reply"] == "hello"
        assert result["action"]["type"] == "wait"

    def test_markdown_json_block(self):
        text = '```json\n{"reply": "hi", "action": null}\n```'
        result = LLMClient._extract_json(text)
        assert result["reply"] == "hi"
        assert result["action"] is None

    def test_markdown_block_no_lang(self):
        text = '```\n{"reply": "yo"}\n```'
        result = LLMClient._extract_json(text)
        assert result["reply"] == "yo"

    def test_json_with_leading_text(self):
        text = 'Sure! Here is the output:\n{"reply": "ok", "action": {"type": "wait"}}'
        result = LLMClient._extract_json(text)
        assert result["reply"] == "ok"

    def test_json_with_trailing_text(self):
        text = '{"reply": "done"}\nHope this helps!'
        result = LLMClient._extract_json(text)
        assert result["reply"] == "done"

    def test_nested_json(self):
        obj = {
            "thought": "user wants apple",
            "reply": "Looking for apple",
            "action": {"type": "search", "object_name": "apple"},
        }
        text = json.dumps(obj)
        result = LLMClient._extract_json(text)
        assert result["action"]["object_name"] == "apple"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            LLMClient._extract_json("this is not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            LLMClient._extract_json("")


# ------------------------------------------------------------------
# /no_think injection (chat method)
# ------------------------------------------------------------------

class TestNoThinkInjection:
    """/no_think should be appended to the last user message without
    mutating the caller's original messages list."""

    @patch("cade.brain.llm_client.OpenAI")
    @patch("cade.brain.llm_client.Config")
    def test_no_think_appended_to_last_user(self, mock_config, mock_openai_cls):
        mock_config.is_cloud_mode.return_value = False
        mock_config.get_llm_config.return_value = {
            "base_url": "http://localhost:1/v1",
            "api_key": "k",
            "model": "qwen3.5-9b-q8-local",
            "temperature": 0.2,
            "max_tokens": 10,
            "timeout": 5,
        }

        client = LLMClient.__new__(LLMClient)
        client.model = "qwen3.5-9b-q8-local"
        client.temperature = 0.2
        client.max_tokens = 10
        client.timeout = 5

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client.client = mock_client

        original_messages = [
            {"role": "system", "content": "You are a robot."},
            {"role": "user", "content": "hello"},
        ]
        # Keep a copy to verify non-mutation
        original_copy = [dict(m) for m in original_messages]

        client.chat(original_messages, enable_thinking=False)

        # Caller's list must NOT be mutated
        assert original_messages == original_copy

        # The call sent to the API must contain /no_think on the user msg
        sent = mock_client.chat.completions.create.call_args
        sent_messages = sent.kwargs["messages"]
        assert "/no_think" in sent_messages[-1]["content"]

    @patch("cade.brain.llm_client.OpenAI")
    @patch("cade.brain.llm_client.Config")
    def test_no_think_not_injected_when_thinking_enabled(self, mock_config, mock_openai_cls):
        mock_config.is_cloud_mode.return_value = False
        mock_config.get_llm_config.return_value = {
            "base_url": "http://localhost:1/v1",
            "api_key": "k",
            "model": "qwen3.5-9b-q8-local",
            "temperature": 0.2,
            "max_tokens": 10,
            "timeout": 5,
        }

        client = LLMClient.__new__(LLMClient)
        client.model = "qwen3.5-9b-q8-local"
        client.temperature = 0.2
        client.max_tokens = 10
        client.timeout = 5

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{}'))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client.client = mock_client

        messages = [{"role": "user", "content": "hello"}]
        client.chat(messages, enable_thinking=True)

        sent = mock_client.chat.completions.create.call_args
        sent_messages = sent.kwargs["messages"]
        assert "/no_think" not in sent_messages[-1]["content"]


# ------------------------------------------------------------------
# response_format forwarded
# ------------------------------------------------------------------

class TestResponseFormat:
    """response_format should be forwarded to the OpenAI SDK call."""

    @patch("cade.brain.llm_client.OpenAI")
    @patch("cade.brain.llm_client.Config")
    def test_json_object_format_forwarded(self, mock_config, mock_openai_cls):
        mock_config.is_cloud_mode.return_value = False
        mock_config.get_llm_config.return_value = {
            "base_url": "http://localhost:1/v1",
            "api_key": "k",
            "model": "qwen3.5-9b-q8-local",
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
        mock_response.choices = [MagicMock(message=MagicMock(content='{}'))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client.client = mock_client

        client.chat(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )

        sent = mock_client.chat.completions.create.call_args
        assert sent.kwargs["response_format"] == {"type": "json_object"}
