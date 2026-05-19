"""
LLM Client - model invocation client.

Wraps OpenAI-compatible APIs and supports both cloud and local backends.
"""

import json
from typing import Optional, List, Dict, Any, Callable, TypeVar
from openai import OpenAI, AsyncOpenAI
from cade.config import Config
from cade.brain.prompts import (
    get_order_listen_prompt,
    get_order_repeat_prompt,
    get_order_check_prompt,
)
from cade.brain.schemas import (
    RobotDecision,
    OrderAction,
    OrderCheckDecision,
    OrderSpeakDecision,
    parse_action,
    parse_order_action,
    parse_order_check_decision,
    parse_order_speak_decision,
)


ParsedModelT = TypeVar("ParsedModelT")


class LLMClient:
    """
    Synchronous LLM client.

    Supports cloud APIs, local Ollama-compatible endpoints, JSON parsing,
    and retry-on-parse-failure behavior.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM client.

        Args:
            config: Custom config. If None, Config.get_llm_config() is used.
        """
        if config is None:
            config = Config.get_llm_config()

        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        self.timeout = config.get("timeout", 30)

        import httpx
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=httpx.Client(trust_env=False)
        )

        print(f"LLM Client initialized")
        print(f"  Mode: {'Cloud' if Config.is_cloud_mode() else 'Local'}")
        print(f"  Model: {self.model}")
        print(f"  Base URL: {self.base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False
    ) -> str:
        """
        Call the LLM chat completion endpoint.
        """
        if not enable_thinking and "qwen3" in self.model.lower():
            if messages and messages[-1].get("role") == "user":
                messages[-1]["content"] = messages[-1]["content"] + " /no_think"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

    def get_decision(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3
    ) -> RobotDecision:
        """
        Ask the LLM for a robot decision.
        """
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_input})
        messages.append({
            "role": "assistant",
            "content": "Let me think step by step.\n\nHere is the JSON output:\n\n"
        })

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.chat(messages)

                if attempt == 0:
                    print(f"\nRaw LLM output:\n{'-'*60}")
                    print(response)
                    print(f"{'-'*60}\n")

                decision_dict = self._extract_json(response)

                if decision_dict.get("action") is not None:
                    action_dict = decision_dict["action"]
                    decision_dict["action"] = parse_action(action_dict)

                decision = RobotDecision(**decision_dict)
                return decision

            except Exception as e:
                last_error = e
                print(f"Parse failed (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    error_msg = (
                        f"Your output format is invalid. Error: {str(e)}\n"
                        f"Please output valid JSON only, using the required schema. Keep all spoken content in English."
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})

        raise ValueError(
            f"Failed to parse LLM output after {max_retries} retries. Last error: {last_error}"
        )

    def _run_structured_request(
        self,
        messages: List[Dict[str, str]],
        parser: Callable[[dict], ParsedModelT],
        max_retries: int = 3,
        raw_log_title: str = "Raw LLM output",
    ) -> ParsedModelT:
        """Run a structured JSON request with retry and parser validation."""
        last_error = None
        response = ""

        for attempt in range(max_retries):
            try:
                response = self.chat(messages)

                if attempt == 0:
                    print(f"\n{raw_log_title}:\n{'-'*60}")
                    print(response)
                    print(f"{'-'*60}\n")

                payload = self._extract_json(response)
                return parser(payload)

            except Exception as exc:
                last_error = exc
                print(f"Parse failed (attempt {attempt + 1}/{max_retries}): {exc}")

                if attempt < max_retries - 1:
                    error_msg = (
                        f"Your output format is invalid. Error: {str(exc)}\n"
                        "Please output valid JSON only and follow the required schema exactly."
                    )
                    if response:
                        messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})

        raise ValueError(
            f"Failed to parse LLM output after {max_retries} retries. Last error: {last_error}"
        )

    def get_order_action(
        self,
        user_input: str,
        food_aliases: Optional[Dict[str, List[str]]] = None,
        max_retries: int = 3,
    ) -> OrderAction:
        """LISTEN stage: parse order items from user utterance."""
        prompt = get_order_listen_prompt(food_aliases or {})
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
        return self._run_structured_request(
            messages=messages,
            parser=parse_order_action,
            max_retries=max_retries,
            raw_log_title="Order LISTEN raw output",
        )

    def get_order_repeat_speak(
        self,
        confirm_instruction: str,
        order_action: OrderAction,
        max_retries: int = 3,
    ) -> OrderSpeakDecision:
        """REPEAT stage: generate speak action for order confirmation."""
        prompt = get_order_repeat_prompt()
        payload = order_action.model_dump()
        user_input = (
            f"Instruction: {confirm_instruction}\n"
            f"Current order JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
        return self._run_structured_request(
            messages=messages,
            parser=parse_order_speak_decision,
            max_retries=max_retries,
            raw_log_title="Order REPEAT raw output",
        )

    def get_order_check_decision(
        self,
        customer_reply: str,
        order_action: OrderAction,
        food_aliases: Optional[Dict[str, List[str]]] = None,
        max_retries: int = 3,
    ) -> OrderCheckDecision:
        """CHECK stage: judge correct/wrong and optional fix_order action."""
        prompt = get_order_check_prompt(food_aliases or {})
        payload = order_action.model_dump()
        user_input = (
            f"Current order JSON:\n{json.dumps(payload, ensure_ascii=False)}\n"
            f"Customer reply:\n{customer_reply}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
        return self._run_structured_request(
            messages=messages,
            parser=parse_order_check_decision,
            max_retries=max_retries,
            raw_log_title="Order CHECK raw output",
        )

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from plain text or Markdown code blocks."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        raise json.JSONDecodeError("Could not extract JSON from text", text, 0)


class AsyncLLMClient:
    """Asynchronous LLM client for web services and other async flows."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = Config.get_llm_config()

        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        self.timeout = config.get("timeout", 30)

        import httpx
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=httpx.AsyncClient(trust_env=False)
        )

        print(f"Async LLM Client initialized")
        print(f"  Mode: {'Cloud' if Config.is_cloud_mode() else 'Local'}")
        print(f"  Model: {self.model}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False
    ) -> str:
        """Call the async LLM chat completion endpoint."""
        if not enable_thinking and "qwen3" in self.model.lower():
            if messages and messages[-1].get("role") == "user":
                messages[-1]["content"] = messages[-1]["content"] + " /no_think"

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

    async def get_decision(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3
    ) -> RobotDecision:
        """Ask the LLM for a robot decision asynchronously."""
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_input})

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.chat(messages)
                decision_dict = self._extract_json(response)

                if decision_dict.get("action") is not None:
                    action_dict = decision_dict["action"]
                    decision_dict["action"] = parse_action(action_dict)

                decision = RobotDecision(**decision_dict)
                return decision

            except Exception as e:
                last_error = e
                print(f"Parse failed (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    error_msg = (
                        f"Your output format is invalid. Error: {str(e)}\n"
                        f"Please output valid JSON only, using the required schema. Keep all spoken content in English."
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})

        raise ValueError(
            f"Failed to parse LLM output after {max_retries} retries. Last error: {last_error}"
        )

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from plain text or Markdown code blocks."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        raise json.JSONDecodeError("Could not extract JSON from text", text, 0)
