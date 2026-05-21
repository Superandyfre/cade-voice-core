"""
LLM Client - model invocation client.

Wraps OpenAI-compatible APIs and supports both cloud and local backends.
"""

import json
import re
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
from cade.brain.response_formats import (
    build_listen_response_format,
    build_repeat_response_format,
    build_check_response_format,
)


ParsedModelT = TypeVar("ParsedModelT")


class LLMClient:
    """
    Synchronous LLM client.

    Supports cloud APIs, local Ollama-compatible endpoints, JSON parsing,
    and retry-on-parse-failure behavior.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = Config.get_llm_config()

        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 256)
        self.timeout = config.get("timeout", 60)

        import httpx
        safe_key = self.api_key if self.api_key else "not-needed"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=safe_key,
            timeout=self.timeout,
            http_client=httpx.Client(trust_env=False)
        )

        print(f"LLM Client initialized")
        print(f"  Mode: {'Cloud' if Config.is_cloud_mode() else 'Local'}")
        print(f"  Model: {self.model}")
        print(f"  Base URL: {self.base_url}")

    # ------------------------------------------------------------------
    # Core chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Call the LLM chat completion endpoint.

        When *enable_thinking* is False and the model name contains "qwen3",
        ``/no_think`` is appended to the **last user message** of a *copy* of
        *messages* so the caller's list is never mutated.
        """
        work_messages = list(messages)  # shallow copy — we only replace dicts

        if not enable_thinking and "qwen3" in self.model.lower():
            for i in range(len(work_messages) - 1, -1, -1):
                if work_messages[i].get("role") == "user":
                    work_messages[i] = {
                        **work_messages[i],
                        "content": work_messages[i]["content"] + " /no_think",
                    }
                    break

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=work_messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content or not content.strip():
            rc = getattr(response.choices[0].message, "reasoning_content", None)
            if rc and rc.strip():
                return rc
        return content if content is not None else ""

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def get_decision(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3,
    ) -> RobotDecision:
        """
        Ask the LLM for a robot decision.

        Uses ``response_format={"type": "json_object"}`` when the backend
        supports it, and avoids the old assistant-prefill pattern that
        confuses Qwen chat templates.
        """
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_input})

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.chat(
                    messages,
                    response_format={"type": "json_object"},
                )

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
                        f"Please output valid JSON only, using the required schema. "
                        f"Keep all spoken content in English."
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})

        raise ValueError(
            f"Failed to parse LLM output after {max_retries} retries. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Structured request (ordering sub-FSM, etc.)
    # ------------------------------------------------------------------

    def _run_structured_request(
        self,
        messages: List[Dict[str, str]],
        parser: Callable[[dict], ParsedModelT],
        max_retries: int = 3,
        raw_log_title: str = "Raw LLM output",
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ParsedModelT:
        """Run a structured JSON request with retry and parser validation.

        If *response_format* is provided, it is forwarded to ``chat()`` for
        schema-constrained sampling.  On API rejection of the schema (e.g.
        backends that do not support ``json_schema``), the method falls back
        through ``{"type": "json_object"}`` then to no response_format at all.
        Fallbacks do not consume retry attempts.
        """
        is_qwen3 = "qwen3" in self.model.lower()

        last_error = None
        response = ""
        effective_rf = response_format
        rf_fallback_stage = 0  # 0=original, 1=json_object, 2=none

        # Qwen3 with /no_think puts everything into reasoning_content and
        # leaves content empty.  For structured requests we need the actual
        # JSON output, so always enable thinking.
        use_thinking = is_qwen3
        # Qwen3 thinking process is verbose; 256 tokens is not enough.
        use_max_tokens = 2048 if is_qwen3 else None

        attempt = 0
        while attempt < max_retries:
            try:
                response = self.chat(
                    messages,
                    response_format=effective_rf,
                    enable_thinking=use_thinking,
                    max_tokens=use_max_tokens,
                )

                if attempt == 0:
                    print(f"\n{raw_log_title}:\n{'-'*60}")
                    print(response)
                    print(f"{'-'*60}\n")

                payload = self._extract_json(response)
                try:
                    return parser(payload)
                except Exception:
                    # The last JSON object may be a schema example echoed by
                    # the model.  Try all candidates from last to first.
                    candidates = LLMClient._extract_all_balanced_json(response)
                    for candidate in reversed(candidates):
                        try:
                            return parser(candidate)
                        except Exception:
                            continue
                    raise

            except Exception as exc:
                last_error = exc
                print(f"Parse failed (attempt {attempt + 1}/{max_retries}): {exc}")

                # Progressive fallback of response_format (does not consume attempt)
                if rf_fallback_stage == 0 and effective_rf and effective_rf.get("type") == "json_schema":
                    print("Warning: json_schema response_format rejected, falling back to json_object")
                    effective_rf = {"type": "json_object"}
                    rf_fallback_stage = 1
                    continue

                if rf_fallback_stage == 1 and effective_rf:
                    print("Warning: json_object response_format also failed, removing response_format")
                    effective_rf = None
                    rf_fallback_stage = 2
                    continue

                attempt += 1

                if attempt < max_retries:
                    error_msg = (
                        f"Your output format is invalid. Error: {str(exc)}\n"
                        "Please output valid JSON only and follow the required schema exactly. "
                        "Do not explain the error. Output only the corrected JSON."
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
        rf = build_listen_response_format(food_aliases)
        return self._run_structured_request(
            messages=messages,
            parser=parse_order_action,
            max_retries=max_retries,
            raw_log_title="Order LISTEN raw output",
            response_format=rf,
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
        rf = build_repeat_response_format()
        return self._run_structured_request(
            messages=messages,
            parser=parse_order_speak_decision,
            max_retries=max_retries,
            raw_log_title="Order REPEAT raw output",
            response_format=rf,
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
        rf = build_check_response_format(food_aliases)
        return self._run_structured_request(
            messages=messages,
            parser=parse_order_check_decision,
            max_retries=max_retries,
            raw_log_title="Order CHECK raw output",
            response_format=rf,
        )

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Extract JSON from plain text, Markdown code blocks, or text with
        a leading/trailing ``{...}`` blob.

        Uses balanced-brace scanning so multiple JSON objects or trailing
        explanation text do not cause greedy over-match.

        Raises ``json.JSONDecodeError`` when nothing works.
        """
        # Strip Qwen-style <thinkClassName>...</thinkClassName> blocks
        cleaned = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
        # Also strip <think/> self-closing tags
        cleaned = re.sub(r"<think[^>]*/>", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            text = cleaned

        # 1. Pure JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. ```json ... ``` block
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 3. ``` ... ``` block (no language tag)
        m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 4. Find all top-level balanced {...} objects and return the last
        #    valid one.  The model's actual output typically appears at the
        #    end, especially in reasoning_content where earlier JSON objects
        #    may be echoed from the prompt.
        candidates = LLMClient._extract_all_balanced_json(text)
        for obj in reversed(candidates):
            return obj

        raise json.JSONDecodeError(
            "Could not extract JSON from text", text, 0
        )

    @staticmethod
    def _extract_balanced_json(text: str, start: int) -> dict | None:
        """Extract the first complete balanced JSON object starting at *start*."""
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _extract_all_balanced_json(text: str) -> list[dict]:
        """Find all top-level balanced JSON objects in *text*."""
        results: list[dict] = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                obj = LLMClient._extract_balanced_json(text, i)
                if obj is not None:
                    results.append(obj)
                    # Skip past this object
                    depth = 0
                    in_string = False
                    escape = False
                    j = i
                    while j < len(text):
                        ch = text[j]
                        if escape:
                            escape = False
                            j += 1
                            continue
                        if ch == "\\" and in_string:
                            escape = True
                            j += 1
                            continue
                        if ch == '"':
                            in_string = not in_string
                        elif not in_string:
                            if ch == "{":
                                depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0:
                                    i = j + 1
                                    break
                        j += 1
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        return results


class AsyncLLMClient:
    """Asynchronous LLM client for web services and other async flows."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = Config.get_llm_config()

        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 256)
        self.timeout = config.get("timeout", 60)

        import httpx
        safe_key = self.api_key if self.api_key else "not-needed"
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=safe_key,
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
        enable_thinking: bool = False,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """Call the async LLM chat completion endpoint."""
        work_messages = list(messages)

        if not enable_thinking and "qwen3" in self.model.lower():
            for i in range(len(work_messages) - 1, -1, -1):
                if work_messages[i].get("role") == "user":
                    work_messages[i] = {
                        **work_messages[i],
                        "content": work_messages[i]["content"] + " /no_think",
                    }
                    break

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=work_messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content or not content.strip():
            rc = getattr(response.choices[0].message, "reasoning_content", None)
            if rc and rc.strip():
                return rc
        return content if content is not None else ""

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
                response = await self.chat(
                    messages,
                    response_format={"type": "json_object"},
                )
                decision_dict = LLMClient._extract_json(response)

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
                        f"Please output valid JSON only, using the required schema. "
                        f"Keep all spoken content in English."
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})

        raise ValueError(
            f"Failed to parse LLM output after {max_retries} retries. Last error: {last_error}"
        )
