"""Structured LLM backend abstraction.

Provides a unified interface for requesting structured JSON output from LLM
endpoints, with automatic capability detection and fallback from
schema-constrained -> json_object -> prompt-only modes.
"""

import time
import logging
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class BackendProfile(str, Enum):
    OPENAI_RESPONSE_FORMAT = "openai"
    VLLM_STRUCTURED_OUTPUTS = "vllm"
    SGLANG_RESPONSE_FORMAT = "sglang"
    LLAMA_CPP_RESPONSE_FORMAT = "llama_cpp"
    PROMPT_ONLY = "prompt_only"


class StructuredCallStats(BaseModel):
    backend: str
    model: str
    schema_name: str
    format_mode: str
    attempts: int
    latency_s: float
    fallback_stage: Optional[str] = None


class StructuredLLMError(BaseModel):
    error_type: str  # api_rejected, invalid_json, schema_invalid, timeout
    message: str
    raw_output: Optional[str] = None


class OpenAICompatibleStructuredBackend:
    """Backend that uses an OpenAI-compatible API with progressive fallback.

    Format priority:
    1. response_format={"type":"json_schema","json_schema":...}
    2. response_format={"type":"json_object"}
    3. prompt-only (no response_format)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 60,
        profile: BackendProfile = BackendProfile.OPENAI_RESPONSE_FORMAT,
    ):
        import httpx

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.profile = profile

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
            http_client=httpx.Client(trust_env=False),
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[ModelT],
        *,
        schema_name: str,
        json_schema: dict,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 2,
    ) -> tuple[ModelT, StructuredCallStats]:
        """Call the LLM and return a parsed Pydantic model + stats.

        Returns (parsed_model, stats) on success.
        Raises StructuredLLMError on failure.
        """
        is_qwen3 = "qwen3" in self.model.lower()
        use_thinking = is_qwen3
        use_max_tokens = 2048 if is_qwen3 else (max_tokens or self.max_tokens)

        rf_stages = [
            {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": json_schema}},
            {"type": "json_object"},
            None,
        ]
        if self.profile == BackendProfile.PROMPT_ONLY:
            rf_stages = [None]

        t0 = time.monotonic()
        last_error: Optional[Exception] = None
        attempt = 0
        fallback_stage: Optional[str] = None

        for rf_idx, rf in enumerate(rf_stages):
            if rf_idx > 0:
                fallback_stage = f"fallback_to_{rf['type'] if rf else 'prompt_only'}"
                logger.info("Structured fallback: stage %d (%s)", rf_idx, rf)

            while attempt < retries + 1:
                try:
                    response = self._chat(
                        messages, response_format=rf,
                        enable_thinking=use_thinking,
                        max_tokens=use_max_tokens,
                        temperature=temperature,
                    )
                    payload = self._extract_json(response)
                    # Try parsing with Pydantic model
                    result = self._try_parse(payload, schema)

                    # If failed, try all balanced JSON candidates
                    if result is None:
                        candidates = self._extract_all_balanced_json(response)
                        for candidate in reversed(candidates):
                            result = self._try_parse(candidate, schema)
                            if result is not None:
                                break

                    if result is not None:
                        stats = StructuredCallStats(
                            backend=self.profile.value,
                            model=self.model,
                            schema_name=schema_name,
                            format_mode=rf.get("type", "prompt_only") if rf else "prompt_only",
                            attempts=attempt + 1,
                            latency_s=round(time.monotonic() - t0, 3),
                            fallback_stage=fallback_stage,
                        )
                        return result, stats

                    last_error = ValueError("Could not parse LLM output as target schema")
                    attempt += 1

                except Exception as exc:
                    last_error = exc
                    # API rejected the schema format -> move to next fallback stage
                    if rf_idx == 0 and rf and rf.get("type") == "json_schema":
                        msg = str(exc).lower()
                        if "400" in msg or "invalid" in msg or "unsupported" in msg:
                            logger.warning("json_schema rejected by API, falling back: %s", exc)
                            break  # break inner loop, try next rf_stage
                    attempt += 1

                if attempt < retries + 1:
                    error_msg = (
                        f"Your output format is invalid. Error: {last_error}\n"
                        "Please output valid JSON only and follow the required schema exactly. "
                        "Do not explain the error. Output only the corrected JSON."
                    )
                    if response:
                        messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})

        raise ValueError(
            f"Failed to parse LLM output after {attempt} attempts. Last error: {last_error}"
        )

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        work_messages = list(messages)
        if not enable_thinking and "qwen3" in self.model.lower():
            for i in range(len(work_messages) - 1, -1, -1):
                if work_messages[i].get("role") == "user":
                    work_messages[i] = {
                        **work_messages[i],
                        "content": work_messages[i]["content"] + " /no_think",
                    }
                    break

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=work_messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if not content or not content.strip():
            rc = getattr(resp.choices[0].message, "reasoning_content", None)
            if rc and rc.strip():
                return rc
        return content if content is not None else ""

    @staticmethod
    def _try_parse(payload: dict, schema: type[ModelT]) -> Optional[ModelT]:
        try:
            return schema(**payload)
        except Exception:
            return None

    @staticmethod
    def _extract_json(text: str) -> dict:
        import json
        import re

        cleaned = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think[^>]*/>", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            text = cleaned

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        candidates = OpenAICompatibleStructuredBackend._extract_all_balanced_json(text)
        for obj in reversed(candidates):
            return obj

        raise json.JSONDecodeError("Could not extract JSON from text", text, 0)

    @staticmethod
    def _extract_balanced_json(text: str, start: int) -> Optional[dict]:
        import json
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
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
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _extract_all_balanced_json(text: str) -> list[dict]:
        results: list[dict] = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                obj = OpenAICompatibleStructuredBackend._extract_balanced_json(text, i)
                if obj is not None:
                    results.append(obj)
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
