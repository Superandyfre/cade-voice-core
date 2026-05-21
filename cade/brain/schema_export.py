"""Export Pydantic models to LLM-facing JSON Schema.

Ensures schemas are compatible with various backend profiles by stripping
unsupported keywords and enforcing `additionalProperties: false`.
"""

from typing import Optional

from pydantic import BaseModel


def export_llm_json_schema(model: type[BaseModel], *, profile: str = "openai") -> dict:
    """Export a Pydantic model to a JSON Schema suitable for LLM structured output.

    Strips fields that confuse backends (anyOf with null, default values)
    and enforces `additionalProperties: false` on all objects.
    """
    raw = model.model_json_schema()

    schema = _clean_schema(raw, profile=profile)
    return schema


def _clean_schema(schema: dict, *, profile: str) -> dict:
    """Recursively clean a JSON schema for LLM compatibility."""
    if not isinstance(schema, dict):
        return schema

    result = {}

    for key, value in schema.items():
        if key == "$defs":
            defs = {}
            for def_name, def_schema in value.items():
                defs[def_name] = _clean_schema(def_schema, profile=profile)
            if defs:
                result[key] = defs
            continue

        if key == "anyOf":
            # Simplify anyOf with null -> make the field optional at the parent level
            non_null = [s for s in value if not (isinstance(s, dict) and s.get("type") == "null")]
            if len(non_null) == 1:
                result = _merge(result, _clean_schema(non_null[0], profile=profile))
            else:
                result[key] = [_clean_schema(s, profile=profile) for s in value]
            continue

        if key == "allOf":
            if len(value) == 1:
                result = _merge(result, _clean_schema(value[0], profile=profile))
            else:
                result[key] = [_clean_schema(s, profile=profile) for s in value]
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {k: _clean_schema(v, profile=profile) for k, v in value.items()}
            continue

        if key == "items" and isinstance(value, dict):
            result[key] = _clean_schema(value, profile=profile)
            continue

        if key == "additionalProperties":
            continue  # will be set explicitly below

        if key in ("title", "description", "default") and profile in ("llama_cpp",):
            continue

        result[key] = value

    # Enforce additionalProperties: false on objects
    if result.get("type") == "object" and "properties" in result:
        result["additionalProperties"] = False

    return result


def _merge(base: dict, override: dict) -> dict:
    """Merge override into base dict."""
    merged = dict(base)
    merged.update(override)
    return merged
