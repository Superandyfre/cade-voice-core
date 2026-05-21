"""JSON Schema builders for constrained LLM output in the ordering sub-FSM.

Each builder produces an OpenAI-compatible ``response_format`` dict with
``type: "json_schema"``.  The schemas are strict:
- ``additionalProperties: false`` everywhere
- No optional / default-valued fields in the LLM-facing shape
- Canonical food names become an ``enum`` when available
"""

from typing import Dict, List, Optional


def _order_item_schema(canonical_names: Optional[List[str]] = None) -> dict:
    name_schema: dict = {"type": "string", "minLength": 1}
    if canonical_names:
        name_schema = {"type": "string", "enum": canonical_names}
    return {
        "type": "object",
        "properties": {
            "name": name_schema,
            "qty": {"type": "integer", "minimum": 1},
        },
        "required": ["name", "qty"],
        "additionalProperties": False,
    }


def build_listen_response_format(
    food_aliases: Optional[Dict[str, List[str]]] = None,
) -> dict:
    canonical = sorted(
        {str(k).strip().lower() for k in (food_aliases or {}).keys() if str(k).strip()}
    )
    item_schema = _order_item_schema(canonical or None)

    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "const": "order"},
            "items": {"type": "array", "items": item_schema},
        },
        "required": ["type", "items"],
        "additionalProperties": False,
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "order_action",
            "strict": True,
            "schema": schema,
        },
    }


def build_repeat_response_format() -> dict:
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "const": "speak"},
                    "content": {"type": "string", "minLength": 1, "maxLength": 500},
                },
                "required": ["type", "content"],
                "additionalProperties": False,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "order_speak",
            "strict": True,
            "schema": schema,
        },
    }


def build_check_response_format(
    food_aliases: Optional[Dict[str, List[str]]] = None,
) -> dict:
    canonical = sorted(
        {str(k).strip().lower() for k in (food_aliases or {}).keys() if str(k).strip()}
    )
    item_schema = _order_item_schema(canonical or None)

    fix_order_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "const": "fix_order"},
            "items": {"type": "array", "items": item_schema, "minItems": 1},
        },
        "required": ["type", "items"],
        "additionalProperties": False,
    }

    schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string", "enum": ["correct", "wrong"]},
            "action": {"oneOf": [fix_order_schema, {"type": "null"}]},
            "reply": {"oneOf": [{"type": "string", "maxLength": 300}, {"type": "null"}]},
        },
        "required": ["result", "action", "reply"],
        "additionalProperties": False,
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "order_check",
            "strict": True,
            "schema": schema,
        },
    }


def build_robot_decision_response_format() -> dict:
    """Build response_format for the general robot decision schema."""
    schema = {
        "type": "object",
        "properties": {
            "thought": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            "reply": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            "action": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["search", "pick", "place", "speak", "wait"]},
                            "object_name": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                            "object_id": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
                            "location": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                            "content": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                            "reason": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                        },
                        "required": ["type"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ],
            },
        },
        "required": ["thought", "reply", "action"],
        "additionalProperties": False,
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "robot_decision",
            "strict": True,
            "schema": schema,
        },
    }
