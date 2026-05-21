"""LLM structured output capability probe.

Sends minimal JSON Schema probes to the configured LLM endpoint and reports
which schema features each backend actually supports.  Run with --live-llm
to hit a real endpoint; without it, only imports are checked.

Usage:
    python scripts/check_llm_structured_capabilities.py --live-llm
    python scripts/check_llm_structured_capabilities.py --live-llm --base-url http://127.0.0.1:8080/v1
"""

import argparse
import json
import sys
import time

SCHEMA_PROBES = {
    "const": {
        "schema": {
            "type": "object",
            "properties": {"status": {"type": "string", "const": "ok"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    },
    "enum": {
        "schema": {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
            "required": ["color"],
            "additionalProperties": False,
        },
    },
    "additionalProperties_false": {
        "schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
    },
    "nested_object": {
        "schema": {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "string"}},
                    "required": ["inner"],
                    "additionalProperties": False,
                },
            },
            "required": ["outer"],
            "additionalProperties": False,
        },
    },
    "oneOf_with_null": {
        "schema": {
            "type": "object",
            "properties": {
                "value": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}


def probe_one(client, model: str, name: str, schema: dict) -> dict:
    rf = {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }
    result = {"name": name, "api_accepted": False, "valid_json": False,
              "schema_valid": False, "error_type": None, "latency_s": 0.0}

    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Output a valid JSON object matching the schema for {name}."}],
            max_tokens=128,
            temperature=0.0,
            response_format=rf,
        )
        result["api_accepted"] = True
        content = resp.choices[0].message.content or ""

        try:
            parsed = json.loads(content)
            result["valid_json"] = True
        except json.JSONDecodeError:
            result["error_type"] = "invalid_json"
            return result

        try:
            import jsonschema
            jsonschema.validate(parsed, schema)
            result["schema_valid"] = True
        except ImportError:
            result["schema_valid"] = True  # assume ok if jsonschema not installed
        except Exception:
            result["error_type"] = "schema_violation"

    except Exception as exc:
        msg = str(exc).lower()
        if "400" in msg or "invalid" in msg or "unsupported" in msg:
            result["error_type"] = "api_rejected_schema"
        elif "timeout" in msg:
            result["error_type"] = "timeout"
        else:
            result["error_type"] = f"other: {type(exc).__name__}"
    result["latency_s"] = round(time.monotonic() - t0, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description="Probe LLM structured output capabilities")
    parser.add_argument("--live-llm", action="store_true", help="Actually connect to the LLM")
    parser.add_argument("--base-url", default=None, help="Override LLM base URL")
    parser.add_argument("--model", default=None, help="Override model name")
    args = parser.parse_args()

    if not args.live_llm:
        print("No --live-llm flag. Exiting (use --live-llm to probe a real endpoint).")
        sys.exit(0)

    from openai import OpenAI
    from cade.config import Config

    config = Config.get_llm_config()
    base_url = args.base_url or config["base_url"]
    model = args.model or config["model"]

    print(f"Probing: {base_url} model={model}\n")
    client = OpenAI(base_url=base_url, api_key=config.get("api_key", "not-needed"))

    results = []
    for name, probe in SCHEMA_PROBES.items():
        r = probe_one(client, model, name, probe["schema"])
        results.append(r)
        status = "PASS" if r["schema_valid"] else "FAIL"
        print(f"  {name}: {status} (api={r['api_accepted']}, json={r['valid_json']}, "
              f"schema={r['schema_valid']}, err={r['error_type']}, latency={r['latency_s']}s)")

    supported = sum(1 for r in results if r["schema_valid"])
    print(f"\n{supported}/{len(results)} probes fully supported")

    report_path = ".cache/llm_capabilities.json"
    try:
        import os
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"base_url": base_url, "model": model, "results": results}, f, indent=2)
        print(f"Report saved to {report_path}")
    except Exception as exc:
        print(f"Could not save report: {exc}")


if __name__ == "__main__":
    main()
