"""Audio replay eval runner for InputPipeline.

Validates that the deterministic parsing pipeline produces the expected
SemanticEvent for each test case in evals/audio/manifest.json.

Usage:
    python scripts/eval_audio_replay.py
    python scripts/eval_audio_replay.py --manifest evals/audio/manifest.json
    python scripts/eval_audio_replay.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cade.fsm.parsing.input_classifier import OrderInputClassifier
from cade.fsm.parsing.menu_context import MenuContextProvider
from cade.fsm.parsing.order_parser import ConfirmationParser, DeterministicOrderParser
from cade.fsm.parsing.pipeline import InputPipeline


DEFAULT_ALIASES = {
    "coke": ["coke", "cola"],
    "water": ["water"],
    "burger": ["burger", "hamburger"],
}

DEFAULT_MANIFEST = PROJECT_ROOT / "evals" / "audio" / "manifest.json"


def make_pipeline(aliases: Dict[str, List[str]] | None = None) -> InputPipeline:
    aliases = aliases or DEFAULT_ALIASES
    provider = MenuContextProvider(aliases)
    return InputPipeline(
        classifier=OrderInputClassifier(aliases),
        order_parser=DeterministicOrderParser(provider),
        confirm_parser=ConfirmationParser(),
        menu_provider=provider,
    )


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def check_match(event_dict: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    errors = []
    for key, expected_val in expected.items():
        if key == "has_fix_order":
            has_fix = event_dict.get("fix_order") is not None
            if has_fix != expected_val:
                errors.append(f"has_fix_order: expected {expected_val}, got {has_fix}")
            continue
        if key == "items":
            actual_items = event_dict.get("items") or []
            if not _items_match(actual_items, expected_val):
                errors.append(f"items mismatch: expected {expected_val}, got {actual_items}")
            continue
        actual_val = event_dict.get(key)
        if actual_val != expected_val:
            errors.append(f"{key}: expected {expected_val!r}, got {actual_val!r}")
    return errors


def _items_match(actual: List[Dict], expected: List[Dict]) -> bool:
    if len(actual) != len(expected):
        return False
    for exp_item in expected:
        found = False
        for act_item in actual:
            if (act_item.get("name") == exp_item.get("name")
                    and act_item.get("qty") == exp_item.get("qty")):
                found = True
                break
        if not found:
            return False
    return True


def run_eval(manifest_path: Path, json_output: bool = False) -> int:
    cases = load_manifest(manifest_path)
    pipeline = make_pipeline()
    results = []
    pass_count = 0
    fail_count = 0

    for case in cases:
        case_id = case["id"]
        text = case["input_text"]
        state = case["fsm_state"]
        expected = case["expected"]

        if state == "LISTEN":
            event = pipeline.process_listen(text, source="eval")
        elif state == "CHECK":
            event = pipeline.process_check(text, source="eval")
        else:
            results.append({"id": case_id, "status": "error", "error": f"unknown state: {state}"})
            fail_count += 1
            continue

        event_dict = event.model_dump()
        errors = check_match(event_dict, expected)

        if errors:
            status = "fail"
            fail_count += 1
        else:
            status = "pass"
            pass_count += 1

        results.append({
            "id": case_id,
            "status": status,
            "input_text": text,
            "fsm_state": state,
            "actual": event_dict,
            "expected": expected,
            "errors": errors or None,
        })

    if json_output:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            marker = "PASS" if r["status"] == "pass" else "FAIL"
            print(f"  [{marker}] {r['id']}: {r['input_text']!r}")
            if r.get("errors"):
                for e in r["errors"]:
                    print(f"         {e}")
        print()
        print(f"Results: {pass_count} passed, {fail_count} failed, {len(results)} total")
        if fail_count == 0:
            print("All tests passed!")

    return 1 if fail_count > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Audio replay eval runner")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.json")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    args = parser.parse_args()
    sys.exit(run_eval(Path(args.manifest), json_output=args.json))


if __name__ == "__main__":
    main()
