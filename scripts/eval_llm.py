"""Eval runner for LLM structured output.

Supports mock mode (no LLM) and live mode.
Reads JSONL test cases from evals/bootstrap/ and outputs traces + summary.

Usage:
    python scripts/eval_llm.py --task order_listen --live-llm
    python scripts/eval_llm.py --all --live-llm
    python scripts/eval_llm.py --task order_check --out evals/results/run.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
BOOTSTRAP_DIR = EVALS_DIR / "bootstrap"
RESULTS_DIR = EVALS_DIR / "results"


def load_cases(task: str) -> list[dict]:
    path = BOOTSTRAP_DIR / f"{task}.jsonl"
    if not path.is_file():
        print(f"No cases file found: {path}")
        return []
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_mock_eval(cases: list[dict], task: str) -> list[dict]:
    """Run eval without LLM — only checks that cases are parseable."""
    results = []
    for case in cases:
        results.append({
            "id": case["id"],
            "task": task,
            "mode": "mock",
            "input": case["input"],
            "expected": case.get("expected"),
            "status": "loaded",
            "tags": case.get("tags", []),
        })
    return results


def run_live_order_listen(llm, cases: list[dict], food_aliases: dict) -> list[dict]:
    results = []
    for case in cases:
        t0 = time.monotonic()
        try:
            order = llm.get_order_action(
                user_input=case["input"],
                food_aliases=food_aliases,
                max_retries=2,
            )
            actual = order.model_dump()
            expected = case.get("expected", {})

            result = evaluate_order(actual, expected)
            result.update({
                "id": case["id"],
                "task": "order_listen",
                "mode": "live",
                "latency_s": round(time.monotonic() - t0, 3),
                "tags": case.get("tags", []),
            })
            results.append(result)
        except Exception as exc:
            results.append({
                "id": case["id"],
                "task": "order_listen",
                "mode": "live",
                "status": "error",
                "error": str(exc),
                "latency_s": round(time.monotonic() - t0, 3),
            })
    return results


def run_live_order_check(llm, cases: list[dict], food_aliases: dict) -> list[dict]:
    results = []
    for case in cases:
        t0 = time.monotonic()
        try:
            current_order = case.get("current_order", {"type": "order", "items": [{"name": "coke", "qty": 1}]})
            from cade.brain.schemas import OrderAction, OrderItem
            order = OrderAction(**current_order)

            decision = llm.get_order_check_decision(
                customer_reply=case["input"],
                order_action=order,
                food_aliases=food_aliases,
                max_retries=2,
            )
            actual = decision.model_dump()
            expected = case.get("expected", {})

            result = evaluate_check(actual, expected)
            result.update({
                "id": case["id"],
                "task": "order_check",
                "mode": "live",
                "latency_s": round(time.monotonic() - t0, 3),
                "tags": case.get("tags", []),
            })
            results.append(result)
        except Exception as exc:
            results.append({
                "id": case["id"],
                "task": "order_check",
                "mode": "live",
                "status": "error",
                "error": str(exc),
                "latency_s": round(time.monotonic() - t0, 3),
            })
    return results


def evaluate_order(actual: dict, expected: dict) -> dict:
    """Compare actual order output to expected."""
    if not actual.get("items"):
        if not expected.get("items"):
            return {"status": "pass", "actual": actual}
        return {"status": "fail", "actual": actual, "reason": "expected items but got empty"}

    actual_items = {i["name"]: i["qty"] for i in actual.get("items", [])}
    expected_items = {i["name"]: i["qty"] for i in expected.get("items", [])}

    if actual_items == expected_items:
        return {"status": "pass", "actual": actual}

    return {
        "status": "fail",
        "actual": actual,
        "expected": expected,
        "reason": f"items mismatch: {actual_items} != {expected_items}",
    }


def evaluate_check(actual: dict, expected: dict) -> dict:
    """Compare actual check decision to expected."""
    if actual.get("result") == expected.get("result"):
        if expected.get("action") and actual.get("action"):
            return {"status": "pass", "actual": actual}
        if not expected.get("action") and not actual.get("action"):
            return {"status": "pass", "actual": actual}
        if expected.get("has_reply") and actual.get("reply"):
            return {"status": "pass", "actual": actual}
        if not expected.get("action") and actual.get("action") is None:
            return {"status": "pass", "actual": actual}
        return {"status": "partial", "actual": actual, "expected": expected}
    return {"status": "fail", "actual": actual, "expected": expected}


def compute_summary(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"total": 0}

    passed = sum(1 for r in results if r.get("status") == "pass")
    failed = sum(1 for r in results if r.get("status") == "fail")
    errors = sum(1 for r in results if r.get("status") == "error")
    latencies = [r["latency_s"] for r in results if "latency_s" in r]

    summary = {
        "total": total,
        "pass": passed,
        "fail": failed,
        "error": errors,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
    }
    if latencies:
        summary["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        summary["p95_latency_s"] = round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 3)
    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM eval runner")
    parser.add_argument("--task", default=None, help="Task name (order_listen, order_check, etc.)")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--live-llm", action="store_true", help="Use real LLM")
    parser.add_argument("--out", default=None, help="Output JSONL path")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tasks = []
    if args.all:
        for f in BOOTSTRAP_DIR.glob("*.jsonl"):
            tasks.append(f.stem)
    elif args.task:
        tasks = [args.task]
    else:
        print("Specify --task <name> or --all")
        sys.exit(1)

    llm = None
    if args.live_llm:
        from cade.brain.llm_client import LLMClient
        llm = LLMClient()

    food_aliases = {
        "water": ["water", "bottle of water"],
        "coke": ["coke", "cola", "coca cola"],
        "juice": ["juice", "orange juice"],
        "coffee": ["coffee", "latte"],
        "tea": ["tea"],
        "burger": ["burger", "hamburger"],
        "pizza": ["pizza"],
        "fries": ["fries", "french fries", "chips"],
        "fried_rice": ["fried rice"],
        "noodles": ["noodles", "ramen"],
    }

    all_results = []
    for task in tasks:
        cases = load_cases(task)
        if not cases:
            continue
        print(f"\n=== {task}: {len(cases)} cases ===")

        if not args.live_llm:
            results = run_mock_eval(cases, task)
        elif task == "order_listen":
            results = run_live_order_listen(llm, cases, food_aliases)
        elif task == "order_check":
            results = run_live_order_check(llm, cases, food_aliases)
        else:
            results = run_mock_eval(cases, task)

        all_results.extend(results)
        summary = compute_summary(results)
        print(f"  Pass: {summary.get('pass', 0)}/{summary.get('total', 0)} "
              f"({summary.get('pass_rate', 0)}%)")
        if summary.get("avg_latency_s"):
            print(f"  Avg latency: {summary['avg_latency_s']}s, "
                  f"P95: {summary.get('p95_latency_s', 'N/A')}s")

    # Write output
    out_path = args.out or str(RESULTS_DIR / f"run_{int(time.time())}.jsonl")
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults written to {out_path}")

    # Final summary
    final = compute_summary(all_results)
    print(f"\nTotal: {final.get('total', 0)} | Pass: {final.get('pass', 0)} | "
          f"Fail: {final.get('fail', 0)} | Error: {final.get('error', 0)}")


if __name__ == "__main__":
    main()
