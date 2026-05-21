# Bootstrap eval cases

First batch of test inputs frozen from the smoke benchmark and manual adversarial cases.
Each file is JSONL with fields: id, task, input, expected, tags.

Usage: `python scripts/eval_llm.py --task order_listen --live-llm`
