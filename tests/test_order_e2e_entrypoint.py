"""Tests for the real-device cade-order-e2e entrypoint."""

from pathlib import Path

import pytest

from cade.fsm import full_pipeline_standalone


def test_removed_fake_cli_options_are_rejected():
    parser = full_pipeline_standalone._build_parser()
    for option in ("--mock", "--text", "--timeout", "--order-text", "--confirm-text"):
        with pytest.raises(SystemExit):
            parser.parse_args([option])


def test_standalone_e2e_contains_no_fake_paths_or_deadline_waits():
    source = Path(full_pipeline_standalone.__file__).read_text(encoding="utf-8")
    forbidden = (
        "MockLLMClient",
        "transcribe_once",
        "--mock",
        "--text",
        "--timeout",
        "--order-text",
        "--confirm-text",
        "I want a coke",
        "yes, that's correct",
        "deadline",
        "RCVTIMEO",
        "SNDTIMEO",
    )
    for token in forbidden:
        assert token not in source
