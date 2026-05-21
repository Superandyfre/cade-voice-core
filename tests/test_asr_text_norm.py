"""Unit tests for ASR text normalization."""

from cade.asr.text_norm import normalize_asr_text


def test_basic_lowercase():
    assert normalize_asr_text("Hello World") == "hello world"


def test_strip_trailing_punctuation():
    assert normalize_asr_text("I would like a Coke.") == "i would like a coke"
    assert normalize_asr_text("Yes!") == "yes"
    assert normalize_asr_text("No, thanks.") == "no, thanks"
    assert normalize_asr_text("Really?") == "really"


def test_empty_string():
    assert normalize_asr_text("") == ""
    assert normalize_asr_text("   ") == ""


def test_default_replacements():
    assert normalize_asr_text("I want a hamburger.") == "i want a burger"
    assert normalize_asr_text("Two cheeseburgers.") == "two burger"
    assert normalize_asr_text("A Coca Cola.") == "a coke"
    assert normalize_asr_text("I want cola.") == "i want coke"


def test_no_replacements():
    text = "I want a hamburger."
    result = normalize_asr_text(text, replacements=[])
    assert result == "i want a hamburger"


def test_custom_replacements():
    custom = [(r"\bsoda\b", "coke")]
    assert normalize_asr_text("I want soda.", custom) == "i want coke"


def test_internal_punctuation_preserved():
    assert normalize_asr_text("I'd like that.") == "i'd like that"
    assert normalize_asr_text("It's fine.") == "it's fine"
