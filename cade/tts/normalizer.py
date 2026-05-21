"""Text normalization for English TTS."""

from __future__ import annotations

import re


_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
_TEENS = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_KNOWN_ACRONYMS = {
    "ASR": "A S R",
    "LLM": "L L M",
    "ZMQ": "Z M Q",
    "USB-C": "U S B C",
    "USB": "U S B",
    "ID": "I D",
}


def normalize_tts_text(text: str, locale: str = "en_US", domain: str = "restaurant") -> str:
    """Normalize text before synthesis.

    The rules are intentionally conservative and restaurant-oriented.  They
    convert clear speech forms without rewriting every digit globally.
    """
    if locale != "en_US":
        return _clean_spaces(text)

    value = _clean_spaces(str(text or ""))
    if not value:
        return ""

    value = _normalize_money(value)
    value = _normalize_number_labels(value)
    value = _normalize_times(value)
    value = _normalize_ids(value)
    value = _normalize_x_quantities(value)
    if domain == "restaurant":
        value = _normalize_item_quantities(value)
    value = _normalize_acronyms(value)
    return _clean_spaces(value)


class TextNormalizer:
    """Small callable wrapper for configurable normalizers."""

    def __init__(self, enabled: bool = True, locale: str = "en_US", domain: str = "restaurant"):
        self.enabled = enabled
        self.locale = locale
        self.domain = domain

    def normalize(self, text: str) -> str:
        if not self.enabled:
            return _clean_spaces(text)
        return normalize_tts_text(text, locale=self.locale, domain=self.domain)


def _normalize_money(text: str) -> str:
    def repl(match: re.Match) -> str:
        dollars = int(match.group(1).replace(",", ""))
        cents_raw = match.group(2)
        cents = int(cents_raw[1:]) if cents_raw else 0
        dollar_word = "dollar" if dollars == 1 else "dollars"
        if cents:
            cent_word = "cent" if cents == 1 else "cents"
            return f"{_number_to_words(dollars)} {dollar_word} and {_number_to_words(cents)} {cent_word}"
        return f"{_number_to_words(dollars)} {dollar_word}"

    return re.sub(r"\$(\d[\d,]*)(\.\d{1,2})?", repl, text)


def _normalize_number_labels(text: str) -> str:
    return re.sub(
        r"\bNo\.\s*(\d+)\b",
        lambda m: f"number {_number_to_words(int(m.group(1)))}",
        text,
        flags=re.IGNORECASE,
    )


def _normalize_times(text: str) -> str:
    def repl(match: re.Match) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if minute == 0:
            return f"{_number_to_words(hour)} o'clock"
        if minute < 10:
            return f"{_number_to_words(hour)} oh {_number_to_words(minute)}"
        return f"{_number_to_words(hour)} {_number_to_words(minute)}"

    return re.sub(r"\b(\d{1,2}):(\d{2})\b", repl, text)


def _normalize_ids(text: str) -> str:
    def repl(match: re.Match) -> str:
        label = match.group(1)
        digits = match.group(2)
        return f"{label} {_digits_to_words(digits)}"

    return re.sub(r"\b(Order|ID|Room|Table)\s+(\d{2,})\b", repl, text, flags=re.IGNORECASE)


def _normalize_x_quantities(text: str) -> str:
    def repl(match: re.Match) -> str:
        item = match.group(1)
        qty = int(match.group(2))
        return f"{_number_to_words(qty)} {_pluralize(item, qty)}"

    return re.sub(r"\b([A-Za-z][A-Za-z-]*)\s+x\s*(\d+)\b", repl, text, flags=re.IGNORECASE)


def _normalize_item_quantities(text: str) -> str:
    def repl(match: re.Match) -> str:
        qty = int(match.group(1))
        item = match.group(2)
        return f"{_number_to_words(qty)} {item}"

    return re.sub(r"\b(\d+)\s+([A-Za-z][A-Za-z-]*)\b", repl, text)


def _normalize_acronyms(text: str) -> str:
    for token in sorted(_KNOWN_ACRONYMS, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(token)}\b", _KNOWN_ACRONYMS[token], text)
    return text


def _number_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _number_to_words(abs(n))
    if n < 10:
        return _ONES[n]
    if n in _TEENS:
        return _TEENS[n]
    if n < 100:
        tens = (n // 10) * 10
        rem = n % 10
        return _TENS[tens] if rem == 0 else f"{_TENS[tens]} {_ONES[rem]}"
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        base = f"{_ONES[hundreds]} hundred"
        return base if rem == 0 else f"{base} {_number_to_words(rem)}"
    if n < 10000:
        thousands = n // 1000
        rem = n % 1000
        base = f"{_number_to_words(thousands)} thousand"
        return base if rem == 0 else f"{base} {_number_to_words(rem)}"
    return _digits_to_words(str(n))


def _digits_to_words(digits: str) -> str:
    return " ".join(_ONES[int(ch)] for ch in str(digits) if ch.isdigit())


def _pluralize(word: str, qty: int) -> str:
    if qty == 1:
        return word.lower()
    lowered = word.lower()
    if lowered.endswith("s"):
        return lowered
    if lowered.endswith("y"):
        return lowered[:-1] + "ies"
    return lowered + "s"


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
