"""Deterministic order and confirmation parsers.

Handles simple, high-frequency ordering patterns without LLM calls.
Uses rule-based parsing for quantity extraction, alias resolution, and
yes/no confirmation.
"""

import re
from difflib import SequenceMatcher
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from cade.brain.schemas import OrderAction, OrderItem
from cade.fsm.parsing.menu_context import MenuContext


class ParseResult(BaseModel):
    order: Optional[OrderAction] = None
    confidence: float = 0.0
    reason_code: str = ""
    unmatched_text: str = ""
    used_aliases: Dict[str, str] = {}


class ConfirmationParseResult(BaseModel):
    result: Literal["correct", "wrong", "unknown", "cancel", "repeat_request"]
    fix_order: Optional[OrderAction] = None
    confidence: float = 0.0
    reply: Optional[str] = None


class OrderInputKind(str, Enum):
    """Classification of user input during ordering."""
    VALID_ORDER = "valid_order"
    MENU_QUESTION = "menu_question"
    OUT_OF_MENU_ITEM = "out_of_menu_item"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    QUANTITY_ERROR = "quantity_error"
    CANCEL_REQUEST = "cancel_request"
    PAUSE_REQUEST = "pause_request"
    REPEAT_REQUEST = "repeat_request"
    EMPTY_OR_NOISE = "empty_or_noise"
    SMALLTALK = "smalltalk"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


class ClassifiedInput(BaseModel):
    """Result of fast input classification."""
    kind: OrderInputKind
    confidence: float = 0.0
    normalized_text: str = ""
    reason: Optional[str] = None
    matched_items: Optional[List[OrderItem]] = None
    out_of_menu_item: Optional[str] = None


# Number word -> integer mapping
_NUMBER_WORDS = {
    "one": 1, "a": 1, "an": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "11": 11, "12": 12, "13": 13, "14": 14, "15": 15,
    "16": 16, "17": 17, "18": 18, "19": 19, "20": 20,
}

# Filler phrases to strip before evaluation
_FILLERS = {
    "please", "can i have", "i want", "i'd like", "i will have",
    "i would like", "give me", "get me", "let me have",
    "could i get", "may i have", "uh", "um", "maybe",
    "actually", "i think", "and", "plus",
}

# Confirmation words
_CORRECT_WORDS = {"yes", "yeah", "yep", "correct", "right", "sure", "ok", "okay", "that's right", "that is right"}
_WRONG_WORDS = {"no", "nope", "wrong", "not correct", "that's wrong"}

_NO_CHANGE_PHRASES = [
    "no change", "nothing changed", "nothing to change",
    "nothing needs to change", "nothing need to change",
    "no changes", "doesn't need to change", "don't need to change",
    "don't change anything", "no need to change", "nothing to be changed",
    "nothing has changed", "nothing is changed",
]

_POSITIVE_CONFIRM_PHRASES = [
    "that's right", "that is right", "that's correct", "that is correct",
    "looks good", "looks fine", "looks great", "sounds good", "sounds right",
    "that's fine", "that is fine", "that's perfect", "keep it", "leave it",
    "that's all", "everything is fine", "everything is good", "it's good",
    "it's fine", "it's correct", "order is correct", "order is fine",
    "order is good", "that's good", "that is good", "all good",
    "no problem", "go ahead", "looks correct", "sounds correct",
]


class DeterministicOrderParser:
    """Rule-based order parser for simple ordering patterns."""

    def __init__(self, menu_context_provider=None):
        self._menu_provider = menu_context_provider

    def parse_order(self, text: str, menu: MenuContext) -> ParseResult:
        """Parse an order from text using rules."""
        text_lower = text.strip().lower()
        if not text_lower:
            return ParseResult(reason_code="empty_input")

        # Build lookup: canonical_name and aliases -> canonical
        lookup: Dict[str, str] = {}
        for candidate in menu.candidates:
            lookup[candidate.canonical] = candidate.canonical
            for alias in candidate.aliases:
                alias_norm = alias.replace(" ", "_")
                lookup[alias_norm] = candidate.canonical
                lookup[alias.replace(" ", "")] = candidate.canonical

        items: Dict[str, int] = {}
        used_aliases: Dict[str, str] = {}
        remaining = text_lower

        # Intermediate measure words: "two cups of coke", "three glasses of water"
        _MEASURE_WORDS = r"(?:\s+(?:cup|cups|glass|glasses|bottle|bottles|piece|pieces|plate|plates|bowl|bowls|slice|slices|serving|servings|portion|portions|order|orders)\s+of)?"

        # Try to match food items with quantities
        # Pattern: [number] [measure?] [food_name] [and|plus|comma] [number] [measure?] [food_name] ...
        # First pass: try explicit number + food patterns
        for canonical in menu.candidates:
            all_names = [canonical.canonical] + [a.replace(" ", "_") for a in canonical.aliases]

            for name in sorted(all_names, key=len, reverse=True):
                # Pattern: number + [measure] + name
                pattern = rf'(?:^|\s)(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|a|an){_MEASURE_WORDS}\s+{re.escape(name)}s?(?:\s|$|,)'
                matches = list(re.finditer(pattern, remaining))
                for m in matches:
                    qty_str = m.group(1)
                    qty = _NUMBER_WORDS.get(qty_str, 1)
                    items[canonical.canonical] = items.get(canonical.canonical, 0) + qty
                    if name != canonical.canonical:
                        used_aliases[name] = canonical.canonical
                    remaining = remaining[:m.start()] + remaining[m.end():]

                # Pattern: just the name (qty=1)
                pattern2 = rf'(?:^|\s){re.escape(name)}s?(?:\s|$|,)'
                if re.search(pattern2, remaining):
                    items[canonical.canonical] = items.get(canonical.canonical, 0) + 1
                    if name != canonical.canonical:
                        used_aliases[name] = canonical.canonical
                    remaining = re.sub(pattern2, " ", remaining)

        if not items:
            return ParseResult(reason_code="no_items_found", confidence=0.0)

        # Check remaining text for confidence
        remaining_clean = re.sub(r"[^a-z\s]", "", remaining).strip()
        remaining_words = [w for w in remaining_clean.split() if w and w not in _FILLERS]

        # Calculate confidence
        if not remaining_words:
            confidence = 0.95
        elif len(remaining_words) <= 2:
            confidence = 0.85
        elif len(remaining_words) <= 4:
            confidence = 0.7
        else:
            confidence = 0.5

        order_items = [OrderItem(name=name, item_id=name, qty=qty) for name, qty in sorted(items.items())]
        return ParseResult(
            order=OrderAction(type="order", items=order_items),
            confidence=confidence,
            reason_code="parsed",
            unmatched_text=" ".join(remaining_words),
            used_aliases=used_aliases,
        )


_CANCEL_CONFIRM_PHRASES = {"cancel", "never mind", "forget it", "stop", "abort"}
_REPEAT_CONFIRM_PHRASES = {"say again", "repeat that", "what did i order", "repeat"}

_MODIFICATION_SIGNALS = {
    "add", "remove", "change", "instead", "actually", "but",
    "make it", "one more", "without", "extra", "less", "more",
    "replace", "swap",
}


class ConfirmationParser:
    """Rule-based confirmation parser with modification-priority logic.

    Priority: cancel > repeat_request > no_change_confirm > modification_signal > negation > positive > unknown.
    Modification signals override positive words: "yes but add fries" is NOT correct.
    """

    def parse(self, text: str, menu: MenuContext) -> ConfirmationParseResult:
        """Parse a confirmation response."""
        text_lower = text.strip().lower()

        if not text_lower:
            return ConfirmationParseResult(result="unknown", confidence=0.0, reply="Could you please confirm?")

        # 1. Cancel
        for phrase in _CANCEL_CONFIRM_PHRASES:
            if text_lower == phrase or text_lower.startswith(phrase):
                return ConfirmationParseResult(result="cancel", confidence=0.95)

        # 2. Repeat request
        for phrase in _REPEAT_CONFIRM_PHRASES:
            if text_lower == phrase or text_lower.startswith(phrase):
                return ConfirmationParseResult(result="repeat_request", confidence=0.9)

        # 3. "No change" / "nothing changed" → confirmation (before modification signal check)
        for phrase in _NO_CHANGE_PHRASES:
            if phrase in text_lower:
                return ConfirmationParseResult(result="correct", confidence=0.92)

        # 4. Modification signals (even with yes/ok prefix) — highest priority after cancel
        if self._has_modification_signal(text_lower):
            fix = self._extract_modification(text_lower, menu)
            if fix:
                return ConfirmationParseResult(
                    result="wrong",
                    fix_order=fix,
                    confidence=0.95,
                )
            return ConfirmationParseResult(
                result="wrong", confidence=0.95,
                reply="What would you like to change?",
            )

        # 4. Explicit negative (no / nope / wrong / not)
        has_negation = any(
            text_lower.startswith(w) or f" {w} " in f" {text_lower} "
            for w in ["no", "nope", "not", "wrong"]
        )

        if has_negation:
            order_parser = DeterministicOrderParser()
            clean_text = text_lower
            for neg in ["nope", "not correct", "no", "wrong", "not"]:
                clean_text = clean_text.replace(neg, "", 1)
            clean_text = clean_text.strip(" ,")
            clean_text = clean_text.replace("instead", "").strip(" ,")

            if clean_text:
                order_result = order_parser.parse_order(clean_text, menu)
                if order_result.order and order_result.order.items:
                    return ConfirmationParseResult(
                        result="wrong",
                        fix_order=order_result.order,
                        confidence=0.9,
                    )

            if "not sure" in text_lower or "maybe" in text_lower or "don't know" in text_lower:
                return ConfirmationParseResult(
                    result="wrong", confidence=0.8,
                    reply="Would you like to change anything?",
                )

            return ConfirmationParseResult(
                result="wrong", confidence=0.85,
                reply="What would you like instead?",
            )

        # 5. Explicit positive (yes / correct / right / sure / ok / extended phrases)
        for word in _CORRECT_WORDS:
            if text_lower == word or text_lower.startswith(word):
                return ConfirmationParseResult(result="correct", confidence=0.99)

        for phrase in _POSITIVE_CONFIRM_PHRASES:
            if text_lower == phrase or text_lower.startswith(phrase + " ") or text_lower.startswith(phrase):
                return ConfirmationParseResult(result="correct", confidence=0.95)

        # 6. Unknown
        return ConfirmationParseResult(result="unknown", confidence=0.3, reply="Could you please confirm?")

    def _has_modification_signal(self, text: str) -> bool:
        # Negated modification signals are NOT modifications
        for neg in _NO_CHANGE_PHRASES:
            if neg in text:
                return False
        for signal in _MODIFICATION_SIGNALS:
            if f" {signal} " in f" {text} " or text.endswith(f" {signal}"):
                return True
        # "no <food>" at start or after space
        for candidate_name in ("coke", "fries", "water", "burger", "coffee", "tea",
                               "juice", "pizza", "sandwich", "salad", "soup", "noodles",
                               "dumplings", "dumpling", "pasta", "fried_rice", "rice"):
            if f"no {candidate_name}" in text:
                return True
        return False

    def _extract_modification(self, text: str, menu: MenuContext) -> Optional[OrderAction]:
        order_parser = DeterministicOrderParser()
        clean_text = text
        # Strip common positive prefixes to get the modification part
        for prefix in ("yes but", "yeah but", "ok but", "yes actually", "yeah actually",
                       "ok actually", "yes", "yeah", "ok", "okay", "sure"):
            if clean_text.startswith(prefix):
                clean_text = clean_text[len(prefix):].strip(" ,")
                break
        clean_text = clean_text.replace("instead", "").strip(" ,")
        if not clean_text:
            return None
        result = order_parser.parse_order(clean_text, menu)
        if result.order and result.order.items:
            return result.order
        return None
