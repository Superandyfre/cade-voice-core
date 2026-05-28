"""
Fast rule-based input classifier for the ordering sub-FSM.

Classifies user text into OrderInputKind before parsing. Zero LLM calls.
Runs in < 1ms for any input.
"""

import re
from typing import Dict, List, Optional, Set

from cade.brain.schemas import OrderAction, OrderItem
from cade.fsm.parsing.order_parser import ClassifiedInput, OrderInputKind


_CANCEL_PHRASES: List[str] = [
    "cancel", "never mind", "forget it", "stop",
    "no thanks", "not anymore", "abort",
]

_CANCEL_EXCLUSIONS: List[str] = [
    "nothing changed", "nothing to change", "nothing needs to change",
    "nothing need to change", "nothing to add", "nothing else",
    "nothing else to add", "nothing more", "nothing else to change",
    "nothing has changed", "nothing is changed", "nothing will change",
]

_REPEAT_REQUEST_PHRASES: List[str] = [
    "say again", "what did you say", "repeat",
    "pardon", "come again", "excuse me",
    "what was that", "i didn't hear",
    "what did i order", "what did i get",
    "repeat that", "repeat order",
]

_MENU_QUESTION_PHRASES: List[str] = [
    "what do you have", "what do you have?",
    "menu", "options", "what's available",
    "what can i get", "what can i order",
    "what do you sell", "what is on the menu",
    "show me the menu", "what kind of",
    "do you have", "is there",
]

_PAUSE_PHRASES: List[str] = [
    "wait", "hold on", "give me a second",
    "one moment", "just a moment", "hang on",
    "one second", "give me a minute",
]

_SMALLTALK_PHRASES: Set[str] = {
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "how are you", "thanks", "thank you",
    "goodbye", "bye", "see you", "yo", "sup",
}

_OUT_OF_SCOPE_PHRASES: List[str] = [
    "where is the bathroom", "where is the toilet",
    "what time", "weather", "where are you",
    "who are you", "what is your name",
    "how old are you", "tell me a joke",
]

_FOOD_WORD_PATTERN = re.compile(
    r"(burger|hamburger|cheeseburger|pizza|sandwich|fries|french fries|"
    r"chips|salad|soup|noodles|ramen|dumpling|dumplings|pasta|spaghetti|"
    r"fried rice|rice|coke|cola|coca cola|water|juice|coffee|latte|"
    r"americano|cappuccino|tea|bread|chicken|beef|fish|steak|ice cream|"
    r"cake|cookie|pie|donut|milk|beer|wine|soda|drink)",
    re.IGNORECASE,
)

_MODIFICATION_SIGNALS: List[str] = [
    "add", "remove", "change", "instead", "actually", "but",
    "make it", "one more", "without", "extra", "less", "more",
    "replace", "swap",
]

_QUANTITY_ERROR_WORDS: Set[str] = {
    "zero", "hundred", "thousand", "million",
    "minus", "negative",
}

_LARGE_NUMBER_RE = re.compile(r"\b(\d{2,})\b")

_FILLERS: Set[str] = {
    "uh", "um", "hmm", "ah", "oh", "er", "mm", "huh",
    "like", "you know", "so", "well",
}

_CONFIRM_WORDS: Set[str] = {
    "no", "nope", "yes", "yeah", "ok", "okay", "sure",
    "right", "correct", "wrong", "not", "yep", "nah",
}


class OrderInputClassifier:
    """Fast rule-based classifier for ordering FSM input."""

    def __init__(self, food_aliases: Dict[str, List[str]]):
        self._canonical_names: Set[str] = set()
        self._all_food_names: Set[str] = set()

        for canonical, aliases in food_aliases.items():
            norm = canonical.lower().replace(" ", "_")
            self._canonical_names.add(norm)
            self._all_food_names.add(norm)
            for alias in aliases:
                self._all_food_names.add(alias.lower().replace(" ", "_"))
                self._all_food_names.add(alias.lower().replace(" ", ""))

    def classify(
        self,
        text: str,
        state: str = "LISTEN",
        current_order: Optional[OrderAction] = None,
    ) -> ClassifiedInput:
        """Classify user input text into an OrderInputKind."""
        raw = str(text or "").strip()
        lower = raw.lower()
        norm = " ".join(lower.split())

        if not norm:
            return ClassifiedInput(
                kind=OrderInputKind.EMPTY_OR_NOISE,
                confidence=0.95,
                normalized_text=norm,
                reason="empty",
            )

        # 1. Cancel
        # "nothing" alone is cancel, but "nothing changed/nothing to change" is NOT
        cancel_excluded = False
        for ex in _CANCEL_EXCLUSIONS:
            if norm == ex or norm.startswith(ex + " ") or norm.startswith(ex):
                cancel_excluded = True
                break

        if not cancel_excluded:
            for phrase in _CANCEL_PHRASES:
                if norm == phrase or norm.startswith(phrase + " ") or norm.startswith(phrase):
                    return ClassifiedInput(
                        kind=OrderInputKind.CANCEL_REQUEST,
                        confidence=0.95,
                        normalized_text=norm,
                        reason=f"match:{phrase}",
                    )
            # "nothing" standalone is cancel
            if norm == "nothing" or norm.startswith("nothing.") or norm.startswith("nothing!"):
                return ClassifiedInput(
                    kind=OrderInputKind.CANCEL_REQUEST,
                    confidence=0.95,
                    normalized_text=norm,
                    reason="match:nothing",
                )

        # 2. Repeat request
        for phrase in _REPEAT_REQUEST_PHRASES:
            if norm == phrase or norm.startswith(phrase):
                return ClassifiedInput(
                    kind=OrderInputKind.REPEAT_REQUEST,
                    confidence=0.9,
                    normalized_text=norm,
                    reason=f"match:{phrase}",
                )

        # 3. Menu question
        for phrase in _MENU_QUESTION_PHRASES:
            if norm == phrase or norm.startswith(phrase):
                return ClassifiedInput(
                    kind=OrderInputKind.MENU_QUESTION,
                    confidence=0.9,
                    normalized_text=norm,
                    reason=f"match:{phrase}",
                )

        # 4. Pause request
        for phrase in _PAUSE_PHRASES:
            if norm == phrase or norm.startswith(phrase + " "):
                return ClassifiedInput(
                    kind=OrderInputKind.PAUSE_REQUEST,
                    confidence=0.85,
                    normalized_text=norm,
                    reason=f"match:{phrase}",
                )

        # 5. Smalltalk (exact or very short match)
        if norm in _SMALLTALK_PHRASES:
            return ClassifiedInput(
                kind=OrderInputKind.SMALLTALK,
                confidence=0.9,
                normalized_text=norm,
            )

        # 6. Out of scope
        for phrase in _OUT_OF_SCOPE_PHRASES:
            if phrase in norm:
                return ClassifiedInput(
                    kind=OrderInputKind.OUT_OF_SCOPE,
                    confidence=0.85,
                    normalized_text=norm,
                    reason=f"match:{phrase}",
                )

        # 7. Check for food names to determine if order-related
        has_food = self._contains_food_name(norm)
        food_match = self._extract_food_names(norm)

        # 8. Quantity errors (zero/minus/extremely large + food)
        if self._has_quantity_error(norm, has_food):
            return ClassifiedInput(
                kind=OrderInputKind.QUANTITY_ERROR,
                confidence=0.85,
                normalized_text=norm,
                reason="quantity_error",
            )

        # 9. If we found food names, classify as VALID_ORDER (let parser handle details)
        if food_match:
            items = [OrderItem(name=n, qty=1) for n in sorted(food_match)]
            return ClassifiedInput(
                kind=OrderInputKind.VALID_ORDER,
                confidence=0.8,
                normalized_text=norm,
                matched_items=items,
            )

        # 10. Ambiguous reference ("that one", "the first one", "it") when order exists
        if current_order and self._has_ambiguous_reference(norm):
            return ClassifiedInput(
                kind=OrderInputKind.AMBIGUOUS_REFERENCE,
                confidence=0.7,
                normalized_text=norm,
                reason="ambiguous_reference",
            )

        # 11. Noise / empty-like (very short, no food, only fillers)
        if self._is_noise(norm):
            return ClassifiedInput(
                kind=OrderInputKind.EMPTY_OR_NOISE,
                confidence=0.8,
                normalized_text=norm,
                reason="noise_or_too_short",
            )

        # 12. Text looks like a food request but no known food match — delegate to LLM
        if self._looks_like_food_request(norm):
            out_item = self._extract_unknown_food(norm)
            return ClassifiedInput(
                kind=OrderInputKind.VALID_ORDER,
                confidence=0.5,
                normalized_text=norm,
                reason="no_known_food_match",
                out_of_menu_item=out_item,
            )

        # 13. Default — unknown, let parser try
        return ClassifiedInput(
            kind=OrderInputKind.UNKNOWN,
            confidence=0.3,
            normalized_text=norm,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _contains_food_name(self, text: str) -> bool:
        for name in self._all_food_names:
            if name in text.replace(" ", "_") or name in text.replace(" ", ""):
                return True
        return False

    def _extract_food_names(self, text: str) -> Set[str]:
        found: Set[str] = set()
        norm_text = text.replace(" ", "_")
        flat_text = text.replace(" ", "")
        for name in self._all_food_names:
            if name in norm_text or name.replace("_", "") in flat_text:
                found.add(name)
        return found

    def _has_quantity_error(self, text: str, has_food: bool) -> bool:
        if not has_food:
            return False
        for word in _QUANTITY_ERROR_WORDS:
            if f" {word} " in f" {text} " or text.startswith(word + " "):
                return True
        m = _LARGE_NUMBER_RE.search(text)
        if m:
            val = int(m.group(1))
            if val > 20:
                return True
        return False

    def _has_ambiguous_reference(self, text: str) -> bool:
        refs = {"that one", "that", "this one", "the first one", "the second one",
                "it", "the same", "same", "that thing"}
        return text in refs

    def _is_noise(self, text: str) -> bool:
        if text in _CONFIRM_WORDS:
            return False
        words = text.split()
        if not words:
            return True
        if len(words) <= 2 and all(w in _FILLERS for w in words):
            return True
        if len(text) <= 1:
            return True
        return False

    def _looks_like_food_request(self, text: str) -> bool:
        m = _FOOD_WORD_PATTERN.search(text)
        if m:
            # Has a food-like word but it wasn't in our menu
            return True
        order_prefixes = ["i want", "i'd like", "give me", "can i get", "i'll have"]
        for prefix in order_prefixes:
            if text.startswith(prefix):
                return True
        return False

    def _extract_unknown_food(self, text: str) -> Optional[str]:
        m = _FOOD_WORD_PATTERN.search(text)
        if m:
            return m.group(1).lower().replace(" ", "_")
        return None

    @staticmethod
    def has_modification_signal(text: str) -> bool:
        """Check if text contains modification signals (used by ConfirmationParser)."""
        lower = text.lower()
        for signal in _MODIFICATION_SIGNALS:
            if signal in lower:
                return True
        # "no <food>" pattern at end of sentence
        for name in ("coke", "fries", "water", "burger", "coffee", "tea",
                     "juice", "pizza", "sandwich", "salad", "soup", "noodles",
                     "dumplings", "pasta", "fried rice"):
            norm_name = name.replace(" ", "_")
            if f"no {name}" in lower or f"no {norm_name}" in lower:
                return True
        return False
