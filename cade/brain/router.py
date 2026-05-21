"""Intent router — classifies user input into order, robot_action, smalltalk, etc."""

import re
from typing import Literal, Optional, List

from pydantic import BaseModel, Field


class IntentSubtask(BaseModel):
    type: Literal["order", "robot_action", "smalltalk", "clarification", "out_of_scope"]
    text: str


class IntentRouterDecision(BaseModel):
    intent: Literal[
        "order",
        "robot_action",
        "smalltalk",
        "clarification",
        "out_of_scope",
        "mixed",
    ]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason_code: str
    subtasks: List[IntentSubtask] = []
    reply: Optional[str] = None


# Keywords for rule-based routing
_ORDER_KEYWORDS = {
    "order", "coke", "burger", "pizza", "fries", "water", "coffee", "tea",
    "noodles", "salad", "soup", "sandwich", "rice", "pasta", "dumplings",
    "drink", "food", "menu", "hungry", "eat", "cola", "chips", "latte",
}
_ROBOT_ACTION_KEYWORDS = {
    "find", "pick", "place", "bring", "search", "go", "move", "come",
    "grab", "put", "carry", "deliver", "navigate", "look for",
}
_SMALLTALK_KEYWORDS = {
    "hello", "hi", "hey", "good morning", "good afternoon",
    "what is your name", "who are you", "how are you",
    "what can you do", "thank", "thanks", "bye", "goodbye",
}
_CLARIFICATION_KEYWORDS = {
    "what do you mean", "i don't understand", "sorry", "repeat",
    "pardon", "excuse me", "can you say that again",
}


class IntentRouter:
    """Rule-based intent router with optional LLM fallback."""

    def route(self, text: str, *, in_ordering_session: bool = False) -> IntentRouterDecision:
        text_lower = text.strip().lower()
        if not text_lower:
            return IntentRouterDecision(
                intent="out_of_scope", confidence=1.0,
                reason_code="empty_input",
            )

        tokens = set(re.findall(r"[a-z']+", text_lower))

        # If in an active ordering session, strongly bias toward order
        if in_ordering_session:
            order_overlap = tokens & _ORDER_KEYWORDS
            robot_overlap = tokens & _ROBOT_ACTION_KEYWORDS
            if order_overlap and not robot_overlap:
                return IntentRouterDecision(
                    intent="order", confidence=0.9,
                    reason_code="ordering_session_keyword",
                    subtasks=[IntentSubtask(type="order", text=text)],
                )

        # Check multi-word patterns in full text
        smalltalk_full_match = any(kw in text_lower for kw in _SMALLTALK_KEYWORDS)
        clarification_full_match = any(kw in text_lower for kw in _CLARIFICATION_KEYWORDS)

        # Score each intent type
        order_score = len(tokens & _ORDER_KEYWORDS)
        robot_score = len(tokens & _ROBOT_ACTION_KEYWORDS)
        smalltalk_score = len(tokens & _SMALLTALK_KEYWORDS) + (1 if smalltalk_full_match else 0)
        clarification_score = len(tokens & _CLARIFICATION_KEYWORDS) + (1 if clarification_full_match else 0)

        max_score = max(order_score, robot_score, smalltalk_score, clarification_score)

        if max_score == 0:
            # No keywords matched — check for ordering session context
            if in_ordering_session:
                return IntentRouterDecision(
                    intent="order", confidence=0.5,
                    reason_code="ordering_session_default",
                    subtasks=[IntentSubtask(type="order", text=text)],
                )
            return IntentRouterDecision(
                intent="out_of_scope", confidence=0.5,
                reason_code="no_keywords",
            )

        # Check for mixed intent
        scores = [
            ("order", order_score),
            ("robot_action", robot_score),
            ("smalltalk", smalltalk_score),
        ]
        non_zero = [(name, score) for name, score in scores if score > 0]

        if len(non_zero) >= 2:
            # Mixed intent
            subtasks = [IntentSubtask(type=name, text=text) for name, _ in non_zero]
            return IntentRouterDecision(
                intent="mixed", confidence=0.7,
                reason_code="multiple_intents",
                subtasks=subtasks,
                reply="I can handle one thing at a time. What would you like me to do first?",
            )

        # Single dominant intent
        if order_score == max_score:
            return IntentRouterDecision(
                intent="order", confidence=min(0.9, 0.6 + order_score * 0.1),
                reason_code="order_keywords",
                subtasks=[IntentSubtask(type="order", text=text)],
            )
        elif robot_score == max_score:
            return IntentRouterDecision(
                intent="robot_action", confidence=min(0.9, 0.6 + robot_score * 0.1),
                reason_code="robot_action_keywords",
                subtasks=[IntentSubtask(type="robot_action", text=text)],
            )
        elif smalltalk_score == max_score:
            return IntentRouterDecision(
                intent="smalltalk", confidence=0.95,
                reason_code="smalltalk_keywords",
                subtasks=[IntentSubtask(type="smalltalk", text=text)],
            )
        elif clarification_score == max_score:
            return IntentRouterDecision(
                intent="clarification", confidence=0.9,
                reason_code="clarification_keywords",
                subtasks=[IntentSubtask(type="clarification", text=text)],
            )

        return IntentRouterDecision(
            intent="out_of_scope", confidence=0.3,
            reason_code="fallback",
        )
