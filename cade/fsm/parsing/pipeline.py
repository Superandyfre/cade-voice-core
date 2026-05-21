"""Input pipeline that transforms raw ASR text into SemanticEvent objects.

Orchestrates: classify -> parse -> resolve -> output SemanticEvent.
The FSM consumes SemanticEvent instead of raw text.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from cade.brain.schemas import OrderAction
from cade.fsm.events import SemanticEvent
from cade.fsm.parsing.input_classifier import OrderInputClassifier
from cade.fsm.parsing.menu_context import MenuContext, MenuContextProvider
from cade.fsm.parsing.order_parser import (
    ClassifiedInput,
    ConfirmationParseResult,
    DeterministicOrderParser,
    OrderInputKind,
)

logger = logging.getLogger(__name__)


class InputPipeline:
    """Transforms raw ASR text into SemanticEvent objects.

    All FSM input processing should go through this pipeline so the FSM
    never directly handles raw text classification or order parsing.
    """

    def __init__(
        self,
        classifier: OrderInputClassifier,
        order_parser: DeterministicOrderParser,
        confirm_parser: "ConfirmationParser",
        menu_provider: MenuContextProvider,
        *,
        rule_parse_enabled: bool = True,
        rule_parse_threshold: float = 0.90,
        confirm_rule_threshold: float = 0.90,
        llm_candidate_top_k: int = 12,
    ):
        self._classifier = classifier
        self._order_parser = order_parser
        self._confirm_parser = confirm_parser
        self._menu_provider = menu_provider
        self._rule_parse_enabled = rule_parse_enabled
        self._rule_parse_threshold = rule_parse_threshold
        self._confirm_rule_threshold = confirm_rule_threshold
        self._llm_candidate_top_k = llm_candidate_top_k

    def process_listen(self, text: str, source: str = "", current_order: Optional[OrderAction] = None) -> SemanticEvent:
        """Full pipeline for LISTEN stage: classify -> parse -> output SemanticEvent."""
        classified = self._classifier.classify(text, state="LISTEN", current_order=current_order)
        base = SemanticEvent(
            event_type=classified.kind.value,
            confidence=classified.confidence,
            items=[item.model_dump() for item in (classified.matched_items or [])],
            raw_text=text,
            source=source,
            reason=classified.reason,
            parse_source="classifier",
            out_of_menu_item=classified.out_of_menu_item,
        )

        if classified.kind == OrderInputKind.VALID_ORDER and self._rule_parse_enabled:
            parsed = self._try_rule_parse_order(text)
            if parsed is not None:
                return parsed

        return base

    def process_check(self, text: str, source: str = "", current_order: Optional[OrderAction] = None) -> SemanticEvent:
        """Full pipeline for CHECK stage: classify -> confirm parse -> output SemanticEvent."""
        classified = self._classifier.classify(text, state="CHECK", current_order=current_order)
        base = SemanticEvent(
            event_type=classified.kind.value,
            confidence=classified.confidence,
            raw_text=text,
            source=source,
            reason=classified.reason,
            parse_source="classifier",
        )

        # Skip parsing for non-order events
        if classified.kind in (
            OrderInputKind.CANCEL_REQUEST,
            OrderInputKind.REPEAT_REQUEST,
            OrderInputKind.EMPTY_OR_NOISE,
            OrderInputKind.SMALLTALK,
            OrderInputKind.OUT_OF_SCOPE,
            OrderInputKind.PAUSE_REQUEST,
            OrderInputKind.MENU_QUESTION,
        ):
            return base

        # Try rule-based confirmation parsing
        if self._rule_parse_enabled:
            parsed = self._try_rule_parse_confirm(text)
            if parsed is not None:
                return parsed

        return base

    def _try_rule_parse_order(self, text: str) -> Optional[SemanticEvent]:
        """Attempt deterministic order parsing. Returns SemanticEvent if successful."""
        try:
            menu_ctx = self._menu_provider.get_candidates(text, top_k=self._llm_candidate_top_k)
            result = self._order_parser.parse_order(text, menu_ctx)
            if result.order and result.order.items and result.confidence >= self._rule_parse_threshold:
                return SemanticEvent(
                    event_type=OrderInputKind.VALID_ORDER.value,
                    confidence=result.confidence,
                    items=[item.model_dump() for item in result.order.items],
                    raw_text=text,
                    parse_source="rule",
                    reason=result.reason_code,
                    candidate_order=result.order.model_dump(),
                )
        except Exception as exc:
            logger.debug("[pipeline] rule order parse failed: %s", exc)
        return None

    def _try_rule_parse_confirm(self, text: str) -> Optional[SemanticEvent]:
        """Attempt deterministic confirmation parsing. Returns SemanticEvent if successful."""
        try:
            menu_ctx = self._menu_provider.get_candidates(text, top_k=self._llm_candidate_top_k)
            result = self._confirm_parser.parse(text, menu_ctx)
            if result.confidence >= self._confirm_rule_threshold:
                event = SemanticEvent(
                    event_type=_confirm_result_to_event_type(result.result),
                    confidence=result.confidence,
                    raw_text=text,
                    parse_source="rule",
                    confirm_result=result.result,
                    fix_order=result.fix_order.model_dump() if result.fix_order else None,
                    fix_reply=result.reply,
                )
                return event
        except Exception as exc:
            logger.debug("[pipeline] rule confirm parse failed: %s", exc)
        return None


def _confirm_result_to_event_type(result: str) -> str:
    mapping = {
        "correct": "confirm_correct",
        "wrong": "confirm_wrong",
        "cancel": "cancel_request",
        "repeat_request": "repeat_request",
        "unknown": "confirm_unknown",
    }
    return mapping.get(result, "confirm_unknown")
