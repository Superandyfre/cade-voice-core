"""Menu context provider for order parsing.

Provides food item candidates from a menu definition, using exact match,
alias match, token normalization, and fuzzy matching.
"""

from difflib import SequenceMatcher
from typing import Dict, List, Optional

from pydantic import BaseModel


class MenuItem(BaseModel):
    canonical: str
    aliases: List[str] = []
    available: bool = True
    category: Optional[str] = None


class MenuContext(BaseModel):
    menu_version: str = "default"
    candidates: List[MenuItem] = []
    canonical_names: List[str] = []


class MenuContextProvider:
    """Provides menu item candidates for order parsing."""

    def __init__(self, food_aliases: Dict[str, List[str]]):
        self._items: List[MenuItem] = []
        for canonical, aliases in food_aliases.items():
            self._items.append(MenuItem(
                canonical=canonical.strip().lower().replace(" ", "_"),
                aliases=[a.strip().lower() for a in aliases],
            ))

    @property
    def all_canonical_names(self) -> List[str]:
        return sorted(item.canonical for item in self._items)

    def get_candidates(self, user_text: str, *, top_k: int = 12) -> MenuContext:
        """Return menu candidates relevant to user_text."""
        text_lower = user_text.strip().lower()
        if not text_lower:
            return MenuContext(canonical_names=self.all_canonical_names)

        tokens = text_lower.split()
        scored: Dict[str, float] = {}

        for item in self._items:
            best_score = 0.0
            for token in tokens:
                # Exact canonical match
                if token == item.canonical:
                    best_score = max(best_score, 1.0)
                # Exact alias match
                if token in item.aliases:
                    best_score = max(best_score, 0.95)
                # Token normalized match
                norm_token = token.replace(" ", "_")
                if norm_token == item.canonical:
                    best_score = max(best_score, 0.9)
                for alias in item.aliases:
                    if norm_token == alias:
                        best_score = max(best_score, 0.85)
                # Fuzzy match
                fuzzy = SequenceMatcher(None, token, item.canonical).ratio()
                best_score = max(best_score, fuzzy * 0.8)
                for alias in item.aliases:
                    fuzzy = SequenceMatcher(None, token, alias).ratio()
                    best_score = max(best_score, fuzzy * 0.75)

            # Also try full text matching for multi-word items
            for alias in item.aliases:
                if alias in text_lower:
                    best_score = max(best_score, 0.95)

            if best_score > 0.4:
                scored[item.canonical] = max(scored.get(item.canonical, 0), best_score)

        # Sort by score descending
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]
        candidates = []
        for canonical, score in ranked:
            for item in self._items:
                if item.canonical == canonical:
                    candidates.append(item)
                    break

        # If very few candidates, supplement with all items
        if len(candidates) < 3:
            existing = {c.canonical for c in candidates}
            for item in self._items:
                if item.canonical not in existing:
                    candidates.append(item)
                if len(candidates) >= top_k:
                    break

        return MenuContext(
            candidates=candidates,
            canonical_names=self.all_canonical_names,
        )
