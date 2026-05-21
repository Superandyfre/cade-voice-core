"""
Text normalization for ASR output.

Nemotron outputs text with capitalization and punctuation
(e.g. "I would like a cheeseburger and a Coke."). Downstream
consumers (FSM, LLM) expect lowercase, punctuation-free text.
"""

import re
from typing import Dict, List, Optional, Tuple

# Default ASR correction replacements: (pattern, replacement)
_DEFAULT_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"\bcoca[\s-]?cola\b", "coke"),
    (r"\bcoke\b", "coke"),
    (r"\bcola\b", "coke"),
    (r"\bhamburgers?\b", "burger"),
    (r"\bcheeseburgers?\b", "burger"),
]


def normalize_asr_text(
    text: str,
    replacements: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Normalize ASR output for downstream consumption.

    Steps:
    1. Strip leading/trailing whitespace
    2. Lowercase
    3. Remove trailing punctuation (.!?,;:)
    4. Apply optional regex replacements
    """
    text = text.strip()
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"""[.!?,;:]+$""", "", text)

    for pattern, repl in (replacements if replacements is not None else _DEFAULT_REPLACEMENTS):
        text = re.sub(pattern, repl, text)

    text = text.strip()
    return text
