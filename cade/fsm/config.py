"""Configuration for the ordering sub-FSM."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv(override=False)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MENU_FILE = _REPO_ROOT / "menu.yml"


class MenuItemConfig(BaseModel):
    id: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    available: bool = True
    max_qty: int = 10
    modifiers: Dict[str, List[str]] = Field(default_factory=dict)
    price: Optional[float] = None


class MenuModifierConfig(BaseModel):
    name: str
    values: List[str] = Field(default_factory=list)


def _default_food_aliases() -> Dict[str, List[str]]:
    return {
        "water": ["water", "bottle of water"],
        "coke": ["coke", "cola", "coca cola"],
        "juice": ["juice", "orange juice", "apple juice"],
        "coffee": ["coffee", "latte", "americano", "cappuccino"],
        "tea": ["tea", "black tea", "green tea", "milk tea"],
        "burger": ["burger", "hamburger", "cheeseburger"],
        "pizza": ["pizza"],
        "sandwich": ["sandwich"],
        "fried_rice": ["fried rice"],
        "noodles": ["noodles", "ramen"],
        "dumplings": ["dumpling", "dumplings"],
        "pasta": ["pasta", "spaghetti"],
        "fries": ["fries", "french fries", "chips"],
        "salad": ["salad"],
        "soup": ["soup"],
    }


def _normalize_menu_key(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "_")


def _aliases_from_menu_items(items: List[MenuItemConfig]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    for item in items:
        key = _normalize_menu_key(item.id or item.name)
        base = [item.name] + list(item.aliases)
        deduped: List[str] = []
        seen = set()
        for alias in base:
            alias_text = str(alias or "").strip()
            if not alias_text or alias_text in seen:
                continue
            seen.add(alias_text)
            deduped.append(alias_text)
        aliases[key] = deduped
    return aliases


def _menu_items_from_alias_map(alias_map: Dict[str, List[str]], default_max_qty: int) -> List[MenuItemConfig]:
    items: List[MenuItemConfig] = []
    for canonical, aliases in alias_map.items():
        key = _normalize_menu_key(canonical)
        display_name = key.replace("_", " ")
        items.append(
            MenuItemConfig(
                id=key,
                name=display_name,
                aliases=list(aliases or []),
                max_qty=default_max_qty,
            )
        )
    return items


def _load_menu_items_from_file(path: Path) -> List[dict]:
    try:
        import yaml
    except ImportError:
        return []
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(item)
    return normalized


def _load_menu_items() -> List[dict]:
    menu_file = os.getenv("CADE_ORDER_MENU_FILE", "").strip()
    if menu_file:
        loaded = _load_menu_items_from_file(Path(menu_file))
        if loaded:
            return loaded
    if _DEFAULT_MENU_FILE.is_file():
        loaded = _load_menu_items_from_file(_DEFAULT_MENU_FILE)
        if loaded:
            return loaded
    return []


def _load_food_aliases() -> Dict[str, List[str]]:
    raw = os.getenv("CADE_ORDER_FOOD_ALIASES", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data:
                return data
        except json.JSONDecodeError:
            pass

    alias_file = os.getenv("CADE_ORDER_FOOD_ALIASES_FILE", "").strip()
    if alias_file:
        path = Path(alias_file)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    return data
            except (json.JSONDecodeError, OSError):
                pass

    loaded_menu = _load_menu_items()
    if loaded_menu:
        return _aliases_from_menu_items([MenuItemConfig(**item) for item in loaded_menu])

    return _default_food_aliases()


class OrderFSMConfig(BaseModel):
    """Configuration for OrderSubFSM, loaded from environment variables."""

    model_config = {"extra": "forbid"}

    order_base_dir: str = Field(default_factory=lambda: os.getenv("CADE_ORDER_BASE_DIR", "data/orders"))
    menu_file: Optional[str] = Field(
        default_factory=lambda: os.getenv("CADE_ORDER_MENU_FILE", str(_DEFAULT_MENU_FILE) if _DEFAULT_MENU_FILE.is_file() else ""),
    )
    menu_items: List[MenuItemConfig] = Field(default_factory=_load_menu_items)
    food_aliases: Dict[str, List[str]] = Field(default_factory=_load_food_aliases)

    ask_prompt: str = Field(default_factory=lambda: os.getenv("CADE_ORDER_ASK_PROMPT", "What would you like to order?"))
    repeat_instruction: str = Field(
        default_factory=lambda: os.getenv(
            "CADE_ORDER_REPEAT_INSTRUCTION",
            "Repeat the order and ask the customer whether it is correct.",
        )
    )
    listen_retry_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "CADE_ORDER_LISTEN_RETRY_PROMPT",
            "Sorry, I did not catch your order. Please tell me your order again.",
        )
    )
    fix_missing_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "CADE_ORDER_FIX_MISSING_PROMPT",
            "Sorry, I didn't catch the changes. Please tell me your updated order.",
        )
    )
    check_retry_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "CADE_ORDER_CHECK_RETRY_PROMPT",
            "Please tell me if the order is correct, or say your updated order.",
        )
    )
    finish_template: str = Field(
        default_factory=lambda: os.getenv("CADE_ORDER_FINISH_TEMPLATE", "OK I'll get {foods} for you")
    )

    input_dedup_window_sec: float = Field(default_factory=lambda: float(os.getenv("CADE_ORDER_DEDUP_WINDOW_SEC", "1.5")))
    llm_max_retries: int = Field(default_factory=lambda: int(os.getenv("CADE_ORDER_LLM_MAX_RETRIES", "3")))
    order_id_proposal_timeout_sec: float = Field(
        default_factory=lambda: float(os.getenv("CADE_ORDER_ID_PROPOSAL_TIMEOUT_SEC", "5.0"))
    )

    zmq_pub_bind: str = Field(default_factory=lambda: os.getenv("CADE_ZMQ_PUB_BIND", "tcp://0.0.0.0:5555"))
    zmq_router_bind: str = Field(default_factory=lambda: os.getenv("CADE_ZMQ_ROUTER_BIND", "tcp://0.0.0.0:5556"))
    zmq_heartbeat_sec: float = Field(default_factory=lambda: float(os.getenv("CADE_ZMQ_HEARTBEAT_SEC", "2.0")))

    input_channel_mode: str = Field(
        default_factory=lambda: os.getenv("CADE_ORDER_INPUT_MODE", "both").strip().lower(),
    )
    rule_parse_enabled: bool = Field(
        default_factory=lambda: os.getenv("CADE_ORDER_RULE_PARSE_ENABLED", "true").lower() in ("1", "true", "yes"),
    )
    rule_parse_threshold: float = Field(default_factory=lambda: float(os.getenv("CADE_ORDER_RULE_PARSE_THRESHOLD", "0.90")))
    confirm_rule_threshold: float = Field(default_factory=lambda: float(os.getenv("CADE_ORDER_CONFIRM_RULE_THRESHOLD", "0.90")))
    llm_candidate_top_k: int = Field(default_factory=lambda: int(os.getenv("CADE_ORDER_LLM_CANDIDATE_TOP_K", "12")))

    listen_max_retries: int = Field(default_factory=lambda: int(os.getenv("CADE_ORDER_LISTEN_MAX_RETRIES", "5")))
    check_max_retries: int = Field(default_factory=lambda: int(os.getenv("CADE_ORDER_CHECK_MAX_RETRIES", "5")))
    empty_input_max: int = Field(default_factory=lambda: int(os.getenv("CADE_ORDER_EMPTY_INPUT_MAX", "3")))

    max_qty_per_item: int = Field(default_factory=lambda: int(os.getenv("CADE_ORDER_MAX_QTY_PER_ITEM", "9")))
    max_total_qty: int = Field(default_factory=lambda: int(os.getenv("CADE_ORDER_MAX_TOTAL_QTY", "20")))
    snapshot_file_name: str = Field(default_factory=lambda: os.getenv("CADE_ORDER_SESSION_SNAPSHOT_FILE", "session_snapshot.json"))
    repeat_use_llm: bool = Field(
        default_factory=lambda: os.getenv("CADE_ORDER_REPEAT_USE_LLM", "false").lower() in ("1", "true", "yes"),
    )
    idempotency_cache_file: str = Field(
        default_factory=lambda: os.getenv("CADE_ORDER_IDEMPOTENCY_CACHE_FILE", ""),
    )
    idempotency_ttl_sec: float = Field(
        default_factory=lambda: float(os.getenv("CADE_ORDER_IDEMPOTENCY_TTL_SEC", "300")),
    )
    outbox_retry_sec: float = Field(
        default_factory=lambda: float(os.getenv("CADE_ORDER_OUTBOX_RETRY_SEC", "30")),
    )
    outbox_max_attempts: int = Field(
        default_factory=lambda: int(os.getenv("CADE_ORDER_OUTBOX_MAX_ATTEMPTS", "10")),
    )

    @field_validator("input_channel_mode")
    @classmethod
    def _validate_input_channel_mode(cls, value: str) -> str:
        allowed = {"primary", "secondary", "both"}
        if value not in allowed:
            raise ValueError(f"CADE_ORDER_INPUT_MODE must be one of {sorted(allowed)}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _sync_menu_aliases(self) -> "OrderFSMConfig":
        if self.menu_items:
            self.menu_items = [
                item if isinstance(item, MenuItemConfig) else MenuItemConfig(**item)
                for item in self.menu_items
            ]
        if self.menu_items:
            derived = _aliases_from_menu_items(self.menu_items)
            merged = {key: list(value) for key, value in derived.items()}
            for key, aliases in self.food_aliases.items():
                norm_key = _normalize_menu_key(key)
                existing = merged.setdefault(norm_key, [])
                seen = set(existing)
                for alias in aliases:
                    alias_text = str(alias or "").strip()
                    if alias_text and alias_text not in seen:
                        existing.append(alias_text)
                        seen.add(alias_text)
            self.food_aliases = merged
        else:
            self.menu_items = _menu_items_from_alias_map(self.food_aliases, self.max_qty_per_item)
        return self
