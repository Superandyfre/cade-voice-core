"""Event and command models for the ordering sub-FSM."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Envelope(BaseModel):
    """Standard JSON envelope for all ZeroMQ messages."""

    v: int = Field(default=1, description="Protocol version")
    type: str = Field(..., description="Message type")
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = Field(default_factory=time.time)
    source: str = Field(default="voice-core")
    session_id: int = Field(default=0)
    client_id: Optional[str] = None
    client_msg_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    event_seq: Optional[int] = None
    last_event_seq: Optional[int] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class OrderStateEvent(BaseModel):
    """Published on every FSM state change."""

    timestamp: float = Field(default_factory=time.time)
    state: str
    reason: str
    serving_state: str
    order: Optional[Dict[str, Any]] = None
    order_id: Optional[str] = None
    order_dir: Optional[str] = None
    session_id: int = 0


class TtsRequestEvent(BaseModel):
    text: str
    session_id: int = 0


class TtsCompletedEvent(BaseModel):
    text: str
    playback_duration_s: Optional[float] = None
    audio_duration_s: Optional[float] = None
    session_id: int = 0


class TtsFailedEvent(BaseModel):
    text: str
    error: str
    session_id: int = 0


class OrderConfirmedEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    source: str = "voice_order_subfsm"
    order: Dict[str, Any]
    foods: List[str]
    foods_with_qty: List[Dict[str, Any]]
    recognized_text: str = ""
    check_text: str = ""
    order_id: Optional[str] = None
    order_dir: Optional[str] = None
    session_id: int = 0
    customer_id: Optional[str] = None
    customer_no: Optional[str] = None
    folder: Optional[str] = None
    customer_folder: Optional[str] = None


class OrderMetricsEvent(BaseModel):
    total_inputs: int = 0
    ignored_inputs: int = 0
    successful_replies: int = 0
    orders_confirmed: int = 0
    order_session_started_total: int = 0
    order_confirmed_total: int = 0
    order_cancelled_total: int = 0
    order_failed_total: int = 0
    order_recovered_total: int = 0
    llm_fallback_count: int = 0
    duplicate_input_count: int = 0
    invalid_state_input_count: int = 0
    tts_hard_fail_count: int = 0
    tts_soft_fail_count: int = 0
    mean_time_to_confirm_ms: float = 0.0
    outbox_pending_total: int = 0
    outbox_published_total: int = 0
    outbox_delivered_total: int = 0
    outbox_dead_letter_total: int = 0
    rule_parse_hit_total: int = 0
    confirm_rule_hit_total: int = 0
    asr_echo_block_total: int = 0
    llm_rule_agree_total: int = 0
    llm_rule_disagree_total: int = 0
    p50_time_to_confirm_ms: float = 0.0
    p95_time_to_confirm_ms: float = 0.0
    session_id: int = 0


class OrderErrorEvent(BaseModel):
    error: str
    stage: str = ""
    session_id: int = 0


class OrderCancelledEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    reason: str = "user_cancel"
    order_id: Optional[str] = None
    session_id: int = 0


class OrderWarningEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    warning_type: str = ""
    message: str = ""
    session_id: int = 0
    stage: str = ""


class InvalidTransitionEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    from_state: str
    event: str
    reason: str
    session_id: int = 0


class ServingStateCommand(BaseModel):
    state: str = "IDLE"
    customer_id: Optional[str] = None
    customer_no: Optional[str] = None
    folder: Optional[str] = None
    customer_folder: Optional[str] = None


class UserTextCommand(BaseModel):
    text: str


class OrderIdCommand(BaseModel):
    order_id: str


class SnapshotCommand(BaseModel):
    pass


class HealthCommand(BaseModel):
    pass


class SessionCancelCommand(BaseModel):
    reason: str = "external_cancel"


class EventsSinceCommand(BaseModel):
    last_event_seq: int = 0


class SemanticEvent(BaseModel):
    event_type: str
    confidence: float = 0.0
    items: Optional[List[Dict[str, Any]]] = None
    raw_text: str = ""
    source: str = ""
    reason: Optional[str] = None
    parse_source: str = "classifier"
    is_candidate: bool = False
    candidate_order: Optional[Dict[str, Any]] = None
    confirm_result: Optional[str] = None
    fix_order: Optional[Dict[str, Any]] = None
    fix_reply: Optional[str] = None
    out_of_menu_item: Optional[str] = None


class OutboxStatus(str, Enum):
    pending = "pending"
    published = "published"
    delivered = "delivered"
    dead_letter = "dead_letter"


class OrderConfirmedAckCommand(BaseModel):
    order_id: str
    status: Literal["delivered", "dead_letter"] = "delivered"


class OutboxRetryCommand(BaseModel):
    order_id: Optional[str] = None
