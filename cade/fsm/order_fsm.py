"""ROS-free ordering sub-FSM."""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from cade.brain.schemas import FixOrderAction, OrderAction, OrderCheckDecision, OrderItem
from cade.fsm.config import MenuItemConfig, OrderFSMConfig
from cade.fsm.events import (
    InvalidTransitionEvent,
    OrderCancelledEvent,
    OrderConfirmedEvent,
    OrderErrorEvent,
    OrderMetricsEvent,
    OrderStateEvent,
    OrderWarningEvent,
    SemanticEvent,
    TtsCompletedEvent,
    TtsFailedEvent,
    TtsRequestEvent,
)
from cade.fsm.input_classifier import OrderInputClassifier
from cade.fsm.menu_context import MenuContextProvider
from cade.fsm.order_parser import ConfirmationParser, DeterministicOrderParser, OrderInputKind
from cade.fsm.states import OrderState
from cade.fsm.storage.journal import TransitionJournal
from cade.fsm.parsing.pipeline import InputPipeline

logger = logging.getLogger(__name__)


class TTSPlaybackError(RuntimeError):
    """Raised when TTS playback did not complete successfully."""


@dataclass
class InputResult:
    accepted: bool
    reason: Optional[str] = None
    state: Optional[str] = None
    session_id: int = 0


@dataclass(frozen=True)
class TransitionRule:
    event: str
    from_states: frozenset[OrderState]
    to_state: OrderState
    before: tuple = ()
    after: tuple = ()


@dataclass
class RawInput:
    text: str
    source: str
    state: OrderState
    session_id: int


@dataclass
class NormalizedInput:
    raw: RawInput
    normalized_text: str


@dataclass
class SemanticInput:
    normalized: NormalizedInput
    kind: OrderInputKind


class LLMClientProtocol(Protocol):
    def get_order_action(
        self,
        user_input: str,
        food_aliases: Optional[Dict[str, List[str]]] = None,
        max_retries: int = 3,
    ) -> OrderAction: ...

    def get_order_repeat_speak(
        self,
        confirm_instruction: str,
        order_action: OrderAction,
        max_retries: int = 3,
    ) -> Any: ...

    def get_order_check_decision(
        self,
        customer_reply: str,
        order_action: OrderAction,
        food_aliases: Optional[Dict[str, List[str]]] = None,
        max_retries: int = 3,
    ) -> Any: ...


class OrderIdProvider(Protocol):
    def propose(self) -> Optional[str]: ...


class OrderStorage(Protocol):
    def create_order_dir(self, order_id: str) -> str: ...
    def save_order_group(self, order_dir: str, payload: dict) -> None: ...
    def load_known_ids(self) -> set: ...


class TTSSink(Protocol):
    def speak(self, text: str, profile: str = "dialogue") -> Any: ...


class EventSink(Protocol):
    def publish(self, topic: str, payload: dict) -> None: ...


class LocalOrderIdProvider:
    def __init__(self, known_ids: Optional[set] = None):
        self._known = known_ids or set()
        self._lock = threading.Lock()

    def propose(self) -> Optional[str]:
        import random

        with self._lock:
            for _ in range(1000):
                candidate = f"{random.randint(0, 99999):05d}"
                if candidate not in self._known:
                    self._known.add(candidate)
                    return candidate
        return None


class LocalOrderStorage:
    """Filesystem-based order storage with atomic writes, event logging, and snapshots."""

    def __init__(self, base_dir: str, snapshot_file_name: str = "session_snapshot.json"):
        self._base_dir = base_dir
        self._snapshot_file_name = snapshot_file_name
        os.makedirs(base_dir, exist_ok=True)

    def create_order_dir(self, order_id: str) -> str:
        order_dir = os.path.join(self._base_dir, order_id)
        os.makedirs(order_dir, exist_ok=True)
        return order_dir

    def save_order_group(self, order_dir: str, payload: dict) -> None:
        self._atomic_json_write(os.path.join(order_dir, "order_group.json"), payload)

    def save_session_snapshot(self, order_dir: str, payload: dict) -> None:
        self._atomic_json_write(os.path.join(order_dir, self._snapshot_file_name), payload)

    def load_session_snapshot(self, order_dir: str) -> Optional[dict]:
        target = Path(order_dir) / self._snapshot_file_name
        if not target.is_file():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list_session_snapshots(self) -> List[dict]:
        snapshots: List[dict] = []
        try:
            for child in sorted(Path(self._base_dir).iterdir()):
                if not child.is_dir():
                    continue
                snapshot = self.load_session_snapshot(str(child))
                if snapshot:
                    snapshots.append({"order_dir": str(child), "order_id": child.name, "snapshot": snapshot})
        except OSError:
            return []
        return snapshots

    def append_event(self, order_dir: str, event: dict) -> None:
        log_path = os.path.join(order_dir, "events.jsonl")
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def load_known_ids(self) -> set:
        known = set()
        try:
            for name in os.listdir(self._base_dir):
                if len(name) == 5 and name.isdigit():
                    known.add(name)
        except OSError:
            pass
        return known

    def append_outbox(self, order_dir: str, entry: dict) -> None:
        log_path = os.path.join(order_dir, "outbox.jsonl")
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def load_outbox(self, order_dir: str) -> List[dict]:
        log_path = Path(order_dir) / "outbox.jsonl"
        if not log_path.is_file():
            return []
        entries = []
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        return entries

    def outbox_pending_count(self) -> int:
        count = 0
        try:
            for child in sorted(Path(self._base_dir).iterdir()):
                if not child.is_dir():
                    continue
                for entry in self.load_outbox(str(child)):
                    if entry.get("status") == "pending":
                        count += 1
        except OSError:
            pass
        return count

    @staticmethod
    def _atomic_json_write(target: str, payload: dict) -> None:
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)


class CallbackTTSSink:
    def __init__(self, callback: Callable[[str], Any]):
        self._callback = callback
        self._accepts_profile = self._detect_profile_kwarg(callback)

    def speak(self, text: str, profile: str = "dialogue") -> Any:
        if self._accepts_profile:
            return self._callback(text, profile=profile)
        return self._callback(text)

    @staticmethod
    def _detect_profile_kwarg(callback: Callable[..., Any]) -> bool:
        try:
            sig = inspect.signature(callback)
        except (TypeError, ValueError):
            return False
        if "profile" in sig.parameters:
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


class CallbackEventSink:
    def __init__(self, callback: Callable[[str, dict], None]):
        self._callback = callback

    def publish(self, topic: str, payload: dict) -> None:
        self._callback(topic, payload)


class OrderSubFSM:
    """Pure business FSM for ordering."""

    VALID_INPUT_STATES = {OrderState.LISTEN, OrderState.CHECK}

    def _hook_reset_fields(self, event: str, reason: str, old_state: OrderState, next_state: OrderState, session_id: Optional[int]) -> None:
        self._session_id += 1
        self._current_order = None
        self._last_listen_text = ""
        self._last_check_text = ""
        self._last_prompt_text = ""
        self._current_order_id = ""
        self._current_order_dir = ""
        self._processing_input = False
        self._listen_retry_count = 0
        self._listen_empty_count = 0
        self._check_retry_count = 0
        self._session_started_mono = 0.0

    def _hook_publish_metrics(self, event: str, reason: str, old_state: OrderState, next_state: OrderState, session_id: Optional[int]) -> None:
        self._publish_metrics()

    def _hook_commit_order(self, event: str, reason: str, old_state: OrderState, next_state: OrderState, session_id: Optional[int]) -> None:
        order = self._current_order
        if not order or not order.items:
            return
        with self._fsm_lock:
            order_dir = str(self._current_order_dir or "").strip()
        if not order_dir:
            return

        journal_path = os.path.join(order_dir, "transition_journal.jsonl")
        journal = TransitionJournal(journal_path) if os.path.isdir(order_dir) else None
        jid = journal.begin_transition("check.correct", session_id or 0) if journal else None
        oid = self._current_order_id or ""
        ikey = f"order-confirmed-{oid}" if oid else None

        if not journal or not journal.is_step_completed(jid, "save_order_group"):
            if journal: journal.write_step(jid, "save_order_group", "started", order_id=oid, idempotency_key=ikey)
            self._save_order_group(order, stage="confirmed")
            if journal: journal.write_step(jid, "save_order_group", "committed", order_id=oid, idempotency_key=ikey)

        if not journal or not journal.is_step_completed(jid, "append_event"):
            if journal: journal.write_step(jid, "append_event", "started", order_id=oid, idempotency_key=ikey)
            self._append_event("order_confirmed", {"check_text": self._last_check_text})
            if journal: journal.write_step(jid, "append_event", "committed", order_id=oid, idempotency_key=ikey)

        if not journal or not journal.is_step_completed(jid, "outbox_pending"):
            if journal: journal.write_step(jid, "outbox_pending", "started", order_id=oid, idempotency_key=ikey)
            self._append_outbox("pending", order)
            self.outbox_pending_total += 1
            if journal: journal.write_step(jid, "outbox_pending", "committed", order_id=oid, idempotency_key=ikey)

        if not journal or not journal.is_step_completed(jid, "publish_confirm"):
            if journal: journal.write_step(jid, "publish_confirm", "started", order_id=oid, idempotency_key=ikey)
            self._publish_order_confirm(order)
            if journal: journal.write_step(jid, "publish_confirm", "committed", order_id=oid, idempotency_key=ikey)

        if not journal or not journal.is_step_completed(jid, "outbox_published"):
            if journal: journal.write_step(jid, "outbox_published", "started", order_id=oid, idempotency_key=ikey)
            self._append_outbox("published", order)
            self.outbox_published_total += 1
            if journal: journal.write_step(jid, "outbox_published", "committed", order_id=oid, idempotency_key=ikey)

        if not journal or not journal.is_step_completed(jid, "tts_finish"):
            if journal: journal.write_step(jid, "tts_finish", "started", order_id=oid, idempotency_key=ikey)
            finish_text = self._build_finish_text(order)
            self._publish_tts_soft(finish_text, profile="order_confirm")
            if journal: journal.write_step(jid, "tts_finish", "committed", order_id=oid, idempotency_key=ikey)

        self.orders_confirmed += 1
        self.order_confirmed_total += 1
        self.successful_replies += 1
        if self._session_started_mono > 0:
            elapsed = max(0.0, time.monotonic() - self._session_started_mono)
            self._confirm_latency_total_s += elapsed
            self._confirm_latency_count += 1
            self._confirm_latency_samples.append(elapsed * 1000.0)
            if len(self._confirm_latency_samples) > 1000:
                self._confirm_latency_samples = self._confirm_latency_samples[-1000:]

    TRANSITIONS: Dict[str, TransitionRule] = {
        "session.permitted": TransitionRule("session.permitted", frozenset({OrderState.NOT_PERMITTED}), OrderState.PERMITTED),
        "ask.begin": TransitionRule("ask.begin", frozenset({OrderState.PERMITTED}), OrderState.ASK),
        "ask.completed": TransitionRule("ask.completed", frozenset({OrderState.ASK}), OrderState.LISTEN),
        "repeat.retry": TransitionRule("repeat.retry", frozenset({OrderState.CHECK}), OrderState.REPEAT),
        "order.extracted": TransitionRule("order.extracted", frozenset({OrderState.LISTEN}), OrderState.REPEAT),
        "repeat.completed": TransitionRule("repeat.completed", frozenset({OrderState.REPEAT}), OrderState.CHECK),
        "order.fixed": TransitionRule("order.fixed", frozenset({OrderState.CHECK}), OrderState.REPEAT),
        "wrong.without_fix": TransitionRule("wrong.without_fix", frozenset({OrderState.CHECK}), OrderState.LISTEN),
        "check.correct": TransitionRule("check.correct", frozenset({OrderState.CHECK}), OrderState.FINISH, after=("_hook_commit_order",)),
        "session.reset": TransitionRule(
            "session.reset",
            frozenset({
                OrderState.PERMITTED,
                OrderState.ASK,
                OrderState.LISTEN,
                OrderState.REPEAT,
                OrderState.CHECK,
                OrderState.FINISH,
                OrderState.NOT_PERMITTED,
            }),
            OrderState.NOT_PERMITTED,
            after=("_hook_reset_fields", "_hook_publish_metrics"),
        ),
    }

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: Optional[OrderFSMConfig] = None,
        order_id_provider: Optional[OrderIdProvider] = None,
        order_storage: Optional[OrderStorage] = None,
        tts_sink: Optional[TTSSink] = None,
        event_sink: Optional[EventSink] = None,
    ):
        self.config = config or OrderFSMConfig()
        self._llm = llm_client
        self._storage = order_storage or LocalOrderStorage(
            self.config.order_base_dir,
            snapshot_file_name=self.config.snapshot_file_name,
        )
        self._known_order_ids = self._storage.load_known_ids()
        self._external_order_id: Optional[str] = None
        self._order_id_provider = order_id_provider or LocalOrderIdProvider(self._known_order_ids)
        self._tts = tts_sink or CallbackTTSSink(lambda text: logger.info("[TTS] %s", text))
        self._events = event_sink or CallbackEventSink(lambda topic, payload: logger.debug("[EVENT] %s: %s", topic, payload))
        self._food_alias_lookup = self._build_food_alias_lookup(self.config.food_aliases)
        self._menu_items = self._build_menu_items(self.config.menu_items)
        self._menu_provider = MenuContextProvider(self.config.food_aliases)
        self._order_parser = DeterministicOrderParser(self._menu_provider)
        self._confirm_parser = ConfirmationParser()
        self._input_classifier = OrderInputClassifier(self.config.food_aliases)
        self._pipeline = InputPipeline(
            classifier=self._input_classifier,
            order_parser=self._order_parser,
            confirm_parser=self._confirm_parser,
            menu_provider=self._menu_provider,
            rule_parse_enabled=self.config.rule_parse_enabled,
            rule_parse_threshold=self.config.rule_parse_threshold,
            confirm_rule_threshold=self.config.confirm_rule_threshold,
            llm_candidate_top_k=self.config.llm_candidate_top_k,
        )
        self._fsm_lock = threading.RLock()
        self._session_id = 0
        self._state = OrderState.NOT_PERMITTED
        self._serving_state = "IDLE"
        self._serving_payload: Dict[str, Any] = {}
        self._current_order: Optional[OrderAction] = None
        self._last_listen_text = ""
        self._last_check_text = ""
        self._last_prompt_text = ""
        self._current_order_id = ""
        self._current_order_dir = ""
        self._processing_input = False
        self._last_input_norm = ""
        self._last_input_mono = 0.0
        self._listen_retry_count = 0
        self._listen_empty_count = 0
        self._check_retry_count = 0
        self._session_started_mono = 0.0
        self._confirm_latency_total_s = 0.0
        self._confirm_latency_count = 0
        self.total_inputs = 0
        self.ignored_inputs = 0
        self.successful_replies = 0
        self.orders_confirmed = 0
        self.order_session_started_total = 0
        self.order_confirmed_total = 0
        self.order_cancelled_total = 0
        self.order_failed_total = 0
        self.llm_fallback_count = 0
        self.duplicate_input_count = 0
        self.tts_hard_fail_count = 0
        self.order_recovered_total = 0
        self.invalid_state_input_count = 0
        self.tts_soft_fail_count = 0
        self.outbox_pending_total = 0
        self.outbox_published_total = 0
        self.outbox_delivered_total = 0
        self.outbox_dead_letter_total = 0
        self.rule_parse_hit_total = 0
        self.confirm_rule_hit_total = 0
        self.asr_echo_block_total = 0
        self.llm_rule_agree_total = 0
        self.llm_rule_disagree_total = 0
        self._confirm_latency_samples: List[float] = []
        self._latest_state_event: Optional[OrderStateEvent] = None
        self._latest_order_snapshot: Optional[Dict[str, Any]] = None
        self._latest_session_snapshot: Optional[Dict[str, Any]] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        self._recover_existing_snapshots()
        logger.info("OrderSubFSM initialized (order_base_dir=%s)", self.config.order_base_dir)

    def handle_serving_state(self, payload: Dict[str, Any]) -> None:
        state_text = str(payload.get("state", "IDLE")).strip().upper() or "IDLE"
        with self._fsm_lock:
            prev_state = self._serving_state
            self._serving_state = state_text
            self._serving_payload = dict(payload)
        if state_text == "PAUSED_ORDERING":
            if prev_state != "PAUSED_ORDERING":
                logger.info("[order_subfsm] serving state entered PAUSED_ORDERING")
            self._start_ordering_session()
            return
        if prev_state == "PAUSED_ORDERING":
            logger.info("[order_subfsm] serving state left PAUSED_ORDERING -> %s", state_text)
        self._reset_to_not_permitted(f"serving_state_changed:{state_text}")

    def handle_user_text(self, text: str, source: str = "primary") -> InputResult:
        raw = RawInput(text=str(text or "").strip(), source=str(source or "primary"), state=self._state, session_id=self._session_id)
        if not raw.text:
            return InputResult(accepted=False, reason="empty_text")
        self.total_inputs += 1

        mode = self.config.input_channel_mode
        if mode != "both":
            normalized_source = raw.source.strip().lower()
            if mode == "primary" and normalized_source not in ("primary", "asr_microphone"):
                self.ignored_inputs += 1
                return InputResult(accepted=False, reason="source_filtered")
            if mode == "secondary" and normalized_source != "secondary":
                self.ignored_inputs += 1
                return InputResult(accepted=False, reason="source_filtered")

        if self._is_duplicate_input(raw.text):
            self.ignored_inputs += 1
            self.duplicate_input_count += 1
            return InputResult(accepted=False, reason="duplicate_input")

        with self._fsm_lock:
            current_state = self._state
            session_id = self._session_id
            if self._serving_state != "PAUSED_ORDERING" or current_state not in self.VALID_INPUT_STATES:
                self.ignored_inputs += 1
                self.invalid_state_input_count += 1
                return InputResult(accepted=False, reason="invalid_state", state=current_state.value, session_id=session_id)
            if self._processing_input:
                self.ignored_inputs += 1
                return InputResult(accepted=False, reason="processing_busy", state=current_state.value, session_id=session_id)
            self._processing_input = True

        thread = threading.Thread(
            target=self._process_order_input_async,
            args=(RawInput(text=raw.text, source=raw.source, state=current_state, session_id=session_id),),
            daemon=True,
        )
        thread.start()
        return InputResult(accepted=True, state=current_state.value, session_id=session_id)

    def handle_order_id(self, candidate: str) -> None:
        raw = str(candidate or "").strip()
        if re.fullmatch(r"\d{5}", raw):
            self._external_order_id = raw
            logger.info("[order_subfsm] received order_id proposal: %s", raw)

    def snapshot(self) -> Dict[str, Any]:
        with self._fsm_lock:
            state_event = self._build_state_event("snapshot")
            metrics = self._build_metrics_event().model_dump()
            outbox_pending = 0
            if hasattr(self._storage, "outbox_pending_count") and callable(getattr(self._storage, "outbox_pending_count")):
                try:
                    result = self._storage.outbox_pending_count()
                    outbox_pending = int(result) if isinstance(result, int) else 0
                except Exception:
                    pass
            return {
                "state_event": state_event.model_dump(),
                "order_snapshot": self._latest_order_snapshot,
                "session_snapshot": self._latest_session_snapshot,
                "metrics": metrics,
                "outbox_pending_count": outbox_pending,
            }

    def cancel(self, reason: str = "external_cancel") -> None:
        self._reset_to_not_permitted(reason)

    def start_heartbeat(self, interval_sec: Optional[float] = None) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        interval = interval_sec or self.config.zmq_heartbeat_sec

        def _heartbeat_loop():
            while not self._heartbeat_stop.wait(interval):
                self._events.publish("order.heartbeat", {"ts": time.time()})

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

    def is_accepting_live_input(self) -> bool:
        with self._fsm_lock:
            return (
                self._serving_state == "PAUSED_ORDERING"
                and self._state in self.VALID_INPUT_STATES
                and not self._processing_input
            )

    def transition(self, event: str, *, session_id: Optional[int] = None, reason: Optional[str] = None, publish: bool = True) -> bool:
        rule = self.TRANSITIONS.get(event)
        if rule is None:
            self._publish_invalid_transition(event, f"unknown_event:{event}")
            return False
        with self._fsm_lock:
            if session_id is not None and self._session_id != session_id:
                return False
            if self._state not in rule.from_states:
                self._publish_invalid_transition(event, f"state={self._state.value}")
                return False
            old_state = self._state
            next_state = rule.to_state
            transition_reason = reason or event
            for hook_name in rule.before:
                getattr(self, hook_name)(event, transition_reason, old_state, next_state, session_id)
            self._persist_session_snapshot_locked(event, "pending", transition_reason, state_override=next_state)
            self._state = next_state
            logger.info("[order_subfsm] %s -> %s (%s)", old_state.value, next_state.value, transition_reason)
            self._persist_session_snapshot_locked(event, "committed", transition_reason, state_override=next_state)
            for hook_name in rule.after:
                getattr(self, hook_name)(event, transition_reason, old_state, next_state, session_id)
        if publish:
            self._publish_state_event(transition_reason)
        return True

    def _start_ordering_session(self) -> None:
        with self._fsm_lock:
            if self._state != OrderState.NOT_PERMITTED:
                return
            self._session_id += 1
            session_id = self._session_id
            self._current_order = None
            self._last_listen_text = ""
            self._last_check_text = ""
            self._last_prompt_text = ""
            self._current_order_id = ""
            self._current_order_dir = ""
            self._processing_input = False
            self._listen_retry_count = 0
            self._listen_empty_count = 0
            self._check_retry_count = 0
            self._session_started_mono = time.monotonic()
            self.order_session_started_total += 1

        order_id = self._acquire_order_id(session_id)
        if order_id is None:
            self.order_failed_total += 1
            return
        order_dir = self._storage.create_order_dir(order_id)
        with self._fsm_lock:
            if not self._is_session_pending(session_id):
                return
            self._current_order_id = order_id
            self._current_order_dir = order_dir
        self._append_event("session_started", {"order_id": order_id, "session_id": session_id})
        self.transition("session.permitted", session_id=session_id, reason=f"serving_state=PAUSED_ORDERING;order_id={order_id}")
        thread = threading.Thread(target=self._run_ask_stage, args=(session_id,), daemon=True)
        thread.start()

    def _acquire_order_id(self, session_id: int) -> Optional[str]:
        timeout = self.config.order_id_proposal_timeout_sec
        deadline = time.monotonic() + timeout
        while self._is_session_pending(session_id):
            candidate = self._external_order_id
            if candidate and re.fullmatch(r"\d{5}", candidate):
                if candidate not in self._known_order_ids:
                    self._known_order_ids.add(candidate)
                    self._external_order_id = None
                    logger.info("[order_subfsm] acquired external order_id=%s", candidate)
                    return candidate
            candidate = self._order_id_provider.propose()
            if candidate:
                self._known_order_ids.add(candidate)
                logger.info("[order_subfsm] acquired local order_id=%s", candidate)
                return candidate
            if time.monotonic() > deadline:
                logger.warning("[order_subfsm] order_id acquisition timed out")
                return None
            time.sleep(0.1)
        return None

    def _run_ask_stage(self, session_id: int) -> None:
        if not self._can_session_continue(session_id):
            return
        if not self.transition("ask.begin", session_id=session_id, reason="ordering_started"):
            return
        self.transition("ask.completed", session_id=session_id, reason="ask_completed")
        if not self._can_session_continue(session_id):
            return
        try:
            self._publish_tts(self.config.ask_prompt, profile="order_prompt")
        except TTSPlaybackError:
            self.order_failed_total += 1

    def _process_order_input_async(self, raw: RawInput) -> None:
        try:
            if not self._can_session_continue(raw.session_id):
                return
            logger.info("[order_subfsm] %s input(%s): %s", raw.state.value, raw.source, raw.text)
            normalized = NormalizedInput(raw=raw, normalized_text=" ".join(raw.text.lower().split()))
            if raw.state == OrderState.LISTEN:
                self._process_listen_stage(normalized)
            elif raw.state == OrderState.CHECK:
                self._process_check_stage(normalized)
        except TTSPlaybackError as exc:
            logger.warning("Ordering sub-FSM TTS failed: %s", exc)
        except Exception as exc:
            logger.warning("Ordering sub-FSM processing failed: %s", exc)
            self.order_failed_total += 1
            self._events.publish(
                "order.error",
                OrderErrorEvent(error=str(exc), stage=raw.state.value, session_id=raw.session_id).model_dump(),
            )
            if self._can_session_continue(raw.session_id):
                self._publish_tts_soft(self.config.check_retry_prompt, profile="error")
        finally:
            with self._fsm_lock:
                if self._session_id == raw.session_id:
                    self._processing_input = False

    def _classify_listen_input(self, normalized: NormalizedInput, current_order: Optional[OrderAction]) -> SemanticEvent:
        return self._pipeline.process_listen(
            normalized.normalized_text,
            source=normalized.raw.source,
            current_order=current_order,
        )

    def _classify_check_input(self, normalized: NormalizedInput, current_order: Optional[OrderAction]) -> SemanticEvent:
        return self._pipeline.process_check(
            normalized.normalized_text,
            source=normalized.raw.source,
            current_order=current_order,
        )

    def _process_listen_stage(self, normalized: NormalizedInput) -> None:
        if not self._can_session_continue(normalized.raw.session_id):
            return
        with self._fsm_lock:
            current_order = self._current_order
        semantic = self._classify_listen_input(normalized, current_order)
        logger.info(
            "[order_subfsm] LISTEN semantic event: %s (confidence=%.2f): %s",
            semantic.event_type,
            semantic.confidence,
            normalized.raw.text,
        )
        kind_str = semantic.event_type
        if kind_str == "cancel_request":
            self._publish_order_cancelled("user_cancel")
            self._reset_to_not_permitted("user_cancelled")
            return
        if kind_str == "repeat_request":
            self._publish_tts_soft(self.config.ask_prompt, profile="order_prompt")
            return
        if kind_str == "pause_request":
            self._publish_tts_soft("Take your time.", profile="order_prompt")
            return
        if kind_str == "menu_question":
            self._publish_tts_soft(self._build_menu_summary(), profile="order_prompt")
            return
        if kind_str == "out_of_menu_item":
            item_name = semantic.reason or "that item"
            self._publish_tts_soft(
                f"Sorry, we don't have {item_name}. {self._build_available_items_text()}",
                profile="order_prompt",
            )
            return
        if kind_str == "ambiguous_reference":
            self._publish_tts_soft("Could you please specify which item you'd like?", profile="order_prompt")
            self._listen_retry_count += 1
            return
        if kind_str == "quantity_error":
            self._publish_tts_soft("Could you please check the quantity?", profile="order_prompt")
            self._listen_retry_count += 1
            return
        if kind_str == "smalltalk":
            self._publish_tts_soft("Hello! I'm ready to take your order.", profile="order_prompt")
            return
        if kind_str == "out_of_scope":
            self._publish_tts_soft("I'm taking orders right now. What would you like?", profile="order_prompt")
            return
        if kind_str == "empty_or_noise":
            self._listen_empty_count += 1
            if self._listen_empty_count > self.config.empty_input_max:
                self._publish_warning("empty_input_exceeded", "Too many empty inputs")
                self._publish_order_cancelled("empty_input_exceeded")
                self._reset_to_not_permitted("empty_input_exceeded")
                return
            self._publish_tts_soft(self.config.listen_retry_prompt, profile="error")
            return

        self._listen_retry_count += 1
        if self._listen_retry_count > self.config.listen_max_retries:
            self._publish_warning("retry_limit", "Listen retry limit exceeded")
            self._publish_order_cancelled("listen_retry_exceeded")
            self._reset_to_not_permitted("listen_retry_exceeded")
            return

        order_action: Optional[OrderAction] = None
        if semantic.parse_source == "rule" and semantic.candidate_order:
            self.rule_parse_hit_total += 1
            order_action = OrderAction(**semantic.candidate_order)
            logger.info("[order_subfsm] pipeline rule parse accepted (confidence=%.2f)", semantic.confidence)

        if order_action is None:
            self.llm_fallback_count += 1
            try:
                order_action = self._llm.get_order_action(
                    user_input=normalized.raw.text,
                    food_aliases=self.config.food_aliases,
                    max_retries=self.config.llm_max_retries,
                )
            except Exception as exc:
                logger.warning("Order LISTEN LLM failed: %s", exc)
                self._publish_tts_soft(self.config.listen_retry_prompt, profile="error")
                return

            # Cross-validate LLM result against rule parser
            if order_action and order_action.items:
                rule_event = self._pipeline._try_rule_parse_order(normalized.raw.text)
                if rule_event and rule_event.candidate_order:
                    rule_order = OrderAction(**rule_event.candidate_order)
                    if self._orders_agree(order_action, rule_order):
                        self.llm_rule_agree_total += 1
                        logger.info("[order_subfsm] LLM and rule parser agree on order")
                    else:
                        self.llm_rule_disagree_total += 1
                        logger.warning(
                            "[order_subfsm] LLM and rule parser disagree: llm=%s rule=%s",
                            [{i.name: i.qty} for i in order_action.items],
                            [{i.name: i.qty} for i in rule_order.items],
                        )
                        self._events.publish("order.llm_candidate", {
                            "stage": "listen",
                            "llm_items": [{"name": i.name, "qty": i.qty} for i in order_action.items],
                            "rule_items": [{"name": i.name, "qty": i.qty} for i in rule_order.items],
                            "text": normalized.raw.text,
                            "session_id": normalized.raw.session_id,
                        })

        normalized_items = self._normalize_order_items(order_action.items)
        if not normalized_items:
            self._publish_tts_soft(self.config.listen_retry_prompt, profile="error")
            return

        violation = self._validate_order_constraints(normalized_items)
        if violation is not None:
            self._publish_tts_soft(violation, profile="order_prompt")
            return

        normalized_order = OrderAction(type="order", items=normalized_items)
        with self._fsm_lock:
            if not self._can_session_continue(normalized.raw.session_id):
                return
            self._current_order = normalized_order
            self._last_listen_text = normalized.raw.text
        if not self.transition("order.extracted", session_id=normalized.raw.session_id, reason="order_extracted"):
            return
        self._append_event("order_parsed", {"text": normalized.raw.text, "items": [item.model_dump() for item in normalized_items]})
        self._save_order_group(normalized_order, stage="listen_parsed")
        self._run_repeat_stage(normalized.raw.session_id)

    def _run_repeat_stage(self, session_id: int) -> None:
        if not self._can_session_continue(session_id):
            return
        with self._fsm_lock:
            order = self._current_order
        if order is None:
            self.transition("wrong.without_fix", session_id=session_id, reason="repeat_without_order")
            return
        speak_text = self._build_repeat_fallback(order)
        if self.config.repeat_use_llm:
            try:
                speak_decision = self._llm.get_order_repeat_speak(
                    confirm_instruction=self.config.repeat_instruction,
                    order_action=order,
                    max_retries=self.config.llm_max_retries,
                )
                llm_text = str(getattr(getattr(speak_decision, "action", None), "content", "") or "").strip()
                if llm_text:
                    speak_text = llm_text
            except Exception as exc:
                logger.warning("Order REPEAT LLM failed, using deterministic template: %s", exc)
        self.transition("repeat.completed", session_id=session_id, reason="repeat_completed")
        with self._fsm_lock:
            if self._session_id == session_id:
                self._processing_input = False
        if not self._can_session_continue(session_id):
            return
        self._publish_tts(speak_text, profile="order_confirm")

    def _process_check_stage(self, normalized: NormalizedInput) -> None:
        if not self._can_session_continue(normalized.raw.session_id):
            return
        with self._fsm_lock:
            order = self._current_order
        if order is None:
            self.transition("wrong.without_fix", session_id=normalized.raw.session_id, reason="check_without_order")
            return

        semantic = self._classify_check_input(normalized, order)
        logger.info(
            "[order_subfsm] CHECK semantic event: %s (confidence=%.2f): %s",
            semantic.event_type,
            semantic.confidence,
            normalized.raw.text,
        )
        kind_str = semantic.event_type
        if kind_str == "cancel_request":
            self._append_event("check_cancelled", {"text": normalized.raw.text})
            self._publish_order_cancelled("user_cancel")
            self._reset_to_not_permitted("user_cancelled")
            return
        if kind_str == "repeat_request":
            if self.transition("repeat.retry", session_id=normalized.raw.session_id, reason="repeat_requested"):
                self._run_repeat_stage(normalized.raw.session_id)
            return
        if kind_str == "menu_question":
            self._publish_tts_soft(f"{self._build_menu_summary()} Please confirm your order.", profile="order_prompt")
            return
        if kind_str == "empty_or_noise":
            self._check_retry_count += 1
            if self._check_retry_count > self.config.check_max_retries:
                self._publish_warning("retry_limit", "Check retry limit exceeded")
                self._publish_order_cancelled("check_retry_exceeded")
                self._reset_to_not_permitted("check_retry_exceeded")
                return
            self._publish_tts_soft(self.config.check_retry_prompt, profile="error")
            return
        if kind_str == "smalltalk":
            self._publish_tts_soft("Please confirm your order: yes or no?", profile="order_prompt")
            return
        if kind_str == "out_of_scope":
            self._publish_tts_soft("I'm confirming your order. Is it correct?", profile="order_prompt")
            return

        self._check_retry_count += 1
        if self._check_retry_count > self.config.check_max_retries:
            self._publish_warning("retry_limit", "Check retry limit exceeded")
            self._publish_order_cancelled("check_retry_exceeded")
            self._reset_to_not_permitted("check_retry_exceeded")
            return

        decision: Optional[OrderCheckDecision] = None
        self._last_check_text = normalized.raw.text
        if semantic.parse_source == "rule" and semantic.confirm_result:
            self.confirm_rule_hit_total += 1
            logger.info("[order_subfsm] pipeline rule confirm (confidence=%.2f): %s", semantic.confidence, semantic.confirm_result)
            if semantic.confirm_result == "cancel":
                self._append_event("check_cancelled", {"text": normalized.raw.text})
                self._publish_order_cancelled("user_cancel")
                self._reset_to_not_permitted("user_cancelled")
                return
            if semantic.confirm_result == "repeat_request":
                if self.transition("repeat.retry", session_id=normalized.raw.session_id, reason="repeat_requested"):
                    self._run_repeat_stage(normalized.raw.session_id)
                return
            if semantic.confirm_result == "correct":
                decision = OrderCheckDecision(result="correct")
            elif semantic.confirm_result == "wrong":
                if semantic.fix_order:
                    fix_items = [OrderItem(**item) for item in (semantic.fix_order.get("items", []))]
                    decision = OrderCheckDecision(
                        result="wrong",
                        action=FixOrderAction(type="fix_order", items=fix_items),
                    )
                else:
                    decision = OrderCheckDecision(result="wrong", reply=semantic.fix_reply or self.config.fix_missing_prompt)

        if decision is None:
            self.llm_fallback_count += 1
            try:
                decision = self._llm.get_order_check_decision(
                    customer_reply=normalized.raw.text,
                    order_action=order,
                    food_aliases=self.config.food_aliases,
                    max_retries=self.config.llm_max_retries,
                )
            except Exception as exc:
                logger.warning("Order CHECK LLM failed: %s", exc)
                self._publish_tts_soft(self.config.check_retry_prompt, profile="error")
                return
            if decision.result == "correct" and self._confirm_parser._has_modification_signal(normalized.normalized_text):
                logger.warning(
                    "[order_subfsm] LLM returned correct but text has modification signal, rejecting: %s",
                    normalized.raw.text,
                )
                self._events.publish("order.llm_candidate", {
                    "llm_result": "correct",
                    "rejected_reason": "modification_signal_present",
                    "text": normalized.raw.text,
                    "session_id": normalized.raw.session_id,
                })
                self._publish_tts_soft("I heard something that might be a change. Could you please just say yes or no?", profile="order_prompt")
                return

            # Cross-validate LLM check result against rule confirmation parser
            if decision.action and decision.action.items:
                rule_confirm = self._pipeline._try_rule_parse_confirm(normalized.raw.text)
                if rule_confirm and rule_confirm.fix_order:
                    rule_fix = OrderAction(**rule_confirm.fix_order)
                    llm_fix = OrderAction(type="order", items=decision.action.items)
                    if self._orders_agree(llm_fix, rule_fix):
                        self.llm_rule_agree_total += 1
                    else:
                        self.llm_rule_disagree_total += 1
                        logger.warning(
                            "[order_subfsm] CHECK LLM and rule disagree on fix: llm=%s rule=%s",
                            [{i.name: i.qty} for i in llm_fix.items],
                            [{i.name: i.qty} for i in rule_fix.items],
                        )
                        self._events.publish("order.llm_candidate", {
                            "stage": "check",
                            "llm_items": [{"name": i.name, "qty": i.qty} for i in llm_fix.items],
                            "rule_items": [{"name": i.name, "qty": i.qty} for i in rule_fix.items],
                            "text": normalized.raw.text,
                            "session_id": normalized.raw.session_id,
                        })

        if decision.result == "correct":
            self._commit_confirmed_order(order, normalized.raw.session_id, normalized.raw.text)
            return

        if decision.action is not None and decision.action.items:
            normalized_items = self._normalize_order_items(decision.action.items)
            violation = self._validate_order_constraints(normalized_items)
            if normalized_items and violation is None:
                with self._fsm_lock:
                    if self._can_session_continue(normalized.raw.session_id):
                        self._current_order = OrderAction(type="order", items=normalized_items)
                if self.transition("order.fixed", session_id=normalized.raw.session_id, reason="order_fixed"):
                    self._save_order_group(self._current_order, stage="order_fixed")
                    self._append_event("order_fixed", {"text": normalized.raw.text, "items": [item.model_dump() for item in normalized_items]})
                    self._run_repeat_stage(normalized.raw.session_id)
                return

        followup = str(decision.reply or "").strip() or self.config.fix_missing_prompt
        self._publish_tts_soft(followup, profile="order_prompt")
        if self._can_session_continue(normalized.raw.session_id):
            self.transition("wrong.without_fix", session_id=normalized.raw.session_id, reason="wrong_without_fix")

    def _commit_confirmed_order(self, order: OrderAction, session_id: int, check_text: str) -> None:
        if not order.items:
            self._publish_tts_soft(self.config.check_retry_prompt, profile="error")
            return
        if not self.transition("check.correct", session_id=session_id, reason="check_correct", publish=False):
            return
        self._publish_state_event("check_correct")
        self._reset_to_not_permitted("finish_completed")

    def _publish_tts(self, text: str, profile: str = "dialogue") -> None:
        text = str(text or "").strip()
        if not text:
            return
        with self._fsm_lock:
            session_id = self._session_id
            self._last_prompt_text = text
        self._events.publish("tts.request", {**TtsRequestEvent(text=text, session_id=session_id).model_dump(), "profile": profile})
        try:
            result = self._tts.speak(text, profile=profile)
        except Exception as exc:
            logger.warning("TTS speak failed: %s", exc)
            self.tts_hard_fail_count += 1
            failed = TtsFailedEvent(text=text, error=str(exc), session_id=session_id).model_dump()
            self._events.publish("tts.failed", failed)
            self._events.publish(
                "order.error",
                OrderErrorEvent(error=f"TTS playback failed: {exc}", stage=self._state.value, session_id=session_id).model_dump(),
            )
            raise TTSPlaybackError(str(exc)) from exc

        playback_duration_s = None
        audio_duration_s = None
        detail_payload: Dict[str, Any] = {"profile": profile}
        if isinstance(result, tuple) and len(result) >= 2:
            try:
                playback_duration_s = float(result[0])
                audio_duration_s = float(result[1])
            except (TypeError, ValueError):
                playback_duration_s = None
                audio_duration_s = None
        elif result is not None:
            playback_duration_s = _float_attr(result, "playback_duration_s")
            audio_duration_s = _float_attr(result, "audio_duration_s")
            for field in (
                "backend",
                "model_id",
                "profile",
                "normalized_text",
                "cache_hit",
                "fallback",
                "synth_latency_s",
                "rtf",
            ):
                if hasattr(result, field):
                    detail_payload[field] = getattr(result, field)
            self._publish_tts_detail_events(text, session_id, detail_payload)

        self._events.publish(
            "tts.completed",
            {
                **TtsCompletedEvent(
                    text=text,
                    playback_duration_s=playback_duration_s,
                    audio_duration_s=audio_duration_s,
                    session_id=session_id,
                ).model_dump(),
                **detail_payload,
            },
        )

    def _publish_tts_detail_events(self, text: str, session_id: int, payload: Dict[str, Any]) -> None:
        base = {"text": text, "session_id": session_id, **payload}
        if payload.get("normalized_text"):
            self._events.publish("tts.normalized", base)
        if payload.get("backend"):
            self._events.publish("tts.backend_selected", base)
        self._events.publish("tts.cache_hit" if payload.get("cache_hit") else "tts.cache_miss", base)
        if payload.get("synth_latency_s") is not None:
            self._events.publish("tts.synthesis_completed", base)
        self._events.publish("tts.playback_completed", base)
        if payload.get("fallback"):
            self._events.publish("tts.fallback_used", base)

    def _publish_state_event(self, reason: str) -> None:
        event = self._build_state_event(reason)
        self._latest_state_event = event
        self._events.publish("order.state", event.model_dump())

    def _publish_order_confirm(self, order: Optional[OrderAction]) -> None:
        if order is None:
            return
        foods = [item.name for item in order.items]
        foods_with_qty = [{"name": item.name, "qty": int(item.qty)} for item in order.items]
        event = OrderConfirmedEvent(
            order=order.model_dump(),
            foods=foods,
            foods_with_qty=foods_with_qty,
            recognized_text=self._last_listen_text,
            check_text=self._last_check_text,
            order_id=self._current_order_id or None,
            order_dir=self._current_order_dir or None,
            session_id=self._session_id,
        )
        for key in ("customer_id", "customer_no", "folder", "customer_folder"):
            if key in self._serving_payload:
                setattr(event, key, self._serving_payload.get(key))
        self._latest_order_snapshot = event.model_dump()
        self._events.publish("order.confirmed", event.model_dump())

    def _publish_order_confirm_from_outbox(self, order_id: str, order_dir: str, outbox_entry: dict) -> None:
        """Re-publish a confirmed event from outbox data (used by outbox.retry)."""
        event = OrderConfirmedEvent(
            order=outbox_entry.get("order", {}),
            foods=outbox_entry.get("foods", []),
            foods_with_qty=outbox_entry.get("foods_with_qty", []),
            order_id=order_id,
            order_dir=order_dir,
            session_id=0,
        )
        self._events.publish("order.confirmed", event.model_dump())

    @staticmethod
    def _orders_agree(order_a: OrderAction, order_b: OrderAction) -> bool:
        """Check if two parsed orders agree on items and quantities."""
        def _item_key(item: OrderItem) -> str:
            return item.item_id or item.name

        items_a = {_item_key(i): i.qty for i in order_a.items}
        items_b = {_item_key(i): i.qty for i in order_b.items}
        return items_a == items_b

    def _build_metrics_event(self) -> OrderMetricsEvent:
        mean_ms = 0.0
        p50_ms = 0.0
        p95_ms = 0.0
        if self._confirm_latency_count:
            mean_ms = (self._confirm_latency_total_s / self._confirm_latency_count) * 1000.0
        if self._confirm_latency_samples:
            sorted_ms = sorted(self._confirm_latency_samples)
            n = len(sorted_ms)
            p50_ms = sorted_ms[int(n * 0.50)]
            p95_ms = sorted_ms[min(int(n * 0.95), n - 1)]
        return OrderMetricsEvent(
            total_inputs=self.total_inputs,
            ignored_inputs=self.ignored_inputs,
            successful_replies=self.successful_replies,
            orders_confirmed=self.orders_confirmed,
            order_session_started_total=self.order_session_started_total,
            order_confirmed_total=self.order_confirmed_total,
            order_cancelled_total=self.order_cancelled_total,
            order_failed_total=self.order_failed_total,
            order_recovered_total=self.order_recovered_total,
            llm_fallback_count=self.llm_fallback_count,
            duplicate_input_count=self.duplicate_input_count,
            invalid_state_input_count=self.invalid_state_input_count,
            tts_hard_fail_count=self.tts_hard_fail_count,
            tts_soft_fail_count=self.tts_soft_fail_count,
            mean_time_to_confirm_ms=mean_ms,
            outbox_pending_total=self.outbox_pending_total,
            outbox_published_total=self.outbox_published_total,
            outbox_delivered_total=self.outbox_delivered_total,
            outbox_dead_letter_total=self.outbox_dead_letter_total,
            rule_parse_hit_total=self.rule_parse_hit_total,
            confirm_rule_hit_total=self.confirm_rule_hit_total,
            asr_echo_block_total=self.asr_echo_block_total,
            llm_rule_agree_total=self.llm_rule_agree_total,
            llm_rule_disagree_total=self.llm_rule_disagree_total,
            p50_time_to_confirm_ms=p50_ms,
            p95_time_to_confirm_ms=p95_ms,
            session_id=self._session_id,
        )

    def _publish_metrics(self) -> None:
        self._events.publish("order.metrics", self._build_metrics_event().model_dump())

    def _publish_tts_soft(self, text: str, profile: str = "dialogue") -> bool:
        try:
            self._publish_tts(text, profile)
            return True
        except TTSPlaybackError:
            self.tts_soft_fail_count += 1
            logger.warning("[order_subfsm] soft TTS failure suppressed (profile=%s)", profile)
            return False

    def _publish_order_cancelled(self, reason: str) -> None:
        self.order_cancelled_total += 1
        event = OrderCancelledEvent(reason=reason, order_id=self._current_order_id or None, session_id=self._session_id)
        self._events.publish("order.cancelled", event.model_dump())

    def _publish_warning(self, warning_type: str, message: str) -> None:
        event = OrderWarningEvent(warning_type=warning_type, message=message, session_id=self._session_id, stage=self._state.value)
        self._events.publish("order.warning", event.model_dump())

    def _publish_invalid_transition(self, event: str, reason: str) -> None:
        payload = InvalidTransitionEvent(
            from_state=self._state.value,
            event=event,
            reason=reason,
            session_id=self._session_id,
        ).model_dump()
        self._events.publish("order.invalid_transition", payload)

    def _append_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        with self._fsm_lock:
            order_dir = str(self._current_order_dir or "").strip()
        if not order_dir:
            return
        event = {"type": event_type, "ts": time.time(), "session_id": self._session_id}
        if hasattr(self._events, "last_event_seq"):
            event["last_event_seq"] = getattr(self._events, "last_event_seq")
        if data:
            event.update(data)
        try:
            self._storage.append_event(order_dir, event)
        except AttributeError:
            pass
        except Exception as exc:
            logger.debug("[order_subfsm] append_event failed: %s", exc)

    def _append_outbox(self, status: str, order: Optional[OrderAction]) -> None:
        with self._fsm_lock:
            order_dir = str(self._current_order_dir or "").strip()
            order_id = str(self._current_order_id or "").strip()
        if not order_dir or not hasattr(self._storage, "append_outbox"):
            return
        now = time.time()
        entry = {
            "status": status,
            "topic": "order.confirmed",
            "order_id": order_id,
            "idempotency_key": f"order-confirmed-{order_id}",
            "attempt_count": 0,
            "last_attempt_ts": now,
            "next_retry_ts": now + self.config.outbox_retry_sec,
            "ts": now,
        }
        if order:
            entry["foods"] = [item.name for item in order.items]
            entry["foods_with_qty"] = [{"name": item.name, "qty": int(item.qty)} for item in order.items]
            entry["order"] = order.model_dump()
        try:
            self._storage.append_outbox(order_dir, entry)
        except Exception as exc:
            logger.debug("[order_subfsm] append_outbox failed: %s", exc)

    def _build_menu_summary(self) -> str:
        names = [name.replace("_", " ") for name, item in sorted(self._menu_items.items()) if item.available]
        if not names:
            return "I'm sorry, I don't have menu information available."
        if len(names) <= 3:
            return f"We have {', '.join(names)}."
        return f"We have {', '.join(names[:-1])}, and {names[-1]}."

    def _build_available_items_text(self) -> str:
        names = [name.replace("_", " ") for name, item in sorted(self._menu_items.items()) if item.available]
        if not names:
            return "Please ask about our available items."
        if len(names) <= 3:
            return f"We have {', '.join(names)}."
        return f"Available items are {', '.join(names[:-1])}, and {names[-1]}."

    def _validate_order_constraints(self, items: List[OrderItem]) -> Optional[str]:
        total = sum(item.qty for item in items)
        if total > self.config.max_total_qty:
            return f"Sorry, I can only take up to {self.config.max_total_qty} items in one order. Please tell me a smaller order."
        for item in items:
            key = getattr(item, "item_id", None) or item.name
            menu_item = self._menu_items.get(key) or self._menu_items.get(item.name)
            if menu_item is not None and not menu_item.available:
                return f"Sorry, {item.name.replace('_', ' ')} is sold out right now. {self._build_available_items_text()}"
            if item.qty > self.config.max_qty_per_item:
                return "That's a bit too many. Could you order fewer?"
            if menu_item and item.qty > int(menu_item.max_qty):
                return f"Sorry, you can order up to {menu_item.max_qty} {item.name.replace('_', ' ')} at a time."
            if item.qty <= 0:
                return "Could you please check the quantity?"
        return None

    def _build_state_event(self, reason: str) -> OrderStateEvent:
        with self._fsm_lock:
            return OrderStateEvent(
                state=self._state.value,
                reason=reason,
                serving_state=self._serving_state,
                order=self._current_order.model_dump() if self._current_order else None,
                order_id=self._current_order_id or None,
                order_dir=self._current_order_dir or None,
                session_id=self._session_id,
            )

    def _is_session_pending(self, session_id: int) -> bool:
        with self._fsm_lock:
            return self._session_id == session_id and self._serving_state == "PAUSED_ORDERING"

    def _can_session_continue(self, session_id: int) -> bool:
        with self._fsm_lock:
            return (
                self._session_id == session_id
                and self._serving_state == "PAUSED_ORDERING"
                and self._state != OrderState.NOT_PERMITTED
            )

    def _reset_to_not_permitted(self, reason: str) -> None:
        with self._fsm_lock:
            current_session_id = self._session_id
        self.transition("session.reset", session_id=current_session_id, reason=reason)

    def _is_duplicate_input(self, text: str) -> bool:
        now = time.monotonic()
        norm = " ".join(str(text or "").strip().lower().split())
        if not norm:
            return True
        with self._fsm_lock:
            if norm == self._last_input_norm and (now - self._last_input_mono) <= max(0.0, self.config.input_dedup_window_sec):
                return True
            self._last_input_norm = norm
            self._last_input_mono = now
        return False

    @staticmethod
    def _normalize_text_token(text: str) -> str:
        lowered = str(text or "").strip().lower()
        lowered = re.sub(r"[^a-z0-9_ ]+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered.replace(" ", "_")

    def _build_food_alias_lookup(self, alias_map: Dict[str, List[str]]) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        if not isinstance(alias_map, dict):
            return lookup
        for canonical, aliases in alias_map.items():
            canonical_norm = self._normalize_text_token(canonical)
            if not canonical_norm:
                continue
            lookup[canonical_norm] = canonical_norm
            if isinstance(aliases, list):
                for alias in aliases:
                    alias_norm = self._normalize_text_token(alias)
                    if alias_norm:
                        lookup[alias_norm] = canonical_norm
        return lookup

    def _build_menu_items(self, menu_items: List[MenuItemConfig]) -> Dict[str, MenuItemConfig]:
        items: Dict[str, MenuItemConfig] = {}
        for item in menu_items or []:
            key = self._normalize_text_token(item.id or item.name)
            if key:
                items[key] = item
        return items

    def _canonicalize_food_name(self, raw_name: str) -> str:
        key = self._normalize_text_token(raw_name)
        if not key:
            return ""
        return self._food_alias_lookup.get(key, key)

    def _normalize_order_items(self, items: List[OrderItem]) -> List[OrderItem]:
        merged: Dict[str, int] = {}
        for item in items or []:
            try:
                raw_name = item.name if hasattr(item, "name") else item.get("name", "")
                raw_qty = item.qty if hasattr(item, "qty") else item.get("qty", 1)
            except Exception:
                continue
            canonical = self._canonicalize_food_name(raw_name)
            if not canonical:
                continue
            try:
                qty = int(raw_qty)
            except Exception:
                qty = 1
            if qty <= 0:
                qty = 1
            merged[canonical] = merged.get(canonical, 0) + qty
        result = []
        for name in sorted(merged.keys()):
            menu_item = self._menu_items.get(name)
            iid = menu_item.id if menu_item else name
            result.append(OrderItem(name=name, item_id=iid, qty=int(merged[name])))
        return result

    def _format_order_for_speech(self, order: Optional[OrderAction]) -> str:
        if order is None or not order.items:
            return "your order"
        parts = []
        for item in order.items:
            label = item.name.replace("_", " ")
            parts.append(f"{item.qty} {label}" if item.qty > 1 else label)
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def _build_repeat_fallback(self, order: Optional[OrderAction]) -> str:
        return f"Let me confirm. You ordered {self._format_order_for_speech(order)}. Is that correct?"

    def _build_finish_text(self, order: Optional[OrderAction]) -> str:
        foods_text = self._format_order_for_speech(order)
        template = str(self.config.finish_template or "").strip() or "OK I'll get {foods} for you"
        try:
            return template.format(foods=foods_text)
        except Exception:
            return f"OK I'll get {foods_text} for you"

    def _save_order_group(self, order: Optional[OrderAction], stage: str) -> None:
        if order is None:
            return
        with self._fsm_lock:
            order_dir = str(self._current_order_dir or "").strip()
            order_id = str(self._current_order_id or "").strip()
            serving_payload = dict(self._serving_payload) if isinstance(self._serving_payload, dict) else {}
            listen_text = self._last_listen_text
            check_text = self._last_check_text
        if not order_dir:
            return
        payload = {
            "timestamp": time.time(),
            "stage": str(stage or "").strip(),
            "order_id": order_id,
            "order": order.model_dump(),
            "recognized_text": listen_text,
            "check_text": check_text,
        }
        for key in ("customer_id", "customer_no", "folder", "customer_folder"):
            if key in serving_payload:
                payload[key] = serving_payload.get(key)
        self._storage.save_order_group(order_dir, payload)

    def _persist_session_snapshot(self, phase: str, commit_status: str, reason: str, *, state_override: Optional[OrderState] = None) -> None:
        with self._fsm_lock:
            self._persist_session_snapshot_locked(phase, commit_status, reason, state_override=state_override)

    def _persist_session_snapshot_locked(self, phase: str, commit_status: str, reason: str, *, state_override: Optional[OrderState] = None) -> None:
        order_dir = str(self._current_order_dir or "").strip()
        if not order_dir or not hasattr(self._storage, "save_session_snapshot"):
            return
        state = (state_override or self._state).value
        snapshot = {
            "timestamp": time.time(),
            "session_id": self._session_id,
            "state": state,
            "serving_state": self._serving_state,
            "order_id": self._current_order_id or None,
            "order_dir": order_dir,
            "order": self._current_order.model_dump() if self._current_order else None,
            "recognized_text": self._last_listen_text,
            "check_text": self._last_check_text,
            "last_prompt": self._last_prompt_text,
            "phase": phase,
            "commit_status": commit_status,
            "reason": reason,
            "last_event_seq": int(getattr(self._events, "last_event_seq", 0) or 0),
        }
        self._latest_session_snapshot = snapshot
        try:
            self._storage.save_session_snapshot(order_dir, snapshot)
        except Exception as exc:
            logger.warning("Failed to save session snapshot (%s): %s", order_dir, exc)

    def _recover_existing_snapshots(self) -> None:
        if not hasattr(self._storage, "list_session_snapshots"):
            return
        try:
            snapshots = self._storage.list_session_snapshots()
        except Exception:
            logger.debug("Failed to scan session snapshots", exc_info=True)
            return
        for entry in snapshots:
            snapshot = entry.get("snapshot") or {}
            state = str(snapshot.get("state") or "")
            order_dir = entry.get("order_dir")
            order_id = entry.get("order_id")
            phase = str(snapshot.get("phase") or "")
            status = str(snapshot.get("commit_status") or "")
            if order_id:
                self._known_order_ids.add(order_id)
            if state == OrderState.NOT_PERMITTED.value:
                continue
            if phase == "finish_confirmed" and status == "committed":
                continue

            if phase == "finish_confirmed" and status == "pending":
                self._recover_pending_confirmed(order_dir, order_id, snapshot)
                continue

            recovery_snapshot = dict(snapshot)
            recovery_snapshot.update(
                {
                    "timestamp": time.time(),
                    "state": OrderState.NOT_PERMITTED.value,
                    "phase": "recovered_cancelled",
                    "commit_status": "committed",
                    "reason": "startup_recovery_cancelled",
                }
            )
            try:
                self._storage.save_session_snapshot(order_dir, recovery_snapshot)
                self._storage.append_event(order_dir, {"type": "session_recovered", "recovery_action": "cancelled_incomplete", "ts": time.time()})
            except Exception:
                logger.debug("Failed to persist recovery snapshot", exc_info=True)

    def _recover_pending_confirmed(self, order_dir: str, order_id: Optional[str], snapshot: dict) -> None:
        logger.info("[order_subfsm] recovering pending confirmed order: %s", order_id)
        self.order_recovered_total += 1
        try:
            order_group_path = Path(order_dir) / "order_group.json"
            if order_group_path.is_file():
                order_data = json.loads(order_group_path.read_text(encoding="utf-8"))
                if order_data.get("stage") != "confirmed":
                    order_data["stage"] = "confirmed"
                    self._storage.save_order_group(order_dir, order_data)

            if hasattr(self._storage, "append_outbox"):
                has_pending_or_published = False
                if hasattr(self._storage, "load_outbox"):
                    for ob in self._storage.load_outbox(order_dir):
                        if ob.get("status") in ("pending", "published") and ob.get("order_id") == order_id:
                            has_pending_or_published = True
                            break
                if not has_pending_or_published:
                    self._storage.append_outbox(order_dir, {
                        "status": "pending",
                        "topic": "order.confirmed",
                        "order_id": order_id or "",
                        "ts": time.time(),
                        "recovery": True,
                    })

            committed_snapshot = dict(snapshot)
            committed_snapshot.update({
                "timestamp": time.time(),
                "state": OrderState.FINISH.value,
                "phase": "finish_confirmed",
                "commit_status": "committed",
                "reason": "recovered_pending_confirmed",
            })
            self._storage.save_session_snapshot(order_dir, committed_snapshot)
            self._storage.append_event(order_dir, {
                "type": "session_recovered",
                "recovery_action": "confirmed_pending_completed",
                "order_id": order_id,
                "ts": time.time(),
            })
        except Exception:
            logger.warning("[order_subfsm] failed to recover pending confirmed order %s", order_id, exc_info=True)


def _float_attr(obj: Any, name: str) -> Optional[float]:
    value = getattr(obj, name, None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
