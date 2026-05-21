"""ZeroMQ runtime for the ordering sub-FSM."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IdempotencyStore:
    """In-memory idempotency cache with optional JSON file persistence and TTL."""

    def __init__(
        self,
        persist_path: Optional[str] = None,
        ttl_sec: float = 300.0,
        flush_interval: int = 50,
    ):
        self._persist_path = persist_path
        self._ttl_sec = max(1.0, ttl_sec)
        self._flush_interval = max(1, flush_interval)
        self._cache: dict[str, dict] = {}
        self._ops_since_flush = 0
        self._lock = threading.Lock()
        if persist_path:
            self._load(persist_path)

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.monotonic() - entry.get("_ts", 0) > self._ttl_sec:
                del self._cache[key]
                return None
            return {k: v for k, v in entry.items() if k != "_ts"}

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._cache[key] = {**value, "_ts": time.monotonic()}
            self._ops_since_flush += 1
            if self._ops_since_flush >= self._flush_interval:
                self._flush_locked()
                self._ops_since_flush = 0

    def evict_expired(self) -> int:
        with self._lock:
            now = time.monotonic()
            expired = [k for k, v in self._cache.items() if now - v.get("_ts", 0) > self._ttl_sec]
            for k in expired:
                del self._cache[k]
            if expired and self._persist_path:
                self._flush_locked()
            return len(expired)

    def __setitem__(self, key: str, value: dict) -> None:
        self.put(key, value)

    def __getitem__(self, key: str) -> dict:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def _load(self, path: str) -> None:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except (OSError, FileNotFoundError):
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        now = time.monotonic()
        base_monotonic = time.time()
        loaded = 0
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            wall_ts = entry.get("_wall_ts", 0)
            age = abs(base_monotonic - wall_ts) if wall_ts else self._ttl_sec + 1
            if age > self._ttl_sec:
                continue
            entry["_ts"] = now - age
            self._cache[key] = entry
            loaded += 1
        if loaded:
            logger.info("IdempotencyStore loaded %d entries from %s", loaded, path)

    def _flush_locked(self) -> None:
        if not self._persist_path:
            return
        now = time.monotonic()
        serializable = {}
        for key, entry in self._cache.items():
            if now - entry.get("_ts", 0) > self._ttl_sec:
                continue
            clean = {k: v for k, v in entry.items() if k != "_ts"}
            clean["_wall_ts"] = time.time()
            serializable[key] = clean
        tmp = self._persist_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(serializable, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._persist_path)
        except OSError as exc:
            logger.debug("IdempotencyStore flush failed: %s", exc)


class ZmqRuntime:
    """ZeroMQ adapter that wraps OrderSubFSM."""

    def __init__(
        self,
        fsm: Any,
        pub_bind: str = "tcp://0.0.0.0:5555",
        router_bind: str = "tcp://0.0.0.0:5556",
        idempotency_path: Optional[str] = None,
        idempotency_ttl_sec: float = 300.0,
    ):
        self._fsm = fsm
        self._pub_bind = pub_bind
        self._router_bind = router_bind
        self._ctx: Optional[Any] = None
        self._pub_sock: Optional[Any] = None
        self._router_sock: Optional[Any] = None
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._event_seq = 0
        self._event_log: list[dict] = []
        self._event_lock = threading.Lock()
        self._ack_cache = IdempotencyStore(
            persist_path=idempotency_path,
            ttl_sec=idempotency_ttl_sec,
        )

    @property
    def last_event_seq(self) -> int:
        with self._event_lock:
            return self._event_seq

    def start(self) -> None:
        import zmq

        self._ctx = zmq.Context()
        self._pub_sock = self._ctx.socket(zmq.PUB)
        self._pub_sock.bind(self._pub_bind)
        logger.info("ZMQ PUB bound on %s", self._pub_bind)

        self._router_sock = self._ctx.socket(zmq.ROUTER)
        self._router_sock.bind(self._router_bind)
        logger.info("ZMQ ROUTER bound on %s", self._router_bind)

        self._fsm._events = _RuntimeEventSink(self)
        self._fsm.start_heartbeat()

        from cade.fsm.storage.outbox import OutboxRetryWorker
        self._outbox_worker = OutboxRetryWorker(
            self._fsm._storage,
            publish_fn=self._outbox_publish,
            retry_sec=self._fsm.config.outbox_retry_sec,
            max_attempts=self._fsm.config.outbox_max_attempts,
        )
        self._outbox_worker.start()

        self._running = True
        self._loop_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._loop_thread.start()
        logger.info("ZMQ runtime started")

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_outbox_worker"):
            self._outbox_worker.stop()
        self._fsm.stop_heartbeat()
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
        if self._pub_sock:
            self._pub_sock.close()
        if self._router_sock:
            self._router_sock.close()
        if self._ctx:
            self._ctx.term()
        logger.info("ZMQ runtime stopped")

    def spin(self) -> None:
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _event_loop(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self._router_sock, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(timeout=500))
                if self._router_sock in events:
                    self._handle_router_message()
            except Exception as exc:
                if self._running:
                    logger.warning("ZMQ event loop error: %s", exc)

    def _handle_router_message(self) -> None:
        import zmq

        frames = self._router_sock.recv_multipart(zmq.NOBLOCK)
        if len(frames) < 3:
            return
        identity = frames[0]
        raw = frames[2]
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON on ROUTER: %s", raw[:200])
            self._send_ack(identity, "error", {"error": "invalid_json"})
            return

        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})
        dedupe_key = self._command_key(msg)
        cached = self._ack_cache.get(dedupe_key)
        if cached is not None:
            duplicate_payload = dict(cached)
            duplicate_payload["duplicate"] = True
            self._send_ack(identity, msg_type, duplicate_payload, msg=msg, duplicate=True)
            return

        handler = self._COMMAND_HANDLERS.get(msg_type)
        if handler is None:
            logger.debug("Unknown command type: %s", msg_type)
            result = {"error": "unknown_command"}
            self._ack_cache[dedupe_key] = dict(result)
            self._send_ack(identity, msg_type, result, msg=msg)
            return

        try:
            result = handler(self, identity, payload, msg)
        except Exception as exc:
            logger.warning("Command handler error (%s): %s", msg_type, exc)
            result = {"error": str(exc)}
        if isinstance(result, dict):
            self._ack_cache[dedupe_key] = dict(result)
        else:
            self._ack_cache[dedupe_key] = {"result": result}
        self._send_ack(identity, msg_type, result, msg=msg)

    def _command_key(self, msg: dict) -> str:
        client_id = str(msg.get("client_id") or msg.get("source") or "client")
        client_msg_id = str(msg.get("client_msg_id") or msg.get("id") or uuid.uuid4().hex)
        explicit = str(msg.get("idempotency_key") or "").strip()
        if explicit:
            return explicit
        return f"{client_id}:{client_msg_id}"

    def _send_ack(self, identity: bytes, msg_type: str, result: Any, *, msg: Optional[dict] = None, duplicate: bool = False) -> None:
        import zmq

        payload = result if isinstance(result, dict) else {"result": result}
        payload = dict(payload)
        if "ok" not in payload and "accepted" not in payload:
            payload.setdefault("ok", payload.get("error") is None)
        payload.setdefault("ok", payload.get("accepted", payload.get("error") is None))
        payload.setdefault("accepted", payload.get("ok", payload.get("error") is None))
        payload.setdefault("reason", payload.get("reason") or payload.get("error"))
        payload.setdefault("duplicate", duplicate)
        payload.setdefault("state", self._fsm._state.value)
        payload.setdefault("session_id", self._fsm._session_id)
        payload.setdefault("last_event_seq", self.last_event_seq)

        envelope = {
            "v": 1,
            "type": f"{msg_type}.ack",
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "source": "voice-core",
            "session_id": self._fsm._session_id,
            "client_id": None if msg is None else msg.get("client_id"),
            "client_msg_id": None if msg is None else msg.get("client_msg_id"),
            "last_event_seq": self.last_event_seq,
            "payload": payload,
        }
        raw = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        try:
            self._router_sock.send_multipart([identity, b"", raw], zmq.NOBLOCK)
        except Exception as exc:
            logger.debug("Failed to send ACK: %s", exc)

    def publish_event(self, topic: str, payload: dict) -> None:
        import zmq

        with self._event_lock:
            self._event_seq += 1
            event_seq = self._event_seq
            envelope = {
                "v": 1,
                "type": topic,
                "id": uuid.uuid4().hex[:12],
                "ts": time.time(),
                "source": "voice-core",
                "session_id": payload.get("session_id", 0),
                "event_seq": event_seq,
                "payload": payload,
            }
            self._event_log.append(envelope)
            if len(self._event_log) > 4096:
                self._event_log = self._event_log[-4096:]
        raw = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        try:
            self._pub_sock.send_multipart([topic.encode("utf-8"), raw], zmq.NOBLOCK)
        except Exception as exc:
            logger.debug("Failed to publish %s: %s", topic, exc)

    def _outbox_publish(self, order_id: str, order_dir: str, outbox_entry: dict) -> None:
        """Publish order.confirmed from outbox retry worker."""
        self._fsm._publish_order_confirm_from_outbox(order_id, order_dir, outbox_entry)

    def _events_since(self, last_event_seq: int, max_events: int = 1000) -> list[dict]:
        with self._event_lock:
            matched = [event for event in self._event_log if int(event.get("event_seq") or 0) > int(last_event_seq)]
            return matched[-max_events:]

    def _handle_serving_state(self, identity: bytes, payload: dict, msg: dict) -> dict:
        self._fsm.handle_serving_state(payload)
        return {"ok": True}

    def _handle_user_text_primary(self, identity: bytes, payload: dict, msg: dict) -> dict:
        text = payload.get("text", "")
        source = payload.get("source") or msg.get("source") or "primary"
        result = self._fsm.handle_user_text(text, source=str(source))
        return {
            "received": True,
            "accepted": result.accepted,
            "reason": result.reason,
            "state": result.state,
            "session_id": result.session_id,
        }

    def _handle_user_text_secondary(self, identity: bytes, payload: dict, msg: dict) -> dict:
        text = payload.get("text", "")
        source = payload.get("source") or msg.get("source") or "secondary"
        result = self._fsm.handle_user_text(text, source=str(source))
        return {
            "received": True,
            "accepted": result.accepted,
            "reason": result.reason,
            "state": result.state,
            "session_id": result.session_id,
        }

    def _handle_order_id_propose(self, identity: bytes, payload: dict, msg: dict) -> dict:
        order_id = payload.get("order_id", "")
        self._fsm.handle_order_id(order_id)
        return {"ok": True}

    def _handle_snapshot_get(self, identity: bytes, payload: dict, msg: dict) -> dict:
        snap = self._fsm.snapshot()
        snap["last_event_seq"] = self.last_event_seq
        return snap

    def _handle_events_get_since(self, identity: bytes, payload: dict, msg: dict) -> dict:
        since = int(payload.get("last_event_seq") or 0)
        max_events = min(int(payload.get("max_events") or 1000), 1000)
        events = self._events_since(since, max_events=max_events)
        has_more = len(self._events_since(since)) > max_events
        to_seq = events[-1].get("event_seq", since) if events else since
        return {
            "events": events,
            "from_seq": since,
            "to_seq": to_seq,
            "next_from_seq": to_seq + 1 if events else since + 1,
            "last_event_seq": self.last_event_seq,
            "has_more": has_more,
            "count": len(events),
        }

    def _handle_health_get(self, identity: bytes, payload: dict, msg: dict) -> dict:
        import cade
        return {
            "status": "ok",
            "version": cade.__version__,
            "state": self._fsm._state.value,
            "session_id": self._fsm._session_id,
            "last_event_seq": self.last_event_seq,
            "uptime_sec": time.time(),
        }

    def _handle_session_cancel(self, identity: bytes, payload: dict, msg: dict) -> dict:
        reason = payload.get("reason", "external_cancel")
        self._fsm.cancel(reason)
        return {"ok": True}

    def _handle_outbox_undelivered(self, identity: bytes, payload: dict, msg: dict) -> dict:
        pending = []
        storage = self._fsm._storage
        if not hasattr(storage, "list_session_snapshots"):
            return {"entries": pending}
        try:
            for entry in storage.list_session_snapshots():
                order_dir = entry.get("order_dir")
                if not order_dir or not hasattr(storage, "load_outbox"):
                    continue
                for ob_entry in storage.load_outbox(order_dir):
                    if ob_entry.get("status") in ("pending", "published"):
                        pending.append(ob_entry)
        except Exception:
            pass
        return {"entries": pending}

    def _handle_metrics_get(self, identity: bytes, payload: dict, msg: dict) -> dict:
        return self._fsm._build_metrics_event().model_dump()

    def _handle_order_confirmed_ack(self, identity: bytes, payload: dict, msg: dict) -> dict:
        order_id = payload.get("order_id", "")
        to_status = payload.get("status", "delivered")
        if to_status not in ("delivered", "dead_letter"):
            return {"ok": False, "error": "status must be 'delivered' or 'dead_letter'"}
        if not order_id:
            return {"ok": False, "error": "order_id required"}

        storage = self._fsm._storage
        if not hasattr(storage, "list_session_snapshots"):
            return {"ok": False, "error": "storage does not support snapshots"}

        found = False
        for entry in storage.list_session_snapshots():
            if entry.get("order_id") == order_id:
                order_dir = entry.get("order_dir", "")
                if order_dir and hasattr(storage, "append_outbox"):
                    from cade.fsm.storage.outbox import OutboxManager
                    mgr = OutboxManager(storage)
                    found = mgr.mark_outbox(order_dir, order_id, to_status)
                    if found:
                        if to_status == "delivered":
                            self._fsm.outbox_delivered_total += 1
                        elif to_status == "dead_letter":
                            self._fsm.outbox_dead_letter_total += 1
                break

        if not found:
            return {"ok": False, "error": f"no pending/published outbox entry for {order_id}"}
        return {"ok": True}

    def _handle_outbox_retry(self, identity: bytes, payload: dict, msg: dict) -> dict:
        target_order_id = payload.get("order_id") or None
        storage = self._fsm._storage
        retried = []

        if not hasattr(storage, "list_session_snapshots"):
            return {"retried": retried}

        for entry in storage.list_session_snapshots():
            order_id = entry.get("order_id", "")
            order_dir = entry.get("order_dir", "")
            if not order_dir or not hasattr(storage, "load_outbox"):
                continue
            if target_order_id and order_id != target_order_id:
                continue
            for ob_entry in storage.load_outbox(order_dir):
                if ob_entry.get("status") in ("pending", "published"):
                    order = ob_entry.get("order") or self._fsm._latest_order_snapshot
                    self._fsm._publish_order_confirm_from_outbox(order_id, order_dir, ob_entry)
                    retried.append({"order_id": order_id})
                    break

        return {"retried": retried}

    _COMMAND_HANDLERS = {
        "serving_state.update": _handle_serving_state,
        "user_text.primary": _handle_user_text_primary,
        "user_text.secondary": _handle_user_text_secondary,
        "order_id.propose": _handle_order_id_propose,
        "snapshot.get": _handle_snapshot_get,
        "events.get_since": _handle_events_get_since,
        "health.get": _handle_health_get,
        "session.cancel": _handle_session_cancel,
        "outbox.undelivered": _handle_outbox_undelivered,
        "metrics.get": _handle_metrics_get,
        "order.confirmed.ack": _handle_order_confirmed_ack,
        "outbox.retry": _handle_outbox_retry,
    }


class _RuntimeEventSink:
    """Event sink implementation that publishes through the runtime."""

    def __init__(self, runtime: ZmqRuntime):
        self._runtime = runtime

    @property
    def last_event_seq(self) -> int:
        return self._runtime.last_event_seq

    def publish(self, topic: str, payload: dict) -> None:
        self._runtime.publish_event(topic, payload)
