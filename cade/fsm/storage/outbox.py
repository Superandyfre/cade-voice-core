"""Outbox management for reliable confirmed-event delivery."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cade.fsm.events import OutboxStatus

logger = logging.getLogger(__name__)


class OutboxManager:
    """Manages outbox entries with status transitions: pending -> published -> delivered/dead_letter."""

    def __init__(self, storage):
        self._storage = storage

    def append_outbox(self, order_dir: str, entry: dict) -> None:
        self._storage.append_outbox(order_dir, entry)

    def load_outbox(self, order_dir: str) -> List[dict]:
        return self._storage.load_outbox(order_dir)

    def mark_outbox(self, order_dir: str, order_id: str, to_status: str) -> bool:
        """Transition the latest pending/published outbox entry for an order to to_status."""
        entries = self.load_outbox(order_dir)
        transitioned = False
        for entry in reversed(entries):
            if entry.get("order_id") == order_id and entry.get("status") in ("pending", "published"):
                if to_status in (OutboxStatus.published.value, OutboxStatus.delivered.value, OutboxStatus.dead_letter.value):
                    new_entry = dict(entry)
                    new_entry["status"] = to_status
                    new_entry["ts"] = _now()
                    self._storage.append_outbox(order_dir, new_entry)
                    transitioned = True
                    break
        return transitioned

    def find_undelivered(self) -> List[dict]:
        """Find all pending or published outbox entries across all orders (latest per order_id)."""
        undelivered = []
        if not hasattr(self._storage, "list_session_snapshots"):
            return undelivered
        try:
            for entry in self._storage.list_session_snapshots():
                order_dir = entry.get("order_dir")
                if not order_dir:
                    continue
                latest_by_order: Dict[str, dict] = {}
                for ob_entry in self.load_outbox(order_dir):
                    oid = ob_entry.get("order_id", "")
                    if oid:
                        latest_by_order[oid] = ob_entry
                for oid, ob_entry in latest_by_order.items():
                    if ob_entry.get("status") in (OutboxStatus.pending.value, OutboxStatus.published.value):
                        ob_entry["order_dir"] = order_dir
                        undelivered.append(ob_entry)
        except Exception:
            pass
        return undelivered

    def find_retryable(self, retry_sec: float, max_attempts: int) -> List[dict]:
        """Find pending/published entries eligible for retry (latest entry per order_id only)."""
        now = time.time()
        retryable = []
        if not hasattr(self._storage, "list_session_snapshots"):
            return retryable
        try:
            for entry in self._storage.list_session_snapshots():
                order_dir = entry.get("order_dir")
                if not order_dir:
                    continue
                latest_by_order: Dict[str, dict] = {}
                for ob_entry in self.load_outbox(order_dir):
                    oid = ob_entry.get("order_id", "")
                    if oid:
                        latest_by_order[oid] = ob_entry
                for oid, ob_entry in latest_by_order.items():
                    if ob_entry.get("status") not in (OutboxStatus.pending.value, OutboxStatus.published.value):
                        continue
                    attempts = int(ob_entry.get("attempt_count", 0))
                    if attempts >= max_attempts:
                        continue
                    next_retry = float(ob_entry.get("next_retry_ts", 0))
                    if now >= next_retry:
                        ob_entry["order_dir"] = order_dir
                        retryable.append(ob_entry)
        except Exception:
            pass
        return retryable

    def count_by_status(self, order_dir: str) -> Dict[str, int]:
        counts = {"pending": 0, "published": 0, "delivered": 0, "dead_letter": 0}
        for entry in self.load_outbox(order_dir):
            status = entry.get("status", "pending")
            if status in counts:
                counts[status] += 1
        return counts

    def count_undelivered(self) -> int:
        return len(self.find_undelivered())


class OutboxRetryWorker:
    """Background worker that retries undelivered outbox entries."""

    def __init__(
        self,
        storage: Any,
        publish_fn: Callable[[str, str, dict], None],
        *,
        retry_sec: float = 30.0,
        max_attempts: int = 10,
    ):
        self._mgr = OutboxManager(storage)
        self._storage = storage
        self._publish_fn = publish_fn
        self._retry_sec = max(1.0, retry_sec)
        self._max_attempts = max(1, max_attempts)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("OutboxRetryWorker started (retry_sec=%.1f, max_attempts=%d)", self._retry_sec, self._max_attempts)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def tick(self) -> int:
        """Single retry pass. Returns number of retried entries."""
        retried = 0
        for entry in self._mgr.find_retryable(self._retry_sec, self._max_attempts):
            order_dir = entry.get("order_dir", "")
            order_id = entry.get("order_id", "")
            if not order_dir or not order_id:
                continue

            attempt_count = int(entry.get("attempt_count", 0)) + 1
            now = time.time()

            if attempt_count >= self._max_attempts:
                self._storage.append_outbox(order_dir, {
                    **entry,
                    "status": "dead_letter",
                    "attempt_count": attempt_count,
                    "last_attempt_ts": now,
                    "ts": now,
                })
                logger.warning("[outbox_retry] order %s exceeded max_attempts=%d -> dead_letter", order_id, self._max_attempts)
                continue

            self._publish_fn(order_id, order_dir, entry)

            self._storage.append_outbox(order_dir, {
                **entry,
                "status": "published",
                "attempt_count": attempt_count,
                "last_attempt_ts": now,
                "next_retry_ts": now + self._retry_sec,
                "ts": now,
                "retry": True,
            })
            retried += 1
            logger.info("[outbox_retry] republished order %s (attempt %d/%d)", order_id, attempt_count, self._max_attempts)

        return retried

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self._retry_sec):
            try:
                self.tick()
            except Exception as exc:
                logger.warning("[outbox_retry] tick failed: %s", exc)


def _now() -> float:
    return time.time()
