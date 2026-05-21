"""Transition hook journal for crash-recovery of multi-step hooks."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TransitionJournal:
    """Append-only JSONL journal for transition hook steps.

    Each entry records: journal_id, event, step name, status, timestamp,
    plus optional transition_id, order_id, idempotency_key.
    Status values: started, committed, failed, skipped.
    On crash recovery, the journal tells which steps already completed.
    """

    def __init__(self, journal_path: str):
        self._path = str(journal_path)

    def begin_transition(self, event: str, session_id: int) -> str:
        """Return a journal_id for this transition (does not write)."""
        return uuid.uuid4().hex[:12]

    def write_step(
        self,
        journal_id: str,
        step_name: str,
        status: str,
        detail: Optional[Dict] = None,
        *,
        order_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        entry: Dict[str, object] = {
            "journal_id": journal_id,
            "step": step_name,
            "status": status,
            "ts": time.time(),
        }
        if order_id:
            entry["order_id"] = order_id
        if idempotency_key:
            entry["idempotency_key"] = idempotency_key
        if detail:
            entry.update(detail)
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def completed_steps(self, journal_id: str) -> Set[str]:
        """Return set of step names that reached 'committed' status for the given journal_id."""
        steps: Set[str] = set()
        for entry in self._load_entries():
            if entry.get("journal_id") == journal_id and entry.get("status") == "committed":
                steps.add(entry.get("step", ""))
        return steps

    def failed_steps(self, journal_id: str) -> Set[str]:
        """Return set of step names that reached 'failed' status for the given journal_id."""
        steps: Set[str] = set()
        for entry in self._load_entries():
            if entry.get("journal_id") == journal_id and entry.get("status") == "failed":
                steps.add(entry.get("step", ""))
        return steps

    def is_step_completed(self, journal_id: str, step_name: str) -> bool:
        return step_name in self.completed_steps(journal_id)

    def is_step_failed(self, journal_id: str, step_name: str) -> bool:
        return step_name in self.failed_steps(journal_id)

    def latest_journal_id(self) -> Optional[str]:
        """Return the journal_id from the most recent entry."""
        entries = self._load_entries()
        if entries:
            return entries[-1].get("journal_id")
        return None

    def latest_transition_id(self, journal_id: str) -> Optional[str]:
        """Return the transition_id from the latest entry for a journal_id, if any."""
        for entry in reversed(self._load_entries()):
            if entry.get("journal_id") == journal_id:
                return entry.get("transition_id")
        return None

    def _load_entries(self) -> List[Dict]:
        path = Path(self._path)
        if not path.is_file():
            return []
        entries = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
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
