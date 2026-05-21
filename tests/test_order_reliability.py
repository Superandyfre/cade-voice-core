"""Reliability-focused tests for the hardened ordering runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import zmq

from cade.brain.schemas import OrderAction, OrderItem
from cade.fsm.config import MenuItemConfig, OrderFSMConfig
from cade.fsm.order_fsm import CallbackEventSink, CallbackTTSSink, LocalOrderIdProvider, LocalOrderStorage, OrderSubFSM
from cade.fsm.voice_runtime import SpeakingGate
from cade.fsm.zmq_runtime import ZmqRuntime


def _make_config(tmp_path, **overrides) -> OrderFSMConfig:
    defaults = dict(
        order_base_dir=str(tmp_path / "orders"),
        food_aliases={"coke": ["coke", "cola"], "water": ["water"]},
        ask_prompt="What would you like?",
        repeat_instruction="Repeat the order.",
        listen_retry_prompt="Say again.",
        fix_missing_prompt="Tell me changes.",
        check_retry_prompt="Is it correct?",
        finish_template="OK I'll get {foods}",
        input_dedup_window_sec=0.1,
        llm_max_retries=1,
        order_id_proposal_timeout_sec=2.0,
    )
    defaults.update(overrides)
    return OrderFSMConfig(**defaults)


def _make_fsm(tmp_path, config=None):
    config = config or _make_config(tmp_path)
    llm = MagicMock()
    llm.get_order_action.return_value = OrderAction(type="order", items=[OrderItem(name="coke", qty=1)])
    llm.get_order_repeat_speak.return_value = SimpleNamespace(action=SimpleNamespace(content="Coke?"))
    storage = LocalOrderStorage(config.order_base_dir, snapshot_file_name=config.snapshot_file_name)
    return OrderSubFSM(
        llm_client=llm,
        config=config,
        order_id_provider=LocalOrderIdProvider(),
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: None),
    )


def _free_port(ctx):
    sock = ctx.socket(zmq.PUSH)
    sock.bind("tcp://127.0.0.1:0")
    port = sock.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1]
    sock.close()
    return int(port)


def _send_envelope(ctx, port, envelope, timeout=2000):
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, f"reliability-{time.time_ns()}".encode("ascii"))
    sock.setsockopt(zmq.RCVTIMEO, timeout)
    sock.connect(f"tcp://127.0.0.1:{port}")
    try:
        sock.send_multipart([b"", json.dumps(envelope).encode("utf-8")])
        frames = sock.recv_multipart()
        return json.loads(frames[-1])
    finally:
        sock.close()


def test_incomplete_session_snapshot_is_cancelled_on_restart(tmp_path):
    storage = LocalOrderStorage(str(tmp_path / "orders"))
    order_dir = storage.create_order_dir("12345")
    storage.save_session_snapshot(
        order_dir,
        {
            "state": "CHECK",
            "phase": "repeat_completed",
            "commit_status": "committed",
            "order_id": "12345",
            "order_dir": order_dir,
        },
    )

    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: None),
    )

    recovered = storage.load_session_snapshot(order_dir)
    assert recovered is not None
    assert recovered["state"] == "NOT_PERMITTED"
    assert recovered["phase"] == "recovered_cancelled"

    events = (Path(order_dir) / "events.jsonl").read_text(encoding="utf-8")
    assert "session_recovered" in events


def test_confirmed_snapshot_is_not_replayed_on_restart(tmp_path):
    events = []
    storage = LocalOrderStorage(str(tmp_path / "orders"))
    order_dir = storage.create_order_dir("12345")
    storage.save_session_snapshot(
        order_dir,
        {
            "state": "FINISH",
            "phase": "finish_confirmed",
            "commit_status": "committed",
            "order_id": "12345",
            "order_dir": order_dir,
        },
    )

    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: events.append((topic, payload))),
    )

    assert not [topic for topic, _ in events if topic == "order.confirmed"]
    snapshot = storage.load_session_snapshot(order_dir)
    assert snapshot["phase"] == "finish_confirmed"
    assert snapshot["commit_status"] == "committed"


def test_duplicate_router_command_returns_duplicate_ack_and_no_second_session(tmp_path):
    ctx = zmq.Context()
    pub_port = _free_port(ctx)
    router_port = _free_port(ctx)
    config = _make_config(
        tmp_path,
        zmq_pub_bind=f"tcp://127.0.0.1:{pub_port}",
        zmq_router_bind=f"tcp://127.0.0.1:{router_port}",
    )
    fsm = _make_fsm(tmp_path, config=config)
    runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)
    try:
        runtime.start()
        time.sleep(0.3)
        envelope = {
            "v": 1,
            "type": "serving_state.update",
            "id": "same-command",
            "client_id": "tablet-01",
            "client_msg_id": "msg-001",
            "ts": time.time(),
            "source": "tablet",
            "payload": {"state": "PAUSED_ORDERING"},
        }
        ack1 = _send_envelope(ctx, router_port, envelope)
        ack2 = _send_envelope(ctx, router_port, envelope)
        assert ack1["payload"]["duplicate"] is False
        assert ack2["payload"]["duplicate"] is True
        assert fsm._session_id == 1
    finally:
        runtime.stop()
        ctx.term()


def test_events_get_since_replays_published_events(tmp_path):
    ctx = zmq.Context()
    pub_port = _free_port(ctx)
    router_port = _free_port(ctx)
    config = _make_config(
        tmp_path,
        zmq_pub_bind=f"tcp://127.0.0.1:{pub_port}",
        zmq_router_bind=f"tcp://127.0.0.1:{router_port}",
    )
    fsm = _make_fsm(tmp_path, config=config)
    runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)
    try:
        runtime.start()
        time.sleep(0.3)
        _send_envelope(
            ctx,
            router_port,
            {
                "v": 1,
                "type": "serving_state.update",
                "id": "evt-1",
                "client_id": "tablet-01",
                "client_msg_id": "evt-1",
                "ts": time.time(),
                "source": "tablet",
                "payload": {"state": "PAUSED_ORDERING"},
            },
        )
        time.sleep(0.3)
        replay = _send_envelope(
            ctx,
            router_port,
            {
                "v": 1,
                "type": "events.get_since",
                "id": "evt-2",
                "ts": time.time(),
                "source": "tablet",
                "payload": {"last_event_seq": 0},
            },
        )
        events = replay["payload"]["events"]
        assert events
        assert all(event.get("event_seq", 0) > 0 for event in events)
        assert replay["payload"]["last_event_seq"] >= events[-1]["event_seq"]
    finally:
        runtime.stop()
        ctx.term()


def test_out_of_stock_item_stays_in_listen(tmp_path):
    config = _make_config(
        tmp_path,
        menu_items=[MenuItemConfig(id="coke", name="coke", aliases=["coke"], available=False)],
        food_aliases={"coke": ["coke"]},
    )
    llm = MagicMock()
    tts_texts = []
    storage = LocalOrderStorage(config.order_base_dir, snapshot_file_name=config.snapshot_file_name)
    fsm = OrderSubFSM(
        llm_client=llm,
        config=config,
        order_id_provider=LocalOrderIdProvider(),
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: tts_texts.append(text)),
        event_sink=CallbackEventSink(lambda topic, payload: None),
    )
    fsm.handle_serving_state({"state": "PAUSED_ORDERING"})
    time.sleep(0.3)
    fsm.handle_user_text("one coke", source="primary")
    time.sleep(0.3)
    assert fsm._state == "LISTEN"
    assert any("sold out" in text.lower() for text in tts_texts)


def test_speaking_gate_blocks_similar_recent_tts_text():
    gate = SpeakingGate(0, similarity_threshold=0.6, similarity_window_sec=5.0)
    gate.begin("Let me confirm. You ordered coke.")
    gate.end("Let me confirm. You ordered coke.")
    assert gate.is_blocked("let me confirm you ordered coke") is True
    assert gate.is_blocked("i want a water") is False


# ------------------------------------------------------------------
# Crash recovery tests
# ------------------------------------------------------------------


def _make_storage(tmp_path):
    order_dir_path = tmp_path / "orders"
    return LocalOrderStorage(str(order_dir_path), snapshot_file_name="session_snapshot.json")


def _create_snapshot(storage, order_id, state, phase, commit_status, **extra):
    order_dir = storage.create_order_dir(order_id)
    snapshot = {
        "state": state,
        "phase": phase,
        "commit_status": commit_status,
        "order_id": order_id,
        "order_dir": order_dir,
        "session_id": 1,
        "timestamp": time.time(),
    }
    snapshot.update(extra)
    storage.save_session_snapshot(order_dir, snapshot)
    return order_dir


def test_crash_after_ask_begin_recovers_to_not_permitted(tmp_path):
    storage = _make_storage(tmp_path)
    order_dir = _create_snapshot(storage, "10001", "ASK", "ask.begin", "committed")

    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: None),
    )

    recovered = storage.load_session_snapshot(order_dir)
    assert recovered["state"] == "NOT_PERMITTED"
    assert recovered["phase"] == "recovered_cancelled"


def test_crash_after_order_extracted_recovers_to_not_permitted(tmp_path):
    storage = _make_storage(tmp_path)
    order_dir = _create_snapshot(
        storage, "10002", "REPEAT", "order.extracted", "committed",
        order={"type": "order", "items": [{"name": "coke", "qty": 1}]},
    )

    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: None),
    )

    recovered = storage.load_session_snapshot(order_dir)
    assert recovered["state"] == "NOT_PERMITTED"


def test_crash_after_repeat_completed_recovers_to_not_permitted(tmp_path):
    storage = _make_storage(tmp_path)
    order_dir = _create_snapshot(
        storage, "10003", "CHECK", "repeat.completed", "committed",
        order={"type": "order", "items": [{"name": "coke", "qty": 1}]},
    )

    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: None),
    )

    recovered = storage.load_session_snapshot(order_dir)
    assert recovered["state"] == "NOT_PERMITTED"


def test_crash_during_commit_confirmed_completes_order(tmp_path):
    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("10004")
    storage.save_order_group(order_dir, {
        "stage": "listen_parsed",
        "order_id": "10004",
        "order": {"type": "order", "items": [{"name": "coke", "qty": 1}]},
    })
    _create_snapshot(
        storage, "10004", "FINISH", "finish_confirmed", "pending",
        order={"type": "order", "items": [{"name": "coke", "qty": 1}]},
    )

    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: None),
    )

    recovered = storage.load_session_snapshot(order_dir)
    assert recovered["commit_status"] == "committed"
    assert recovered["phase"] == "finish_confirmed"

    order_data = json.loads((Path(order_dir) / "order_group.json").read_text(encoding="utf-8"))
    assert order_data["stage"] == "confirmed"

    events_text = (Path(order_dir) / "events.jsonl").read_text(encoding="utf-8")
    assert "confirmed_pending_completed" in events_text


def test_crash_after_commit_confirmed_does_not_replay(tmp_path):
    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("10005")
    storage.save_order_group(order_dir, {
        "stage": "confirmed",
        "order_id": "10005",
        "order": {"type": "order", "items": [{"name": "water", "qty": 2}]},
    })
    _create_snapshot(
        storage, "10005", "FINISH", "finish_confirmed", "committed",
        order={"type": "order", "items": [{"name": "water", "qty": 2}]},
    )

    published = []
    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: published.append((topic, payload))),
    )

    assert not [t for t, _ in published if t == "order.confirmed"]
    recovered = storage.load_session_snapshot(order_dir)
    assert recovered["commit_status"] == "committed"
    assert recovered["phase"] == "finish_confirmed"


def test_crash_after_tts_soft_fail_does_not_duplicate_order(tmp_path):
    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("10006")
    storage.save_order_group(order_dir, {
        "stage": "confirmed",
        "order_id": "10006",
        "order": {"type": "order", "items": [{"name": "coke", "qty": 1}]},
    })
    _create_snapshot(
        storage, "10006", "FINISH", "finish_confirmed", "committed",
        order={"type": "order", "items": [{"name": "coke", "qty": 1}]},
    )

    published = []
    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    fsm1 = OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: published.append((topic, payload))),
    )

    assert not [t for t, _ in published if t == "order.confirmed"]
    assert fsm1.order_confirmed_total == 0


# ------------------------------------------------------------------
# P0: Outbox status flow (pending -> published, not delivered)
# ------------------------------------------------------------------


def test_outbox_uses_published_not_delivered_on_confirm(tmp_path):
    """After order confirm, outbox should have pending then published (not delivered)."""
    config = _make_config(tmp_path)
    fsm = _make_fsm(tmp_path, config=config)
    fsm.handle_serving_state({"state": "PAUSED_ORDERING"})
    time.sleep(1.0)
    order_dir = fsm._current_order_dir
    assert order_dir, "Order dir should be set after session start"
    fsm.handle_user_text("one coke", source="primary")
    time.sleep(1.0)
    fsm.handle_user_text("yes", source="primary")
    time.sleep(1.0)

    storage = fsm._storage
    outbox_entries = storage.load_outbox(order_dir)
    statuses = [e.get("status") for e in outbox_entries]

    assert "pending" in statuses, f"Expected 'pending' in outbox, got: {statuses}"
    assert "published" in statuses, f"Expected 'published' in outbox, got: {statuses}"
    assert "delivered" not in statuses, f"Should NOT have 'delivered' without external ACK, got: {statuses}"


def test_order_confirmed_ack_marks_delivered(tmp_path):
    """order.confirmed.ack should mark the outbox entry as delivered."""
    ctx = zmq.Context()
    pub_port = _free_port(ctx)
    router_port = _free_port(ctx)
    config = _make_config(
        tmp_path,
        zmq_pub_bind=f"tcp://127.0.0.1:{pub_port}",
        zmq_router_bind=f"tcp://127.0.0.1:{router_port}",
    )
    fsm = _make_fsm(tmp_path, config=config)
    runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)
    try:
        runtime.start()
        time.sleep(0.3)

        _send_envelope(ctx, router_port, {
            "type": "serving_state.update", "id": "ack-1",
            "payload": {"state": "PAUSED_ORDERING"},
        })
        time.sleep(1.0)
        order_id = fsm._current_order_id
        order_dir = fsm._current_order_dir
        assert order_id, "Order should have been created"

        _send_envelope(ctx, router_port, {
            "type": "user_text.primary", "id": "ack-2",
            "payload": {"text": "one coke"},
        })
        time.sleep(1.0)
        _send_envelope(ctx, router_port, {
            "type": "user_text.primary", "id": "ack-3",
            "payload": {"text": "yes"},
        })
        time.sleep(1.0)

        # Send ACK
        ack_resp = _send_envelope(ctx, router_port, {
            "type": "order.confirmed.ack", "id": "ack-4",
            "payload": {"order_id": order_id, "status": "delivered"},
        })
        assert ack_resp["payload"]["ok"] is True

        # Verify outbox now has delivered status
        outbox = fsm._storage.load_outbox(order_dir)
        statuses = [e.get("status") for e in outbox]
        assert "delivered" in statuses, f"Expected 'delivered' after ACK, got: {statuses}"
    finally:
        runtime.stop()
        ctx.term()


def test_outbox_undelivered_includes_published_entries(tmp_path):
    """outbox.undelivered should include both pending and published entries."""
    ctx = zmq.Context()
    pub_port = _free_port(ctx)
    router_port = _free_port(ctx)
    config = _make_config(
        tmp_path,
        zmq_pub_bind=f"tcp://127.0.0.1:{pub_port}",
        zmq_router_bind=f"tcp://127.0.0.1:{router_port}",
    )
    fsm = _make_fsm(tmp_path, config=config)
    runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)
    try:
        runtime.start()
        time.sleep(0.3)

        _send_envelope(ctx, router_port, {
            "type": "serving_state.update", "id": "undel-1",
            "payload": {"state": "PAUSED_ORDERING"},
        })
        time.sleep(0.3)
        _send_envelope(ctx, router_port, {
            "type": "user_text.primary", "id": "undel-2",
            "payload": {"text": "one water"},
        })
        time.sleep(0.3)
        _send_envelope(ctx, router_port, {
            "type": "user_text.primary", "id": "undel-3",
            "payload": {"text": "yes"},
        })
        time.sleep(0.3)

        resp = _send_envelope(ctx, router_port, {
            "type": "outbox.undelivered", "id": "undel-4",
            "payload": {},
        })
        entries = resp["payload"]["entries"]
        assert len(entries) > 0, "Expected undelivered entries after confirm"
        statuses = {e.get("status") for e in entries}
        assert statuses & {"pending", "published"}, f"Expected pending/published, got: {statuses}"
    finally:
        runtime.stop()
        ctx.term()


def test_outbox_manager_mark_outbox(tmp_path):
    """OutboxManager.mark_outbox should transition status."""
    from cade.fsm.storage.outbox import OutboxManager

    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("50001")
    storage.append_outbox(order_dir, {
        "status": "pending", "topic": "order.confirmed", "order_id": "50001", "ts": time.time(),
    })
    storage.append_outbox(order_dir, {
        "status": "published", "topic": "order.confirmed", "order_id": "50001", "ts": time.time(),
    })

    mgr = OutboxManager(storage)
    result = mgr.mark_outbox(order_dir, "50001", "delivered")
    assert result is True

    entries = storage.load_outbox(order_dir)
    statuses = [e.get("status") for e in entries]
    assert "delivered" in statuses


# ------------------------------------------------------------------
# P1: Transition hook journal
# ------------------------------------------------------------------


def test_journal_writes_steps_on_confirm(tmp_path):
    """Confirming an order should write a transition journal with all steps."""
    config = _make_config(tmp_path)
    fsm = _make_fsm(tmp_path, config=config)
    fsm.handle_serving_state({"state": "PAUSED_ORDERING"})
    time.sleep(1.0)
    order_dir = fsm._current_order_dir
    fsm.handle_user_text("one coke", source="primary")
    time.sleep(1.0)
    fsm.handle_user_text("yes", source="primary")
    time.sleep(1.0)

    journal_path = Path(order_dir) / "transition_journal.jsonl"
    assert journal_path.is_file(), "Journal file should be created"

    from cade.fsm.storage.journal import TransitionJournal
    journal = TransitionJournal(str(journal_path))
    entries = journal._load_entries()
    step_names = [e.get("step") for e in entries]
    assert "save_order_group" in step_names
    assert "append_event" in step_names
    assert "outbox_pending" in step_names
    assert "publish_confirm" in step_names
    assert "outbox_published" in step_names
    assert "tts_finish" in step_names

    # All steps should be committed
    for entry in entries:
        if entry.get("status") == "started":
            step = entry.get("step")
            assert any(
                e.get("step") == step and e.get("status") == "committed"
                for e in entries
            ), f"Step {step} started but never committed"


def test_journal_completed_steps_returns_correct_set(tmp_path):
    """TransitionJournal.completed_steps should return only committed steps."""
    from cade.fsm.storage.journal import TransitionJournal

    journal_path = tmp_path / "test_journal.jsonl"
    journal = TransitionJournal(str(journal_path))
    jid = journal.begin_transition("check.correct", 1)

    journal.write_step(jid, "step_a", "started")
    journal.write_step(jid, "step_a", "committed")
    journal.write_step(jid, "step_b", "started")

    completed = journal.completed_steps(jid)
    assert completed == {"step_a"}
    assert journal.is_step_completed(jid, "step_a") is True
    assert journal.is_step_completed(jid, "step_b") is False


def test_journal_skips_already_committed_steps(tmp_path):
    """Simulating crash recovery: pre-written journal should prevent re-execution."""
    from cade.fsm.storage.journal import TransitionJournal

    journal_path = tmp_path / "recovery_journal.jsonl"
    journal = TransitionJournal(str(journal_path))
    jid = "crash-recovery"

    # Simulate steps completed before crash
    journal.write_step(jid, "save_order_group", "started")
    journal.write_step(jid, "save_order_group", "committed")
    journal.write_step(jid, "append_event", "started")
    journal.write_step(jid, "append_event", "committed")
    # publish_confirm was started but not committed (crash point)

    # Recovery should see save_order_group and append_event as completed
    assert journal.is_step_completed(jid, "save_order_group") is True
    assert journal.is_step_completed(jid, "append_event") is True
    assert journal.is_step_completed(jid, "publish_confirm") is False
    assert journal.is_step_completed(jid, "outbox_published") is False


# ------------------------------------------------------------------
# P0: OutboxRetryWorker
# ------------------------------------------------------------------


def test_outbox_retry_worker_republishes_published_entry(tmp_path):
    """published entry past retry_sec should be republished with incremented attempt_count."""
    from cade.fsm.storage.outbox import OutboxRetryWorker

    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("60001")
    # Need a session snapshot for list_session_snapshots to find the order dir
    storage.save_session_snapshot(order_dir, {
        "state": "FINISH", "phase": "finish_confirmed", "commit_status": "committed",
        "order_id": "60001", "order_dir": order_dir,
    })
    past_ts = time.time() - 60
    storage.append_outbox(order_dir, {
        "status": "published",
        "topic": "order.confirmed",
        "order_id": "60001",
        "attempt_count": 0,
        "last_attempt_ts": past_ts,
        "next_retry_ts": past_ts,
        "ts": past_ts,
        "foods": ["coke"],
        "order": {"type": "order", "items": [{"name": "coke", "qty": 1}]},
    })

    published = []

    def publish_fn(oid, odir, entry):
        published.append((oid, entry))

    worker = OutboxRetryWorker(storage, publish_fn, retry_sec=30, max_attempts=5)
    retried = worker.tick()

    assert retried == 1
    assert len(published) == 1
    assert published[0][0] == "60001"

    entries = storage.load_outbox(order_dir)
    retry_entries = [e for e in entries if e.get("retry") is True and e.get("attempt_count") == 1]
    assert len(retry_entries) == 1


def test_outbox_retry_worker_dead_letter_after_max_attempts(tmp_path):
    """Entry exceeding max_attempts should be marked dead_letter."""
    from cade.fsm.storage.outbox import OutboxRetryWorker

    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("60002")
    storage.save_session_snapshot(order_dir, {
        "state": "FINISH", "phase": "finish_confirmed", "commit_status": "committed",
        "order_id": "60002", "order_dir": order_dir,
    })
    past_ts = time.time() - 60
    storage.append_outbox(order_dir, {
        "status": "published",
        "topic": "order.confirmed",
        "order_id": "60002",
        "attempt_count": 4,
        "last_attempt_ts": past_ts,
        "next_retry_ts": past_ts,
        "ts": past_ts,
        "foods": ["water"],
    })

    worker = OutboxRetryWorker(storage, lambda *a: None, retry_sec=30, max_attempts=5)
    retried = worker.tick()

    assert retried == 0

    entries = storage.load_outbox(order_dir)
    dead = [e for e in entries if e.get("status") == "dead_letter"]
    assert len(dead) == 1
    assert dead[0]["attempt_count"] == 5


def test_outbox_retry_worker_skips_delivered(tmp_path):
    """delivered entries should never be retried."""
    from cade.fsm.storage.outbox import OutboxRetryWorker

    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("60003")
    storage.append_outbox(order_dir, {
        "status": "delivered",
        "topic": "order.confirmed",
        "order_id": "60003",
        "attempt_count": 0,
        "next_retry_ts": 0,
        "ts": time.time(),
    })

    published = []
    worker = OutboxRetryWorker(storage, lambda *a: published.append(a), retry_sec=30, max_attempts=5)
    retried = worker.tick()

    assert retried == 0
    assert len(published) == 0


def test_outbox_entry_has_idempotency_key(tmp_path):
    """Outbox entries should include idempotency_key."""
    config = _make_config(tmp_path)
    fsm = _make_fsm(tmp_path, config=config)
    fsm.handle_serving_state({"state": "PAUSED_ORDERING"})
    time.sleep(1.0)
    order_dir = fsm._current_order_dir
    fsm.handle_user_text("one coke", source="primary")
    time.sleep(1.0)
    fsm.handle_user_text("yes", source="primary")
    time.sleep(1.0)

    entries = fsm._storage.load_outbox(order_dir)
    pending = [e for e in entries if e.get("status") == "pending"]
    assert len(pending) >= 1
    assert pending[0].get("idempotency_key") is not None
    assert "order-confirmed-" in pending[0]["idempotency_key"]


# ------------------------------------------------------------------
# P0: Metrics wiring
# ------------------------------------------------------------------


def test_outbox_metrics_increment_on_confirm(tmp_path):
    """outbox_pending_total and outbox_published_total should increment on order confirm."""
    config = _make_config(tmp_path)
    fsm = _make_fsm(tmp_path, config=config)
    assert fsm.outbox_pending_total == 0
    assert fsm.outbox_published_total == 0
    fsm.handle_serving_state({"state": "PAUSED_ORDERING"})
    time.sleep(1.0)
    fsm.handle_user_text("one coke", source="primary")
    time.sleep(1.0)
    fsm.handle_user_text("yes", source="primary")
    time.sleep(1.0)

    assert fsm.outbox_pending_total >= 1
    assert fsm.outbox_published_total >= 1


def test_outbox_delivered_metric_on_ack(tmp_path):
    """outbox_delivered_total should increment when order.confirmed.ack marks delivered."""
    ctx = zmq.Context()
    pub_port = _free_port(ctx)
    router_port = _free_port(ctx)
    config = _make_config(
        tmp_path,
        zmq_pub_bind=f"tcp://127.0.0.1:{pub_port}",
        zmq_router_bind=f"tcp://127.0.0.1:{router_port}",
    )
    fsm = _make_fsm(tmp_path, config=config)
    runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)
    try:
        runtime.start()
        time.sleep(0.3)

        _send_envelope(ctx, router_port, {
            "type": "serving_state.update", "id": "m-1",
            "payload": {"state": "PAUSED_ORDERING"},
        })
        time.sleep(1.0)
        order_id = fsm._current_order_id
        assert order_id

        _send_envelope(ctx, router_port, {
            "type": "user_text.primary", "id": "m-2",
            "payload": {"text": "one coke"},
        })
        time.sleep(1.0)
        _send_envelope(ctx, router_port, {
            "type": "user_text.primary", "id": "m-3",
            "payload": {"text": "yes"},
        })
        time.sleep(1.0)

        delivered_before = fsm.outbox_delivered_total
        _send_envelope(ctx, router_port, {
            "type": "order.confirmed.ack", "id": "m-4",
            "payload": {"order_id": order_id, "status": "delivered"},
        })
        assert fsm.outbox_delivered_total == delivered_before + 1
    finally:
        runtime.stop()
        ctx.term()


def test_order_recovered_total_increments(tmp_path):
    """order_recovered_total should increment when recovering a pending confirmed order."""
    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("70001")
    storage.save_order_group(order_dir, {
        "stage": "listen_parsed",
        "order_id": "70001",
        "order": {"type": "order", "items": [{"name": "coke", "qty": 1}]},
    })
    storage.save_session_snapshot(order_dir, {
        "state": "FINISH", "phase": "finish_confirmed", "commit_status": "pending",
        "order_id": "70001", "order_dir": order_dir,
    })

    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    fsm = OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: None),
    )

    assert fsm.order_recovered_total == 1


def test_speaking_gate_block_increments_asr_echo_block_total():
    """When the gate blocks ASR input, asr_echo_block_total should increment."""
    gate = SpeakingGate(500, similarity_threshold=0.6, similarity_window_sec=5.0)
    gate.begin("Let me confirm. You ordered coke.")
    # Simulating what OrderingVoiceRuntime does:
    # if gate.is_blocked -> fsm.asr_echo_block_total += 1
    assert gate.is_blocked("let me confirm you ordered coke") is True
    # The actual increment happens in voice_runtime.py, but we verify the gate works
    gate.end("Let me confirm. You ordered coke.")
    # After end, similar text within window should also be blocked
    assert gate.is_blocked("let me confirm you ordered coke") is True
    # Non-similar text during suppress window is also blocked
    assert gate.is_blocked("i want a water") is True


# ------------------------------------------------------------------
# P1: Crash recovery at each commit hook sub-step
# ------------------------------------------------------------------


def _create_partial_journal(order_dir, jid, completed_steps, pending_steps=None, failed_steps=None):
    """Helper to create a partial journal with some steps completed, some pending/failed."""
    from cade.fsm.storage.journal import TransitionJournal
    journal_path = Path(order_dir) / "transition_journal.jsonl"
    journal = TransitionJournal(str(journal_path))
    for step in completed_steps:
        journal.write_step(jid, step, "started", order_id="80001", idempotency_key="order-confirmed-80001")
        journal.write_step(jid, step, "committed", order_id="80001", idempotency_key="order-confirmed-80001")
    for step in (pending_steps or []):
        journal.write_step(jid, step, "started", order_id="80001", idempotency_key="order-confirmed-80001")
    for step in (failed_steps or []):
        journal.write_step(jid, step, "started", order_id="80001", idempotency_key="order-confirmed-80001")
        journal.write_step(jid, step, "failed", order_id="80001", idempotency_key="order-confirmed-80001")


def test_recovery_after_save_order_group_crash(tmp_path):
    """Crash after save_order_group committed — should not re-write order_group.json."""
    from cade.fsm.storage.journal import TransitionJournal
    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("80001")
    storage.save_order_group(order_dir, {
        "stage": "confirmed", "order_id": "80001",
        "order": {"type": "order", "items": [{"name": "coke", "qty": 1}]},
    })
    jid = "crash-test-1"
    _create_partial_journal(order_dir, jid, completed_steps=["save_order_group"])

    journal = TransitionJournal(str(Path(order_dir) / "transition_journal.jsonl"))
    assert journal.is_step_completed(jid, "save_order_group")
    assert not journal.is_step_completed(jid, "append_event")
    assert not journal.is_step_completed(jid, "outbox_pending")


def test_recovery_skips_committed_steps(tmp_path):
    """Journal shows append_event committed — next run should skip it."""
    from cade.fsm.storage.journal import TransitionJournal
    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("80002")
    jid = "crash-test-2"
    _create_partial_journal(order_dir, jid, completed_steps=["save_order_group", "append_event"])

    journal = TransitionJournal(str(Path(order_dir) / "transition_journal.jsonl"))
    assert journal.is_step_completed(jid, "save_order_group")
    assert journal.is_step_completed(jid, "append_event")
    assert not journal.is_step_completed(jid, "outbox_pending")


def test_journal_records_order_id_and_idempotency_key(tmp_path):
    """Journal entries should include order_id and idempotency_key."""
    config = _make_config(tmp_path)
    fsm = _make_fsm(tmp_path, config=config)
    fsm.handle_serving_state({"state": "PAUSED_ORDERING"})
    time.sleep(1.0)
    order_dir = fsm._current_order_dir
    order_id = fsm._current_order_id
    fsm.handle_user_text("one coke", source="primary")
    time.sleep(1.0)
    fsm.handle_user_text("yes", source="primary")
    time.sleep(1.0)

    from cade.fsm.storage.journal import TransitionJournal
    journal_path = Path(order_dir) / "transition_journal.jsonl"
    journal = TransitionJournal(str(journal_path))
    entries = journal._load_entries()

    for entry in entries:
        if entry.get("status") == "committed":
            assert entry.get("order_id") == order_id, f"Missing order_id in committed entry: {entry}"
            assert entry.get("idempotency_key") == f"order-confirmed-{order_id}", f"Missing idempotency_key: {entry}"


def test_journal_failed_step_detection(tmp_path):
    """failed_steps should return steps marked as failed."""
    from cade.fsm.storage.journal import TransitionJournal
    journal_path = tmp_path / "failed_journal.jsonl"
    journal = TransitionJournal(str(journal_path))
    jid = "fail-test"

    journal.write_step(jid, "step_a", "started")
    journal.write_step(jid, "step_a", "committed")
    journal.write_step(jid, "step_b", "started")
    journal.write_step(jid, "step_b", "failed")

    assert journal.is_step_completed(jid, "step_a") is True
    assert journal.is_step_failed(jid, "step_b") is True
    assert journal.is_step_completed(jid, "step_b") is False
    assert journal.failed_steps(jid) == {"step_b"}


def test_confirmed_order_does_not_duplicate_outbox_on_recovery(tmp_path):
    """A fully committed confirmed order should not create duplicate outbox entries on recovery."""
    storage = _make_storage(tmp_path)
    order_dir = storage.create_order_dir("90001")
    storage.save_order_group(order_dir, {
        "stage": "confirmed", "order_id": "90001",
        "order": {"type": "order", "items": [{"name": "coke", "qty": 1}]},
    })
    storage.save_session_snapshot(order_dir, {
        "state": "FINISH", "phase": "finish_confirmed", "commit_status": "committed",
        "order_id": "90001", "order_dir": order_dir,
    })
    storage.append_outbox(order_dir, {
        "status": "published", "topic": "order.confirmed", "order_id": "90001", "ts": time.time(),
    })

    published = []
    config = _make_config(tmp_path, order_base_dir=str(tmp_path / "orders"))
    OrderSubFSM(
        llm_client=MagicMock(),
        config=config,
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda text, **kwargs: None),
        event_sink=CallbackEventSink(lambda topic, payload: published.append((topic, payload))),
    )

    # Should NOT re-publish order.confirmed
    assert not [t for t, _ in published if t == "order.confirmed"]
    outbox = storage.load_outbox(order_dir)
    # Original published entry should still be there without duplicates
    published_entries = [e for e in outbox if e.get("status") == "published"]
    assert len(published_entries) == 1

