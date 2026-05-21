"""Tests for ZmqRuntime — uses real ZeroMQ with temporary ports."""

import json
import threading
import time
import pytest
from unittest.mock import MagicMock

import zmq

from cade.brain.schemas import OrderAction, OrderItem, OrderCheckDecision, OrderSpeakDecision, SpeakAction
from cade.fsm.config import OrderFSMConfig
from cade.fsm.order_fsm import OrderSubFSM, LocalOrderIdProvider, CallbackTTSSink, CallbackEventSink
from cade.fsm.zmq_runtime import ZmqRuntime


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _free_port(ctx):
    """Find a free TCP port by binding to port 0."""
    sock = ctx.socket(zmq.PUSH)
    sock.bind("tcp://127.0.0.1:0")
    port = sock.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1]
    sock.close()
    return int(port)


def _make_config(pub_port, router_port) -> OrderFSMConfig:
    return OrderFSMConfig(
        order_base_dir="/tmp/test_orders_zmq",
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
        zmq_pub_bind=f"tcp://127.0.0.1:{pub_port}",
        zmq_router_bind=f"tcp://127.0.0.1:{router_port}",
        zmq_heartbeat_sec=0.5,
    )


def _make_fsm(config, llm=None):
    llm = llm or MagicMock()
    storage = MagicMock()
    storage.load_known_ids.return_value = set()
    storage.create_order_dir.return_value = "/tmp/test_orders_zmq/00001"

    return OrderSubFSM(
        llm_client=llm,
        config=config,
        order_id_provider=LocalOrderIdProvider(),
        order_storage=storage,
        tts_sink=CallbackTTSSink(lambda t: None),
        event_sink=CallbackEventSink(lambda t, p: None),
    )


def _send_command(ctx, port, msg_type, payload=None, timeout=2000):
    """Send a command to the ROUTER socket and wait for ACK."""
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, b"test-client")
    sock.setsockopt(zmq.RCVTIMEO, timeout)
    sock.setsockopt(zmq.SNDTIMEO, 1000)
    sock.connect(f"tcp://127.0.0.1:{port}")

    try:
        envelope = {
            "v": 1,
            "type": msg_type,
            "id": "test-001",
            "ts": time.time(),
            "source": "test",
            "session_id": 0,
            "payload": payload or {},
        }
        raw = json.dumps(envelope).encode("utf-8")
        sock.send_multipart([b"", raw])

        frames = sock.recv_multipart()
        if len(frames) >= 2:
            return json.loads(frames[-1])
        return None
    finally:
        sock.close()


def _subscribe_and_receive(ctx, port, topic, timeout=3000):
    """Subscribe to a PUB topic and receive one message."""
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO, timeout)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    sub.connect(f"tcp://127.0.0.1:{port}")

    try:
        frames = sub.recv_multipart()
        if len(frames) >= 2:
            return json.loads(frames[1])
        return None
    except zmq.Again:
        return None
    finally:
        sub.close()


# ------------------------------------------------------------------
# Command ACK
# ------------------------------------------------------------------

class TestCommandAck:

    def test_health_get_returns_ok(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            ack = _send_command(ctx, router_port, "health.get")
            assert ack is not None
            assert ack["type"] == "health.get.ack"
            assert ack["payload"]["status"] == "ok"
        finally:
            runtime.stop()
            ctx.term()

    def test_serving_state_update_ack(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            ack = _send_command(ctx, router_port, "serving_state.update", {"state": "IDLE"})
            assert ack is not None
            assert ack["payload"]["ok"] is True
            # Unified ACK schema
            for field in ("ok", "state", "session_id", "last_event_seq", "duplicate"):
                assert field in ack["payload"], f"ACK missing field: {field}"
        finally:
            runtime.stop()
            ctx.term()

    def test_user_text_ack(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            ack = _send_command(ctx, router_port, "user_text.primary", {"text": "hello"})
            assert ack is not None
            assert ack["payload"]["received"] is True
            assert "accepted" in ack["payload"]
            # Unified ACK schema: must include all standard fields
            for field in ("ok", "accepted", "reason", "state", "session_id", "last_event_seq", "duplicate"):
                assert field in ack["payload"], f"ACK payload missing field: {field}"
        finally:
            runtime.stop()
            ctx.term()

    def test_snapshot_get_returns_state(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            ack = _send_command(ctx, router_port, "snapshot.get")
            assert ack is not None
            assert "state_event" in ack["payload"]
            assert "metrics" in ack["payload"]
        finally:
            runtime.stop()
            ctx.term()

    def test_session_cancel_ack(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            ack = _send_command(ctx, router_port, "session.cancel", {"reason": "test"})
            assert ack is not None
            assert ack["payload"]["ok"] is True
        finally:
            runtime.stop()
            ctx.term()

    def test_unknown_command_returns_error(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            ack = _send_command(ctx, router_port, "unknown.type")
            assert ack is not None
            assert "error" in ack["payload"]
        finally:
            runtime.stop()
            ctx.term()


# ------------------------------------------------------------------
# PUB events
# ------------------------------------------------------------------

class TestPubEvents:

    def test_serving_state_triggers_state_event(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.5)

            # Subscribe to all events
            sub = ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.RCVTIMEO, 5000)
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(f"tcp://127.0.0.1:{pub_port}")

            # Give SUB time to connect
            time.sleep(0.5)

            # Trigger a state change
            _send_command(ctx, router_port, "serving_state.update", {"state": "PAUSED_ORDERING"})
            time.sleep(1.0)

            # Should receive at least one state event
            received = []
            while True:
                try:
                    frames = sub.recv_multipart(zmq.NOBLOCK)
                    if len(frames) >= 2:
                        received.append(json.loads(frames[1]))
                except zmq.Again:
                    break

            sub.close()

            # We should have received events
            assert len(received) > 0
            # Check that state events were published
            state_events = [e for e in received if e.get("type") == "order.state"]
            assert len(state_events) > 0

        finally:
            runtime.stop()
            ctx.term()

    def test_heartbeat_published(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            sub = ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.RCVTIMEO, 3000)
            sub.setsockopt(zmq.SUBSCRIBE, b"order.heartbeat")
            sub.connect(f"tcp://127.0.0.1:{pub_port}")

            time.sleep(0.2)

            # Wait for heartbeat (config says 0.5s)
            time.sleep(1.5)

            received = []
            while True:
                try:
                    frames = sub.recv_multipart(zmq.NOBLOCK)
                    if len(frames) >= 2:
                        received.append(json.loads(frames[1]))
                except zmq.Again:
                    break

            sub.close()

            assert len(received) >= 1
            assert received[0]["type"] == "order.heartbeat"

        finally:
            runtime.stop()
            ctx.term()


# ------------------------------------------------------------------
# Snapshot state replay
# ------------------------------------------------------------------

class TestSnapshotReplay:

    def test_new_client_can_get_snapshot(self):
        """A new client sends snapshot.get, gets current state, then subscribes."""
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            # Step 1: Get snapshot
            ack = _send_command(ctx, router_port, "snapshot.get")
            assert ack is not None
            snap = ack["payload"]
            assert "state_event" in snap
            assert snap["state_event"]["state"] == "NOT_PERMITTED"

            # Step 2: Subscribe (simulating new client)
            sub = ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.RCVTIMEO, 2000)
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(f"tcp://127.0.0.1:{pub_port}")

            # Step 3: Trigger a state change
            time.sleep(0.2)
            _send_command(ctx, router_port, "serving_state.update", {"state": "PAUSED_ORDERING"})
            time.sleep(1.0)

            # Step 4: Should receive the new state event
            received = []
            while True:
                try:
                    frames = sub.recv_multipart(zmq.NOBLOCK)
                    if len(frames) >= 2:
                        received.append(json.loads(frames[1]))
                except zmq.Again:
                    break

            sub.close()

            assert len(received) > 0

        finally:
            runtime.stop()
            ctx.term()


# ------------------------------------------------------------------
# Multi-client subscription
# ------------------------------------------------------------------

class TestMultiClientSubscription:

    def test_multiple_subscribers_receive_events(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.5)

            subs = []
            for _ in range(3):
                sub = ctx.socket(zmq.SUB)
                sub.setsockopt(zmq.RCVTIMEO, 5000)
                sub.setsockopt(zmq.SUBSCRIBE, b"")
                sub.connect(f"tcp://127.0.0.1:{pub_port}")
                subs.append(sub)

            time.sleep(0.5)

            _send_command(ctx, router_port, "serving_state.update", {"state": "PAUSED_ORDERING"})
            time.sleep(1.5)

            for sub in subs:
                received = []
                while True:
                    try:
                        frames = sub.recv_multipart(zmq.NOBLOCK)
                        if len(frames) >= 2:
                            received.append(json.loads(frames[1]))
                    except zmq.Again:
                        break
                assert len(received) > 0, "Subscriber should have received events"
                sub.close()

        finally:
            runtime.stop()
            ctx.term()


# ------------------------------------------------------------------
# Invalid JSON
# ------------------------------------------------------------------

class TestInvalidJSON:

    def test_invalid_json_returns_error(self):
        ctx = zmq.Context()
        pub_port = _free_port(ctx)
        router_port = _free_port(ctx)
        config = _make_config(pub_port, router_port)
        fsm = _make_fsm(config)
        runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)

        try:
            runtime.start()
            time.sleep(0.3)

            sock = ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.IDENTITY, b"bad-client")
            sock.setsockopt(zmq.RCVTIMEO, 2000)
            sock.setsockopt(zmq.SNDTIMEO, 1000)
            sock.connect(f"tcp://127.0.0.1:{router_port}")

            sock.send_multipart([b"", b"not valid json{{{"])
            frames = sock.recv_multipart()
            ack = json.loads(frames[-1])
            assert "error" in ack["type"] or "error" in ack.get("payload", {})

            sock.close()
        finally:
            runtime.stop()
            ctx.term()


def test_events_get_since_pagination_fields(tmp_path):
    """events.get_since should return from_seq, to_seq, next_from_seq, count."""
    ctx = zmq.Context()
    pub_port = _free_port(ctx)
    router_port = _free_port(ctx)
    config = _make_config(pub_port, router_port)
    fsm = _make_fsm(config)
    runtime = ZmqRuntime(fsm=fsm, pub_bind=config.zmq_pub_bind, router_bind=config.zmq_router_bind)
    try:
        runtime.start()
        time.sleep(0.3)
        _send_command(ctx, router_port, "serving_state.update", {"state": "PAUSED_ORDERING"})
        time.sleep(0.5)

        # Use a unique id to avoid idempotency cache hit
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, b"pg-client")
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(f"tcp://127.0.0.1:{router_port}")
        envelope = {
            "v": 1, "type": "events.get_since",
            "id": "pg-unique-001", "ts": time.time(), "source": "test",
            "payload": {"last_event_seq": 0},
        }
        sock.send_multipart([b"", json.dumps(envelope).encode("utf-8")])
        frames = sock.recv_multipart()
        replay = json.loads(frames[-1])
        sock.close()

        payload = replay.get("payload", replay)
        assert "from_seq" in payload, f"Missing from_seq in: {list(payload.keys())}"
        assert "to_seq" in payload, f"Missing to_seq in: {list(payload.keys())}"
        assert "next_from_seq" in payload, f"Missing next_from_seq in: {list(payload.keys())}"
        assert "count" in payload, f"Missing count in: {list(payload.keys())}"
        assert payload["from_seq"] == 0
        assert payload["count"] == len(payload["events"])
        if payload["events"]:
            assert payload["to_seq"] == payload["events"][-1]["event_seq"]
            assert payload["next_from_seq"] == payload["to_seq"] + 1
    finally:
        runtime.stop()
        ctx.term()
