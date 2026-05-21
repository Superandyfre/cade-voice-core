#!/usr/bin/env python3
"""Real-device CADE ordering E2E harness."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import zmq

from cade.config import Config
from cade.fsm.voice_runtime import (
    VoiceRuntimeError,
    build_ordering_voice_runtime,
    configure_runtime_mode,
    load_sounddevice,
    print_asr_input,
    print_tts_output,
)

logger = logging.getLogger(__name__)


class E2EFailure(RuntimeError):
    """Raised when a real dependency or verification condition fails."""


class E2EMonitor:
    """Thread-safe record of evidence required for a real E2E pass."""

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self.current_state = "NOT_PERMITTED"
        self.states_seen = set()
        self.transcript_count = 0
        self.order_confirmed: Optional[Dict[str, Any]] = None
        self.last_tts_completed: Optional[Dict[str, Any]] = None
        self.final_not_permitted = False

    def record_transcript(self, text: str) -> None:
        with self._lock:
            self.transcript_count += 1

    def record_event(self, msg: Dict[str, Any]) -> None:
        payload = msg.get("payload", {})
        msg_type = msg.get("type", "")
        with self._lock:
            if msg_type == "order.state":
                state = str(payload.get("state", ""))
                self.current_state = state
                if state:
                    self.states_seen.add(state)
                if state == "NOT_PERMITTED" and self.order_confirmed:
                    self.final_not_permitted = True
            elif msg_type == "tts.completed":
                self.last_tts_completed = dict(payload)
            elif msg_type == "order.confirmed":
                self.order_confirmed = dict(payload)

    def can_accept_transcript(self) -> bool:
        with self._lock:
            return self.current_state in {"LISTEN", "CHECK"}

    def is_complete(self) -> bool:
        with self._lock:
            required_states = {"LISTEN", "CHECK", "FINISH"}
            return (
                self.transcript_count > 0
                and required_states.issubset(self.states_seen)
                and self.order_confirmed is not None
                and self.last_tts_completed is not None
                and self.final_not_permitted
            )

    def waiting_for(self) -> str:
        with self._lock:
            missing = []
            for state in ("LISTEN", "CHECK", "FINISH"):
                if state not in self.states_seen:
                    missing.append(state)
            if self.transcript_count == 0:
                missing.append("microphone transcript")
            if self.order_confirmed is None:
                missing.append("order.confirmed")
            if self.last_tts_completed is None:
                missing.append("tts.completed")
            if not self.final_not_permitted:
                missing.append("final NOT_PERMITTED")
            return ", ".join(missing) or "verification"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cade-order-e2e",
        description="Real microphone/speaker ordering E2E verification.",
    )
    parser.add_argument("--pub", default="tcp://127.0.0.1:5555")
    parser.add_argument("--router", default="tcp://127.0.0.1:5556")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--log-level", default="INFO")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="Use local OpenAI-compatible LLM.")
    mode.add_argument("--cloud", action="store_true", help="Use cloud LLM.")
    return parser


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def _envelope(msg_type: str, payload: Optional[dict] = None, source: str = "e2e") -> bytes:
    return json.dumps(
        {
            "v": 1,
            "type": msg_type,
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "source": source,
            "session_id": 0,
            "payload": payload or {},
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _send_command(ctx: zmq.Context, router_addr: str, msg_type: str, payload: Optional[dict] = None, source: str = "e2e") -> dict:
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, f"e2e-{uuid.uuid4().hex[:8]}".encode("ascii"))
    sock.connect(router_addr)
    try:
        sock.send_multipart([b"", _envelope(msg_type, payload, source=source)])
        frames = sock.recv_multipart()
        if len(frames) < 2:
            raise E2EFailure(f"Malformed ACK for {msg_type}: {frames!r}")
        ack = json.loads(frames[-1])
        if "error" in ack.get("payload", {}):
            raise E2EFailure(f"Command {msg_type} failed: {ack['payload']['error']}")
        return ack
    finally:
        sock.close()


def _recv_pub_event(sub: zmq.Socket) -> Dict[str, Any]:
    frames = sub.recv_multipart()
    if len(frames) < 2:
        raise E2EFailure(f"Malformed PUB event: {frames!r}")
    return json.loads(frames[1])


def _wait_for_first_heartbeat(sub: zmq.Socket) -> None:
    print("Waiting for ZeroMQ subscriber heartbeat...")
    while True:
        msg = _recv_pub_event(sub)
        if msg.get("type") == "order.heartbeat":
            print("Subscriber is receiving voice-core events.")
            return


def _verify_order_file(payload: Dict[str, Any]) -> None:
    order_dir = payload.get("order_dir")
    order_id = payload.get("order_id")
    if not order_dir or not order_id:
        raise E2EFailure("order.confirmed is missing order_dir or order_id")

    target = Path(order_dir) / "order_group.json"
    if not target.is_file():
        raise E2EFailure(f"Missing order file: {target}")

    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("order_id") != order_id:
        raise E2EFailure("order_group.json order_id does not match order.confirmed")
    if data.get("recognized_text") != payload.get("recognized_text"):
        raise E2EFailure("order_group.json recognized_text does not match order.confirmed")
    if data.get("check_text") != payload.get("check_text"):
        raise E2EFailure("order_group.json check_text does not match order.confirmed")

    file_items = data.get("order", {}).get("items", [])
    file_foods = [item.get("name") for item in file_items]
    if file_foods != payload.get("foods"):
        raise E2EFailure("order_group.json foods do not match order.confirmed")


def _verify_final_snapshot(ctx: zmq.Context, router_addr: str) -> None:
    ack = _send_command(ctx, router_addr, "snapshot.get")
    state = ack.get("payload", {}).get("state_event", {}).get("state")
    if state != "NOT_PERMITTED":
        raise E2EFailure(f"Final snapshot state is {state}, expected NOT_PERMITTED")


def _print_event(msg: Dict[str, Any]) -> None:
    msg_type = msg.get("type", "")
    payload = msg.get("payload", {})
    if msg_type == "order.state":
        print(f"[STATE] {payload.get('state')} ({payload.get('reason')})")
    elif msg_type == "tts.request":
        print(f"[TTS] request: {payload.get('text')}")
    elif msg_type == "tts.completed":
        print(
            "[TTS] completed: "
            f"playback={payload.get('playback_duration_s')}s "
            f"audio={payload.get('audio_duration_s')}s"
        )
    elif msg_type == "tts.failed":
        print(f"[TTS] failed: {payload.get('error')}")
    elif msg_type == "order.confirmed":
        print(f"[ORDER] confirmed: foods={payload.get('foods')} order_id={payload.get('order_id')}")
    elif msg_type == "order.error":
        print(f"[ERROR] {payload.get('stage')}: {payload.get('error')}")


def run_real_e2e(args: argparse.Namespace) -> int:
    configure_runtime_mode(
        cloud=args.cloud,
        local=args.local,
        input_device=args.input_device,
        output_device=args.output_device,
    )

    warnings = Config.validate()
    for warning in warnings:
        print(f"WARNING: {warning}")

    monitor = E2EMonitor()
    runtime = build_ordering_voice_runtime(
        pub_bind=args.pub,
        router_bind=args.router,
        input_device=args.input_device,
        output_device=args.output_device,
        monitor=monitor,
        verify_audio=True,
    )

    sd = load_sounddevice()
    print("Real audio devices verified.")
    print_asr_input(sd, runtime.input_source)
    print_tts_output(sd, runtime.output)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    try:
        runtime.start()
        sub.connect(args.pub)
        _wait_for_first_heartbeat(sub)

        print("Triggering PAUSED_ORDERING. Speak only after LISTEN is printed.")
        _send_command(
            ctx,
            args.router,
            "serving_state.update",
            {"state": "PAUSED_ORDERING", "customer_id": "audio_e2e"},
        )

        last_waiting = ""
        while True:
            if not runtime.asr_errors.empty():
                raise E2EFailure(f"ASR failed: {runtime.asr_errors.get()}")

            msg = _recv_pub_event(sub)
            _print_event(msg)
            if msg.get("type") == "order.error":
                payload = msg.get("payload", {})
                raise E2EFailure(f"FSM error at {payload.get('stage')}: {payload.get('error')}")
            if msg.get("type") == "tts.failed":
                raise E2EFailure(f"TTS failed: {msg.get('payload', {}).get('error')}")

            monitor.record_event(msg)
            waiting = monitor.waiting_for()
            if waiting != last_waiting and msg.get("type") == "order.heartbeat":
                print(f"Waiting for: {waiting}")
                last_waiting = waiting

            if monitor.is_complete():
                payload = monitor.order_confirmed or {}
                _verify_order_file(payload)
                _verify_final_snapshot(ctx, args.router)
                print("Real device E2E passed.")
                print(f"Order ID: {payload.get('order_id')}")
                print(f"Foods: {payload.get('foods')}")
                return 0
    finally:
        sub.close()
        ctx.term()
        runtime.stop()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)

    try:
        raise SystemExit(run_real_e2e(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except (E2EFailure, VoiceRuntimeError) as exc:
        print(f"Real device E2E failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
