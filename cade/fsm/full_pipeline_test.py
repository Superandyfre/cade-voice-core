#!/usr/bin/env python3
"""
全链路集成测试脚本 — 端到端验证点餐 sub-FSM + ZeroMQ 通信。

使用方法:
    # 1. 先启动 FSM 服务端
    cade-order-fsm

    # 2. 另一个终端运行全链路测试
    python -m cade.fsm.full_pipeline_test

    # 或者指定 ZeroMQ 地址
    python -m cade.fsm.full_pipeline_test --pub tcp://192.168.1.100:5555 --router tcp://192.168.1.100:5556

测试流程:
    1. health.get — 检查服务存活
    2. snapshot.get — 获取初始状态 (NOT_PERMITTED)
    3. serving_state.update(IDLE) — 模拟空闲
    4. serving_state.update(PAUSED_ORDERING) — 触发点餐会话
    5. 订阅 order.state，等待状态进入 LISTEN
    6. user_text.primary("I want a coke and two waters") — 模拟 ASR 输入
    7. 订阅 tts.request，验证 TTS 输出
    8. 订阅 order.state，等待进入 CHECK
    9. user_text.primary("yes, that's correct") — 确认订单
   10. 订阅 order.confirmed，验证订单确认
   11. 验证最终状态回到 NOT_PERMITTED
"""

import argparse
import json
import sys
import time
import uuid

import zmq


def make_envelope(msg_type: str, payload: dict = None) -> bytes:
    """构造标准 JSON envelope。"""
    envelope = {
        "v": 1,
        "type": msg_type,
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "source": "full-pipeline-test",
        "session_id": 0,
        "payload": payload or {},
    }
    return json.dumps(envelope, ensure_ascii=False).encode("utf-8")


def send_command(sock: zmq.Socket, msg_type: str, payload: dict = None, timeout_ms: int = 5000) -> dict:
    """发送命令并等待 ACK。"""
    raw = make_envelope(msg_type, payload)
    sock.send_multipart([b"", raw])

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    events = dict(poller.poll(timeout_ms))

    if sock in events:
        frames = sock.recv_multipart()
        if len(frames) >= 2:
            return json.loads(frames[-1])
    return {"error": "timeout"}


def drain_events(sub: zmq.Socket, timeout_ms: int = 100) -> list:
    """非阻塞地收取所有待处理事件。"""
    received = []
    while True:
        try:
            frames = sub.recv_multipart(zmq.NOBLOCK)
            if len(frames) >= 2:
                received.append(json.loads(frames[1]))
        except zmq.Again:
            break
    return received


def wait_for_event(sub: zmq.Socket, event_type: str, timeout_sec: float = 10.0) -> dict:
    """等待特定类型的事件。"""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        remaining_ms = max(100, int((deadline - time.monotonic()) * 1000))
        sub.setsockopt(zmq.RCVTIMEO, remaining_ms)
        try:
            frames = sub.recv_multipart()
            if len(frames) >= 2:
                msg = json.loads(frames[1])
                if msg.get("type") == event_type:
                    return msg
        except zmq.Again:
            continue
    return None


def wait_for_state(sub: zmq.Socket, target_state: str, timeout_sec: float = 10.0) -> dict:
    """等待 FSM 进入指定状态。"""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        remaining_ms = max(100, int((deadline - time.monotonic()) * 1000))
        sub.setsockopt(zmq.RCVTIMEO, remaining_ms)
        try:
            frames = sub.recv_multipart()
            if len(frames) >= 2:
                msg = json.loads(frames[1])
                if msg.get("type") == "order.state":
                    state = msg.get("payload", {}).get("state", "")
                    print(f"  [STATE] {state} (reason: {msg['payload'].get('reason', '')})")
                    if state == target_state:
                        return msg
        except zmq.Again:
            continue
    return None


def step(num: int, desc: str):
    """打印步骤标题。"""
    print(f"\n{'='*60}")
    print(f"  步骤 {num}: {desc}")
    print(f"{'='*60}")


def ok(msg: str):
    print(f"  ✓ {msg}")


def fail(msg: str):
    print(f"  ✗ {msg}")


def main():
    parser = argparse.ArgumentParser(description="全链路集成测试")
    parser.add_argument("--pub", default="tcp://127.0.0.1:5555", help="PUB 地址")
    parser.add_argument("--router", default="tcp://127.0.0.1:5556", help="ROUTER 地址")
    parser.add_argument("--timeout", type=float, default=10.0, help="事件等待超时(秒)")
    parser.add_argument("--order-text", default="I want a coke and two waters",
                        help="模拟点餐文本")
    parser.add_argument("--confirm-text", default="yes, that's correct",
                        help="模拟确认文本")
    args = parser.parse_args()

    ctx = zmq.Context()

    # ROUTER (DEALER 模式)
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, b"pipeline-test")
    dealer.setsockopt(zmq.RCVTIMEO, 5000)
    dealer.setsockopt(zmq.SNDTIMEO, 2000)
    dealer.connect(args.router)

    # SUB (订阅所有事件)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(args.pub)

    # 等待连接建立
    time.sleep(0.5)

    passed = 0
    failed = 0
    total_steps = 8

    print("\n" + "█" * 60)
    print("  CADE 点餐 Sub-FSM 全链路集成测试")
    print("  PUB:    " + args.pub)
    print("  ROUTER: " + args.router)
    print("█" * 60)

    try:
        # ──────────────────────────────────────────────
        step(1, "health.get — 检查服务存活")
        # ──────────────────────────────────────────────
        ack = send_command(dealer, "health.get")
        if ack.get("payload", {}).get("status") == "ok":
            ok(f"服务状态: {ack['payload']}")
            for field in ("ok", "state", "session_id", "last_event_seq"):
                if field not in ack.get("payload", {}):
                    fail(f"ACK missing field: {field}")
            passed += 1
        else:
            fail(f"健康检查失败: {ack}")
            failed += 1
            print("\n服务未启动? 请先运行: cade-order-fsm")
            sys.exit(1)

        # ──────────────────────────────────────────────
        step(2, "snapshot.get — 获取初始状态")
        # ──────────────────────────────────────────────
        ack = send_command(dealer, "snapshot.get")
        snap = ack.get("payload", {})
        init_state = snap.get("state_event", {}).get("state", "UNKNOWN")
        if init_state == "NOT_PERMITTED":
            ok(f"初始状态: {init_state}")
            passed += 1
        else:
            fail(f"期望 NOT_PERMITTED, 实际: {init_state}")
            failed += 1

        # ──────────────────────────────────────────────
        step(3, "serving_state.update(PAUSED_ORDERING) — 触发点餐")
        # ──────────────────────────────────────────────
        # 清空旧事件
        time.sleep(0.2)
        drain_events(sub)

        ack = send_command(dealer, "serving_state.update", {
            "state": "PAUSED_ORDERING",
            "customer_id": "test_cust_001",
            "customer_no": "5",
        })
        if ack.get("payload", {}).get("ok"):
            ok("已发送 PAUSED_ORDERING")
            passed += 1
        else:
            fail(f"发送失败: {ack}")
            failed += 1

        # ──────────────────────────────────────────────
        step(4, "等待 FSM 进入 LISTEN 状态")
        # ──────────────────────────────────────────────
        state_msg = wait_for_state(sub, "LISTEN", timeout_sec=args.timeout)
        if state_msg:
            ok(f"已进入 LISTEN (order_id={state_msg['payload'].get('order_id', 'N/A')})")
            passed += 1
        else:
            fail(f"超时未进入 LISTEN 状态")
            failed += 1

        # ──────────────────────────────────────────────
        step(5, f"发送点餐文本: \"{args.order_text}\"")
        # ──────────────────────────────────────────────
        # 清空旧事件
        drain_events(sub)

        ack = send_command(dealer, "user_text.primary", {"text": args.order_text})
        if ack.get("payload", {}).get("accepted"):
            ok("点餐文本已发送")
            for field in ("accepted", "reason", "state", "session_id", "last_event_seq", "duplicate", "ok"):
                if field not in ack.get("payload", {}):
                    fail(f"ACK missing field: {field}")
                    break
            passed += 1
        else:
            fail(f"发送失败: {ack}")
            failed += 1

        # ──────────────────────────────────────────────
        step(6, "等待 TTS 确认 + 进入 CHECK 状态")
        # ──────────────────────────────────────────────
        tts_msg = wait_for_event(sub, "tts.request", timeout_sec=args.timeout)
        if tts_msg:
            tts_text = tts_msg.get("payload", {}).get("text", "")
            ok(f"TTS: \"{tts_text}\"")
        else:
            fail("未收到 TTS 请求")
            failed += 1

        state_msg = wait_for_state(sub, "CHECK", timeout_sec=args.timeout)
        if state_msg:
            ok("已进入 CHECK 状态")
            passed += 1
        else:
            fail("超时未进入 CHECK 状态")
            failed += 1

        # ──────────────────────────────────────────────
        step(7, f"发送确认文本: \"{args.confirm_text}\"")
        # ──────────────────────────────────────────────
        drain_events(sub)

        ack = send_command(dealer, "user_text.primary", {"text": args.confirm_text})
        if ack.get("payload", {}).get("accepted"):
            ok("确认文本已发送")
        else:
            fail(f"发送失败: {ack}")

        # ──────────────────────────────────────────────
        step(8, "验证订单确认 + 状态回到 NOT_PERMITTED")
        # ──────────────────────────────────────────────
        confirm_msg = wait_for_event(sub, "order.confirmed", timeout_sec=args.timeout)
        if confirm_msg:
            payload = confirm_msg.get("payload", {})
            ok(f"订单确认!")
            ok(f"  foods: {payload.get('foods', [])}")
            ok(f"  foods_with_qty: {payload.get('foods_with_qty', [])}")
            ok(f"  order_id: {payload.get('order_id', 'N/A')}")
            ok(f"  customer_id: {payload.get('customer_id', 'N/A')}")
            passed += 1
        else:
            fail("未收到订单确认事件")
            failed += 1

        # 等待最终状态
        state_msg = wait_for_state(sub, "NOT_PERMITTED", timeout_sec=args.timeout)
        if state_msg:
            ok("状态已回到 NOT_PERMITTED")
        else:
            fail("状态未回到 NOT_PERMITTED")

        # 收集最终 metrics
        time.sleep(0.5)
        events = drain_events(sub)
        metrics = [e for e in events if e.get("type") == "order.metrics"]
        if metrics:
            m = metrics[-1].get("payload", {})
            ok(f"最终指标: confirmed={m.get('orders_confirmed', 0)}, "
               f"total_inputs={m.get('total_inputs', 0)}")

    except KeyboardInterrupt:
        print("\n\n中断")
    finally:
        dealer.close()
        sub.close()
        ctx.term()

    # ──────────────────────────────────────────────
    # 汇总
    # ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  测试结果: {passed}/{passed + failed} 通过")
    if failed == 0:
        print("  ✓ 全链路测试通过!")
    else:
        print(f"  ✗ {failed} 项失败")
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
