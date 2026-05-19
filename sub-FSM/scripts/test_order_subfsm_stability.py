#!/usr/bin/env python3
"""
Offline stability test for the MHRC ordering sub-FSM.

This script does not require ROS master. It mocks rospy/std_msgs and runs
end-to-end state transitions on RosVoiceBridge with a fake LLM backend.
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Callable, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def install_fake_ros(params: Dict[str, object], verbose: bool = True):
    fake_rospy = types.ModuleType("rospy")
    fake_rospy._params = dict(params)
    fake_rospy._publishers = {}
    fake_rospy._subscribers = []
    fake_rospy._logs = []

    class FakeTime:
        def __init__(self, secs: float = 0.0):
            self._secs = float(secs)

        def to_sec(self) -> float:
            return self._secs

        @staticmethod
        def now():
            return FakeTime(time.time())

    class FakePublisher:
        def __init__(self, topic, msg_type=None, queue_size=10, latch=False):
            self.topic = topic
            self.msg_type = msg_type
            self.queue_size = queue_size
            self.latch = latch
            self.messages = []
            fake_rospy._publishers[topic] = self

        def publish(self, msg):
            self.messages.append(msg)

    class FakeSubscriber:
        def __init__(self, topic, msg_type, callback, queue_size=10):
            self.topic = topic
            self.msg_type = msg_type
            self.callback = callback
            self.queue_size = queue_size
            fake_rospy._subscribers.append(self)

    def _log(level: str, message: str, *args):
        if args:
            try:
                message = message % args
            except Exception:
                message = f"{message} {args}"
        fake_rospy._logs.append((level, message))
        if verbose:
            print(f"[fake_rospy][{level}] {message}")

    fake_rospy.Time = FakeTime
    fake_rospy.Publisher = FakePublisher
    fake_rospy.Subscriber = FakeSubscriber
    fake_rospy.Duration = lambda x: x
    fake_rospy.init_node = lambda *a, **k: None
    fake_rospy.get_param = lambda name, default=None: fake_rospy._params.get(name, default)
    fake_rospy.loginfo = lambda msg, *a: _log("INFO", msg, *a)
    fake_rospy.logwarn = lambda msg, *a: _log("WARN", msg, *a)
    fake_rospy.logdebug = lambda msg, *a: _log("DEBUG", msg, *a)
    fake_rospy.logerr = lambda msg, *a: _log("ERROR", msg, *a)
    fake_rospy.spin = lambda: None
    fake_rospy.get_node_uri = lambda: "fake://node"
    fake_rospy.ROSInterruptException = RuntimeError

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")

    class String:
        __slots__ = ("data",)

        def __init__(self, data=""):
            self.data = data

    std_msgs_msg.String = String
    std_msgs.msg = std_msgs_msg

    sys.modules["rospy"] = fake_rospy
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg

    return fake_rospy, String


def wait_until(cond: Callable[[], bool], timeout: float, desc: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError(f"Timeout waiting for: {desc}")


def parse_states(state_pub) -> List[str]:
    states = []
    for msg in state_pub.messages:
        try:
            payload = json.loads(msg.data)
        except Exception:
            continue
        state = str(payload.get("state", "")).strip().upper()
        if not state:
            continue
        if not states or states[-1] != state:
            states.append(state)
    return states


def parse_tts_texts(tts_pub) -> List[str]:
    texts = []
    for msg in tts_pub.messages:
        text = getattr(msg, "data", "")
        if isinstance(text, str):
            texts.append(text)
    return texts


def run_suite(rounds: int, verbose: bool = True):
    params = {
        "~order_input_mode": "both",
        "~primary_input_topic": "/asr",
        "~secondary_input_topic": "/person_following/pause_reply_text",
        "~serving_customer_state_topic": "/person_following/serving_customer_state",
        "~order_confirm_topic": "/person_following/order_confirm_json",
        "~order_subfsm_state_topic": "/mhrc/order_subfsm_state",
        "~order_input_dedup_window_sec": 1.2,
    }
    fake_rospy, String = install_fake_ros(params, verbose=verbose)

    if "dotenv" not in sys.modules:
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda *a, **k: None
        sys.modules["dotenv"] = fake_dotenv

    from body.robot_interface import RobotState
    from brain.schemas import OrderAction, OrderCheckDecision, OrderItem, OrderSpeakDecision, SpeakAction, FixOrderAction
    from bridge import ros_voice_bridge as bridge_mod

    class FakeRobot:
        def __init__(self, name="fake"):
            self.name = name
            self.state = RobotState.IDLE

        def set_state(self, state):
            self.state = state

        def is_busy(self):
            return self.state in (RobotState.THINKING, RobotState.SPEAKING, RobotState.EXECUTING)

    class FakeLLM:
        @staticmethod
        def _extract_qty(text: str, keyword: str, default_qty: int = 1) -> int:
            text_l = text.lower()
            idx = text_l.find(keyword)
            if idx < 0:
                return 0
            prefix = text_l[:idx].strip().split()
            if not prefix:
                return default_qty
            token = prefix[-1]
            mapping = {
                "a": 1,
                "an": 1,
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
            }
            if token.isdigit():
                return max(1, int(token))
            return mapping.get(token, default_qty)

        def get_order_action(self, user_input, food_aliases=None, max_retries=3):
            text = str(user_input).lower()
            items = []
            if "coke" in text or "cola" in text:
                qty = self._extract_qty(text, "coke", 1)
                if qty == 0:
                    qty = self._extract_qty(text, "cola", 1)
                items.append(OrderItem(name="cola", qty=max(1, qty)))
            if "burger" in text:
                qty = self._extract_qty(text, "burger", 1)
                items.append(OrderItem(name="burger", qty=max(1, qty)))
            if "water" in text:
                qty = self._extract_qty(text, "water", 1)
                items.append(OrderItem(name="water", qty=max(1, qty)))
            if "tea" in text:
                qty = self._extract_qty(text, "tea", 1)
                items.append(OrderItem(name="tea", qty=max(1, qty)))
            return OrderAction(type="order", items=items)

        def get_order_repeat_speak(self, confirm_instruction, order_action, max_retries=3):
            parts = []
            for it in order_action.items:
                label = it.name.replace("_", " ")
                if it.qty > 1:
                    parts.append(f"{it.qty} {label}")
                else:
                    parts.append(label)
            summary = ", ".join(parts) if parts else "your order"
            content = f"Let me confirm. You ordered {summary}. Is that correct?"
            return OrderSpeakDecision(action=SpeakAction(type="speak", content=content))

        def get_order_check_decision(self, customer_reply, order_action, food_aliases=None, max_retries=3):
            text = str(customer_reply).lower()
            if "yes" in text or "correct" in text or "right" in text:
                return OrderCheckDecision(result="correct", action=None, reply=None)

            if "wrong" in text or "change" in text or "instead" in text:
                if "water" in text:
                    qty = self._extract_qty(text, "water", 1)
                    fix = FixOrderAction(type="fix_order", items=[OrderItem(name="water", qty=max(1, qty))])
                    return OrderCheckDecision(result="wrong", action=fix, reply=None)
                if "coke" in text:
                    qty = self._extract_qty(text, "coke", 1)
                    fix = FixOrderAction(type="fix_order", items=[OrderItem(name="coke", qty=max(1, qty))])
                    return OrderCheckDecision(result="wrong", action=fix, reply=None)
                return OrderCheckDecision(
                    result="wrong",
                    action=None,
                    reply="Please tell me your updated order.",
                )

            return OrderCheckDecision(result="wrong", action=None, reply="Please confirm your order.")

    class FakeRobotController:
        def __init__(self, robot=None, llm_client=None, prompt_mode="default", show_thought=True):
            self.robot = robot or FakeRobot()
            self.llm_client = FakeLLM()
            self.system_prompt = f"fake prompt mode={prompt_mode}"

    bridge_mod.Robot = FakeRobot
    bridge_mod.RobotController = FakeRobotController
    bridge_mod.time.sleep = lambda _: None

    bridge = bridge_mod.RosVoiceBridge(prompt_mode="default", show_thought=False, environment_context="test")

    def set_serving_state(state: str, extra: Dict[str, object] = None):
        payload = {"state": state}
        if extra:
            payload.update(extra)
        bridge._on_serving_state_message(String(json.dumps(payload)))

    def start_order_session(customer_no: str = "1"):
        set_serving_state("TRACKING", {"customer_no": customer_no, "customer_id": customer_no})
        wait_until(lambda: bridge.subfsm_state == "NOT_PERMITTED", 0.5, "state NOT_PERMITTED")
        set_serving_state("PAUSED_ORDERING", {"customer_no": customer_no, "customer_id": customer_no})
        wait_until(lambda: bridge.subfsm_state == "LISTEN", 1.0, "state LISTEN")

    def send_primary(text: str):
        bridge._on_primary_message(String(text))

    def send_secondary(text: str):
        bridge._on_secondary_message(String(text))

    # Case 1: main happy path.
    start_order_session("100")
    send_primary("I'd like two cola and one burger")
    wait_until(lambda: bridge.subfsm_state == "CHECK", 1.0, "state CHECK after LISTEN")
    send_secondary("yes, that is correct")
    wait_until(lambda: bridge.subfsm_state == "NOT_PERMITTED", 1.0, "state NOT_PERMITTED after FINISH")

    # Case 2: wrong with fix_order branch.
    start_order_session("101")
    send_primary("two coke")
    wait_until(lambda: bridge.subfsm_state == "CHECK", 1.0, "state CHECK before fix_order")
    send_secondary("wrong, change to one water")
    wait_until(lambda: bridge.subfsm_state == "CHECK", 1.0, "state CHECK after fix REPEAT")
    send_secondary("yes correct")
    wait_until(lambda: bridge.subfsm_state == "NOT_PERMITTED", 1.0, "state NOT_PERMITTED after fixed finish")

    # Case 3: wrong without fix info goes back to LISTEN.
    start_order_session("102")
    send_primary("one tea")
    wait_until(lambda: bridge.subfsm_state == "CHECK", 1.0, "state CHECK for wrong-without-fix")
    send_secondary("wrong")
    wait_until(lambda: bridge.subfsm_state == "LISTEN", 1.0, "state LISTEN after wrong-without-fix")
    set_serving_state("TRACKING", {"customer_no": "102", "customer_id": "102"})
    wait_until(lambda: bridge.subfsm_state == "NOT_PERMITTED", 0.8, "forced reset to NOT_PERMITTED")

    # Case 4: duplicate text from dual channel should be deduped.
    start_order_session("103")
    ignored_before = bridge.ignored_inputs
    send_primary("one coke")
    send_secondary("one coke")
    wait_until(lambda: bridge.subfsm_state == "CHECK", 1.0, "state CHECK in dedup case")
    send_primary("yes")
    wait_until(lambda: bridge.subfsm_state == "NOT_PERMITTED", 1.0, "state NOT_PERMITTED after dedup case")
    if bridge.ignored_inputs <= ignored_before:
        raise AssertionError("Expected ignored_inputs to increase in dedup case")

    # Stress rounds.
    for i in range(rounds):
        customer_no = str(200 + i)
        start_order_session(customer_no)
        if i % 3 == 0:
            send_primary("two coke and one burger")
        elif i % 3 == 1:
            send_primary("one tea")
        else:
            send_primary("one water")
        wait_until(lambda: bridge.subfsm_state == "CHECK", 1.0, f"stress round {i} -> CHECK")

        if i % 5 == 0:
            send_secondary("wrong, change to one water")
            wait_until(lambda: bridge.subfsm_state == "CHECK", 1.0, f"stress round {i} -> CHECK after fix")

        send_secondary("yes correct")
        wait_until(lambda: bridge.subfsm_state == "NOT_PERMITTED", 1.0, f"stress round {i} finish")

    # Assertions and summary.
    order_msgs = bridge.order_confirm_publisher.messages
    if len(order_msgs) < (rounds + 3):
        raise AssertionError(
            f"Expected at least {rounds + 3} order confirm messages, got {len(order_msgs)}"
        )

    states = parse_states(bridge.subfsm_state_publisher)
    if "ASK" not in states or "LISTEN" not in states or "CHECK" not in states or "FINISH" not in states:
        raise AssertionError(f"Incomplete state coverage: {states}")

    tts_texts = parse_tts_texts(bridge.tts_publisher)
    if not any("What would you like to order" in x for x in tts_texts):
        raise AssertionError("ASK TTS not observed")
    if not any("OK I'll get" in x for x in tts_texts):
        raise AssertionError("FINISH TTS not observed")

    print("\n================= TEST SUMMARY =================")
    print(f"Stress rounds: {rounds}")
    print(f"Total inputs observed by bridge: {bridge.total_inputs}")
    print(f"Ignored inputs: {bridge.ignored_inputs}")
    print(f"Orders confirmed: {bridge.orders_confirmed}")
    print(f"TTS publishes: {len(tts_texts)}")
    print(f"Order confirm publishes: {len(order_msgs)}")
    print(f"State sequence snapshot: {states[:20]} ... total={len(states)}")
    print("Result: PASS")
    print("===============================================\n")


def main():
    parser = argparse.ArgumentParser(description="Offline stability test for order sub-FSM")
    parser.add_argument("--rounds", type=int, default=30, help="Stress rounds (default: 30)")
    parser.add_argument("--quiet", action="store_true", help="Only print final summary")
    args = parser.parse_args()

    try:
        run_suite(max(1, int(args.rounds)), verbose=not args.quiet)
    except Exception as exc:
        print("\n================= TEST SUMMARY =================")
        print(f"Result: FAIL - {exc}")
        print("===============================================\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
