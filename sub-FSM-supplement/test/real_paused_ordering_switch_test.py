#!/usr/bin/env python3
"""Manual Y/N switch controller for PAUSED_ORDERING.

Y: publish PAUSED_ORDERING
N: publish TRACKING
Q: quit
"""

import json
import time
import argparse

import rospy
from std_msgs.msg import String


class ManualSwitch:
    def __init__(self, state_topic: str, subfsm_topic: str):
        self.state_topic = state_topic
        self.subfsm_topic = subfsm_topic
        self.pub = rospy.Publisher(self.state_topic, String, queue_size=20)
        rospy.Subscriber(self.subfsm_topic, String, self._on_subfsm_state, queue_size=200)

    def _on_subfsm_state(self, msg: String):
        raw = str(msg.data or "").strip()
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                state = str(payload.get("state", "")).strip().upper()
                serving = str(payload.get("serving_state", "")).strip().upper()
                reason = str(payload.get("reason", "")).strip()
                ts = payload.get("timestamp", "")
                print(
                    f"[manual-switch] subfsm_ack state={state or '-'} "
                    f"serving={serving or '-'} reason={reason or '-'} ts={ts}"
                )
                return
        except Exception:
            pass
        print(f"[manual-switch] subfsm_ack raw={raw}")

    def publish_state(self, state: str, customer_id: str):
        payload = {
            "timestamp": time.time(),
            "state": str(state).strip().upper(),
            "customer_id": customer_id,
            "customer_no": customer_id,
            "folder": "",
            "source": "manual_y_n_switch",
        }
        self.pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        print(f"[manual-switch] publish state={payload['state']} customer_id={customer_id}")


def parse_args():
    parser = argparse.ArgumentParser(description="Manual Y/N PAUSED_ORDERING switch")
    parser.add_argument(
        "--state-topic",
        type=str,
        default="/person_following/serving_customer_state",
        help="serving_customer_state topic",
    )
    parser.add_argument(
        "--subfsm-topic",
        type=str,
        default="/mhrc/order_subfsm_state",
        help="order sub-FSM state topic",
    )
    parser.add_argument(
        "--customer-id",
        type=str,
        default="manual",
        help="customer_id/customer_no used in published payload",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("manual_paused_ordering_switch", anonymous=True)

    ctl = ManualSwitch(args.state_topic, args.subfsm_topic)

    print("[manual-switch] Ready. Input Y/N/Q then press Enter:")
    print("  Y -> PAUSED_ORDERING")
    print("  N -> TRACKING")
    print("  Q -> quit")

    while not rospy.is_shutdown():
        try:
            cmd = input("[manual-switch] command(Y/N/Q): ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n[manual-switch] exit")
            break

        if cmd == "Y":
            ctl.publish_state("PAUSED_ORDERING", args.customer_id)
        elif cmd == "N":
            ctl.publish_state("TRACKING", args.customer_id)
        elif cmd == "Q":
            print("[manual-switch] exit")
            break
        elif not cmd:
            continue
        else:
            print("[manual-switch] invalid input, use Y/N/Q")


if __name__ == "__main__":
    main()
