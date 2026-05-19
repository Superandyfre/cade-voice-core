"""
ROS Voice Bridge - ordering sub-state-machine driven by PAUSED_ORDERING.

This bridge acts as the ordering sub-FSM for Task5:
NOT_PERMITTED -> PERMITTED -> ASK -> LISTEN -> REPEAT -> CHECK -> FINISH -> NOT_PERMITTED
"""

import json
import os
import re
import threading
import time
from typing import Dict, List, Optional

import rospy
from std_msgs.msg import String

from body.robot import Robot
from body.robot_interface import RobotState
from brain.schemas import OrderAction, OrderItem
from config import Config
from robot_controller import RobotController


def _default_food_aliases() -> Dict[str, List[str]]:
    return {
        "water": ["water", "bottle of water"],
        "coke": ["coke", "cola", "coca cola"],
        "juice": ["juice", "orange juice", "apple juice"],
        "coffee": ["coffee", "latte", "americano", "cappuccino"],
        "tea": ["tea", "black tea", "green tea", "milk tea"],
        "burger": ["burger", "hamburger", "cheeseburger"],
        "pizza": ["pizza"],
        "sandwich": ["sandwich"],
        "fried_rice": ["fried rice"],
        "noodles": ["noodles", "ramen"],
        "dumplings": ["dumpling", "dumplings"],
        "pasta": ["pasta", "spaghetti"],
        "fries": ["fries", "french fries", "chips"],
        "salad": ["salad"],
        "soup": ["soup"],
    }


class RosVoiceBridge:
    """Ordering sub-FSM bridge for ROS ASR/TTS topics."""

    VALID_INPUT_STATES = {"LISTEN", "CHECK"}

    def __init__(
        self,
        prompt_mode: str = "default",
        show_thought: bool = True,
        environment_context: Optional[str] = None,
    ):
        rospy.init_node("cade_voice_bridge", anonymous=True)

        self.robot = Robot(name=Config.ROBOT_NAME)
        self.controller = RobotController(
            robot=self.robot,
            prompt_mode=prompt_mode,
            show_thought=show_thought,
        )
        if environment_context:
            self._inject_environment_context(environment_context)

        self._state_lock = threading.Lock()
        self._fsm_lock = threading.RLock()

        self.primary_input_topic = rospy.get_param("~primary_input_topic", "/asr")
        self.secondary_input_topic = rospy.get_param(
            "~secondary_input_topic",
            "/person_following/pause_reply_text",
        )
        self.input_channel_mode = str(rospy.get_param("~order_input_mode", "both")).strip().lower()
        if self.input_channel_mode not in ("primary", "secondary", "both"):
            rospy.logwarn("Invalid ~order_input_mode=%s, fallback to both", self.input_channel_mode)
            self.input_channel_mode = "both"

        self.tts_topic = rospy.get_param("~tts_topic", "/tts")
        self.serving_state_topic = rospy.get_param(
            "~serving_customer_state_topic",
            "/person_following/serving_customer_state",
        )
        self.order_confirm_topic = rospy.get_param(
            "~order_confirm_topic",
            "/person_following/order_confirm_json",
        )
        self.subfsm_state_topic = rospy.get_param(
            "~order_subfsm_state_topic",
            "/mhrc/order_subfsm_state",
        )
        self.order_session_id_topic = rospy.get_param(
            "~order_session_id_topic",
            "/mhrc/random_order_id",
        )
        self.order_voice_base_dir = rospy.get_param(
            "~order_voice_base_dir",
            "/home/nvidia/taskfive/voice",
        )
        self.order_id_wait_poll_sec = float(rospy.get_param("~order_id_wait_poll_sec", 0.2))

        self.ask_prompt = rospy.get_param("~order_ask_prompt", "What would you like to order?")
        self.repeat_instruction = rospy.get_param(
            "~order_repeat_instruction",
            "Repeat the order and ask the customer whether it is correct.",
        )
        self.listen_retry_prompt = rospy.get_param(
            "~order_listen_retry_prompt",
            "Sorry, I did not catch your order. Please tell me your order again.",
        )
        self.fix_missing_prompt = rospy.get_param(
            "~order_fix_missing_prompt",
            "Sorry, I didn't catch the changes. Please tell me your updated order.",
        )
        self.check_retry_prompt = rospy.get_param(
            "~order_check_retry_prompt",
            "Please tell me if the order is correct, or say your updated order.",
        )
        self.finish_template = rospy.get_param(
            "~order_finish_template",
            "OK I'll get {foods} for you",
        )
        self.input_dedup_window_sec = float(rospy.get_param("~order_input_dedup_window_sec", 1.5))
        self.order_llm_max_retries = int(rospy.get_param("~order_llm_max_retries", 3))
        self.food_aliases = rospy.get_param("~food_aliases", _default_food_aliases())
        self.food_alias_lookup = self._build_food_alias_lookup(self.food_aliases)

        self.total_inputs = 0
        self.ignored_inputs = 0
        self.successful_replies = 0
        self.orders_confirmed = 0

        self.subfsm_state = "NOT_PERMITTED"
        self.active_serving_state = "IDLE"
        self.latest_serving_payload: Dict[str, object] = {}
        self.current_order: Optional[OrderAction] = None
        self.last_listen_text = ""
        self.last_check_text = ""
        self.current_order_id = ""
        self.current_order_dir = ""
        self._session_id = 0
        self._processing_input = False
        self._last_input_norm = ""
        self._last_input_mono = 0.0
        self._id_lock = threading.Condition()
        self._latest_random_order_id = ""
        self._known_order_ids = self._load_existing_order_ids(self.order_voice_base_dir)

        self.tts_publisher = rospy.Publisher(self.tts_topic, String, queue_size=10)
        self.order_confirm_publisher = rospy.Publisher(self.order_confirm_topic, String, queue_size=10)
        self.subfsm_state_publisher = rospy.Publisher(
            self.subfsm_state_topic,
            String,
            queue_size=10,
            latch=True,
        )

        self.serving_state_subscriber = rospy.Subscriber(
            self.serving_state_topic,
            String,
            self._on_serving_state_message,
            queue_size=20,
        )
        self.order_id_subscriber = rospy.Subscriber(
            self.order_session_id_topic,
            String,
            self._on_order_id_message,
            queue_size=200,
        )

        self.primary_subscriber = None
        self.secondary_subscriber = None
        if self.input_channel_mode in ("primary", "both"):
            self.primary_subscriber = rospy.Subscriber(
                self.primary_input_topic,
                String,
                self._on_primary_message,
                queue_size=20,
            )
        if self.input_channel_mode in ("secondary", "both"):
            self.secondary_subscriber = rospy.Subscriber(
                self.secondary_input_topic,
                String,
                self._on_secondary_message,
                queue_size=20,
            )

        self._publish_subfsm_state("init")

        rospy.loginfo("=" * 60)
        rospy.loginfo("ROS Voice Bridge (ordering sub-FSM) initialized")
        rospy.loginfo("  Robot: %s", Config.ROBOT_NAME)
        rospy.loginfo("  Run mode: %s", "Cloud" if Config.is_cloud_mode() else "Local")
        rospy.loginfo("  Model: %s", Config.get_llm_config()["model"])
        rospy.loginfo("  Input mode: %s", self.input_channel_mode)
        rospy.loginfo("  Primary input: %s", self.primary_input_topic)
        rospy.loginfo("  Secondary input: %s", self.secondary_input_topic)
        rospy.loginfo("  Serving state topic: %s", self.serving_state_topic)
        rospy.loginfo("  TTS topic: %s", self.tts_topic)
        rospy.loginfo("  Order confirm topic: %s", self.order_confirm_topic)
        rospy.loginfo("  State publish topic: %s", self.subfsm_state_topic)
        rospy.loginfo("  Random order ID topic: %s", self.order_session_id_topic)
        rospy.loginfo("  Voice order base dir: %s", self.order_voice_base_dir)
        rospy.loginfo("=" * 60)

    def _load_existing_order_ids(self, base_dir: str) -> set:
        known = set()
        try:
            os.makedirs(base_dir, exist_ok=True)
            for name in os.listdir(base_dir):
                if len(name) == 5 and name.isdigit():
                    known.add(name)
        except Exception as exc:
            rospy.logwarn("Failed to scan existing order IDs in %s: %s", base_dir, exc)
        return known

    def _on_order_id_message(self, msg: String):
        raw = str(msg.data or "").strip()
        if not re.fullmatch(r"\d{5}", raw):
            return
        with self._id_lock:
            self._latest_random_order_id = raw
            self._id_lock.notify_all()

    def _can_wait_for_order_id(self, session_id: int) -> bool:
        with self._fsm_lock:
            return self._session_id == session_id and self.active_serving_state == "PAUSED_ORDERING"

    def _acquire_unique_order_id(self, session_id: int) -> Optional[tuple]:
        last_rejected = ""
        while self._can_wait_for_order_id(session_id) and not rospy.is_shutdown():
            candidate = ""
            with self._id_lock:
                candidate = str(self._latest_random_order_id or "").strip()
                if not re.fullmatch(r"\d{5}", candidate):
                    self._id_lock.wait(timeout=max(0.05, self.order_id_wait_poll_sec))
                    continue

            with self._fsm_lock:
                duplicated = candidate in self._known_order_ids
            if duplicated:
                if candidate != last_rejected:
                    rospy.loginfo("[order_subfsm] duplicate order_id=%s, waiting next random id", candidate)
                    last_rejected = candidate
                with self._id_lock:
                    self._id_lock.wait(timeout=max(0.05, self.order_id_wait_poll_sec))
                continue

            order_dir = os.path.join(self.order_voice_base_dir, candidate)
            try:
                os.makedirs(order_dir, exist_ok=False)
            except FileExistsError:
                with self._fsm_lock:
                    self._known_order_ids.add(candidate)
                if candidate != last_rejected:
                    rospy.loginfo("[order_subfsm] order dir exists for id=%s, waiting next random id", candidate)
                    last_rejected = candidate
                with self._id_lock:
                    self._id_lock.wait(timeout=max(0.05, self.order_id_wait_poll_sec))
                continue
            except Exception as exc:
                rospy.logwarn("Failed to create order dir for id=%s: %s", candidate, exc)
                with self._id_lock:
                    self._id_lock.wait(timeout=max(0.05, self.order_id_wait_poll_sec))
                continue

            with self._fsm_lock:
                self._known_order_ids.add(candidate)
            rospy.loginfo("[order_subfsm] acquired unique order_id=%s dir=%s", candidate, order_dir)
            return candidate, order_dir
        return None

    def _inject_environment_context(self, context: str):
        self.controller.system_prompt += f"\n\n## Current Environment\n{context}"
        rospy.loginfo("Injected environment context.")

    def _normalize_text_token(self, text: str) -> str:
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

    def _canonicalize_food_name(self, raw_name: str) -> str:
        key = self._normalize_text_token(raw_name)
        if not key:
            return ""
        return self.food_alias_lookup.get(key, key)

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

        normalized: List[OrderItem] = []
        for name in sorted(merged.keys()):
            normalized.append(OrderItem(name=name, qty=int(merged[name])))
        return normalized

    def _format_order_for_speech(self, order: Optional[OrderAction]) -> str:
        if order is None or not order.items:
            return "your order"

        parts = []
        for item in order.items:
            label = item.name.replace("_", " ")
            if item.qty > 1:
                parts.append(f"{item.qty} {label}")
            else:
                parts.append(label)

        if not parts:
            return "your order"
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def _build_repeat_fallback(self, order: Optional[OrderAction]) -> str:
        foods_text = self._format_order_for_speech(order)
        return f"Let me confirm. You ordered {foods_text}. Is that correct?"

    def _build_finish_text(self, order: Optional[OrderAction]) -> str:
        foods_text = self._format_order_for_speech(order)
        template = str(self.finish_template or "").strip()
        if not template:
            template = "OK I'll get {foods} for you"
        try:
            return template.format(foods=foods_text)
        except Exception:
            return f"OK I'll get {foods_text} for you"

    def _safe_json_loads(self, raw_text: str) -> Dict[str, object]:
        try:
            payload = json.loads(raw_text)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    def _parse_serving_state(self, raw_text: str) -> Dict[str, object]:
        payload = self._safe_json_loads(raw_text)
        if payload:
            state_text = str(payload.get("state", "IDLE")).strip().upper() or "IDLE"
            payload["state"] = state_text
            return payload
        return {"state": str(raw_text or "IDLE").strip().upper() or "IDLE"}

    def _publish_subfsm_state(self, reason: str):
        with self._fsm_lock:
            payload = {
                "timestamp": rospy.Time.now().to_sec(),
                "state": self.subfsm_state,
                "reason": str(reason),
                "serving_state": self.active_serving_state,
            }
            if self.current_order is not None:
                payload["order"] = self.current_order.model_dump()
            if self.current_order_id:
                payload["order_id"] = self.current_order_id
            if self.current_order_dir:
                payload["order_dir"] = self.current_order_dir

        try:
            self.subfsm_state_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        except Exception as exc:
            rospy.logwarn("Failed to publish order sub-FSM state: %s", exc)

    def _set_subfsm_state_locked(self, next_state: str, reason: str):
        old_state = self.subfsm_state
        self.subfsm_state = str(next_state).strip().upper()
        rospy.loginfo(
            "[order_subfsm] %s -> %s (%s)",
            old_state,
            self.subfsm_state,
            reason,
        )
        self._publish_subfsm_state(reason)

    def _set_subfsm_state(self, next_state: str, reason: str):
        with self._fsm_lock:
            self._set_subfsm_state_locked(next_state, reason)

    def _is_session_active(self, session_id: int) -> bool:
        with self._fsm_lock:
            return (
                self._session_id == session_id
                and self.active_serving_state == "PAUSED_ORDERING"
                and self.subfsm_state != "NOT_PERMITTED"
            )

    def _reset_to_not_permitted(self, reason: str):
        with self._fsm_lock:
            self._session_id += 1
            self.current_order = None
            self.last_listen_text = ""
            self.last_check_text = ""
            self.current_order_id = ""
            self.current_order_dir = ""
            self._processing_input = False
            self._set_subfsm_state_locked("NOT_PERMITTED", reason)

    def _start_ordering_session(self):
        with self._fsm_lock:
            if self.subfsm_state != "NOT_PERMITTED":
                return
            self._session_id += 1
            session_id = self._session_id
            self.current_order = None
            self.last_listen_text = ""
            self.last_check_text = ""
            self.current_order_id = ""
            self.current_order_dir = ""
            self._processing_input = False

        acquired = self._acquire_unique_order_id(session_id)
        if acquired is None:
            return
        order_id, order_dir = acquired

        with self._fsm_lock:
            if not self._can_wait_for_order_id(session_id):
                return
            self.current_order_id = order_id
            self.current_order_dir = order_dir
            self._set_subfsm_state_locked(
                "PERMITTED",
                f"serving_state=PAUSED_ORDERING;order_id={order_id}",
            )

        thread = threading.Thread(target=self._run_ask_stage, args=(session_id,), daemon=True)
        thread.start()

    def _run_ask_stage(self, session_id: int):
        if not self._is_session_active(session_id):
            return

        self._set_subfsm_state("ASK", "ordering_started")
        self._publish_tts(self.ask_prompt)

        if not self._is_session_active(session_id):
            return
        self._set_subfsm_state("LISTEN", "ask_completed")

    def _on_serving_state_message(self, msg: String):
        payload = self._parse_serving_state(msg.data)
        state_text = str(payload.get("state", "IDLE")).strip().upper() or "IDLE"

        with self._fsm_lock:
            prev_state = self.active_serving_state
            self.active_serving_state = state_text
            self.latest_serving_payload = payload

        if state_text == "PAUSED_ORDERING":
            if prev_state != "PAUSED_ORDERING":
                rospy.loginfo("[order_subfsm] serving state entered PAUSED_ORDERING")
            self._start_ordering_session()
            return

        if prev_state == "PAUSED_ORDERING":
            rospy.loginfo("[order_subfsm] serving state left PAUSED_ORDERING -> %s", state_text)
        self._reset_to_not_permitted(f"serving_state_changed:{state_text}")

    def _is_duplicate_input(self, text: str) -> bool:
        now = time.monotonic()
        norm = " ".join(str(text or "").strip().lower().split())
        if not norm:
            return True

        with self._fsm_lock:
            if (
                norm == self._last_input_norm
                and (now - self._last_input_mono) <= max(0.0, self.input_dedup_window_sec)
            ):
                return True
            self._last_input_norm = norm
            self._last_input_mono = now
        return False

    def _on_primary_message(self, msg: String):
        self._on_user_input(msg, "primary")

    def _on_secondary_message(self, msg: String):
        self._on_user_input(msg, "secondary")

    def _on_user_input(self, msg: String, source: str):
        text = str(msg.data or "").strip()
        if not text:
            return

        self.total_inputs += 1
        if self._is_duplicate_input(text):
            self.ignored_inputs += 1
            return

        with self._fsm_lock:
            current_state = self.subfsm_state
            session_id = self._session_id
            if (
                self.active_serving_state != "PAUSED_ORDERING"
                or current_state not in self.VALID_INPUT_STATES
            ):
                self.ignored_inputs += 1
                rospy.logdebug(
                    "[order_subfsm] ignore input in state=%s serving_state=%s",
                    current_state,
                    self.active_serving_state,
                )
                return

            if self._processing_input:
                self.ignored_inputs += 1
                rospy.logdebug("[order_subfsm] ignore input because previous turn still running")
                return

            self._processing_input = True

        thread = threading.Thread(
            target=self._process_order_input_async,
            args=(text, source, session_id, current_state),
            daemon=True,
        )
        thread.start()

    def _process_order_input_async(self, text: str, source: str, session_id: int, stage: str):
        try:
            if not self._is_session_active(session_id):
                return

            rospy.loginfo("[order_subfsm] %s input(%s): %s", stage, source, text)
            if stage == "LISTEN":
                self._process_listen_stage(text, session_id)
            elif stage == "CHECK":
                self._process_check_stage(text, session_id)

        except Exception as exc:
            rospy.logwarn("Ordering sub-FSM processing failed: %s", exc)
            if self._is_session_active(session_id):
                self._publish_tts(self.check_retry_prompt)
        finally:
            with self._fsm_lock:
                if self._session_id == session_id:
                    self._processing_input = False

    def _process_listen_stage(self, text: str, session_id: int):
        if not self._is_session_active(session_id):
            return

        with self._state_lock:
            self.robot.set_state(RobotState.THINKING)

        try:
            order_action = self.controller.llm_client.get_order_action(
                user_input=text,
                food_aliases=self.food_aliases,
                max_retries=self.order_llm_max_retries,
            )
        except Exception as exc:
            rospy.logwarn("Order LISTEN LLM failed: %s", exc)
            with self._state_lock:
                self.robot.set_state(RobotState.IDLE)
            self._publish_tts(self.listen_retry_prompt)
            return

        normalized_items = self._normalize_order_items(order_action.items)
        if not normalized_items:
            with self._state_lock:
                self.robot.set_state(RobotState.IDLE)
            self._publish_tts(self.listen_retry_prompt)
            return

        normalized_order = OrderAction(type="order", items=normalized_items)

        with self._fsm_lock:
            if not self._is_session_active(session_id):
                with self._state_lock:
                    self.robot.set_state(RobotState.IDLE)
                return
            self.current_order = normalized_order
            self.last_listen_text = text
            self._set_subfsm_state_locked("REPEAT", "order_extracted")
        self._save_order_group_json(normalized_order, stage="listen_parsed")

        with self._state_lock:
            self.robot.set_state(RobotState.IDLE)

        self._run_repeat_stage(session_id)

    def _run_repeat_stage(self, session_id: int):
        if not self._is_session_active(session_id):
            return

        with self._fsm_lock:
            order = self.current_order
        if order is None:
            self._set_subfsm_state("LISTEN", "repeat_without_order")
            return

        with self._state_lock:
            self.robot.set_state(RobotState.THINKING)

        speak_text = ""
        try:
            speak_decision = self.controller.llm_client.get_order_repeat_speak(
                confirm_instruction=self.repeat_instruction,
                order_action=order,
                max_retries=self.order_llm_max_retries,
            )
            speak_text = str(speak_decision.action.content or "").strip()
        except Exception as exc:
            rospy.logwarn("Order REPEAT LLM failed: %s", exc)

        with self._state_lock:
            self.robot.set_state(RobotState.IDLE)

        if not speak_text:
            speak_text = self._build_repeat_fallback(order)

        self._publish_tts(speak_text)
        if not self._is_session_active(session_id):
            return
        self._set_subfsm_state("CHECK", "repeat_completed")

    def _process_check_stage(self, text: str, session_id: int):
        if not self._is_session_active(session_id):
            return

        with self._fsm_lock:
            order = self.current_order
        if order is None:
            self._set_subfsm_state("LISTEN", "check_without_order")
            return

        with self._state_lock:
            self.robot.set_state(RobotState.THINKING)

        try:
            decision = self.controller.llm_client.get_order_check_decision(
                customer_reply=text,
                order_action=order,
                food_aliases=self.food_aliases,
                max_retries=self.order_llm_max_retries,
            )
        except Exception as exc:
            rospy.logwarn("Order CHECK LLM failed: %s", exc)
            with self._state_lock:
                self.robot.set_state(RobotState.IDLE)
            self._publish_tts(self.check_retry_prompt)
            return

        with self._state_lock:
            self.robot.set_state(RobotState.IDLE)

        self.last_check_text = text
        if decision.result == "correct":
            self._set_subfsm_state("FINISH", "check_correct")
            finish_text = self._build_finish_text(order)
            self._publish_tts(finish_text)
            self._publish_order_confirm(order)
            self.orders_confirmed += 1
            self.successful_replies += 1
            self._reset_to_not_permitted("finish_completed")
            return

        if decision.action is not None and decision.action.items:
            normalized_items = self._normalize_order_items(decision.action.items)
            if normalized_items:
                with self._fsm_lock:
                    if self._is_session_active(session_id):
                        self.current_order = OrderAction(type="order", items=normalized_items)
                        self._set_subfsm_state_locked("REPEAT", "order_fixed")
                self._save_order_group_json(self.current_order, stage="order_fixed")
                self._run_repeat_stage(session_id)
                return

        followup = str(decision.reply or "").strip()
        if not followup:
            followup = self.fix_missing_prompt
        self._publish_tts(followup)
        if self._is_session_active(session_id):
            self._set_subfsm_state("LISTEN", "wrong_without_fix")

    def _publish_order_confirm(self, order: Optional[OrderAction]):
        if order is None:
            return

        foods_with_qty = []
        foods = []
        for item in order.items:
            foods.append(item.name)
            foods_with_qty.append({"name": item.name, "qty": int(item.qty)})

        payload = {
            "timestamp": rospy.Time.now().to_sec(),
            "source": "voice_order_subfsm",
            "order": order.model_dump(),
            "foods": foods,
            "foods_with_qty": foods_with_qty,
            "recognized_text": self.last_listen_text,
            "check_text": self.last_check_text,
        }
        with self._fsm_lock:
            if self.current_order_id:
                payload["order_id"] = self.current_order_id
            if self.current_order_dir:
                payload["order_dir"] = self.current_order_dir

        if isinstance(self.latest_serving_payload, dict):
            for key in ("customer_id", "customer_no", "folder", "customer_folder"):
                if key in self.latest_serving_payload:
                    payload[key] = self.latest_serving_payload.get(key)

        try:
            self.order_confirm_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        except Exception as exc:
            rospy.logwarn("Failed to publish order confirm JSON: %s", exc)

    def _save_order_group_json(self, order: Optional[OrderAction], stage: str):
        if order is None:
            return
        with self._fsm_lock:
            order_dir = str(self.current_order_dir or "").strip()
            order_id = str(self.current_order_id or "").strip()
            serving_payload = dict(self.latest_serving_payload) if isinstance(self.latest_serving_payload, dict) else {}
            listen_text = self.last_listen_text
            check_text = self.last_check_text
        if not order_dir:
            return

        payload = {
            "timestamp": rospy.Time.now().to_sec(),
            "stage": str(stage or "").strip(),
            "order_id": order_id,
            "order": order.model_dump(),
            "recognized_text": listen_text,
            "check_text": check_text,
        }
        for key in ("customer_id", "customer_no", "folder", "customer_folder"):
            if key in serving_payload:
                payload[key] = serving_payload.get(key)

        target = os.path.join(order_dir, "order_group.json")
        try:
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            rospy.logwarn("Failed to save order_group.json (%s): %s", target, exc)

    def _publish_tts(self, text: str):
        text = str(text or "").strip()
        if not text:
            return

        with self._state_lock:
            self.robot.set_state(RobotState.SPEAKING)

        rospy.loginfo("[TTS] %s", text)
        self.tts_publisher.publish(String(data=text))

        estimated_duration = max(1.0, len(text) * 0.1)
        time.sleep(estimated_duration)

        with self._state_lock:
            self.robot.set_state(RobotState.IDLE)

    def spin(self):
        rospy.loginfo("Ordering sub-FSM listening...")
        rospy.spin()

    def print_statistics(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("Statistics:")
        rospy.loginfo("  Total inputs: %s", self.total_inputs)
        rospy.loginfo("  Ignored inputs: %s", self.ignored_inputs)
        rospy.loginfo("  Successful replies: %s", self.successful_replies)
        rospy.loginfo("  Orders confirmed: %s", self.orders_confirmed)
        rospy.loginfo("  Current sub-FSM state: %s", self.subfsm_state)
        rospy.loginfo("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CADE ROS Voice Bridge")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "simple", "compact", "debug"],
        default="default",
        help="Prompt mode",
    )
    parser.add_argument(
        "--no-thought",
        action="store_true",
        help="Do not show LLM reasoning output",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="You are sitting on a table in the Fedora lab. At the moment, you can only interact with people through voice.",
        help="Environment context",
    )

    args = parser.parse_args()

    try:
        bridge = RosVoiceBridge(
            prompt_mode=args.mode,
            show_thought=not args.no_thought,
            environment_context=args.env,
        )
        bridge.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logerr("Startup failed: %s", exc)
        import traceback

        traceback.print_exc()
