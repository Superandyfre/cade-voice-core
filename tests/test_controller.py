"""Unit tests for RobotController — broadcast priority, history limit, return shape."""

import pytest
from unittest.mock import MagicMock, patch

from cade.controller import RobotController, _MAX_HISTORY_TURNS
from cade.brain.schemas import RobotDecision, SpeakAction, WaitAction, ActionResult
from cade.body.safety import ActionSafetyGate
from cade.body.world_state import RobotWorldStateProvider


def _make_controller():
    """Create a RobotController with mocked LLM and robot."""
    ctrl = RobotController.__new__(RobotController)
    ctrl.llm_client = MagicMock()
    ctrl.robot = MagicMock()
    ctrl.robot.state = MagicMock()
    ctrl.robot.state.value = "IDLE"
    ctrl.robot.current_position = "home"
    ctrl.robot.holding_object = None
    ctrl.robot.known_objects = {"apple": {"name": "apple", "location": "table", "position": [4.0, 1.0, 0.8]}}
    ctrl.robot.known_locations = {"table": [4.0, 1.0, 0.0]}
    ctrl.robot.forbidden_objects = []
    ctrl.system_prompt = "test prompt"
    ctrl.conversation_history = []
    ctrl.show_thought = False
    ctrl.total_interactions = 0
    ctrl.successful_actions = 0
    ctrl.failed_actions = 0
    ctrl._world_state_provider = RobotWorldStateProvider(ctrl.robot)
    ctrl._safety_gate = ActionSafetyGate()
    return ctrl


# ------------------------------------------------------------------
# Broadcast priority: reply > speak-action content > nothing
# ------------------------------------------------------------------

class TestBroadcastPriority:

    def test_reply_takes_priority_over_speak_action(self):
        ctrl = _make_controller()
        decision = RobotDecision(
            reply="I will speak this",
            action=SpeakAction(content="Not this"),
        )
        ctrl.llm_client.get_decision.return_value = decision

        result = ctrl.process_input("test")
        assert result["spoken_text"] == "I will speak this"

    def test_speak_action_content_used_when_no_reply(self):
        ctrl = _make_controller()
        decision = RobotDecision(
            reply=None,
            action=SpeakAction(content="Fallback content"),
        )
        ctrl.llm_client.get_decision.return_value = decision

        result = ctrl.process_input("test")
        assert result["spoken_text"] == "Fallback content"

    def test_no_spoken_text_for_non_speak_action_without_reply(self):
        ctrl = _make_controller()
        decision = RobotDecision(
            reply=None,
            action=WaitAction(reason="idle"),
        )
        ctrl.llm_client.get_decision.return_value = decision

        result = ctrl.process_input("test")
        assert result["spoken_text"] is None


# ------------------------------------------------------------------
# Return shape
# ------------------------------------------------------------------

class TestReturnShape:

    def test_result_contains_required_keys(self):
        ctrl = _make_controller()
        decision = RobotDecision(reply="hi", action=WaitAction())
        ctrl.llm_client.get_decision.return_value = decision

        result = ctrl.process_input("hello")
        assert "decision" in result
        assert "action_success" in result
        assert "spoken_text" in result
        assert "timings" in result
        assert "llm_latency_s" in result["timings"]

    def test_timings_have_llm_latency(self):
        ctrl = _make_controller()
        decision = RobotDecision(reply="ok")
        ctrl.llm_client.get_decision.return_value = decision

        result = ctrl.process_input("test")
        assert result["timings"]["llm_latency_s"] >= 0


# ------------------------------------------------------------------
# History limit
# ------------------------------------------------------------------

class TestHistoryLimit:

    def test_history_bounded_by_max_turns(self):
        ctrl = _make_controller()
        # Feed many turns
        for i in range(_MAX_HISTORY_TURNS + 10):
            decision = RobotDecision(reply=f"reply {i}")
            ctrl.llm_client.get_decision.return_value = decision
            ctrl.process_input(f"input {i}")

        # Each turn = 2 messages (user + assistant)
        assert len(ctrl.conversation_history) <= _MAX_HISTORY_TURNS * 2

    def test_history_contains_most_recent_turns(self):
        ctrl = _make_controller()
        for i in range(_MAX_HISTORY_TURNS + 5):
            decision = RobotDecision(reply=f"reply {i}")
            ctrl.llm_client.get_decision.return_value = decision
            ctrl.process_input(f"input {i}")

        # The last user message should be the most recent input
        last_user = ctrl.conversation_history[-2]
        assert last_user["role"] == "user"
        assert "input " in last_user["content"]

    def test_history_does_not_store_thought(self):
        ctrl = _make_controller()
        decision = RobotDecision(
            thought="internal reasoning",
            reply="spoken reply",
        )
        ctrl.llm_client.get_decision.return_value = decision
        ctrl.process_input("test")

        assistant_msg = ctrl.conversation_history[-1]["content"]
        assert "internal reasoning" not in assistant_msg
        assert "spoken reply" in assistant_msg


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------

class TestReset:

    def test_reset_clears_everything(self):
        ctrl = _make_controller()
        decision = RobotDecision(reply="hi")
        ctrl.llm_client.get_decision.return_value = decision
        ctrl.process_input("test")
        ctrl.reset()

        assert ctrl.conversation_history == []
        assert ctrl.total_interactions == 0
        assert ctrl.successful_actions == 0
        assert ctrl.failed_actions == 0
