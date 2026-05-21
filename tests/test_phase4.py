"""Tests for Phase 4: Intent router, context builder, agent state, graph."""

import pytest
from cade.brain.router import IntentRouter, IntentRouterDecision, IntentSubtask
from cade.brain.context import ContextBuilder, TaskState, ConversationMemory
from cade.agent.state import AgentState
from cade.agent.graph import Graph


class TestIntentRouter:

    def setup_method(self):
        self.router = IntentRouter()

    def test_order_intent(self):
        result = self.router.route("I want a coke")
        assert result.intent == "order"
        assert result.confidence > 0.5

    def test_robot_action_intent(self):
        result = self.router.route("find the cup")
        assert result.intent == "robot_action"

    def test_smalltalk_intent(self):
        result = self.router.route("hello")
        assert result.intent == "smalltalk"
        assert result.confidence >= 0.9

    def test_mixed_intent(self):
        result = self.router.route("can I get a coke and can you come closer")
        assert result.intent == "mixed"
        assert len(result.subtasks) >= 2

    def test_ordering_session_bias(self):
        result = self.router.route("bring me the cup", in_ordering_session=True)
        # In ordering session, should still route to robot_action for non-food
        assert result.intent in ("robot_action", "mixed")

    def test_ordering_session_with_food(self):
        result = self.router.route("coke please", in_ordering_session=True)
        assert result.intent == "order"

    def test_empty_input(self):
        result = self.router.route("")
        assert result.intent == "out_of_scope"

    def test_clarification_intent(self):
        result = self.router.route("what do you mean")
        assert result.intent == "clarification"

    def test_out_of_scope(self):
        result = self.router.route("the weather is nice today")
        assert result.intent == "out_of_scope"


class TestContextBuilder:

    def test_empty_context(self):
        cb = ContextBuilder()
        ctx = cb.build_context_section()
        assert ctx == ""

    def test_task_context(self):
        cb = ContextBuilder()
        cb.update_task(active_task="fetch", target_object="cup")
        ctx = cb.build_context_section()
        assert "fetch" in ctx
        assert "cup" in ctx

    def test_action_result_recorded(self):
        cb = ContextBuilder()
        cb.record_action_result({"type": "pick", "object_name": "cup"}, False, "object_not_found")
        ctx = cb.build_context_section()
        assert "failed" in ctx

    def test_recent_turns_kept(self):
        cb = ContextBuilder()
        for i in range(20):
            cb.add_turn("user", f"message {i}")
        assert len(cb.memory.recent_turns) <= 8

    def test_forbidden_repeats_kept(self):
        cb = ContextBuilder()
        for i in range(10):
            cb.record_action_result({"type": "pick"}, False, f"fail_{i}")
        assert len(cb.task_state.forbidden_repeats) == 5

    def test_history_messages(self):
        cb = ContextBuilder()
        cb.add_turn("user", "hello")
        cb.add_turn("assistant", "hi")
        msgs = cb.get_history_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"

    def test_reset_clears_state(self):
        cb = ContextBuilder()
        cb.update_task(active_task="test")
        cb.add_turn("user", "hello")
        cb.reset()
        assert cb.task_state.active_task is None
        assert len(cb.memory.recent_turns) == 0


class TestAgentState:

    def test_default_state(self):
        state = AgentState()
        assert state.session_id == ""
        assert state.route is None
        assert state.errors == []

    def test_with_route(self):
        route = IntentRouterDecision(
            intent="order", confidence=0.9, reason_code="test",
        )
        state = AgentState(session_id="1", raw_user_text="coke", route=route)
        assert state.route.intent == "order"


class TestGraph:

    def test_linear_execution(self):
        graph = Graph("test")

        def node_a(state: AgentState) -> AgentState:
            state.tts_text = "processed"
            return state

        graph.add_node("a", node_a)
        result = graph.run(AgentState(raw_user_text="hello"))
        assert result.tts_text == "processed"

    def test_multiple_nodes(self):
        graph = Graph("test")

        def node_a(state: AgentState) -> AgentState:
            state.session_id = "a"
            return state

        def node_b(state: AgentState) -> AgentState:
            state.tts_text = f"done_{state.session_id}"
            return state

        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        result = graph.run(AgentState())
        assert result.tts_text == "done_a"

    def test_error_handling(self):
        graph = Graph("test")

        def failing_node(state: AgentState) -> AgentState:
            raise RuntimeError("boom")

        def safe_node(state: AgentState) -> AgentState:
            state.tts_text = "recovered"
            return state

        graph.add_node("fail", failing_node)
        graph.add_node("safe", safe_node)
        result = graph.run(AgentState())
        assert len(result.errors) == 1
        assert result.tts_text == "recovered"

    def test_tracing(self, tmp_path):
        graph = Graph("test")
        graph.enable_tracing(str(tmp_path))

        def node_a(state: AgentState) -> AgentState:
            return state

        graph.add_node("a", node_a)
        graph.run(AgentState(session_id="123"))

        trace_file = tmp_path / "session_123.jsonl"
        assert trace_file.exists()
