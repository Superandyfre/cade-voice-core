"""Lightweight graph runtime for agent execution.

Provides a simple node-based execution model inspired by LangGraph,
but without external dependencies. Each node is a function that takes
and returns AgentState.
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

from cade.agent.state import AgentState

logger = logging.getLogger(__name__)


class GraphNode:
    """A node in the execution graph."""

    def __init__(self, name: str, fn: Callable[[AgentState], AgentState]):
        self.name = name
        self.fn = fn

    def run(self, state: AgentState) -> AgentState:
        return self.fn(state)


class Graph:
    """Simple linear graph executor."""

    def __init__(self, name: str = "cade_agent"):
        self.name = name
        self.nodes: List[GraphNode] = []
        self._trace_dir: Optional[Path] = None

    def add_node(self, name: str, fn: Callable[[AgentState], AgentState]) -> "Graph":
        self.nodes.append(GraphNode(name, fn))
        return self

    def enable_tracing(self, trace_dir: str = "logs/agent_traces") -> None:
        self._trace_dir = Path(trace_dir)
        self._trace_dir.mkdir(parents=True, exist_ok=True)

    def run(self, initial_state: AgentState) -> AgentState:
        state = initial_state
        for node in self.nodes:
            t0 = time.monotonic()
            try:
                state = node.run(state)
            except Exception as exc:
                logger.error("Node %s failed: %s", node.name, exc)
                state.errors.append(f"{node.name}: {exc}")
            latency = round(time.monotonic() - t0, 3)
            logger.debug("Node %s completed in %ss", node.name, latency)

            if self._trace_dir:
                self._write_trace(state.session_id, node.name, state, latency)

        return state

    def _write_trace(self, session_id: str, node_name: str, state: AgentState, latency: float) -> None:
        if not self._trace_dir:
            return
        trace_entry = {
            "ts": time.time(),
            "session_id": session_id,
            "node": node_name,
            "latency_s": latency,
            "errors": state.errors,
        }
        trace_file = self._trace_dir / f"session_{session_id}.jsonl"
        try:
            with open(trace_file, "a") as f:
                f.write(json.dumps(trace_entry) + "\n")
        except Exception:
            pass
