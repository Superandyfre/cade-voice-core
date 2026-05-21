"""Context builder — replaces raw sliding window with task state + summary."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class TaskState(BaseModel):
    active_task: Optional[str] = None
    target_object: Optional[str] = None
    target_location: Optional[str] = None
    last_action: Optional[Dict] = None
    last_action_success: Optional[bool] = None
    last_action_status: Optional[str] = None
    forbidden_repeats: List[Dict] = []
    confirmed_order: Optional[Dict] = None
    user_location: Optional[str] = None


class ConversationMemory(BaseModel):
    summary: str = ""
    recent_turns: List[Dict[str, str]] = []


class ContextBuilder:
    """Builds prompt context from task state and conversation memory."""

    MAX_RECENT_TURNS = 4
    MAX_FORBIDDEN_REPEATS = 5

    def __init__(self):
        self.task_state = TaskState()
        self.memory = ConversationMemory()

    def add_turn(self, role: str, content: str) -> None:
        self.memory.recent_turns.append({"role": role, "content": content})
        if len(self.memory.recent_turns) > self.MAX_RECENT_TURNS * 2:
            self.memory.recent_turns = self.memory.recent_turns[-(self.MAX_RECENT_TURNS * 2):]

    def record_action_result(self, action: Dict, success: bool, status: str) -> None:
        self.task_state.last_action = action
        self.task_state.last_action_success = success
        self.task_state.last_action_status = status
        if not success:
            self.task_state.forbidden_repeats.append({
                "action": action,
                "status": status,
            })
            if len(self.task_state.forbidden_repeats) > self.MAX_FORBIDDEN_REPEATS:
                self.task_state.forbidden_repeats = self.task_state.forbidden_repeats[-self.MAX_FORBIDDEN_REPEATS:]

    def update_task(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.task_state, key):
                setattr(self.task_state, key, value)

    def build_context_section(self) -> str:
        """Build the context section for the system prompt."""
        parts = []

        if self.task_state.active_task:
            parts.append(f"Current task: {self.task_state.active_task}")

        if self.task_state.target_object:
            parts.append(f"Target object: {self.task_state.target_object}")

        if self.task_state.target_location:
            parts.append(f"Target location: {self.task_state.target_location}")

        if self.task_state.last_action is not None:
            status = "success" if self.task_state.last_action_success else f"failed ({self.task_state.last_action_status})"
            parts.append(f"Last action: {self.task_state.last_action.get('type', 'unknown')} - {status}")

        if self.task_state.forbidden_repeats:
            recent_fails = self.task_state.forbidden_repeats[-3:]
            fail_descriptions = [f"{f['action'].get('type', '?')}({f['status']})" for f in recent_fails]
            parts.append(f"Recent failed actions to avoid repeating: {', '.join(fail_descriptions)}")

        if self.memory.summary:
            parts.append(f"Conversation summary: {self.memory.summary}")

        if not parts:
            return ""

        return "## Current Context\n\n" + "\n".join(f"- {p}" for p in parts) + "\n"

    def get_history_messages(self) -> List[Dict[str, str]]:
        """Get recent turns as chat messages."""
        return list(self.memory.recent_turns)

    def reset(self) -> None:
        self.task_state = TaskState()
        self.memory = ConversationMemory()
