"""Agent state model for the lightweight graph runtime."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from cade.brain.schemas import (
    RobotDecision, ActionResult, WorldState, SafetyResult,
)
from cade.brain.router import IntentRouterDecision


class AgentState(BaseModel):
    """Complete state for one agent turn."""
    session_id: str = ""
    raw_user_text: str = ""
    route: Optional[IntentRouterDecision] = None
    decision: Optional[RobotDecision] = None
    action_result: Optional[ActionResult] = None
    safety_result: Optional[SafetyResult] = None
    world_state: Optional[WorldState] = None
    tts_text: Optional[str] = None
    errors: List[str] = []
