"""
Data schemas for robot decisions and ordering sub-FSM decisions.

Pydantic models define:
1) Generic robot action space for normal LLM loop.
2) Dedicated ordering action space for the PAUSED_ORDERING sub-FSM.
3) World state, safety results, and action results for safe execution.
"""

from typing import Literal, Union, Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator


# ==================== Generic Action Types ====================

class PickAction(BaseModel):
    """Pick action: pick up an object."""
    model_config = {"extra": "forbid"}

    type: Literal["pick"] = "pick"
    object_name: str = Field(..., description="Object name")
    object_id: Optional[int] = Field(None, description="Object ID when multiple objects have the same name")


class PlaceAction(BaseModel):
    """Place action: place the held object at a target location."""
    model_config = {"extra": "forbid"}

    type: Literal["place"] = "place"
    location: Union[str, List[float]] = Field(
        ...,
        description="Placement location, either a semantic label such as 'table' or coordinates.",
    )


class SearchAction(BaseModel):
    """Search action: look for an object."""
    model_config = {"extra": "forbid"}

    type: Literal["search"] = "search"
    object_name: str = Field(..., description="Name of the object to search for")


class SpeakAction(BaseModel):
    """Speak action: produce spoken output."""
    model_config = {"extra": "forbid"}

    type: Literal["speak"] = "speak"
    content: str = Field(..., description="Text to speak")


class WaitAction(BaseModel):
    """Wait action: stay in the current state."""
    model_config = {"extra": "forbid"}

    type: Literal["wait"] = "wait"
    reason: Optional[str] = Field(None, description="Reason for waiting")


RobotAction = Union[
    PickAction,
    PlaceAction,
    SearchAction,
    SpeakAction,
    WaitAction,
]


class RobotDecision(BaseModel):
    """Robot decision output schema produced by the LLM."""
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    thought: Optional[str] = Field(None, description="Internal reasoning in English")
    reply: Optional[str] = Field(None, description="Natural-language reply to the user in English")
    action: Optional[RobotAction] = Field(None, description="Action to execute, or None for pure conversation")


def parse_action(action_dict: dict) -> RobotAction:
    """Parse a generic robot action dictionary by its type field."""
    action_type = action_dict.get("type")

    action_map = {
        "pick": PickAction,
        "place": PlaceAction,
        "search": SearchAction,
        "speak": SpeakAction,
        "wait": WaitAction,
    }

    if action_type not in action_map:
        raise ValueError(
            f"Unknown action type: {action_type}. "
            f"Available actions: {list(action_map.keys())}"
        )

    return action_map[action_type](**action_dict)


# ==================== Ordering Sub-FSM Schemas ====================

class OrderItem(BaseModel):
    """One normalized order item."""
    model_config = {"extra": "forbid"}

    name: str = Field(..., description="Canonical food name, e.g. coke, fried_rice")
    item_id: Optional[str] = Field(None, description="Stable menu item ID, e.g. coke, fried_rice")
    qty: int = Field(..., ge=1, description="Quantity, must be >= 1")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        text = str(value).strip().lower().replace(" ", "_")
        if not text:
            raise ValueError("Order item name cannot be empty")
        return text


class OrderAction(BaseModel):
    """LISTEN stage output: parsed order items."""
    model_config = {"extra": "forbid"}

    type: Literal["order"] = "order"
    items: List[OrderItem] = Field(default_factory=list, description="Ordered items")


class FixOrderAction(BaseModel):
    """CHECK stage output when user wants to modify an order."""
    model_config = {"extra": "forbid"}

    type: Literal["fix_order"] = "fix_order"
    items: List[OrderItem] = Field(..., min_length=1, description="Updated order items")


class OrderCheckDecision(BaseModel):
    """CHECK stage decision with semantic validation."""
    model_config = {"extra": "forbid"}

    result: Literal["correct", "wrong"]
    action: Optional[FixOrderAction] = None
    reply: Optional[str] = Field(None, description="Optional short follow-up reply text")

    @model_validator(mode="after")
    def _validate_semantics(self) -> "OrderCheckDecision":
        if self.result == "correct" and self.action is not None:
            raise ValueError("When result is 'correct', action must be null")
        if self.result == "wrong" and self.action is not None:
            if not self.action.items:
                raise ValueError("fix_order items must not be empty")
        return self


class OrderSpeakDecision(BaseModel):
    """REPEAT stage output used for TTS broadcasting."""
    model_config = {"extra": "forbid"}

    action: SpeakAction


def parse_order_action(action_dict: dict) -> OrderAction:
    """Parse LISTEN stage order action. Requires type='order' in the raw dict."""
    raw_type = action_dict.get("type")
    if raw_type != "order":
        raise ValueError(f"Expected type='order', got type={raw_type!r}")
    model = OrderAction(**action_dict)
    return model


def parse_order_check_decision(payload: dict) -> OrderCheckDecision:
    """Parse CHECK stage decision payload."""
    return OrderCheckDecision(**payload)


def parse_order_speak_decision(payload: dict) -> OrderSpeakDecision:
    """Parse REPEAT stage speak payload."""
    raw_action = payload.get("action")
    if isinstance(raw_action, dict) and raw_action.get("type") != "speak":
        raise ValueError(
            f"Expected action.type='speak', got {raw_action.get('type')!r}"
        )
    return OrderSpeakDecision(**payload)


# ==================== World State & Safety ====================

class WorldObject(BaseModel):
    """An object in the robot's world model."""
    name: str
    object_id: Optional[int] = None
    location: Optional[str] = None
    position: Optional[List[float]] = None
    visible: bool = True
    graspable: bool = True
    forbidden: bool = False


class WorldState(BaseModel):
    """Snapshot of the robot's current world state."""
    robot_state: str = "IDLE"
    current_position: Optional[str] = None
    holding_object: Optional[str] = None
    visible_objects: List[WorldObject] = []
    known_locations: dict[str, List[float]] = {}
    forbidden_objects: List[str] = []
    last_action_result: Optional["ActionResult"] = None


class SafetyResult(BaseModel):
    """Result of safety gate validation."""
    approved: bool
    reason_code: str
    reason: str
    allowed_next_actions: List[str] = []


class ActionResult(BaseModel):
    """Standardized result after executing a robot action."""
    model_config = {"extra": "forbid"}

    action_id: str = ""
    action: dict = {}
    success: bool
    status: Literal[
        "completed",
        "object_not_found",
        "object_not_visible",
        "object_forbidden",
        "already_holding_object",
        "not_holding_object",
        "location_unreachable",
        "grasp_failed",
        "timeout",
        "blocked_by_safety",
        "execution_error",
        "user_interrupted",
    ]
    observation: str = ""
    world_delta: dict = {}
    suggested_next_actions: List[str] = []
