"""
Data schemas for robot decisions and ordering sub-FSM decisions.

Pydantic models define:
1) Generic robot action space for normal LLM loop.
2) Dedicated ordering action space for the PAUSED_ORDERING sub-FSM.
"""

from typing import Literal, Union, Optional, List
from pydantic import BaseModel, Field, field_validator


# ==================== Generic Action Types ====================

class PickAction(BaseModel):
    """Pick action: pick up an object."""
    type: Literal["pick"] = "pick"
    object_name: str = Field(..., description="Object name")
    object_id: Optional[int] = Field(None, description="Object ID when multiple objects have the same name")


class PlaceAction(BaseModel):
    """Place action: place the held object at a target location."""
    type: Literal["place"] = "place"
    location: Union[str, List[float]] = Field(
        ...,
        description="Placement location, either a semantic label such as 'table' or coordinates.",
    )


class SearchAction(BaseModel):
    """Search action: look for an object."""
    type: Literal["search"] = "search"
    object_name: str = Field(..., description="Name of the object to search for")


class SpeakAction(BaseModel):
    """Speak action: produce spoken output."""
    type: Literal["speak"] = "speak"
    content: str = Field(..., description="Text to speak")


class WaitAction(BaseModel):
    """Wait action: stay in the current state."""
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
    """
    Robot decision output schema produced by the LLM.
    """
    thought: Optional[str] = Field(None, description="Internal reasoning in English")
    reply: Optional[str] = Field(None, description="Natural-language reply to the user in English")
    action: Optional[RobotAction] = Field(None, description="Action to execute, or None for pure conversation")

    class Config:
        arbitrary_types_allowed = True


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

    name: str = Field(..., description="Canonical food name, e.g. coke, fried_rice")
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

    type: Literal["order"] = "order"
    items: List[OrderItem] = Field(..., description="Ordered items")


class FixOrderAction(BaseModel):
    """CHECK stage output when user wants to modify an order."""

    type: Literal["fix_order"] = "fix_order"
    items: List[OrderItem] = Field(..., min_length=1, description="Updated order items")


class OrderCheckDecision(BaseModel):
    """CHECK stage decision."""

    result: Literal["correct", "wrong"]
    action: Optional[FixOrderAction] = None
    reply: Optional[str] = Field(None, description="Optional short follow-up reply text")


class OrderSpeakDecision(BaseModel):
    """REPEAT stage output used for TTS broadcasting."""

    action: SpeakAction


def parse_order_action(action_dict: dict) -> OrderAction:
    """Parse LISTEN stage order action."""
    model = OrderAction(**action_dict)
    if model.type != "order":
        raise ValueError("Order action type must be 'order'")
    return model


def parse_order_check_decision(payload: dict) -> OrderCheckDecision:
    """Parse CHECK stage decision payload."""
    return OrderCheckDecision(**payload)


def parse_order_speak_decision(payload: dict) -> OrderSpeakDecision:
    """Parse REPEAT stage speak payload."""
    return OrderSpeakDecision(**payload)
