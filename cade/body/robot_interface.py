"""
Robot interface definitions.

Defines a common robot abstraction that both mock and real robot classes must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum


class RobotState(str, Enum):
    """Robot state."""
    IDLE = "IDLE"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


class RobotInterface(ABC):
    """Abstract robot interface. All robot implementations must provide these methods."""

    def __init__(self):
        self.state = RobotState.IDLE
        self.current_position: Optional[str] = None
        self.holding_object: Optional[str] = None

    @abstractmethod
    def search(self, object_name: str) -> Optional[dict]:
        pass

    @abstractmethod
    def pick(self, object_name: str, object_id: Optional[int] = None) -> bool:
        pass

    @abstractmethod
    def place(self, location) -> bool:
        pass

    @abstractmethod
    def speak(self, content: str) -> bool:
        pass

    @abstractmethod
    def wait(self, reason: Optional[str] = None) -> bool:
        pass

    def get_state(self) -> RobotState:
        return self.state

    def set_state(self, state: RobotState):
        self.state = state
