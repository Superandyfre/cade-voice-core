"""
CADE Robot V2 (ROS-free).

This implementation keeps the internal semantic map, object database,
and echo-suppression state checks in English.
"""

import logging
import time
import random
from typing import Optional, Dict, Any

from cade.body.robot_interface import RobotInterface, RobotState
from cade.config import Config

logger = logging.getLogger(__name__)


class Robot(RobotInterface):
    """CADE Robot V2 with English-only runtime logs and semantic labels."""

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or Config.ROBOT_NAME
        self.current_position = "home"
        self.holding_object = None

        self.known_locations: Dict[str, list] = {
            "home": [0.0, 0.0, 0.0],
            "start_point": [0.0, 0.0, 0.0],
            "kitchen": [5.0, 2.0, 0.0],
            "living_room": [3.0, -1.0, 0.0],
            "bedroom": [-2.0, 4.0, 0.0],
            "table": [4.0, 1.0, 0.0],
            "desk": [1.0, 3.0, 0.0],
        }

        self.known_objects: Dict[str, Dict[str, Any]] = {
            "apple": {"name": "apple", "location": "table", "position": [4.0, 1.0, 0.8]},
            "bottle": {"name": "bottle", "location": "kitchen", "position": [5.0, 2.0, 1.0]},
            "cup": {"name": "cup", "location": "table", "position": [4.2, 1.0, 0.8]},
            "book": {"name": "book", "location": "desk", "position": [1.0, 3.0, 0.9]},
        }

        logger.info(f"{self.name} initialized")
        logger.info(f"   Current position: {self.current_position}")
        logger.info(f"   Known locations: {list(self.known_locations.keys())}")
        logger.info(f"   Known objects: {list(self.known_objects.keys())}")


    def search(self, object_name: str) -> Optional[dict]:
        self.set_state(RobotState.EXECUTING)
        logger.info(f"[SEARCH] Looking for: {object_name}")
        time.sleep(0.8)

        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            logger.info(f"Found {object_name} at {obj['location']}")
            logger.info(f"   Position: {obj['position']}")
            self.set_state(RobotState.IDLE)
            return obj

        if random.random() < 0.3:
            logger.info(f"Could not find {object_name}")
            self.set_state(RobotState.IDLE)
            return None

        new_obj = {
            "name": object_name,
            "location": self.current_position,
            "position": [
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(0.5, 1.5),
            ],
        }
        self.known_objects[object_name] = new_obj
        logger.info(f"Found {object_name} near the current position")
        self.set_state(RobotState.IDLE)
        return new_obj

    def pick(self, object_name: str, object_id: Optional[int] = None) -> bool:
        self.set_state(RobotState.EXECUTING)
        logger.info(f"[PICK] Trying to pick: {object_name}")

        if self.holding_object:
            logger.info(f"Already holding: {self.holding_object}")
            self.set_state(RobotState.ERROR)
            return False

        time.sleep(0.6)

        if object_name in self.known_objects:
            self.holding_object = object_name
            logger.info(f"Picked up {object_name}")
            self.set_state(RobotState.IDLE)
            return True

        logger.info(f"Object does not exist: {object_name}")
        self.set_state(RobotState.ERROR)
        return False

    def place(self, location) -> bool:
        self.set_state(RobotState.EXECUTING)
        logger.info(f"[PLACE] Placing object at: {location}")

        if not self.holding_object:
            logger.info("Not holding any object")
            self.set_state(RobotState.ERROR)
            return False

        time.sleep(0.5)
        logger.info(f"Placed {self.holding_object} at {location}")

        if self.holding_object in self.known_objects:
            self.known_objects[self.holding_object]["location"] = str(location)

        self.holding_object = None
        self.set_state(RobotState.IDLE)
        return True

    def speak(self, content: str) -> bool:
        self.set_state(RobotState.SPEAKING)
        logger.info(f"[SPEAK] Voice output: \"{content}\"")
        return True

    def wait(self, reason: Optional[str] = None) -> bool:
        msg = "[WAIT]"
        if reason:
            msg += f" Reason: {reason}"
        logger.info(msg)
        self.set_state(RobotState.IDLE)
        return True

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "position": self.current_position,
            "holding": self.holding_object,
        }

    def print_status(self):
        status = self.get_status()
        logger.info(f"Robot status: {status}")

    def is_busy(self) -> bool:
        return self.state in (RobotState.THINKING, RobotState.SPEAKING, RobotState.EXECUTING)
