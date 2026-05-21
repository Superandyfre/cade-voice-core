"""World state provider — builds WorldState from robot internals."""

from typing import Protocol

from cade.body.robot_interface import RobotState
from cade.brain.schemas import WorldObject, WorldState


class WorldStateProvider(Protocol):
    def get_world_state(self) -> WorldState: ...


class RobotWorldStateProvider:
    """Builds WorldState from a Robot instance."""

    def __init__(self, robot):
        self._robot = robot

    def get_world_state(self) -> WorldState:
        robot = self._robot
        state_str = "IDLE"
        if hasattr(robot, "state"):
            try:
                state_str = robot.state.value if isinstance(robot.state, RobotState) else str(robot.state)
            except Exception:
                pass

        current_position = getattr(robot, "current_position", None)
        holding_object = getattr(robot, "holding_object", None)

        visible_objects = []
        known_objects = getattr(robot, "known_objects", {})
        if isinstance(known_objects, dict):
            for name, info in known_objects.items():
                if isinstance(info, dict):
                    visible_objects.append(WorldObject(
                        name=str(name),
                        location=info.get("location"),
                        position=info.get("position"),
                        visible=True,
                    ))

        known_locations = {}
        locs = getattr(robot, "known_locations", {})
        if isinstance(locs, dict):
            for k, v in locs.items():
                if isinstance(v, (list, tuple)):
                    known_locations[str(k)] = list(v)

        forbidden_objects = list(getattr(robot, "forbidden_objects", []))

        return WorldState(
            robot_state=state_str,
            current_position=current_position,
            holding_object=holding_object,
            visible_objects=visible_objects,
            known_locations=known_locations,
            forbidden_objects=forbidden_objects,
        )
