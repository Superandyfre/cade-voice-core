"""Tests for ActionSafetyGate, WorldState, and ActionResult."""

import pytest
from cade.brain.schemas import (
    PickAction, PlaceAction, SearchAction, SpeakAction, WaitAction,
    WorldState, WorldObject, SafetyResult, ActionResult,
)
from cade.body.safety import ActionSafetyGate
from cade.body.world_state import RobotWorldStateProvider


def _make_world(**overrides) -> WorldState:
    defaults = dict(
        robot_state="IDLE",
        current_position="home",
        holding_object=None,
        visible_objects=[
            WorldObject(name="apple", location="table", visible=True, graspable=True),
            WorldObject(name="cup", location="table", visible=True, graspable=True),
            WorldObject(name="knife", location="table", visible=True, graspable=True, forbidden=True),
        ],
        known_locations={"table": [4.0, 1.0, 0.0], "kitchen": [5.0, 2.0, 0.0]},
        forbidden_objects=["knife"],
    )
    defaults.update(overrides)
    return WorldState(**defaults)


class TestActionSafetyGate:

    def setup_method(self):
        self.gate = ActionSafetyGate()

    def test_wait_always_allowed(self):
        world = _make_world()
        result = self.gate.validate(WaitAction(reason="idle"), world)
        assert result.approved is True

    def test_search_allowed(self):
        world = _make_world()
        result = self.gate.validate(SearchAction(object_name="apple"), world)
        assert result.approved is True

    def test_search_empty_name_rejected(self):
        world = _make_world()
        result = self.gate.validate(SearchAction(object_name=""), world)
        assert result.approved is False
        assert result.reason_code == "empty_object_name"

    def test_pick_visible_object_allowed(self):
        world = _make_world()
        result = self.gate.validate(PickAction(object_name="apple"), world)
        assert result.approved is True

    def test_pick_forbidden_object_blocked(self):
        world = _make_world()
        result = self.gate.validate(PickAction(object_name="knife"), world)
        assert result.approved is False
        assert result.reason_code == "object_forbidden"

    def test_pick_while_holding_blocked(self):
        world = _make_world(holding_object="cup")
        result = self.gate.validate(PickAction(object_name="apple"), world)
        assert result.approved is False
        assert result.reason_code == "already_holding_object"

    def test_pick_unknown_object_blocked(self):
        world = _make_world()
        result = self.gate.validate(PickAction(object_name="unicorn"), world)
        assert result.approved is False
        assert result.reason_code == "object_not_found"

    def test_pick_invisible_object_blocked(self):
        world = _make_world(visible_objects=[
            WorldObject(name="apple", visible=False),
        ])
        result = self.gate.validate(PickAction(object_name="apple"), world)
        assert result.approved is False
        assert result.reason_code == "object_not_visible"

    def test_place_with_holding_allowed(self):
        world = _make_world(holding_object="cup")
        result = self.gate.validate(PlaceAction(location="table"), world)
        assert result.approved is True

    def test_place_without_holding_blocked(self):
        world = _make_world(holding_object=None)
        result = self.gate.validate(PlaceAction(location="table"), world)
        assert result.approved is False
        assert result.reason_code == "not_holding_object"

    def test_place_unknown_location_blocked(self):
        world = _make_world(holding_object="cup")
        result = self.gate.validate(PlaceAction(location="mars"), world)
        assert result.approved is False
        assert result.reason_code == "location_unreachable"

    def test_speak_allowed(self):
        world = _make_world()
        result = self.gate.validate(SpeakAction(content="hello"), world)
        assert result.approved is True

    def test_speak_empty_content_blocked(self):
        world = _make_world()
        result = self.gate.validate(SpeakAction(content=""), world)
        assert result.approved is False
        assert result.reason_code == "empty_content"


class TestWorldStateProvider:

    def test_builds_from_robot(self):
        from cade.body.robot import Robot
        robot = Robot(name="TEST")
        provider = RobotWorldStateProvider(robot)
        state = provider.get_world_state()

        assert state.robot_state == "IDLE"
        assert state.current_position == "home"
        assert state.holding_object is None
        assert len(state.visible_objects) > 0
        assert "table" in state.known_locations

    def test_holding_object_reflected(self):
        from cade.body.robot import Robot
        robot = Robot(name="TEST")
        robot.holding_object = "cup"
        provider = RobotWorldStateProvider(robot)
        state = provider.get_world_state()
        assert state.holding_object == "cup"


class TestActionResult:

    def test_success_result(self):
        r = ActionResult(success=True, status="completed", observation="picked apple")
        assert r.success is True
        assert r.status == "completed"

    def test_blocked_by_safety(self):
        r = ActionResult(
            success=False, status="blocked_by_safety",
            observation="object forbidden",
            suggested_next_actions=["speak", "wait"],
        )
        assert r.success is False
        assert "speak" in r.suggested_next_actions

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            ActionResult(success=True, status="completed", unexpected_field=True)
