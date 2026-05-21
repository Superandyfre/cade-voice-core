"""Safety gate — validates robot actions before execution."""

from cade.brain.schemas import (
    SafetyResult,
    WorldState,
    RobotAction,
    PickAction,
    PlaceAction,
    SearchAction,
    SpeakAction,
    WaitAction,
)


class ActionSafetyGate:
    """Validates proposed robot actions against the current world state."""

    def validate(self, action: RobotAction, world: WorldState) -> SafetyResult:
        if isinstance(action, WaitAction):
            return SafetyResult(approved=True, reason_code="wait_allowed", reason="wait is always allowed")

        if isinstance(action, SearchAction):
            return self._check_search(action, world)

        if isinstance(action, PickAction):
            return self._check_pick(action, world)

        if isinstance(action, PlaceAction):
            return self._check_place(action, world)

        if isinstance(action, SpeakAction):
            return self._check_speak(action, world)

        return SafetyResult(approved=False, reason_code="unknown_action", reason=f"unknown action type")

    def _check_search(self, action: SearchAction, world: WorldState) -> SafetyResult:
        if not action.object_name or not action.object_name.strip():
            return SafetyResult(
                approved=False, reason_code="empty_object_name",
                reason="search requires a non-empty object_name",
                allowed_next_actions=["speak", "wait"],
            )
        return SafetyResult(approved=True, reason_code="search_allowed", reason="search allowed")

    def _check_pick(self, action: PickAction, world: WorldState) -> SafetyResult:
        if not action.object_name or not action.object_name.strip():
            return SafetyResult(
                approved=False, reason_code="empty_object_name",
                reason="pick requires a non-empty object_name",
                allowed_next_actions=["search", "speak", "wait"],
            )

        if world.holding_object:
            return SafetyResult(
                approved=False, reason_code="already_holding_object",
                reason=f"already holding {world.holding_object}",
                allowed_next_actions=["place", "speak", "wait"],
            )

        name_lower = action.object_name.strip().lower()
        if name_lower in [f.lower() for f in world.forbidden_objects]:
            return SafetyResult(
                approved=False, reason_code="object_forbidden",
                reason=f"object {action.object_name} is forbidden",
                allowed_next_actions=["speak", "wait"],
            )

        for obj in world.visible_objects:
            if obj.name.lower() == name_lower:
                if obj.forbidden:
                    return SafetyResult(
                        approved=False, reason_code="object_forbidden",
                        reason=f"object {action.object_name} is forbidden",
                        allowed_next_actions=["speak", "wait"],
                    )
                if not obj.visible:
                    return SafetyResult(
                        approved=False, reason_code="object_not_visible",
                        reason=f"object {action.object_name} is not currently visible",
                        allowed_next_actions=["search", "speak", "wait"],
                    )
                if not obj.graspable:
                    return SafetyResult(
                        approved=False, reason_code="not_graspable",
                        reason=f"object {action.object_name} is not graspable",
                        allowed_next_actions=["speak", "wait"],
                    )
                return SafetyResult(approved=True, reason_code="pick_allowed", reason="pick allowed")

        known_names = [obj.name.lower() for obj in world.visible_objects]
        if name_lower not in known_names:
            return SafetyResult(
                approved=False, reason_code="object_not_found",
                reason=f"object {action.object_name} not in known objects",
                allowed_next_actions=["search", "speak", "wait"],
            )

        return SafetyResult(approved=True, reason_code="pick_allowed", reason="pick allowed")

    def _check_place(self, action: PlaceAction, world: WorldState) -> SafetyResult:
        if not world.holding_object:
            return SafetyResult(
                approved=False, reason_code="not_holding_object",
                reason="not holding any object to place",
                allowed_next_actions=["search", "pick", "speak", "wait"],
            )

        if isinstance(action.location, str):
            loc_lower = action.location.strip().lower()
            known_loc_names = [k.lower() for k in world.known_locations]
            if loc_lower not in known_loc_names:
                return SafetyResult(
                    approved=False, reason_code="location_unreachable",
                    reason=f"location {action.location} not in known locations",
                    allowed_next_actions=["speak", "wait"],
                )
        elif isinstance(action.location, list):
            coords = action.location
            if len(coords) not in (2, 3):
                return SafetyResult(
                    approved=False, reason_code="invalid_coordinates",
                    reason="coordinates must have 2 or 3 elements",
                    allowed_next_actions=["speak", "wait"],
                )

        return SafetyResult(approved=True, reason_code="place_allowed", reason="place allowed")

    def _check_speak(self, action: SpeakAction, world: WorldState) -> SafetyResult:
        if not action.content or not action.content.strip():
            return SafetyResult(
                approved=False, reason_code="empty_content",
                reason="speak content cannot be empty",
                allowed_next_actions=["wait"],
            )
        if len(action.content) > 500:
            return SafetyResult(
                approved=False, reason_code="content_too_long",
                reason=f"speak content exceeds 500 characters ({len(action.content)})",
                allowed_next_actions=["speak", "wait"],
            )
        return SafetyResult(approved=True, reason_code="speak_allowed", reason="speak allowed")
