"""
Robot Controller - main robot control loop.

Combines the LLM brain and robot body into a complete
perception-decision-execution loop.
"""

import time
import logging
from typing import List, Dict, Optional, Any
from cade.config import Config
from cade.brain.llm_client import LLMClient
from cade.brain.prompts import get_system_prompt
from cade.brain.schemas import RobotDecision, RobotAction, ActionResult, SafetyResult
from cade.body.robot import Robot
from cade.body.robot_interface import RobotInterface, RobotState
from cade.body.safety import ActionSafetyGate
from cade.body.world_state import WorldStateProvider, RobotWorldStateProvider

logger = logging.getLogger(__name__)

# Maximum conversation turns to keep (each turn = 1 user + 1 assistant message).
_MAX_HISTORY_TURNS = 8


class RobotController:
    """
    Main robot controller.

    Responsibilities:
    1. Receive user input.
    2. Ask the LLM for a decision.
    3. Execute robot actions.
    4. Maintain conversation history (bounded).
    """

    def __init__(
        self,
        robot: Optional[RobotInterface] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_mode: str = "default",
        show_thought: bool = True,
        world_state_provider: Optional[WorldStateProvider] = None,
        safety_gate: Optional[ActionSafetyGate] = None,
    ):
        self.robot = robot or Robot(name=Config.ROBOT_NAME)
        self.llm_client = llm_client or LLMClient()
        self.system_prompt = get_system_prompt(prompt_mode)
        self.conversation_history: List[Dict[str, str]] = []
        self.show_thought = show_thought

        self._world_state_provider = world_state_provider or RobotWorldStateProvider(self.robot)
        self._safety_gate = safety_gate or ActionSafetyGate()

        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0

        logger.info("Robot controller initialized")

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process one user input through the full robot loop.

        Returns a dict with:
            decision:        RobotDecision
            action_success:  bool | None
            spoken_text:     str | None   (what should be spoken via TTS)
            timings:         dict with llm_latency_s, action_latency_s
        """
        logger.info(f"User: {user_input}")
        self.total_interactions += 1

        self.robot.set_state(RobotState.THINKING)

        timings: Dict[str, float] = {}
        t0 = time.monotonic()

        try:
            decision = self.llm_client.get_decision(
                user_input=user_input,
                system_prompt=self.system_prompt,
                conversation_history=self.conversation_history,
            )
            timings["llm_latency_s"] = time.monotonic() - t0

            if self.show_thought and decision.thought:
                logger.info(f"Reasoning: {decision.thought}")
            if decision.reply:
                logger.info(f"Reply: {decision.reply}")
            if decision.action:
                logger.info(f"Planned action: {decision.action.type}")

            # Execute action
            action_success: Optional[bool] = None
            action_result: Optional[ActionResult] = None
            if decision.action:
                t1 = time.monotonic()
                action_result = self._execute_action(decision.action)
                timings["action_latency_s"] = time.monotonic() - t1
                action_success = action_result.success
                if action_success:
                    self.successful_actions += 1
                else:
                    self.failed_actions += 1

            # Determine spoken text — reply takes priority over speak-action content
            spoken_text: Optional[str] = None
            if decision.reply:
                spoken_text = decision.reply
            elif decision.action and decision.action.type == "speak":
                spoken_text = decision.action.content

            # Append to history (compact: no verbose thought)
            self._append_history(user_input, decision, action_result)

            return {
                "decision": decision,
                "action_success": action_success,
                "action_result": action_result,
                "spoken_text": spoken_text,
                "timings": timings,
            }

        except Exception as e:
            logger.error(f"Error: {e}")
            self.robot.set_state(RobotState.ERROR)
            raise

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _append_history(self, user_input: str, decision: RobotDecision, action_result: Optional[ActionResult] = None) -> None:
        """Append a turn to conversation history and trim if needed."""
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })

        parts: List[str] = []
        if decision.reply:
            parts.append(decision.reply)
        if decision.action:
            parts.append(f"[action: {decision.action.model_dump_json()}]")
        if action_result and not action_result.success:
            parts.append(f"[action_result: success=false, status={action_result.status}, observation={action_result.observation}]")

        self.conversation_history.append({
            "role": "assistant",
            "content": "\n".join(parts) if parts else "(no output)",
        })

        # Trim to last N turns (2 messages per turn)
        max_messages = _MAX_HISTORY_TURNS * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _execute_action(self, action: RobotAction) -> ActionResult:
        """Execute a concrete robot action with safety gate check."""
        action_type = action.type
        action_dump = action.model_dump() if hasattr(action, "model_dump") else dict(action)

        # Safety gate check
        world_state = self._world_state_provider.get_world_state()
        safety = self._safety_gate.validate(action, world_state)
        if not safety.approved:
            logger.warning(f"Action blocked by safety gate: {safety.reason_code} - {safety.reason}")
            return ActionResult(
                success=False,
                status="blocked_by_safety",
                observation=safety.reason,
                action=action_dump,
                suggested_next_actions=safety.allowed_next_actions,
            )

        try:
            if action_type == "search":
                result = self.robot.search(action.object_name)
                if result is not None:
                    return ActionResult(success=True, status="completed", observation=f"found {action.object_name}", action=action_dump)
                else:
                    return ActionResult(success=False, status="object_not_found", observation=f"could not find {action.object_name}", action=action_dump)
            elif action_type == "pick":
                ok = self.robot.pick(action.object_name, action.object_id)
                if ok:
                    return ActionResult(success=True, status="completed", observation=f"picked up {action.object_name}", action=action_dump)
                else:
                    return ActionResult(success=False, status="grasp_failed", observation=f"failed to pick {action.object_name}", action=action_dump)
            elif action_type == "place":
                ok = self.robot.place(action.location)
                if ok:
                    return ActionResult(success=True, status="completed", observation=f"placed at {action.location}", action=action_dump)
                else:
                    return ActionResult(success=False, status="execution_error", observation=f"failed to place at {action.location}", action=action_dump)
            elif action_type == "speak":
                ok = self.robot.speak(action.content)
                return ActionResult(success=bool(ok), status="completed" if ok else "execution_error", observation=f"spoke: {action.content[:50]}", action=action_dump)
            elif action_type == "wait":
                ok = self.robot.wait(action.reason)
                return ActionResult(success=bool(ok), status="completed", observation="waited", action=action_dump)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return ActionResult(success=False, status="execution_error", observation=f"unknown action type: {action_type}", action=action_dump)
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return ActionResult(success=False, status="execution_error", observation=str(e), action=action_dump)

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    def interactive_mode(self):
        """Run the command-line interactive mode."""
        print(f"Entering interactive mode (type 'quit' to exit)\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.print_statistics()
                    break

                if user_input.lower() == 'status':
                    self.robot.print_status()
                    continue

                if user_input.lower() == 'stats':
                    self.print_statistics()
                    continue

                result = self.process_input(user_input)
                if result["spoken_text"]:
                    print(f"Robot: {result['spoken_text']}")

            except KeyboardInterrupt:
                self.print_statistics()
                break

            except Exception as e:
                logger.error(f"Runtime error: {e}")

    def run_test_scenario(self, scenarios: List[str]):
        """Run scripted test scenarios."""
        for i, scenario in enumerate(scenarios, 1):
            try:
                self.process_input(scenario)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Test failed: {e}")

        self.print_statistics()

    def print_statistics(self):
        """Print controller statistics."""
        logger.info(
            f"Stats - interactions: {self.total_interactions}, "
            f"success: {self.successful_actions}, "
            f"failed: {self.failed_actions}"
        )

    def reset(self):
        """Reset controller state."""
        self.conversation_history.clear()
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0
