"""
Robot Controller - main robot control loop.

Combines the LLM brain and robot body into a complete
perception-decision-execution loop.
"""

import time
import logging
from typing import List, Dict, Optional
from cade.config import Config
from cade.brain.llm_client import LLMClient
from cade.brain.prompts import get_system_prompt
from cade.brain.schemas import RobotDecision, RobotAction
from cade.body.robot import Robot
from cade.body.robot_interface import RobotInterface, RobotState

logger = logging.getLogger(__name__)


class RobotController:
    """
    Main robot controller.

    Responsibilities:
    1. Receive user input.
    2. Ask the LLM for a decision.
    3. Execute robot actions.
    4. Maintain conversation history.
    """

    def __init__(
        self,
        robot: Optional[RobotInterface] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_mode: str = "default",
        show_thought: bool = True
    ):
        self.robot = robot or Robot(name=Config.ROBOT_NAME)
        self.llm_client = llm_client or LLMClient()
        self.system_prompt = get_system_prompt(prompt_mode)
        self.conversation_history: List[Dict[str, str]] = []
        self.show_thought = show_thought

        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0

        logger.info("Robot controller initialized")

    def process_input(self, user_input: str) -> RobotDecision:
        """Process one user input through the full robot loop."""
        logger.info(f"User: {user_input}")
        self.total_interactions += 1

        self.robot.set_state(RobotState.THINKING)

        try:
            decision = self.llm_client.get_decision(
                user_input=user_input,
                system_prompt=self.system_prompt,
                conversation_history=self.conversation_history
            )

            if self.show_thought and decision.thought:
                logger.info(f"Reasoning: {decision.thought}")
            if decision.reply:
                logger.info(f"Reply: {decision.reply}")
            if decision.action:
                logger.info(f"Planned action: {decision.action.type}")

            if decision.action:
                success = self._execute_action(decision.action)
                if success:
                    self.successful_actions += 1
                else:
                    self.failed_actions += 1

            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })

            assistant_response_parts = []
            if decision.thought:
                assistant_response_parts.append(f"Reasoning: {decision.thought}")
            if decision.reply:
                assistant_response_parts.append(f"Reply: {decision.reply}")
            if decision.action:
                assistant_response_parts.append(f"Action: {decision.action.model_dump_json()}")

            assistant_response = "\n".join(assistant_response_parts)

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })

            return decision

        except Exception as e:
            logger.error(f"Error: {e}")
            self.robot.set_state(RobotState.ERROR)
            raise

    def _execute_action(self, action: RobotAction) -> bool:
        """Execute a concrete robot action."""
        action_type = action.type

        try:
            if action_type == "search":
                result = self.robot.search(action.object_name)
                return result is not None
            elif action_type == "pick":
                return self.robot.pick(action.object_name, action.object_id)
            elif action_type == "place":
                return self.robot.place(action.location)
            elif action_type == "speak":
                return self.robot.speak(action.content)
            elif action_type == "wait":
                return self.robot.wait(action.reason)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False

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

                self.process_input(user_input)

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
