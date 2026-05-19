#!/usr/bin/env python3
"""
CADE ROS Entry Point - ROS voice mode startup entry

Usage:
    rosrun cade main_ros.py
Or
    python main_ros.py

Purpose:
    Start the ROS voice bridge for the voice interaction loop
"""

import sys
import argparse
from bridge.ros_voice_bridge import RosVoiceBridge
from config import Config


def print_banner():
    """Print the welcome banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 CADE - Embodied Robot System (ROS Voice Mode)           ║
║   Cognitive Agent for Domestic Environment                ║
║                                                           ║
║   Project: Project LARA                                      ║
║   Version: 0.1.0                                             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"Run mode: {'☁️  Cloud' if Config.is_cloud_mode() else '💻 Local'}")
    print(f"Model: {Config.get_llm_config()['model']}")
    print(f"Robot: {Config.ROBOT_NAME}")
    print(f"Input source: ROS /asr topic")
    print(f"Output target: ROS /tts topic")
    print()


def print_test_cases():
    """Print test cases"""
    test_cases = """
╔═══════════════════════════════════════════════════════════╗
║                    📋 Test Phrases (say these aloud)                ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  [Basic Conversation]                                              ║
║   1. "hello"                                               ║
║   2. "what is your name"                                        ║
║   3. "what can you do"                                          ║
║                                                           ║
║  [Navigation Tasks]                                              ║
║   4. "go to the kitchen"                                             ║
║   5. "go back to the start point"                                           ║
║                                                           ║
║  [Search Tasks]                                              ║
║   6. "help me find the apple"                                         ║
║   7. "find the cup"                                           ║
║                                                           ║
║  [Multi-step Tasks]                                              ║
║   8. "put the apple on the table"                                    ║
║   9. "I am thirsty, please bring me a bottle of water"                                  ║
║                                                           ║
║  [Small Talk]                                                  ║
║  10. "how is the weather today"                                      ║
║  11. "tell me a joke"                                        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(test_cases)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CADE embodied robot system - ROS voice mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_ros.py                    # default mode
    python main_ros.py --mode debug       # debug mode
    python main_ros.py --no-thought       # hide reasoning output
    python main_ros.py --env "you are at home"   # custom environment context

Prerequisites:
    1. ROS master is running
    2. ASR node is publishing to /asr
    3. TTS node is subscribing to /tts

Startup order:
    1. roscore
    2. roslaunch asr_tts speech.launch   # start ASR and TTS nodes
    3. python main_ros.py                 # start the CADE controller
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['default', 'simple', 'compact', 'debug'],
        default='default',
        help='Prompt mode (default=standard, compact=short, debug=verbose)'
    )

    parser.add_argument(
        '--no-thought',
        action='store_true',
        help='Do not show LLM reasoning output'
    )

    parser.add_argument(
        '--env',
        type=str,
        default="You are sitting on a table in the Fedora lab. At the moment, you can only interact with people through voice.",
        help='Environment context injected into the system prompt'
    )

    args = parser.parse_args()

    print_banner()
    print_test_cases()

    try:
        # Create and start the bridge
        bridge = RosVoiceBridge(
            prompt_mode=args.mode,
            show_thought=not args.no_thought,
            environment_context=args.env
        )

        print("\n✓ CADE is ready and waiting for voice input...\n")
        print("Tips:")
        print("  - Speak into the microphone. You can use the test phrases above.")
        print("  - Watch the terminal output to confirm speech recognition.")
        print("  - Press Ctrl+C to exit.")
        print()

        # Enter the ROS main loop
        bridge.spin()

    except KeyboardInterrupt:
        print("\n\n👋 Program exited")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
