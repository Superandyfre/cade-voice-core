"""Regression: ensure no ROS imports in production code."""

import subprocess
import pytest


def test_no_ros_imports_in_production():
    """Verify the cade/ package has no rospy/std_msgs/roslaunch references."""
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "-l",
         "-E", r"(import rospy|from rospy|import std_msgs|from std_msgs|import roslaunch|from roslaunch)",
         "cade/"],
        capture_output=True,
        text=True,
        cwd="/home/pinggu/audio/cade-voice-core",
    )
    # grep returns 1 when no matches found (which is what we want)
    assert result.returncode == 1, (
        f"Found ROS imports in production code:\n{result.stdout}"
    )
