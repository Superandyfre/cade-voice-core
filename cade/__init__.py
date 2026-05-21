"""CADE Voice Core — standalone voice interaction module (ROS-free)."""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("cade-voice-core")
except Exception:
    __version__ = "0.2.0"
