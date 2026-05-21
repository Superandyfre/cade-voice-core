"""
Audio device selection utilities (shared by ASR and TTS).
"""

from typing import Optional


def select_input_device(devices, target_name: str) -> Optional[int]:
    return select_device(devices, target_name, "input")


def select_output_device(devices, target_name: str) -> Optional[int]:
    return select_device(devices, target_name, "output")


def select_device(devices, target_name: str, direction: str) -> Optional[int]:
    """
    Select a sounddevice index by name.

    Args:
        devices: List of device dicts from sd.query_devices().
        target_name: Device name to search for.
        direction: "input" or "output".

    Returns:
        Device index or None.
    """
    target = str(target_name or "default").strip()
    key = "max_input_channels" if direction == "input" else "max_output_channels"

    def has_channels(device):
        return int(device.get(key, 0) or 0) > 0

    if target.isdigit():
        idx = int(target)
        if 0 <= idx < len(devices) and has_channels(devices[idx]):
            return idx
        return None

    target_lower = target.lower()
    preferred = []
    if target_lower == "default":
        idx = _sounddevice_default_index(devices, direction)
        if idx is not None and has_channels(devices[idx]):
            return idx
        preferred = ["default", "pulse"]
    elif target_lower in ("pulse", "pulseaudio"):
        preferred = ["pulse", "default"]

    for name in preferred:
        idx = _find_exact(devices, name, has_channels)
        if idx is not None:
            return idx

    idx = _find_exact(devices, target_lower, has_channels)
    if idx is not None:
        return idx

    if not preferred:
        idx = _find_substring(devices, target_lower, has_channels)
        if idx is not None:
            return idx

    return None


def _sounddevice_default_index(devices, direction) -> Optional[int]:
    try:
        import sounddevice as sd
    except ImportError:
        return None

    default_device = getattr(sd.default, "device", None)
    if default_device is None:
        return None
    try:
        idx = default_device[0] if direction == "input" else default_device[1]
    except (TypeError, IndexError):
        idx = default_device
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(devices):
        return idx
    return None


def _find_exact(devices, target_lower, predicate):
    for i, device in enumerate(devices):
        if str(device.get("name", "")).strip().lower() == target_lower and predicate(device):
            return i
    return None


def _find_substring(devices, target_lower, predicate):
    for i, device in enumerate(devices):
        if target_lower in str(device.get("name", "")).lower() and predicate(device):
            return i
    return None
