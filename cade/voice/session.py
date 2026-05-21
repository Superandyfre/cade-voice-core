"""
VoiceSession — the main voice interaction loop.

Wires ASR -> RobotController (LLM + actions) -> TTS into a continuous
microphone-driven conversation.

Usage::

    from cade.voice import VoiceSession
    session = VoiceSession()
    session.run_forever()  # blocks
"""

import time
import logging
import threading
from enum import Enum
from typing import Optional, Callable, Any, Dict

from cade.config import Config
from cade.controller import RobotController
from cade.asr.engine import ASREngine
from cade.tts.engine import TTSEngine

logger = logging.getLogger(__name__)

_FALLBACK_REPLY = "Sorry, I had trouble processing that."


class TTSPlaybackState(str, Enum):
    IDLE = "IDLE"
    SYNTHESIZING = "SYNTHESIZING"
    PLAYING = "PLAYING"
    TAIL_SUPPRESS = "TAIL_SUPPRESS"
    INTERRUPTED = "INTERRUPTED"


class VoiceSession:
    """
    End-to-end voice session.

    Lifecycle:
    1. ``run_forever()`` starts ASR continuous listening on the calling thread.
    2. Each recognized transcript calls ``process_transcript()``.
    3. ``process_transcript()`` runs the LLM decision -> action -> TTS pipeline.
    4. Echo suppression prevents the robot from hearing itself.
    """

    def __init__(
        self,
        controller: Optional[RobotController] = None,
        asr: Optional[ASREngine] = None,
        tts: Optional[TTSEngine] = None,
        echo_suppress_ms: Optional[int] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_decision: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.controller = controller or RobotController()
        self.asr = asr or ASREngine(
            model_dir=Config.ASR_MODEL_DIR,
            vad_model=Config.VAD_MODEL,
            model_type=Config.ASR_MODEL_TYPE,
            provider=Config.ASR_PROVIDER,
            fallback_model_type=Config.ASR_FALLBACK_MODEL_TYPE or None,
            fallback_model_dir=Config.ASR_FALLBACK_MODEL_DIR or None,
        )
        self.tts = tts or TTSEngine(
            model_dir=Config.TTS_MODEL_DIR,
            provider=Config.TTS_PROVIDER,
        )
        self.echo_suppress_ms = echo_suppress_ms if echo_suppress_ms is not None else Config.ECHO_SUPPRESS_AFTER_MS

        # Callbacks for observability
        self._on_transcript = on_transcript
        self._on_decision = on_decision

        # Echo-suppression state
        self._is_speaking = False
        self._speaking_until: float = 0.0
        self._tts_state = TTSPlaybackState.IDLE
        self._lock = threading.Lock()

        # Per-session metrics
        self.total_turns = 0
        self.total_errors = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_forever(
        self,
        device_name: Optional[str] = None,
        pulse_source: Optional[str] = None,
    ) -> None:
        """
        Start the voice session. Blocks the calling thread.

        Args:
            device_name: Audio input device override for sounddevice.
                         Defaults to Config.INPUT_DEVICE.
            pulse_source: PipeWire/PulseAudio source name (e.g. ``nx_remapped_out``).
                          When provided, uses ``parec`` instead of sounddevice so
                          virtual PipeWire sources are accessible.
                          Defaults to Config.INPUT_SOURCE.
        """
        logger.info("Voice session starting — speak into the microphone")
        logger.info(f"  Echo suppress: {self.echo_suppress_ms}ms")

        source = pulse_source or Config.INPUT_SOURCE
        if source:
            logger.info(f"  Input: PipeWire source '{source}' (parec)")
            self._run_pulse(source)
        else:
            device = device_name or Config.INPUT_DEVICE
            resolved = self._resolve_pulse_default(device)
            if resolved:
                logger.info(f"  Input: PipeWire default source '{resolved}' (parec)")
                self._run_pulse(resolved)
            else:
                logger.info(f"  Input: sounddevice '{device}'")
                self._run_sounddevice(device)

    @staticmethod
    def _resolve_pulse_default(device_name: str) -> Optional[str]:
        if device_name.lower() not in ("default", "pulse", "pulseaudio", "pipewire"):
            return None
        try:
            import subprocess
            result = subprocess.run(
                ["pactl", "info"], capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                return None
            for line in result.stdout.splitlines():
                if line.startswith("Default Source:"):
                    name = line.split(":", 1)[1].strip()
                    if name:
                        return name
        except FileNotFoundError:
            pass
        return None

    def _run_sounddevice(self, device_name: str) -> None:
        try:
            self.asr.start_listening(
                callback=self._on_asr_callback,
                device_name=device_name,
            )
        except KeyboardInterrupt:
            logger.info("Voice session stopped by user")

    def _run_pulse(self, source_name: str) -> None:
        try:
            self.asr.start_listening_pulse(
                callback=self._on_asr_callback,
                source_name=source_name,
            )
        except KeyboardInterrupt:
            logger.info("Voice session stopped by user")

    # ------------------------------------------------------------------
    # ASR callback (called from ASR thread)
    # ------------------------------------------------------------------

    def _on_asr_callback(self, transcript: str) -> None:
        """Called by the ASR engine for each recognized utterance."""
        if not transcript.strip():
            return

        # Echo gate: ignore audio captured while / just after we are speaking
        with self._lock:
            current_tts_state = getattr(self, "_tts_state", TTSPlaybackState.IDLE)
            is_speaking = self._is_speaking or current_tts_state in {
                TTSPlaybackState.SYNTHESIZING,
                TTSPlaybackState.PLAYING,
            }
            in_tail = time.monotonic() < self._speaking_until
            if is_speaking and Config.BARGE_IN_ENABLED:
                self._tts_state = TTSPlaybackState.INTERRUPTED
                self._is_speaking = False
                self._speaking_until = 0.0
                try:
                    self.tts.stop()
                except Exception:
                    logger.debug("Ignoring TTS stop failure during barge-in", exc_info=True)
            elif is_speaking or in_tail:
                logger.debug(f"Echo gate active, ignoring: {transcript!r}")
                return

        if self._on_transcript:
            self._on_transcript(transcript)

        self.process_transcript(transcript)

    # ------------------------------------------------------------------
    # Transcript processing
    # ------------------------------------------------------------------

    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Run the full pipeline for one transcript.

        Returns the same dict as ``RobotController.process_input()``,
        with an extra ``fallback`` key when the error path was taken.
        """
        self.total_turns += 1
        t_start = time.monotonic()

        try:
            # 1. LLM decision + action
            result = self.controller.process_input(transcript)
            spoken_text = result.get("spoken_text")

            if self._on_decision:
                self._on_decision(result)

            # 2. TTS playback
            if spoken_text:
                self._speak(spoken_text, profile="dialogue")

            return result

        except Exception:
            logger.exception("Error in voice turn")
            self.total_errors += 1
            self._speak(_FALLBACK_REPLY, profile="error")
            return {
                "decision": None,
                "action_success": False,
                "spoken_text": _FALLBACK_REPLY,
                "timings": {"total_s": time.monotonic() - t_start},
                "fallback": True,
            }

    # ------------------------------------------------------------------
    # TTS + echo suppression
    # ------------------------------------------------------------------

    def _speak(self, text: str, profile: str = "dialogue") -> None:
        """Speak text and manage echo-suppression window."""
        with self._lock:
            self._is_speaking = True
            self._tts_state = TTSPlaybackState.SYNTHESIZING

        try:
            playback_s, audio_s = self.tts.speak(
                text,
                device=self._resolve_output_device(),
                profile=profile,
            )
            with self._lock:
                self._tts_state = TTSPlaybackState.PLAYING
            logger.info(f"Spoke ({audio_s:.1f}s audio): {text}")
        except Exception:
            logger.exception("TTS playback failed")
        finally:
            with self._lock:
                self._is_speaking = False
                self._tts_state = TTSPlaybackState.TAIL_SUPPRESS
                # Suppress ASR for a short window after playback ends
                self._speaking_until = (
                    time.monotonic() + self.echo_suppress_ms / 1000.0
                )

    @staticmethod
    def _resolve_output_device() -> Optional[int]:
        """Resolve Config.OUTPUT_DEVICE to a sounddevice index (or None)."""
        target = Config.OUTPUT_DEVICE
        if target in ("default", ""):
            return None
        try:
            import sounddevice as sd
            from cade.tts.audio_utils import select_output_device
            devices = sd.query_devices()
            return select_output_device(devices, target)
        except Exception:
            return None
