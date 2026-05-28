"""Shared real-device runtime helpers for ordering voice flows."""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from cade.config import Config, RunMode

logger = logging.getLogger(__name__)


class VoiceRuntimeError(RuntimeError):
    """Raised when the real-device ordering runtime cannot be constructed."""


class VoiceMonitor(Protocol):
    """Optional observer hooks for the shared ordering voice runtime."""

    def record_transcript(self, text: str) -> None: ...

    def can_accept_transcript(self) -> bool: ...


@dataclass
class ResolvedAudioDevice:
    kind: str
    value: str | int
    rate: int | None = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"kind": self.kind}
        if self.kind == "pulse":
            payload["source" if self.rate is not None else "sink"] = self.value
            if self.rate is not None:
                payload["rate"] = self.rate
        else:
            payload["device"] = self.value
        return payload


class SpeakingGate:
    """Tracks TTS playback and rejects likely echo transcripts."""

    def __init__(
        self,
        echo_suppress_ms: int,
        *,
        similarity_threshold: float = 0.75,
        similarity_window_sec: float = 2.5,
        recent_text_limit: int = 4,
    ):
        self._echo_suppress_s = max(0.0, echo_suppress_ms / 1000.0)
        self._similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self._similarity_window_sec = max(0.0, similarity_window_sec)
        self._is_speaking = False
        self._suppress_until = 0.0
        self._recent_texts: deque[tuple[float, str]] = deque(maxlen=max(1, recent_text_limit))
        self._lock = threading.Lock()

    def begin(self, text: str = "") -> None:
        with self._lock:
            self._is_speaking = True
            self._remember_text_locked(text)

    def end(self, text: str = "") -> None:
        with self._lock:
            self._is_speaking = False
            self._suppress_until = time.monotonic() + self._echo_suppress_s
            self._remember_text_locked(text)

    def is_blocked(self, text: Optional[str] = None) -> bool:
        with self._lock:
            if self._is_speaking or time.monotonic() < self._suppress_until:
                return True
            if text and self._looks_like_recent_tts_locked(text):
                return True
            return False

    def _remember_text_locked(self, text: str) -> None:
        normalized = self._normalize_text(text)
        if normalized:
            self._recent_texts.append((time.monotonic(), normalized))

    def _looks_like_recent_tts_locked(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False
        now = time.monotonic()
        for ts, prior in list(self._recent_texts):
            if (now - ts) > self._similarity_window_sec:
                continue
            if prior == normalized:
                return True
            ratio = SequenceMatcher(None, normalized, prior).ratio()
            if ratio >= self._similarity_threshold:
                return True
        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = " ".join(str(text or "").strip().lower().split())
        if not lowered:
            return ""
        filtered = []
        for ch in lowered:
            filtered.append(ch if ch.isalnum() or ch.isspace() else " ")
        return " ".join("".join(filtered).split())


class RealTTSSink:
    """Blocking TTS sink used by the FSM success path."""

    def __init__(self, engine: Any, output: Dict[str, Any], gate: SpeakingGate):
        self._engine = engine
        self._output = output
        self._gate = gate

    def speak(self, text: str, profile: str = "dialogue"):
        self._gate.begin(text)
        try:
            if self._output["kind"] == "pulse":
                fd, wav_path = tempfile.mkstemp(prefix="cade-order-tts-", suffix=".wav")
                os.close(fd)
                try:
                    return self._engine.speak_detailed(
                        text,
                        save_path=wav_path,
                        pulse_sink=self._output["sink"],
                        profile=profile,
                    )
                finally:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
            return self._engine.speak_detailed(
                text,
                device=self._output["device"],
                profile=profile,
            )
        finally:
            self._gate.end(text)


class OrderingVoiceRuntime:
    """Owns the real-device ordering stack used by voice and E2E entrypoints."""

    def __init__(
        self,
        *,
        asr_engine: Any,
        tts_engine: Any,
        llm: Any,
        fsm: Any,
        zmq_runtime: Any,
        gate: SpeakingGate,
        input_source: Dict[str, Any],
        output: Dict[str, Any],
        monitor: Optional[VoiceMonitor] = None,
    ):
        self.asr_engine = asr_engine
        self.tts_engine = tts_engine
        self.llm = llm
        self.fsm = fsm
        self.zmq_runtime = zmq_runtime
        self.gate = gate
        self.input_source = input_source
        self.output = output
        self.monitor = monitor
        self.asr_errors: queue.Queue = queue.Queue()
        self._asr_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self.zmq_runtime.start()
        self._start_asr_thread()
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.zmq_runtime.stop()
        try:
            self.tts_engine.close()
        except Exception:
            logger.debug("Ignoring TTS close failure", exc_info=True)

    def spin(self) -> None:
        try:
            while True:
                if not self.asr_errors.empty():
                    raise VoiceRuntimeError(f"ASR failed: {self.asr_errors.get()}")
                time.sleep(0.2)
        except KeyboardInterrupt:
            raise

    def _start_asr_thread(self) -> None:
        def on_transcript(text: str) -> None:
            transcript = str(text or "").strip()
            if not transcript:
                return
            if self.gate.is_blocked(transcript):
                logger.info("Ignoring ASR transcript during TTS playback/echo window: %r", transcript)
                self.fsm.asr_echo_block_total += 1
                return
            if self.monitor and hasattr(self.monitor, "can_accept_transcript"):
                try:
                    if not self.monitor.can_accept_transcript():
                        logger.info("Ignoring ASR transcript while monitor blocks input: %r", transcript)
                        return
                except Exception:
                    logger.debug("Monitor can_accept_transcript failed", exc_info=True)
            if not self.fsm.is_accepting_live_input():
                logger.info("Ignoring ASR transcript outside LISTEN/CHECK: %r", transcript)
                return
            if self.monitor:
                try:
                    self.monitor.record_transcript(transcript)
                except Exception:
                    logger.debug("Monitor record_transcript failed", exc_info=True)
            self.fsm.handle_user_text(transcript, source="asr_microphone")

        def run() -> None:
            try:
                if self.input_source["kind"] == "pulse":
                    self.asr_engine.start_listening_pulse(
                        callback=on_transcript,
                        source_name=self.input_source["source"],
                        source_rate=self.input_source["rate"],
                    )
                else:
                    self.asr_engine.start_listening(
                        callback=on_transcript,
                        device_name=str(self.input_source["device"]),
                    )
            except Exception as exc:
                self.asr_errors.put(exc)

        self._asr_thread = threading.Thread(target=run, name="cade-order-asr", daemon=True)
        self._asr_thread.start()


def configure_runtime_mode(*, cloud: bool = False, local: bool = False, input_device: Optional[str] = None, output_device: Optional[str] = None) -> None:
    if cloud:
        Config.MODE = RunMode.CLOUD
    elif local:
        Config.MODE = RunMode.LOCAL
    if input_device:
        Config.INPUT_DEVICE = input_device
    if output_device:
        Config.OUTPUT_DEVICE = output_device


def load_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise VoiceRuntimeError("sounddevice is required for real audio device verification") from exc
    return sd


def pulse_sinks() -> list[str]:
    return _pulse_list("sinks")


def pulse_sources() -> list[str]:
    return _pulse_list("sources")


def _pulse_list(kind: str) -> list[str]:
    if not shutil.which("pactl"):
        return []
    result = subprocess.run(["pactl", "list", "short", kind], check=False, text=True, capture_output=True)
    if result.returncode != 0:
        return []
    names = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.append(parts[1])
    return names


def pulse_default_sink() -> Optional[str]:
    return _pulse_default("Default Sink:")


def pulse_default_source() -> Optional[str]:
    return _pulse_default("Default Source:")


def _pulse_default(prefix: str) -> Optional[str]:
    if not shutil.which("pactl"):
        return None
    result = subprocess.run(["pactl", "info"], check=False, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith(prefix):
            name = line.split(":", 1)[1].strip()
            return name or None
    return None


def resolve_audio_device(sd: Any, selector: str, direction: str) -> int:
    from cade.tts.audio_utils import select_input_device, select_output_device

    devices = sd.query_devices()
    if direction == "input":
        idx = select_input_device(devices, selector)
        key = "max_input_channels"
    else:
        idx = select_output_device(devices, selector)
        key = "max_output_channels"

    if idx is None:
        raise VoiceRuntimeError(f"Audio {direction} device not found: {selector}")
    device = devices[idx]
    if int(device.get(key, 0) or 0) <= 0:
        raise VoiceRuntimeError(f"Audio {direction} device has no usable channels: {selector}")
    return int(idx)


def resolve_asr_input(sd: Any, selector: str) -> Dict[str, Any]:
    target = str(selector or "default").strip()
    target_lower = target.lower()
    sources = pulse_sources()

    if not target.isdigit() and sources:
        if target_lower in ("default", "pulse", "pulseaudio", "pipewire"):
            default_source = pulse_default_source()
            if default_source in sources:
                return {"kind": "pulse", "source": default_source, "rate": 48000}
        for source in sources:
            if source.endswith(".monitor"):
                continue
            source_lower = source.lower()
            if target_lower == source_lower or target_lower in source_lower:
                return {"kind": "pulse", "source": source, "rate": 48000}

    return {"kind": "sounddevice", "device": resolve_audio_device(sd, target, "input")}


def resolve_tts_output(sd: Any, selector: str) -> Dict[str, Any]:
    target = str(selector or "default").strip()
    target_lower = target.lower()
    sinks = pulse_sinks()

    if not target.isdigit() and sinks:
        if target_lower in ("default", "pulse", "pulseaudio", "pipewire"):
            default_sink = pulse_default_sink()
            if default_sink in sinks:
                return {"kind": "pulse", "sink": default_sink}
        for sink in sinks:
            sink_lower = sink.lower()
            if target_lower == sink_lower or target_lower in sink_lower:
                return {"kind": "pulse", "sink": sink}

    return {"kind": "sounddevice", "device": resolve_audio_device(sd, target, "output")}


def probe_audio_device(sd: Any, idx: int, direction: str) -> None:
    device = sd.query_devices(idx)
    sample_rate = int(device["default_samplerate"])
    if direction == "input":
        with sd.InputStream(device=idx, channels=1, samplerate=sample_rate, dtype="float32"):
            return
    with sd.OutputStream(device=idx, channels=1, samplerate=sample_rate, dtype="float32"):
        return


def probe_asr_input(input_source: Dict[str, Any]) -> None:
    if input_source["kind"] != "pulse":
        return
    if not shutil.which("parec"):
        raise VoiceRuntimeError("parec is required for Pulse/PipeWire input")
    if input_source["source"] not in pulse_sources():
        raise VoiceRuntimeError(f"Pulse/PipeWire source not found: {input_source['source']}")


def probe_tts_output(output: Dict[str, Any]) -> None:
    if output["kind"] != "pulse":
        return
    if not shutil.which("paplay"):
        raise VoiceRuntimeError("paplay is required for Pulse/PipeWire output")
    if output["sink"] not in pulse_sinks():
        raise VoiceRuntimeError(f"Pulse/PipeWire sink not found: {output['sink']}")


def print_audio_target(label: str, sd: Any, device_idx: int) -> None:
    device = sd.query_devices(device_idx)
    print(f"{label}: [{device_idx}] {device['name']} @ {int(device['default_samplerate'])} Hz")


def print_asr_input(sd: Any, input_source: Dict[str, Any]) -> None:
    if input_source["kind"] == "pulse":
        print(f"Input: Pulse/PipeWire source {input_source['source']}")
        return
    print_audio_target("Input", sd, int(input_source["device"]))


def print_tts_output(sd: Any, output: Dict[str, Any]) -> None:
    if output["kind"] == "pulse":
        print(f"Output: Pulse/PipeWire sink {output['sink']}")
        return
    print_audio_target("Output", sd, int(output["device"]))


def build_ordering_voice_runtime(
    *,
    pub_bind: str,
    router_bind: str,
    input_device: Optional[str] = None,
    output_device: Optional[str] = None,
    monitor: Optional[VoiceMonitor] = None,
    verify_audio: bool = True,
) -> OrderingVoiceRuntime:
    sd = load_sounddevice()
    input_selector = input_device or Config.INPUT_DEVICE
    output_selector = output_device or Config.OUTPUT_DEVICE
    input_source = resolve_asr_input(sd, input_selector)
    output = resolve_tts_output(sd, output_selector)

    if verify_audio:
        if input_source["kind"] == "sounddevice":
            probe_audio_device(sd, int(input_source["device"]), "input")
        else:
            probe_asr_input(input_source)
        if output["kind"] == "sounddevice":
            probe_audio_device(sd, int(output["device"]), "output")
        else:
            probe_tts_output(output)

    from cade.asr.engine import ASREngine
    from cade.brain.llm_client import LLMClient
    from cade.fsm.config import OrderFSMConfig
    from cade.fsm.order_fsm import CallbackTTSSink, OrderSubFSM
    from cade.fsm.zmq_runtime import ZmqRuntime
    from cade.tts.engine import TTSEngine

    llm = LLMClient()
    asr_engine = ASREngine(
        model_dir=Config.ASR_MODEL_DIR,
        vad_model=Config.VAD_MODEL,
        model_type=Config.ASR_MODEL_TYPE,
        provider=Config.ASR_PROVIDER,
        fallback_model_type=Config.ASR_FALLBACK_MODEL_TYPE or None,
        fallback_model_dir=Config.ASR_FALLBACK_MODEL_DIR or None,
    )
    tts_engine = TTSEngine(model_dir=Config.TTS_MODEL_DIR, provider=Config.TTS_PROVIDER)
    gate = SpeakingGate(
        Config.ECHO_SUPPRESS_MS,
        similarity_threshold=float(getattr(Config, "ECHO_SIMILARITY_THRESHOLD", 0.75)),
        similarity_window_sec=float(getattr(Config, "ECHO_SIMILARITY_WINDOW_SEC", 2.5)),
    )
    fsm_config = OrderFSMConfig(zmq_pub_bind=pub_bind, zmq_router_bind=router_bind)
    fsm = OrderSubFSM(
        llm_client=llm,
        config=fsm_config,
        tts_sink=CallbackTTSSink(RealTTSSink(tts_engine, output, gate).speak),
    )
    runtime = ZmqRuntime(
        fsm=fsm,
        pub_bind=pub_bind,
        router_bind=router_bind,
        idempotency_path=fsm_config.idempotency_cache_file or None,
        idempotency_ttl_sec=fsm_config.idempotency_ttl_sec,
    )
    return OrderingVoiceRuntime(
        asr_engine=asr_engine,
        tts_engine=tts_engine,
        llm=llm,
        fsm=fsm,
        zmq_runtime=runtime,
        gate=gate,
        input_source=input_source,
        output=output,
        monitor=monitor,
    )
