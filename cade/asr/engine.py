"""
ASR Engine — standalone speech recognition engine (ROS-free).

Uses sherpa-onnx with Silero VAD for offline or streaming speech recognition.
"""

import wave
import time
import logging
import queue
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Callable, Tuple

import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = logging.getLogger(__name__)


class ASREngine:
    """
    Standalone ASR engine with VAD-based voice activity detection.

    Supports multiple model architectures via sherpa-onnx:
    Whisper, SenseVoice, Paraformer, Transducer, Streaming Zipformer.
    """

    def __init__(
        self,
        model_dir: str,
        vad_model: Optional[str] = None,
        model_type: str = "whisper",
        provider: str = "cpu",
        sample_rate: int = 16000,
        num_threads: int = 2,
        language: str = "en",
        fallback_model_type: Optional[str] = None,
        fallback_model_dir: Optional[str] = None,
        text_replacements: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Initialize the ASR engine.

        Args:
            model_dir: Path to the model directory.
            vad_model: Path to silero_vad.onnx. If None, looks in model_dir/../silero_vad.onnx.
            model_type: Model architecture: whisper, sense_voice, paraformer, transducer,
                        streaming_zipformer, streaming_nemotron.
            provider: Inference provider: cpu, cuda, coreml.
            sample_rate: Target sample rate.
            num_threads: Number of compute threads.
            language: Language code for Whisper models.
            fallback_model_type: If set, attempt loading this model when primary fails.
            fallback_model_dir: Model directory for the fallback model.
        """
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate
        self._text_replacements = text_replacements
        self._needs_text_norm = model_type == "streaming_nemotron"
        self._metrics = {
            "decode_count": 0,
            "decode_total_s": 0.0,
            "endpoint_count": 0,
            "endpoint_nonempty_count": 0,
            "endpoint_empty_count": 0,
            "consecutive_empty_finals": 0,
            "max_consecutive_empty": 0,
        }

        if sherpa_onnx is None:
            raise ImportError("sherpa-onnx is required for ASR. Install with: pip install sherpa-onnx")
        if sd is None:
            raise ImportError("sounddevice is required for audio capture. Install with: pip install sounddevice")

        if vad_model is None:
            vad_model = str(self.model_dir.parent / "silero_vad.onnx")
        self._assert_file(vad_model)

        self.active_model_name = model_type
        try:
            self._is_streaming = model_type in ("streaming_zipformer", "streaming_nemotron")
            self._recognizer = self._create_recognizer(
                model_type, provider, num_threads, language
            )
        except Exception as exc:
            if fallback_model_type and fallback_model_dir:
                logger.warning(f"Failed to load primary ASR model {model_type}: {exc}")
                logger.info(f"Attempting fallback to {fallback_model_type} from {fallback_model_dir}")
                self.model_dir = Path(fallback_model_dir)
                self._is_streaming = fallback_model_type in ("streaming_zipformer", "streaming_nemotron")
                self._recognizer = self._create_recognizer(
                    fallback_model_type, provider, num_threads, language
                )
                self.active_model_name = fallback_model_type
                logger.info(f"ASR fallback successful: using {fallback_model_type}")
            else:
                raise

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model
        vad_config.silero_vad.min_silence_duration = 0.25
        vad_config.sample_rate = sample_rate
        self._vad_window_size = vad_config.silero_vad.window_size
        self._vad_config = vad_config

        logger.info(f"ASR engine initialized: {self.active_model_name}, provider={provider}, "
                     f"streaming={self._is_streaming}")

    def _normalize_text(self, text: str) -> str:
        if not self._needs_text_norm:
            return text
        from cade.asr.text_norm import normalize_asr_text
        return normalize_asr_text(text, self._text_replacements)

    def get_metrics(self) -> dict:
        return dict(self._metrics)

    def _assert_file(self, path: str):
        assert Path(path).is_file(), f"File not found: {path}"

    def _create_recognizer(self, model_type, provider, num_threads, language):
        """Create a sherpa-onnx recognizer based on model type."""
        d = self.model_dir

        if model_type == "whisper":
            encoder = next(d.glob("*encoder*"), None)
            decoder = next(d.glob("*decoder*"), None)
            tokens = next(d.glob("*tokens*"), None)
            if not all([encoder, decoder, tokens]):
                raise FileNotFoundError(f"Whisper model files not found in {d}")
            return sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(encoder),
                decoder=str(decoder),
                tokens=str(tokens),
                num_threads=num_threads,
                language=language,
                provider=provider,
            )

        elif model_type == "sense_voice":
            model = next(d.glob("*model*"), None)
            tokens = next(d.glob("*tokens*"), None)
            if not all([model, tokens]):
                raise FileNotFoundError(f"SenseVoice model files not found in {d}")
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model),
                tokens=str(tokens),
                num_threads=num_threads,
                use_itn=True,
                provider=provider,
            )

        elif model_type == "paraformer":
            model = next(d.glob("*model*"), None)
            tokens = next(d.glob("*tokens*"), None)
            if not all([model, tokens]):
                raise FileNotFoundError(f"Paraformer model files not found in {d}")
            return sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(model),
                tokens=str(tokens),
                num_threads=num_threads,
                sample_rate=self.sample_rate,
                provider=provider,
            )

        elif model_type == "transducer":
            encoder = next(d.glob("*encoder*"), None)
            decoder = next(d.glob("*decoder*"), None)
            joiner = next(d.glob("*joiner*"), None)
            tokens = next(d.glob("*tokens*"), None)
            if not all([encoder, decoder, joiner, tokens]):
                raise FileNotFoundError(f"Transducer model files not found in {d}")
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                num_threads=num_threads,
                sample_rate=self.sample_rate,
                provider=provider,
            )

        elif model_type == "streaming_zipformer":
            encoder = next(d.glob("*encoder*"), None)
            decoder = next(d.glob("*decoder*"), None)
            joiner = next(d.glob("*joiner*"), None)
            tokens = next(d.glob("*tokens*"), None)
            if not all([encoder, decoder, joiner, tokens]):
                raise FileNotFoundError(
                    f"Streaming Zipformer model files not found in {d}"
                )
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                num_threads=num_threads,
                sample_rate=self.sample_rate,
                provider=provider,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=20,
            )

        elif model_type == "streaming_nemotron":
            encoder = next(d.glob("*encoder*"), None)
            decoder = next(d.glob("*decoder*"), None)
            joiner = next(d.glob("*joiner*"), None)
            tokens = next(d.glob("*tokens*"), None)
            if not all([encoder, decoder, joiner, tokens]):
                raise FileNotFoundError(
                    f"Streaming Nemotron model files not found in {d}"
                )
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                num_threads=num_threads,
                sample_rate=self.sample_rate,
                provider=provider,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=20,
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ------------------------------------------------------------------
    # File transcription (offline models only)
    # ------------------------------------------------------------------

    def transcribe_file(self, wav_path: str) -> str:
        """
        Transcribe a WAV file (offline recognizer only).

        Args:
            wav_path: Path to WAV file.

        Returns:
            Recognized text.
        """
        if self._is_streaming:
            return self._transcribe_file_streaming(wav_path)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, self._read_wav(wav_path))
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def _transcribe_file_streaming(self, wav_path: str) -> str:
        """Transcribe a WAV file using the streaming recognizer."""
        samples = self._read_wav(wav_path)
        stream = self._recognizer.create_stream()

        # Feed in chunks matching the streaming model's expected frame size
        chunk_size = 3200  # 0.2s at 16kHz — typical for streaming zipformer
        for start in range(0, len(samples), chunk_size):
            chunk = samples[start : start + chunk_size]
            stream.accept_waveform(self.sample_rate, chunk)
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)

        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        return self._recognizer.get_result(stream).strip()

    def _read_wav(self, path: str) -> np.ndarray:
        """Read WAV file and return float32 samples."""
        with wave.open(path, "rb") as f:
            assert f.getnchannels() == 1, "Only mono WAV supported"
            assert f.getsampwidth() == 2, "Only 16-bit WAV supported"
            samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            return samples.astype(np.float32) / 32768.0

    # ------------------------------------------------------------------
    # Continuous listening (VAD + recognizer)
    # ------------------------------------------------------------------

    def start_listening(
        self,
        callback: Callable[[str], None],
        device_name: str = "default",
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Start continuous listening with VAD.

        Blocks the calling thread. Call from a separate thread if needed.

        Args:
            callback: Function called with each recognized text.
            device_name: Audio input device name (e.g. "default", "pulse").
            save_dir: Optional directory to save audio segments.
        """
        if self._is_streaming:
            self._start_listening_streaming(callback, device_name, save_dir)
        else:
            self._start_listening_offline(callback, device_name, save_dir)

    def start_listening_pulse(
        self,
        callback: Callable[[str], None],
        source_name: str,
        source_rate: int = 48000,
    ) -> None:
        """
        Start continuous listening from a PulseAudio/PipeWire source via parec.

        This is used when the desired input is a virtual/remote source such as
        NoMachine, which may not appear as a PortAudio input device.
        """
        if not shutil.which("parec"):
            raise RuntimeError("parec is required for Pulse/PipeWire ASR input")
        if self._is_streaming:
            self._start_listening_pulse_streaming(callback, source_name, source_rate)
        else:
            self._start_listening_pulse_offline(callback, source_name, source_rate)

    def _start_listening_offline(
        self,
        callback: Callable[[str], None],
        device_name: str,
        save_dir: Optional[str],
    ) -> None:
        """Continuous listening using offline recognizer + VAD."""
        devices = sd.query_devices()
        device_idx = self._select_device(devices, device_name, "input")

        if device_idx is None:
            raise RuntimeError(f"Audio input device not found: {device_name}")

        device_sample_rate = int(devices[device_idx]['default_samplerate'])
        logger.info(f"Using input device [{device_idx}]: {devices[device_idx]['name']}")

        vad = sherpa_onnx.VoiceActivityDetector(self._vad_config, buffer_size_in_seconds=100)
        audio_queue: queue.Queue = queue.Queue(maxsize=1000)
        buffer = np.array([], dtype=np.float32)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            try:
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

        segment_count = 0

        with sd.InputStream(
            device=device_idx,
            channels=1,
            callback=audio_callback,
            samplerate=device_sample_rate,
            dtype='float32',
        ):
            logger.info("Listening... (Ctrl+C to stop)")
            try:
                while True:
                    indata = audio_queue.get()
                    samples = indata.flatten()

                    if device_sample_rate != self.sample_rate:
                        samples = self._resample(samples, device_sample_rate, self.sample_rate)

                    buffer = np.concatenate([buffer, samples])
                    while len(buffer) > self._vad_window_size:
                        vad.accept_waveform(buffer[:self._vad_window_size])
                        buffer = buffer[self._vad_window_size:]

                    while not vad.empty():
                        speech = vad.front.samples
                        vad.pop()

                        stream = self._recognizer.create_stream()
                        stream.accept_waveform(self.sample_rate, speech)
                        self._recognizer.decode_stream(stream)

                        text = stream.result.text.strip()
                        if text:
                            norm_text = self._normalize_text(text)
                            logger.info(f"Recognized: {text!r} -> {norm_text!r}")
                            callback(norm_text)

                            if save_dir:
                                segment_count += 1
                                self._save_wav(
                                    Path(save_dir) / f"segment-{segment_count}.wav",
                                    speech,
                                )

            except KeyboardInterrupt:
                logger.info("Stopped listening")

    def _start_listening_streaming(
        self,
        callback: Callable[[str], None],
        device_name: str,
        save_dir: Optional[str],
    ) -> None:
        """
        Continuous listening using streaming recognizer.

        For streaming models, we feed audio directly to the recognizer
        (VAD is not used — endpoint detection is built into the recognizer).
        """
        devices = sd.query_devices()
        device_idx = self._select_device(devices, device_name, "input")

        if device_idx is None:
            raise RuntimeError(f"Audio input device not found: {device_name}")

        device_sample_rate = int(devices[device_idx]['default_samplerate'])
        logger.info(f"Using input device [{device_idx}]: {devices[device_idx]['name']}")

        audio_queue: queue.Queue = queue.Queue(maxsize=1000)
        stream = self._recognizer.create_stream()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            try:
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

        segment_count = 0

        with sd.InputStream(
            device=device_idx,
            channels=1,
            callback=audio_callback,
            samplerate=device_sample_rate,
            dtype='float32',
        ):
            logger.info("Listening (streaming)... (Ctrl+C to stop)")
            try:
                while True:
                    indata = audio_queue.get()
                    samples = indata.flatten()

                    if device_sample_rate != self.sample_rate:
                        samples = self._resample(
                            samples, device_sample_rate, self.sample_rate
                        )
                    samples = samples.astype(np.float32)

                    stream.accept_waveform(self.sample_rate, samples)

                    t_decode_start = time.monotonic()
                    while self._recognizer.is_ready(stream):
                        self._recognizer.decode_stream(stream)
                    decode_s = time.monotonic() - t_decode_start
                    self._metrics["decode_count"] += 1
                    self._metrics["decode_total_s"] += decode_s

                    # Check for endpoint (utterance boundary)
                    if self._recognizer.is_endpoint(stream):
                        self._metrics["endpoint_count"] += 1
                        text = self._recognizer.get_result(stream).strip()
                        if text:
                            self._metrics["endpoint_nonempty_count"] += 1
                            self._metrics["consecutive_empty_finals"] = 0
                            norm_text = self._normalize_text(text)
                            logger.info(f"Recognized: {text!r} -> {norm_text!r}")
                            callback(norm_text)

                            if save_dir:
                                segment_count += 1
                                # For streaming we don't have the raw segment;
                                # saving is skipped unless VAD is also used.
                        else:
                            self._metrics["endpoint_empty_count"] += 1
                            self._metrics["consecutive_empty_finals"] += 1
                            if self._metrics["consecutive_empty_finals"] > self._metrics["max_consecutive_empty"]:
                                self._metrics["max_consecutive_empty"] = self._metrics["consecutive_empty_finals"]
                            if self._metrics["consecutive_empty_finals"] >= 5:
                                logger.warning(f"Consecutive empty finals: {self._metrics['consecutive_empty_finals']}")

                        self._recognizer.reset(stream)

            except KeyboardInterrupt:
                logger.info("Stopped listening")

    def _start_listening_pulse_streaming(
        self,
        callback: Callable[[str], None],
        source_name: str,
        source_rate: int,
    ) -> None:
        """Continuous listening from Pulse/PipeWire using the streaming recognizer."""
        cmd = [
            "parec",
            f"--device={source_name}",
            "--raw",
            "--format=float32le",
            f"--rate={int(source_rate)}",
            "--channels=1",
        ]
        logger.info("Using Pulse/PipeWire input source: %s", source_name)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stream = self._recognizer.create_stream()
        chunk_frames = max(1, int(source_rate * 0.1))
        chunk_bytes = chunk_frames * 4

        try:
            while True:
                raw = proc.stdout.read(chunk_bytes) if proc.stdout else b""
                if not raw:
                    code = proc.poll()
                    if code is not None:
                        raise RuntimeError(f"parec exited with code {code}")
                    continue

                samples = np.frombuffer(raw, dtype=np.float32)
                if source_rate != self.sample_rate:
                    samples = self._resample(samples, source_rate, self.sample_rate)
                samples = samples.astype(np.float32)

                stream.accept_waveform(self.sample_rate, samples)

                t_decode_start = time.monotonic()
                while self._recognizer.is_ready(stream):
                    self._recognizer.decode_stream(stream)
                decode_s = time.monotonic() - t_decode_start
                self._metrics["decode_count"] += 1
                self._metrics["decode_total_s"] += decode_s

                if self._recognizer.is_endpoint(stream):
                    self._metrics["endpoint_count"] += 1
                    text = self._recognizer.get_result(stream).strip()
                    if text:
                        self._metrics["endpoint_nonempty_count"] += 1
                        self._metrics["consecutive_empty_finals"] = 0
                        norm_text = self._normalize_text(text)
                        logger.info(f"Recognized: {text!r} -> {norm_text!r}")
                        callback(norm_text)
                    else:
                        self._metrics["endpoint_empty_count"] += 1
                        self._metrics["consecutive_empty_finals"] += 1
                        if self._metrics["consecutive_empty_finals"] > self._metrics["max_consecutive_empty"]:
                            self._metrics["max_consecutive_empty"] = self._metrics["consecutive_empty_finals"]
                        if self._metrics["consecutive_empty_finals"] >= 5:
                            logger.warning(f"Consecutive empty finals: {self._metrics['consecutive_empty_finals']}")
                    self._recognizer.reset(stream)
        except KeyboardInterrupt:
            logger.info("Stopped listening")
        finally:
            proc.terminate()

    def _start_listening_pulse_offline(
        self,
        callback: Callable[[str], None],
        source_name: str,
        source_rate: int,
    ) -> None:
        """Continuous listening from Pulse/PipeWire using VAD + offline recognizer."""
        cmd = [
            "parec",
            f"--device={source_name}",
            "--raw",
            "--format=float32le",
            f"--rate={int(source_rate)}",
            "--channels=1",
        ]
        logger.info("Using Pulse/PipeWire input source: %s", source_name)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        vad = sherpa_onnx.VoiceActivityDetector(self._vad_config, buffer_size_in_seconds=100)
        buffer = np.array([], dtype=np.float32)
        chunk_frames = max(1, int(source_rate * 0.1))
        chunk_bytes = chunk_frames * 4

        try:
            while True:
                raw = proc.stdout.read(chunk_bytes) if proc.stdout else b""
                if not raw:
                    code = proc.poll()
                    if code is not None:
                        raise RuntimeError(f"parec exited with code {code}")
                    continue

                samples = np.frombuffer(raw, dtype=np.float32)
                if source_rate != self.sample_rate:
                    samples = self._resample(samples, source_rate, self.sample_rate)

                buffer = np.concatenate([buffer, samples.astype(np.float32)])
                while len(buffer) > self._vad_window_size:
                    vad.accept_waveform(buffer[:self._vad_window_size])
                    buffer = buffer[self._vad_window_size:]

                while not vad.empty():
                    speech = vad.front.samples
                    vad.pop()

                    stream = self._recognizer.create_stream()
                    stream.accept_waveform(self.sample_rate, speech)
                    self._recognizer.decode_stream(stream)

                    text = stream.result.text.strip()
                    if text:
                        norm_text = self._normalize_text(text)
                        logger.info(f"Recognized: {text!r} -> {norm_text!r}")
                        callback(norm_text)
        except KeyboardInterrupt:
            logger.info("Stopped listening")
        finally:
            proc.terminate()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        if from_rate == to_rate:
            return audio
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    @staticmethod
    def _save_wav(filepath: Path, audio: np.ndarray) -> None:
        audio_int16 = (np.array(audio) * 32767).astype(np.int16)
        with wave.open(str(filepath), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(audio_int16.tobytes())

    @staticmethod
    def _select_device(devices, target_name, direction):
        """Select audio device by name."""
        from cade.tts.audio_utils import select_device
        return select_device(devices, target_name, direction)
