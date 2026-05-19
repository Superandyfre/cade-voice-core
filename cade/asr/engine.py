"""
ASR Engine — standalone speech recognition engine (ROS-free).

Extracted from asr_node.py. Uses sherpa-onnx with Silero VAD for
offline (non-streaming) speech recognition.
"""

import wave
import logging
import queue
from pathlib import Path
from typing import Optional, Callable

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
    Whisper, SenseVoice, Paraformer, Transducer, Moonshine.
    """

    def __init__(
        self,
        model_dir: str,
        vad_model: Optional[str] = None,
        model_type: str = "whisper",
        provider: str = "cuda",
        sample_rate: int = 16000,
        num_threads: int = 2,
        language: str = "en",
    ):
        """
        Initialize the ASR engine.

        Args:
            model_dir: Path to the model directory.
            vad_model: Path to silero_vad.onnx. If None, looks in model_dir/../silero_vad.onnx.
            model_type: Model architecture: whisper, sense_voice, paraformer, transducer, moonshine.
            provider: Inference provider: cpu, cuda, coreml.
            sample_rate: Target sample rate.
            num_threads: Number of compute threads.
            language: Language code for Whisper models.
        """
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate

        if sherpa_onnx is None:
            raise ImportError("sherpa-onnx is required for ASR. Install with: pip install sherpa-onnx")
        if sd is None:
            raise ImportError("sounddevice is required for audio capture. Install with: pip install sounddevice")

        if vad_model is None:
            vad_model = str(self.model_dir.parent / "silero_vad.onnx")
        self._assert_file(vad_model)

        self._recognizer = self._create_recognizer(
            model_type, provider, num_threads, language
        )

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model
        vad_config.silero_vad.min_silence_duration = 0.25
        vad_config.sample_rate = sample_rate
        self._vad_window_size = vad_config.silero_vad.window_size
        self._vad_config = vad_config

        logger.info(f"ASR engine initialized: {model_type}, provider={provider}")

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

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def transcribe_file(self, wav_path: str) -> str:
        """
        Transcribe a WAV file.

        Args:
            wav_path: Path to WAV file.

        Returns:
            Recognized text.
        """
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, self._read_wav(wav_path))
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def _read_wav(self, path: str) -> np.ndarray:
        """Read WAV file and return float32 samples."""
        with wave.open(path, "rb") as f:
            assert f.getnchannels() == 1, "Only mono WAV supported"
            assert f.getsampwidth() == 2, "Only 16-bit WAV supported"
            samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            return samples.astype(np.float32) / 32768.0

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
        devices = sd.query_devices()
        device_idx = self._select_device(devices, device_name, "input")

        if device_idx is None:
            raise RuntimeError(f"Audio input device not found: {device_name}")

        device_sample_rate = int(devices[device_idx]['default_samplerate'])
        logger.info(f"Using input device [{device_idx}]: {devices[device_idx]['name']}")

        vad = sherpa_onnx.VoiceActivityDetector(self._vad_config, buffer_size_in_seconds=100)
        audio_queue = queue.Queue(maxsize=1000)
        buffer = np.array([])

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
                            logger.info(f"Recognized: {text}")
                            callback(text)

                            if save_dir:
                                segment_count += 1
                                self._save_wav(
                                    Path(save_dir) / f"segment-{segment_count}.wav",
                                    speech
                                )

            except KeyboardInterrupt:
                logger.info("Stopped listening")

    @staticmethod
    def _resample(audio, from_rate, to_rate):
        if from_rate == to_rate:
            return audio
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(new_indices, old_indices, audio)

    @staticmethod
    def _save_wav(filepath, audio):
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
