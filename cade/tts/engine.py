"""
TTS Engine — standalone text-to-speech engine (ROS-free).

Extracted from tts_node.py. Uses sherpa-onnx with Kokoro/VITS models.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Standalone TTS engine supporting Kokoro and VITS architectures.

    Auto-detects model type from the model directory path.
    """

    def __init__(
        self,
        model_dir: str,
        provider: str = "cuda",
        num_threads: int = 1,
        sid: int = 0,
        speed: float = 1.0,
    ):
        """
        Initialize the TTS engine.

        Args:
            model_dir: Path to the TTS model directory (containing model.onnx, tokens.txt, etc.).
            provider: Inference provider: cpu, cuda, coreml.
            num_threads: Number of compute threads.
            sid: Speaker ID (for multi-speaker models).
            speed: Speech speed multiplier.
        """
        if sherpa_onnx is None:
            raise ImportError("sherpa-onnx is required for TTS. Install with: pip install sherpa-onnx")
        if sf is None:
            raise ImportError("soundfile is required for TTS. Install with: pip install soundfile")
        if sd is None:
            raise ImportError("sounddevice is required for audio playback. Install with: pip install sounddevice")

        self.model_dir = Path(model_dir)
        self.sid = sid
        self.speed = speed

        self._tts = self._create_tts(provider, num_threads)
        logger.info(f"TTS engine initialized from {model_dir}")

    def _find_file(self, *patterns):
        """Find a file in model_dir matching any pattern."""
        for pattern in patterns:
            match = list(self.model_dir.glob(pattern))
            if match:
                return str(match[0])
        return ""

    def _create_tts(self, provider, num_threads):
        """Create a sherpa-onnx TTS instance (Kokoro or VITS)."""
        model_path = str(self.model_dir).lower()
        model_file = self._find_file("*.onnx")
        tokens_file = self._find_file("tokens.txt")

        if not model_file or not tokens_file:
            raise FileNotFoundError(f"Model files not found in {self.model_dir}")

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            provider=provider,
            num_threads=num_threads,
            debug=False,
        )

        if "vits" in model_path:
            lexicon = self._find_file("lexicon.txt")
            model_config.vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_file,
                tokens=tokens_file,
                lexicon=lexicon,
                noise_scale=0.667,
                noise_scale_w=0.8,
            )
        else:
            # Default to Kokoro
            voices_file = self._find_file("voices.bin")
            data_dir = self._find_file("espeak-ng-data")
            if data_dir and not Path(data_dir).is_dir():
                data_dir = str(self.model_dir / "espeak-ng-data")
            dict_dir = str(self.model_dir) + "/"

            model_config.kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=model_file,
                voices=voices_file,
                tokens=tokens_file,
                data_dir=data_dir,
                dict_dir=dict_dir,
                length_scale=1.0,
            )

        tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config)

        if not tts_config.validate():
            raise ValueError("Invalid TTS configuration. Check model file paths.")

        return sherpa_onnx.OfflineTts(tts_config)

    def synthesize(self, text: str) -> tuple:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech.

        Returns:
            Tuple of (audio_samples, sample_rate).
        """
        import time
        start = time.time()
        audio = self._tts.generate(text, sid=self.sid, speed=self.speed)

        if len(audio.samples) == 0:
            raise ValueError("TTS generation failed: empty audio")

        elapsed = time.time() - start
        duration = len(audio.samples) / audio.sample_rate
        logger.info(f"TTS: {elapsed:.3f}s generation, {duration:.3f}s audio, RTF={elapsed/duration:.3f}")

        return audio.samples, audio.sample_rate

    def speak(
        self,
        text: str,
        device: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Synthesize and play speech.

        Args:
            text: Text to speak.
            device: Output device index. None for system default.
            save_path: Optional path to save WAV file.
        """
        audio, sample_rate = self.synthesize(text)

        if save_path:
            sf.write(save_path, audio, samplerate=sample_rate, subtype="PCM_16")
            logger.info(f"Audio saved to {save_path}")

        self._play_audio(audio, sample_rate, device, save_path)

    def _play_audio(self, audio, sample_rate, device=None, wav_path=None):
        """Play audio, preferring system player over sounddevice."""
        if wav_path and Path(wav_path).exists():
            try:
                subprocess.run(['paplay', wav_path], check=True)
                return
            except Exception:
                pass

        try:
            if device is not None:
                device_info = sd.query_devices(device)
                device_sr = int(device_info['default_samplerate'])
                if sample_rate != device_sr:
                    ratio = device_sr / sample_rate
                    new_length = int(len(audio) * ratio)
                    old_indices = np.arange(len(audio))
                    new_indices = np.linspace(0, len(audio) - 1, new_length)
                    audio = np.interp(new_indices, old_indices, audio)
                    sample_rate = device_sr
                sd.play(audio, sample_rate, device=device)
            else:
                sd.play(audio, sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
