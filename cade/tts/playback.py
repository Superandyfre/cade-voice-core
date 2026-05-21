"""Playback manager for TTS audio."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import soundfile as sf
except ImportError:
    sf = None


logger = logging.getLogger(__name__)


class PlaybackManager:
    """Blocking playback facade with a reusable sounddevice stream."""

    def __init__(
        self,
        backend: str = "sounddevice_stream",
        *,
        use_paplay_fallback: bool = True,
    ):
        self.backend = backend
        self.use_paplay_fallback = use_paplay_fallback
        self._stream = None
        self._stream_key: tuple[int, Optional[int]] | None = None
        self._lock = threading.RLock()
        self._interrupted = False

    def play_blocking(
        self,
        samples,
        sample_rate: int,
        *,
        device: Optional[int] = None,
        wav_path: Optional[str] = None,
        pulse_sink: Optional[str] = None,
    ) -> float:
        """Play audio and return wall-clock playback duration."""
        start = time.monotonic()
        audio = np.asarray(samples, dtype=np.float32)
        with self._lock:
            self._interrupted = False

        if pulse_sink:
            self._play_paplay(audio, sample_rate, wav_path=wav_path, pulse_sink=pulse_sink)
            return time.monotonic() - start

        if self.backend == "paplay" and wav_path:
            self._play_paplay(audio, sample_rate, wav_path=wav_path, pulse_sink=None)
            return time.monotonic() - start

        try:
            self._play_sounddevice_stream(audio, sample_rate, device=device)
        except Exception as exc:
            if not self.use_paplay_fallback or not wav_path:
                logger.error("Audio playback failed: %s", exc)
                raise
            logger.warning("sounddevice playback failed, falling back to paplay: %s", exc)
            self._play_paplay(audio, sample_rate, wav_path=wav_path, pulse_sink=None)
        return time.monotonic() - start

    def stop(self) -> None:
        """Interrupt current playback when supported by the active backend."""
        with self._lock:
            self._interrupted = True
            stream = self._stream
        try:
            if stream is not None and hasattr(stream, "abort"):
                stream.abort()
            elif sd is not None and hasattr(sd, "stop"):
                sd.stop()
        except Exception:
            logger.debug("Ignoring playback stop failure", exc_info=True)

    def close(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._stream_key = None
        try:
            if stream is not None:
                stream.close()
        except Exception:
            logger.debug("Ignoring playback stream close failure", exc_info=True)

    def _play_paplay(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        wav_path: Optional[str],
        pulse_sink: Optional[str],
    ) -> None:
        cleanup_path: str | None = None
        target = wav_path
        if not target:
            if sf is None:
                raise ImportError("soundfile is required to create a temporary paplay WAV")
            fd, target = tempfile.mkstemp(prefix="cade-tts-", suffix=".wav")
            os.close(fd)
            cleanup_path = target
            sf.write(target, audio, samplerate=sample_rate, subtype="PCM_16")

        try:
            cmd = ["paplay"]
            if pulse_sink:
                cmd.append(f"--device={pulse_sink}")
            cmd.append(str(target))
            subprocess.run(cmd, check=True)
        finally:
            if cleanup_path:
                try:
                    os.unlink(cleanup_path)
                except OSError:
                    pass

    def _play_sounddevice_stream(self, audio: np.ndarray, sample_rate: int, *, device: Optional[int]) -> None:
        if sd is None:
            raise ImportError("sounddevice is required for audio playback. Install with: pip install sounddevice")

        playback_rate = int(sample_rate)
        if device is not None:
            device_info = sd.query_devices(device)
            device_sr = int(device_info["default_samplerate"])
            if playback_rate != device_sr:
                audio = _resample_linear(audio, playback_rate, device_sr)
                playback_rate = device_sr

        if self.backend == "sounddevice_stream":
            stream = self._ensure_stream(playback_rate, device)
            try:
                stream.write(audio.reshape(-1, 1))
                return
            except Exception:
                logger.debug("Reusable sounddevice stream failed; falling back to sd.play", exc_info=True)
                self.close()

        if device is not None:
            sd.play(audio, playback_rate, device=device)
        else:
            sd.play(audio, playback_rate)
        sd.wait()

    def _ensure_stream(self, sample_rate: int, device: Optional[int]):
        key = (int(sample_rate), device)
        with self._lock:
            if self._stream is not None and self._stream_key == key:
                return self._stream
            self.close()
            stream = sd.OutputStream(
                samplerate=int(sample_rate),
                channels=1,
                dtype="float32",
                device=device,
            )
            stream.start()
            self._stream = stream
            self._stream_key = key
            return stream


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32)
    ratio = dst_rate / src_rate
    new_length = max(1, int(len(audio) * ratio))
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(new_indices, old_indices, audio).astype(np.float32)
