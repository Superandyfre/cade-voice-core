#!/usr/bin/env python3
"""
Quick validation: load Nemotron INT8 model and transcribe test WAVs.
Compares against Zipformer 20M baseline.
"""

import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

MODELS_ROOT = Path("/home/pinggu/audio/models/asr")
NEMOTRON_DIR = MODELS_ROOT / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
ZIPFORMER_DIR = MODELS_ROOT / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17-mobile"


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return samples.astype(np.float32) / 32768.0


def create_nemotron():
    d = NEMOTRON_DIR
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"),
        tokens=str(d / "tokens.txt"),
        num_threads=4,
        sample_rate=16000,
        provider="cpu",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
    )


def create_zipformer():
    d = ZIPFORMER_DIR
    encoder = next(d.glob("*encoder*"))
    decoder = next(d.glob("*decoder*"))
    joiner = next(d.glob("*joiner*"))
    tokens = next(d.glob("*tokens*"))
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        tokens=str(tokens),
        num_threads=4,
        sample_rate=16000,
        provider="cpu",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
    )


def transcribe_streaming(recognizer, wav_path: str) -> tuple[str, float]:
    samples = read_wav(wav_path)
    audio_dur = len(samples) / 16000.0

    stream = recognizer.create_stream()
    chunk_size = 3200  # 0.2s at 16kHz

    t0 = time.monotonic()
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start:start + chunk_size]
        stream.accept_waveform(16000, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    elapsed = time.monotonic() - t0
    text = recognizer.get_result(stream).strip()
    rtf = elapsed / audio_dur if audio_dur > 0 else float("inf")
    return text, rtf, audio_dur, elapsed


def main():
    test_wavs = sorted(NEMOTRON_DIR.glob("test_wavs/*.wav"))
    if not test_wavs:
        print("No test WAVs found!")
        return

    print("=" * 70)
    print("NEMOTRON 0.6B INT8 — Model Loading")
    print("=" * 70)

    t0 = time.monotonic()
    nemotron = create_nemotron()
    nemotron_load = time.monotonic() - t0
    print(f"  Load time: {nemotron_load:.2f}s")

    t0 = time.monotonic()
    zipformer = create_zipformer()
    zipformer_load = time.monotonic() - t0
    print(f"  Zipformer load time: {zipformer_load:.2f}s")

    print()
    print("=" * 70)
    print("TRANSCRIPTION COMPARISON")
    print("=" * 70)
    print(f"{'File':<20} {'Metric':<12} {'Nemotron':<40} {'Zipformer':<40}")
    print("-" * 112)

    for wav_path in test_wavs:
        fname = wav_path.name

        n_text, n_rtf, n_dur, n_elapsed = transcribe_streaming(nemotron, str(wav_path))
        z_text, z_rtf, z_dur, z_elapsed = transcribe_streaming(zipformer, str(wav_path))

        print(f"{fname:<20} {'Text':<12} {n_text:<40} {z_text:<40}")
        print(f"{'':<20} {'RTF':<12} {n_rtf:<40.4f} {z_rtf:<40.4f}")
        print(f"{'':<20} {'Time':<12} {n_elapsed:<40.3f} {z_elapsed:<40.3f}")
        print()

    # Verdict
    print("=" * 70)
    n_avg_rtf = sum(transcribe_streaming(nemotron, str(w))[1] for w in test_wavs) / len(test_wavs)
    print(f"Nemotron avg RTF: {n_avg_rtf:.4f}")
    if n_avg_rtf < 0.5:
        print("PASS — RTF < 0.5, acceptable for real-time use")
    else:
        print("WARN — RTF >= 0.5, may not be real-time on this CPU")


if __name__ == "__main__":
    main()
