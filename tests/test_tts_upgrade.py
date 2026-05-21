"""Tests for CPU-only TTS router/cache/normalizer/playback helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from cade.tts.backends.null_backend import NullBackend
from cade.tts.cache import TTSCache
from cade.tts.chunker import SentenceChunker
from cade.tts.normalizer import normalize_tts_text
from cade.tts.playback import PlaybackManager
from cade.tts.router import TTSRouter


def test_normalizer_restaurant_examples():
    assert normalize_tts_text("2 burgers") == "two burgers"
    assert normalize_tts_text("$12.50") == "twelve dollars and fifty cents"
    assert normalize_tts_text("No. 3") == "number three"
    assert normalize_tts_text("Coke x2") == "two cokes"
    assert normalize_tts_text("10:30") == "ten thirty"
    assert normalize_tts_text("Order 2048") == "Order two zero four eight"
    assert normalize_tts_text("ASR LLM ZMQ USB-C") == "A S R L L M Z M Q U S B C"


def test_sentence_chunker_splits_and_flushes():
    chunker = SentenceChunker(min_chars=20, max_chars=80, comma_chars=30)

    assert chunker.feed("Sure, I can help with that. What would you like") == [
        "Sure, I can help with that."
    ]
    assert chunker.flush() == ["What would you like"]


def test_sentence_chunker_forces_max_length():
    chunker = SentenceChunker(max_chars=20)
    chunks = chunker.feed("one two three four five six seven eight")
    assert chunks
    assert len(chunks[0]) <= 20


def test_router_uses_fast_backend_for_short_text():
    router = TTSRouter(
        default_backend=NullBackend("default"),
        fast_backend=NullBackend("fast"),
        fallback_backend=NullBackend("fallback"),
        load_threshold=1.1,
    )

    result = router.synthesize("OK.", profile="dialogue")

    assert result.backend == "fast"


def test_router_uses_fallback_on_backend_error():
    class FailingBackend(NullBackend):
        def synthesize(self, *args, **kwargs):
            raise RuntimeError("backend down")

    router = TTSRouter(
        default_backend=FailingBackend("default"),
        fast_backend=NullBackend("fast"),
        fallback_backend=NullBackend("fallback"),
        short_text_chars=0,
        load_threshold=1.1,
    )

    result = router.synthesize("This should use the default path first.", profile="dialogue")

    assert result.backend == "fallback"
    assert result.fallback is True
    assert router.fallback_count == 1


def test_router_uses_fallback_under_high_load(monkeypatch):
    monkeypatch.setattr(TTSRouter, "_system_load", staticmethod(lambda: 0.95))
    router = TTSRouter(
        default_backend=NullBackend("default"),
        fast_backend=NullBackend("fast"),
        fallback_backend=NullBackend("fallback"),
        load_threshold=0.85,
    )

    result = router.synthesize("A normal reply.", profile="dialogue")

    assert result.backend == "fallback"


def test_tts_cache_hit_miss_and_corrupt_file(tmp_path, monkeypatch):
    import cade.tts.cache as cache_mod

    store = {}

    class FakeSoundFile:
        @staticmethod
        def write(path, samples, samplerate, subtype=None):
            arr = np.asarray(samples, dtype=np.float32)
            store[str(path)] = (arr, samplerate)
            if str(path).endswith(".tmp.wav"):
                store[str(path).replace(".tmp.wav", ".wav")] = (arr, samplerate)
            Path(path).write_bytes(b"fake-wav")

        @staticmethod
        def read(path, dtype="float32", always_2d=False):
            if str(path).endswith("corrupt.wav"):
                raise RuntimeError("bad wav")
            return store[str(path)]

    monkeypatch.setattr(cache_mod, "sf", FakeSoundFile)
    backend = NullBackend("fast")
    cache = TTSCache(str(tmp_path), enabled=True)

    assert cache.get(backend, "hello") is None
    result = backend.synthesize("hello")
    cache.put(backend, "hello", result)
    hit = cache.get(backend, "hello")

    assert hit is not None
    assert hit.cache_hit is True
    assert hit.backend == "fast"

    key = cache.key_for(backend, "corrupt")
    corrupt = cache._path_for(backend, key)
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"bad")
    monkeypatch.setattr(FakeSoundFile, "read", staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad wav"))))
    assert cache.get(backend, "corrupt") is None


def test_playback_manager_uses_reusable_output_stream(monkeypatch):
    writes = []

    class FakeStream:
        def start(self):
            return None

        def write(self, audio):
            writes.append(audio.copy())

        def close(self):
            return None

        def abort(self):
            writes.append("abort")

    class FakeSD:
        @staticmethod
        def OutputStream(**kwargs):
            return FakeStream()

        @staticmethod
        def play(*args, **kwargs):
            raise AssertionError("sd.play should not be used when stream works")

        @staticmethod
        def wait():
            return None

    import cade.tts.playback as playback_mod

    monkeypatch.setattr(playback_mod, "sd", FakeSD)
    manager = PlaybackManager("sounddevice_stream")

    elapsed = manager.play_blocking(np.zeros(100, dtype=np.float32), 16000)
    manager.stop()

    assert elapsed >= 0
    assert len(writes) == 2
