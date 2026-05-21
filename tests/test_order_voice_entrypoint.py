"""Tests for the real ASR/TTS `cade-order-voice` entrypoint wiring."""

from __future__ import annotations

from types import SimpleNamespace

from cade.fsm import cli


class _FakeRuntime:
    def __init__(self):
        self.llm = SimpleNamespace(model="fake-local")
        self.fsm = SimpleNamespace(config=SimpleNamespace(order_base_dir="/tmp/orders"), _publish_metrics=lambda: None)
        self.input_source = {"kind": "sounddevice", "device": 0}
        self.output = {"kind": "sounddevice", "device": 1}
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def spin(self):
        raise KeyboardInterrupt()

    def stop(self):
        self.stopped = True


def test_order_voice_uses_shared_runtime_builder(monkeypatch):
    fake_runtime = _FakeRuntime()
    captured = {}

    def _fake_builder(**kwargs):
        captured["kwargs"] = kwargs
        return fake_runtime

    monkeypatch.setattr(
        "cade.fsm.voice_runtime.build_ordering_voice_runtime",
        _fake_builder,
    )
    monkeypatch.setattr("cade.fsm.voice_runtime.load_sounddevice", lambda: object())
    monkeypatch.setattr("cade.fsm.voice_runtime.print_asr_input", lambda *args, **kwargs: None)
    monkeypatch.setattr("cade.fsm.voice_runtime.print_tts_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.Config, "validate", classmethod(lambda cls: []))

    cli.cmd_order_voice(["--local", "--skip-audio-probe"])

    assert captured["kwargs"]["verify_audio"] is False
    assert fake_runtime.started is True
    assert fake_runtime.stopped is True
