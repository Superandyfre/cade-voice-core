"""
CADE CLI — entry points for text-chat, voice-chat, and benchmarks.

Console scripts (defined in pyproject.toml):
    cade-text-chat   — text-only debugging loop (no ASR/TTS)
    cade-voice-chat  — full microphone voice loop
    cade-bench       — LLM / TTS / end-to-end smoke tests
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

from cade.config import Config


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ======================================================================
# cade-text-chat
# ======================================================================

def cmd_text_chat() -> None:
    """Text-only REPL — useful for testing LLM decisions without audio."""
    _setup_logging()
    from cade.controller import RobotController
    from cade.brain.prompts import get_system_prompt

    warnings = Config.validate()
    for w in warnings:
        print(f"WARNING: {w}")

    ctrl = RobotController(prompt_mode="compact")
    print("CADE text chat — type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                ctrl.print_statistics()
                break

            result = ctrl.process_input(user_input)
            if result["spoken_text"]:
                print(f"Robot: {result['spoken_text']}")
            if result.get("timings"):
                llm_ms = result["timings"].get("llm_latency_s", 0) * 1000
                print(f"  [{llm_ms:.0f}ms LLM]")

        except KeyboardInterrupt:
            ctrl.print_statistics()
            break
        except Exception as e:
            print(f"Error: {e}")


# ======================================================================
# cade-voice-chat
# ======================================================================

def cmd_voice_chat() -> None:
    """Full voice loop: microphone -> ASR -> LLM -> action -> TTS."""
    _setup_logging()
    from cade.voice.session import VoiceSession

    warnings = Config.validate()
    for w in warnings:
        print(f"WARNING: {w}")

    session = VoiceSession()
    print("CADE voice chat — speak into the microphone (Ctrl+C to stop)\n")
    session.run_forever()


# ======================================================================
# cade-bench
# ======================================================================

def cmd_bench() -> None:
    """Run smoke / benchmark tests."""
    _setup_logging()

    parser = argparse.ArgumentParser(prog="cade-bench", description="CADE smoke & benchmark tests")
    parser.add_argument("--llm-smoke", action="store_true", help="Test LLM JSON decision (requires running llama-server)")
    parser.add_argument("--order-llm-smoke", action="store_true", help="Test ordering sub-FSM LLM with adversarial inputs")
    parser.add_argument("--tts-smoke", action="store_true", help="Test TTS synthesis")
    parser.add_argument("--tts-cpu", action="store_true", help="Benchmark CPU-only TTS backends")
    parser.add_argument("--asr-smoke", action="store_true", help="Test ASR on a sample WAV")
    parser.add_argument("--full", action="store_true", help="Run all smoke tests")
    parser.add_argument("--wav", type=str, default=None, help="WAV file for ASR smoke test")
    parser.add_argument("--model-type", type=str, default=None, help="ASR model type override (e.g. streaming_nemotron)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path for benchmark results")
    args = parser.parse_args()

    if not any([args.llm_smoke, args.order_llm_smoke, args.tts_smoke, args.tts_cpu, args.asr_smoke, args.full]):
        parser.print_help()
        return

    if args.full:
        args.llm_smoke = args.tts_smoke = args.asr_smoke = True

    if args.llm_smoke:
        _bench_llm()
    if args.order_llm_smoke:
        _bench_order_llm()
    if args.tts_smoke:
        _bench_tts()
    if args.tts_cpu:
        _bench_tts_cpu(args.output_json)
    if args.asr_smoke:
        _bench_asr(args.wav, args.model_type)


def _bench_llm() -> None:
    """Smoke-test: send a short command to the LLM and check JSON success."""
    from cade.brain.llm_client import LLMClient
    from cade.brain.prompts import get_system_prompt

    print("=" * 60)
    print("LLM SMOKE TEST")
    print("=" * 60)

    client = LLMClient()
    prompt = get_system_prompt("compact")

    test_inputs = [
        "hello",
        "find the cup",
        "pick up the apple",
        "place the bottle on the table",
        "what can you do?",
        "bring me the book from the desk",
    ]

    successes = 0
    total_latency = 0.0

    for inp in test_inputs:
        t0 = time.monotonic()
        try:
            decision = client.get_decision(inp, prompt, max_retries=1)
            latency = time.monotonic() - t0
            total_latency += latency
            successes += 1
            action_type = decision.action.type if decision.action else "none"
            print(f"  OK  [{latency:.2f}s] {inp!r:40s} -> action={action_type}, reply={decision.reply!r:.40s}")
        except Exception as e:
            latency = time.monotonic() - t0
            total_latency += latency
            print(f"  FAIL [{latency:.2f}s] {inp!r:40s} -> {e}")

    n = len(test_inputs)
    print(f"\nResult: {successes}/{n} JSON successes, avg latency {total_latency/n:.2f}s")
    if successes == n:
        print("PASS")
    else:
        print("FAIL — some inputs did not produce valid JSON")
        sys.exit(1)


def _bench_order_llm() -> None:
    """Smoke-test: adversarial ordering inputs through all three stages."""
    from cade.brain.llm_client import LLMClient
    from cade.brain.schemas import OrderAction, OrderItem

    print("=" * 60)
    print("ORDER LLM SMOKE TEST (adversarial inputs)")
    print("=" * 60)

    client = LLMClient()
    food_aliases = {
        "water": ["water", "bottle of water"],
        "coke": ["coke", "cola", "coca cola"],
        "burger": ["burger", "hamburger", "cheeseburger"],
        "fried_rice": ["fried rice"],
    }

    listen_cases = [
        ("I want a coke", True, ["coke"]),
        ("two burgers and a water", True, ["burger", "water"]),
        ("uh can you hear me", False, []),
        ("I maybe want cola and two bottle water", True, ["coke", "water"]),
        ("give me the blue one", False, []),
        ("no, actually two waters", True, ["water"]),
        ("hello what is your name", False, []),
    ]

    check_cases = [
        ("yes", "correct"),
        ("yeah that is correct", "correct"),
        ("no, two waters instead", "wrong"),
        ("no", "wrong"),
        ("not sure", "wrong"),
    ]

    successes = 0
    total = 0

    # LISTEN stage tests
    print("\n--- LISTEN stage ---")
    for text, should_have_items, _expected_names in listen_cases:
        total += 1
        try:
            order = client.get_order_action(
                user_input=text,
                food_aliases=food_aliases,
                max_retries=2,
            )
            has_items = len(order.items) > 0
            ok = has_items == should_have_items
            status = "OK  " if ok else "MISMATCH"
            item_str = ", ".join(f"{i.name}x{i.qty}" for i in order.items) or "(empty)"
            print(f"  {status} {text!r:45s} -> {item_str}")
            if ok:
                successes += 1
        except Exception as e:
            print(f"  FAIL {text!r:45s} -> {e}")

    # REPEAT stage test
    print("\n--- REPEAT stage ---")
    try:
        order = OrderAction(type="order", items=[
            OrderItem(name="coke", qty=2),
            OrderItem(name="water", qty=1),
        ])
        speak = client.get_order_repeat_speak(
            confirm_instruction="Repeat the order and ask if it is correct.",
            order_action=order,
            max_retries=2,
        )
        content = speak.action.content
        total += 1
        if content and len(content) > 10:
            print(f"  OK   repeat speak -> {content!r:.80s}")
            successes += 1
        else:
            print(f"  MISMATCH repeat speak too short -> {content!r}")
    except Exception as e:
        total += 1
        print(f"  FAIL repeat speak -> {e}")

    # CHECK stage tests
    print("\n--- CHECK stage ---")
    order = OrderAction(type="order", items=[OrderItem(name="coke", qty=1)])
    for reply_text, expected_result in check_cases:
        total += 1
        try:
            decision = client.get_order_check_decision(
                customer_reply=reply_text,
                order_action=order,
                food_aliases=food_aliases,
                max_retries=2,
            )
            ok = decision.result == expected_result
            status = "OK  " if ok else "MISMATCH"
            action_str = "fix_order" if decision.action else "null"
            print(f"  {status} {reply_text!r:40s} -> result={decision.result}, action={action_str}")
            if ok:
                successes += 1
        except Exception as e:
            print(f"  FAIL {reply_text!r:40s} -> {e}")

    print(f"\nResult: {successes}/{total} passed")
    if successes == total:
        print("PASS")
    else:
        print("FAIL — some adversarial inputs did not produce expected results")
        sys.exit(1)


def _bench_tts() -> None:
    """Smoke-test: synthesize a few sentences and report RTF."""
    from cade.tts.engine import TTSEngine

    print("=" * 60)
    print("TTS SMOKE TEST")
    print("=" * 60)

    engine = TTSEngine(
        model_dir=Config.TTS_MODEL_DIR,
        provider=Config.TTS_PROVIDER,
    )

    sentences = [
        "Hello, I am a robot.",
        "I found the cup on the table.",
        "Sure, I can help with that.",
        "Sorry, I had trouble processing that.",
        "I have picked up the apple.",
    ]

    total_gen = 0.0
    total_audio = 0.0

    for text in sentences:
        try:
            t0 = time.monotonic()
            samples, sr = engine.synthesize(text)
            gen_s = time.monotonic() - t0
            audio_s = len(samples) / sr
            total_gen += gen_s
            total_audio += audio_s
            rtf = gen_s / audio_s if audio_s > 0 else float("inf")
            print(f"  OK  [{gen_s:.3f}s gen, {audio_s:.3f}s audio, RTF={rtf:.3f}] {text!r}")
        except Exception as e:
            print(f"  FAIL {text!r} -> {e}")

    if total_audio > 0:
        print(f"\nOverall RTF: {total_gen/total_audio:.3f}")
    print("PASS" if total_audio > 0 else "FAIL")


def _bench_tts_cpu(output_json: str = None) -> None:
    """Benchmark CPU-only TTS candidates without playing audio."""
    from cade.tts.backends.sherpa_kokoro import SherpaKokoroBackend
    from cade.tts.backends.sherpa_vits import SherpaVitsBackend
    from cade.tts.cache import TTSCache
    from cade.tts.normalizer import TextNormalizer

    try:
        import psutil
    except ImportError:
        psutil = None

    print("=" * 60)
    print("TTS CPU BENCHMARK")
    print("=" * 60)

    texts = [
        ("short", "OK."),
        ("short", "Sure."),
        ("short", "Could you repeat that?"),
        ("prompt", "What would you like to order?"),
        ("order", "You ordered two cheeseburgers, one large fries, and a coke."),
        ("money", "Your total is twelve dollars and fifty cents."),
        ("long", "I can help you with the order, but I need you to confirm one more detail."),
        ("technical", "The ASR module uses ONNX Runtime and ZeroMQ for communication."),
    ]
    candidates = [
        ("vits", Config.TTS_VITS_MODEL_DIR, SherpaVitsBackend),
        ("piper", Config.TTS_PIPER_MODEL_DIR, SherpaVitsBackend),
        ("kokoro", Config.TTS_KOKORO_MODEL_DIR, SherpaKokoroBackend),
    ]
    normalizer = TextNormalizer(enabled=Config.TTS_TEXT_NORMALIZE)
    cache = TTSCache(Config.TTS_CACHE_DIR, enabled=Config.TTS_CACHE_ENABLED)
    process = psutil.Process() if psutil else None
    rows = []

    for backend_name, model_dir, cls in candidates:
        if not Path(model_dir).is_dir():
            print(f"  SKIP {backend_name}: model dir not found: {model_dir}")
            continue
        try:
            t0 = time.monotonic()
            backend = cls(
                model_dir,
                name=backend_name,
                provider=Config.TTS_PROVIDER,
                num_threads=Config.TTS_NUM_THREADS,
                sid=Config.TTS_SID,
                speed=Config.TTS_SPEED,
            )
            init_s = time.monotonic() - t0
        except Exception as exc:
            print(f"  FAIL {backend_name}: init failed: {exc}")
            continue

        for text_type, text in texts:
            normalized = normalizer.normalize(text)
            cache_key = cache.key_for(backend, normalized)
            rss_before = _rss_mb(process)
            cpu_before = _cpu_percent(process)
            try:
                t_first = time.monotonic()
                first = backend.synthesize(normalized, profile="bench")
                first_call_s = time.monotonic() - t_first
                cache.put(backend, normalized, first)

                t_warm = time.monotonic()
                warm = backend.synthesize(normalized, profile="bench")
                warm_call_s = time.monotonic() - t_warm

                t_cache = time.monotonic()
                cached = cache.get(backend, normalized, profile="bench")
                cache_ms = (time.monotonic() - t_cache) * 1000.0
                cpu_after = _cpu_percent(process)
                rss_after = _rss_mb(process)
                row = {
                    "backend": backend_name,
                    "text_type": text_type,
                    "init_time_s": init_s,
                    "first_call_latency_s": first_call_s,
                    "warm_call_latency_s": warm_call_s,
                    "audio_duration_s": first.audio_duration_s,
                    "rtf": first.rtf,
                    "cache_hit_latency_ms": cache_ms if cached else None,
                    "peak_rss_mb": max(rss_before, rss_after),
                    "cpu_percent_avg": cpu_after,
                    "cpu_percent_peak": max(cpu_before, cpu_after),
                    "fallback_count": 0,
                    "cache_key": cache_key,
                    "text": text,
                }
                rows.append(row)
                cache_text = (
                    f"{row['cache_hit_latency_ms']:.1f}ms"
                    if row["cache_hit_latency_ms"] is not None
                    else "n/a"
                )
                print(
                    f"{backend_name:10s} {text_type:9s} "
                    f"first={first_call_s:.3f}s warm={warm_call_s:.3f}s "
                    f"audio={first.audio_duration_s:.2f}s rtf={first.rtf:.3f} "
                    f"cache={cache_text} rss={row['peak_rss_mb']:.0f}MB"
                )
            except Exception as exc:
                print(f"  FAIL {backend_name} {text_type}: {exc}")

    if output_json:
        Path(output_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON benchmark results to {output_json}")
    print("PASS" if rows else "FAIL")


def _rss_mb(process) -> float:
    if process is None:
        return 0.0
    try:
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _cpu_percent(process) -> float:
    if process is None:
        return 0.0
    try:
        return float(process.cpu_percent(interval=None))
    except Exception:
        return 0.0


def _bench_asr(wav_path: str = None, model_type: str = None) -> None:
    """Smoke-test: transcribe a WAV file."""
    from cade.asr.engine import ASREngine

    print("=" * 60)
    print("ASR SMOKE TEST")
    print("=" * 60)

    if wav_path is None:
        print("No --wav provided, skipping ASR smoke test.")
        print("Usage: cade-bench --asr-smoke --wav /path/to/test.wav")
        return

    if not Path(wav_path).is_file():
        print(f"WAV file not found: {wav_path}")
        sys.exit(1)

    mt = model_type or Config.ASR_MODEL_TYPE
    model_dir = Config.ASR_MODEL_DIR

    if mt == "streaming_nemotron":
        nemotron_dir = str(Path(Config.ASR_MODEL_DIR).parent / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25")
        model_dir = os.getenv("CADE_NEMOTRON_MODEL_DIR", nemotron_dir)

    engine = ASREngine(
        model_dir=model_dir,
        vad_model=Config.VAD_MODEL,
        model_type=mt,
        provider=Config.ASR_PROVIDER,
    )

    print(f"  Model: {mt} ({model_dir})")

    t0 = time.monotonic()
    text = engine.transcribe_file(wav_path)
    elapsed = time.monotonic() - t0

    print(f"  File: {wav_path}")
    print(f"  Text: {text!r}")
    print(f"  Time: {elapsed:.3f}s")
    print("PASS" if text else "FAIL — empty transcription")


# ======================================================================
# Entry-point wrappers (called by console_scripts)
# ======================================================================

def main_text_chat():
    cmd_text_chat()


def main_voice_chat():
    cmd_voice_chat()


def main_bench():
    cmd_bench()
