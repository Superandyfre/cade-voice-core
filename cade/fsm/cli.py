"""CLI entry points for the ordering sub-FSM."""

from __future__ import annotations

import argparse
import logging
import time

from cade.config import Config


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_voice_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cade-order-voice",
        description="Real ASR/TTS ordering runtime with ZeroMQ control/event sockets.",
    )
    parser.add_argument("--pub", default="tcp://127.0.0.1:5555")
    parser.add_argument("--router", default="tcp://127.0.0.1:5556")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-audio-probe", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="Use local OpenAI-compatible LLM.")
    mode.add_argument("--cloud", action="store_true", help="Use cloud LLM.")
    return parser


# =====================================================================
# cade-order-fsm
# =====================================================================


def cmd_order_fsm() -> None:
    """Headless ZeroMQ ordering FSM."""
    _setup_logging()
    from cade.brain.llm_client import LLMClient
    from cade.fsm.config import OrderFSMConfig
    from cade.fsm.order_fsm import OrderSubFSM
    from cade.fsm.zmq_runtime import ZmqRuntime

    fsm_config = OrderFSMConfig()
    warnings = Config.validate()
    for warning in warnings:
        print(f"WARNING: {warning}")

    llm = LLMClient()
    fsm = OrderSubFSM(llm_client=llm, config=fsm_config)
    runtime = ZmqRuntime(
        fsm=fsm,
        pub_bind=fsm_config.zmq_pub_bind,
        router_bind=fsm_config.zmq_router_bind,
        idempotency_path=fsm_config.idempotency_cache_file or None,
        idempotency_ttl_sec=fsm_config.idempotency_ttl_sec,
    )

    print("=" * 60)
    print("CADE Ordering Sub-FSM (headless ZeroMQ)")
    print(f"  Robot: {Config.ROBOT_NAME}")
    print(f"  LLM: {'Cloud' if Config.is_cloud_mode() else 'Local'} ({llm.model})")
    print(f"  PUB:  {fsm_config.zmq_pub_bind}")
    print(f"  ROUTER: {fsm_config.zmq_router_bind}")
    print(f"  Order dir: {fsm_config.order_base_dir}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop.\n")

    try:
        runtime.start()
        runtime.spin()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        fsm._publish_metrics()
        print("\nStopped.")


# =====================================================================
# cade-order-voice
# =====================================================================


def cmd_order_voice(argv: list[str] | None = None) -> None:
    """Real ASR/TTS + ZeroMQ ordering runtime."""
    parser = _build_voice_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    from cade.fsm.voice_runtime import (
        VoiceRuntimeError,
        build_ordering_voice_runtime,
        configure_runtime_mode,
        load_sounddevice,
        print_asr_input,
        print_tts_output,
    )

    configure_runtime_mode(
        cloud=args.cloud,
        local=args.local,
        input_device=args.input_device,
        output_device=args.output_device,
    )

    warnings = Config.validate()
    for warning in warnings:
        print(f"WARNING: {warning}")

    runtime = build_ordering_voice_runtime(
        pub_bind=args.pub,
        router_bind=args.router,
        input_device=args.input_device,
        output_device=args.output_device,
        verify_audio=not args.skip_audio_probe,
    )

    sd = load_sounddevice()
    print("=" * 60)
    print("CADE Ordering Sub-FSM (voice + ZeroMQ)")
    print(f"  Robot: {Config.ROBOT_NAME}")
    print(f"  LLM: {'Cloud' if Config.is_cloud_mode() else 'Local'} ({runtime.llm.model})")
    print(f"  PUB:  {args.pub}")
    print(f"  ROUTER: {args.router}")
    print(f"  Order dir: {runtime.fsm.config.order_base_dir}")
    print_asr_input(sd, runtime.input_source)
    print_tts_output(sd, runtime.output)
    print("=" * 60)
    print("\nPress Ctrl+C to stop.\n")

    try:
        runtime.start()
        runtime.spin()
    except KeyboardInterrupt:
        pass
    except VoiceRuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        runtime.stop()
        runtime.fsm._publish_metrics()
        print("\nStopped.")


# =====================================================================
# Entry-point wrappers
# =====================================================================


def main_order_fsm() -> None:
    cmd_order_fsm()


def main_order_voice() -> None:
    cmd_order_voice()


def main_order_doctor() -> None:
    from cade.fsm.diagnostics import main as diag_main
    diag_main()
