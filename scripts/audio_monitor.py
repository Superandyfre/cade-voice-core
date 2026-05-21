#!/usr/bin/env python3
"""
Real-time PipeWire/PulseAudio source monitor.

Shows audio levels and basic spectrum info for one or more sources,
so you can verify that NoMachine (or any virtual device) is feeding
audio correctly.

Usage:
    python scripts/audio_monitor.py                      # all PipeWire sources
    python scripts/audio_monitor.py nx_remapped_out      # specific source
    python scripts/audio_monitor.py nx_voice_out.monitor # sink monitor
"""

import subprocess
import sys
import time
import signal
import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 1
FORMAT_BYTES = 4  # float32le
CHUNK_SEC = 0.1
CHUNK_BYTES = int(SAMPLE_RATE * CHANNELS * FORMAT_BYTES * CHUNK_SEC)

BAR_WIDTH = 40


def list_pipewire_sources():
    """Return list of (index, name, state) for PipeWire sources."""
    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"],
            text=True,
        )
    except FileNotFoundError:
        print("Error: pactl not found. Is PipeWire/PulseAudio installed?")
        sys.exit(1)

    sources = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and "monitor" not in parts[1].lower().split(".")[-1]:
            sources.append((parts[0], parts[1], parts[3] if len(parts) > 3 else "?"))
    return sources


def bar(level_db, width=BAR_WIDTH):
    """Render a VU-meter-style bar from dB value."""
    # Map -60..0 dB to 0..width
    norm = max(0.0, min(1.0, (level_db + 60) / 60))
    filled = int(norm * width)
    empty = width - filled

    if level_db > -6:
        color = "\033[91m"  # red
    elif level_db > -20:
        color = "\033[93m"  # yellow
    else:
        color = "\033[92m"  # green

    return f"{color}{'█' * filled}{'░' * empty}\033[0m"


def db_str(rms):
    if rms < 1e-10:
        return " -inf"
    db = 20 * np.log10(rms)
    return f"{db:6.1f}"


def monitor_source(source_name):
    """Open parec and continuously print audio level."""
    cmd = [
        "parec",
        f"--device={source_name}",
        "--raw",
        "--format=float32le",
        f"--rate={SAMPLE_RATE}",
        f"--channels={CHANNELS}",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Give parec a moment to start
    time.sleep(0.3)
    if proc.poll() is not None:
        err = proc.stderr.read().decode()
        print(f"  \033[91mparec failed\033[0m: {err.strip()}")
        return

    try:
        while True:
            raw = proc.stdout.read(CHUNK_BYTES)
            if not raw:
                if proc.poll() is not None:
                    print(f"  \033[91mparec exited (code {proc.returncode})\033[0m")
                    return
                continue

            samples = np.frombuffer(raw, dtype=np.float32)
            rms = np.sqrt(np.mean(samples**2))
            peak = np.max(np.abs(samples))
            db_rms = 20 * np.log10(rms) if rms > 1e-10 else -120
            db_peak = 20 * np.log10(peak) if peak > 1e-10 else -120

            b = bar(db_rms)
            ts = time.strftime("%H:%M:%S")
            print(
                f"\r  {ts}  RMS {db_str(rms)} dB  Peak {db_str(peak)} dB  {b}  ",
                end="",
                flush=True,
            )

    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait()
        print()


def main():
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    if len(sys.argv) > 1:
        sources_to_check = [(None, name, "?") for name in sys.argv[1:]]
    else:
        all_sources = list_pipewire_sources()
        # Always include key virtual sources
        virtual = []
        for name in ["nx_remapped_out", "nx_voice_out.monitor"]:
            if not any(s[1] == name for s in all_sources):
                virtual.append((None, name, "?"))
        sources_to_check = all_sources + virtual

    if not sources_to_check:
        print("No PipeWire sources found.")
        return

    print("PipeWire audio source monitor — Ctrl+C to stop\n")
    print(f"{'Source':<55s} {'RMS dB':>8s} {'Peak dB':>8s}")
    print("─" * 55 + "─" * 17)

    for idx, name, state in sources_to_check:
        print(f"\n\033[1m{idx or '?':>4s}  {name}\033[0m  ({state})")
        monitor_source(name)


if __name__ == "__main__":
    main()
