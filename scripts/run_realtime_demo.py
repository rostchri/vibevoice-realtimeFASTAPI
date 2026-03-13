#!/usr/bin/env python3
"""
Run the VibeVoice realtime demo server locally.

This is a backward-compatible wrapper that delegates to the generic
``scripts/run_server.py`` launcher with ``--model realtime-0.5b``.

All original CLI flags are preserved.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``runner`` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runner.adapters.realtime_demo import detect_device, run_realtime_demo_subprocess  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VibeVoice realtime demo server locally")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host/interface to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/VibeVoice-Realtime-0.5B",
        help="Path to the model directory (default: models/VibeVoice-Realtime-0.5B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: auto-detect, prefers cuda on Linux, mps on macOS)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (for development)",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=5,
        help="Number of inference steps (default: 5)",
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Defer model initialization until the first speech request",
    )
    parser.add_argument(
        "--startup-warmup",
        dest="startup_warmup",
        action="store_true",
        default=None,
        help="Warm the model during startup",
    )
    parser.add_argument(
        "--no-startup-warmup",
        dest="startup_warmup",
        action="store_false",
        help="Skip startup warmup",
    )
    args = parser.parse_args()

    # Auto-detect device if not specified
    device = args.device
    if device is None:
        device = detect_device()
        print(f"🔍 Auto-detected device: {device}")

    run_realtime_demo_subprocess(
        project_root=_PROJECT_ROOT,
        model_path=Path(args.model_path),
        device=device,
        host=args.host,
        port=args.port,
        inference_steps=args.inference_steps,
        lazy_load=args.lazy_load,
        startup_warmup=args.startup_warmup,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
