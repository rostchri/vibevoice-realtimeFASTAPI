#!/usr/bin/env python3
"""Generic multi-model launcher for the VibeVoice TTS runner.

Usage examples::

    uv run python scripts/run_server.py --model realtime-0.5b --port 8000
    uv run python scripts/run_server.py --model tts-1.5b --port 8000

For the realtime adapter this behaves identically to the legacy
``scripts/run_realtime_demo.py`` script.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``runner`` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runner.adapter_factory import make_adapter  # noqa: E402
from runner.adapters.longform_native import LongformNativeAdapter  # noqa: E402
from runner.adapters.realtime_demo import RealtimeDemoAdapter, detect_device  # noqa: E402
from runner.errors import UnknownModelError  # noqa: E402
from runner.model_registry import get_model_profile, resolve_model_key  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic VibeVoice TTS server launcher")
    parser.add_argument(
        "--model",
        type=str,
        default="realtime-0.5b",
        help="Model key or alias (default: realtime-0.5b)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override path to the model directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host/interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device (default: auto-detect)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development)",
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
        help="Defer model initialization until first request",
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

    # Resolve model
    try:
        model_key = resolve_model_key(args.model)
    except UnknownModelError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    profile = get_model_profile(model_key)
    print(f"📦 Model: {model_key} (family={profile.family})")

    # Device
    device = args.device
    if device is None:
        device = detect_device()
        print(f"🔍 Auto-detected device: {device}")

    # Model path
    model_path = Path(args.model_path) if args.model_path else Path(profile.default_local_dir)

    # Build adapter
    adapter = make_adapter(
        model_key,
        model_path=model_path,
        device=device,
    )

    # --- Realtime adapter: launch via existing subprocess demo ---
    if isinstance(adapter, RealtimeDemoAdapter):
        adapter.launch(
            project_root=_PROJECT_ROOT,
            host=args.host,
            port=args.port,
            inference_steps=args.inference_steps,
            lazy_load=args.lazy_load,
            startup_warmup=args.startup_warmup,
            reload=args.reload,
        )
        return

    # --- Longform adapter: fail early if backend is unavailable ---
    if isinstance(adapter, LongformNativeAdapter):
        if not adapter.is_available():
            print(f"❌ {profile.key} backend is not available.")
            print(f"   {adapter._backend_error}")
            print("\n💡 A compatible long-form backend must be installed.")
            print("   See README for details.")
            sys.exit(1)

        # TODO: launch longform-compatible serving path when implemented
        print(f"🚧 Long-form serving for {model_key} is not yet implemented.")
        sys.exit(1)

    print(f"❌ No launch logic for adapter type: {type(adapter).__name__}")
    sys.exit(1)


if __name__ == "__main__":
    main()
