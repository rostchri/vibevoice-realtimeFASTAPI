#!/usr/bin/env python3
"""
Download a VibeVoice model from Hugging Face.

Supports multiple model profiles via the runner registry.

Examples::

    uv run python scripts/download_model.py
    uv run python scripts/download_model.py --model realtime-0.5b
    uv run python scripts/download_model.py --model tts-1.5b
    uv run python scripts/download_model.py --model tts-7b
    uv run python scripts/download_model.py --model tts-7b --model-id some/custom-repo
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``runner`` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runner.errors import UnknownModelError  # noqa: E402
from runner.model_registry import (  # noqa: E402
    get_model_profile,
    list_model_keys,
    resolve_model_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a VibeVoice model from Hugging Face"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="realtime-0.5b",
        help=(
            "Model key or alias (default: realtime-0.5b). "
            f"Available: {', '.join(list_model_keys())}"
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override Hugging Face model ID (default: from registry)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Override local directory to save the model (default: from registry)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var). Required if model is gated.",
    )
    args = parser.parse_args()

    # Resolve model
    try:
        model_key = resolve_model_key(args.model)
    except UnknownModelError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    profile = get_model_profile(model_key)

    model_id = args.model_id or profile.hf_model_id
    local_dir = Path(args.local_dir).resolve() if args.local_dir else Path(profile.default_local_dir).resolve()

    # Get token from args, env var, or None
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"📥 Downloading model: {model_id} (profile: {model_key})")
    print(f"📁 Destination: {local_dir}")

    # Check if model already exists
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"⚠️  Directory {local_dir} already exists and is not empty.")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != "y":
            print("✅ Skipping download (using existing model)")
            return

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Downloaded model: {model_id}")
        print(f"✅ Model saved to: {local_dir}")
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            print(f"❌ Authentication error: {error_msg}")
            print("\n💡 To download gated models, you need to:")
            print("   1. Log in to Hugging Face: huggingface-cli login")
            print("   2. Or set HF_TOKEN environment variable")
            print("   3. Or pass --token <your_token>")
            sys.exit(1)
        else:
            print(f"❌ Error downloading model: {error_msg}")
            raise


if __name__ == "__main__":
    main()
