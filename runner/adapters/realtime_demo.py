"""Realtime adapter – wraps the vendored subprocess-demo workflow.

This adapter preserves the existing architecture where
``scripts/run_realtime_demo.py`` copies override files into the vendored
``third_party/VibeVoice/demo/`` tree and then launches the demo script as a
subprocess.  The heavy lifting stays in the existing code-paths; this adapter
merely exposes reusable helpers and satisfies the :class:`EngineAdapter` API.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from runner.adapters.base import EngineAdapter
from runner.model_registry import ModelProfile
from runner.types import SpeechRequest

# ---------------------------------------------------------------------------
# Reusable helper functions (extracted from scripts/run_realtime_demo.py)
# ---------------------------------------------------------------------------


def detect_device() -> str:
    """Auto-detect the best compute device (CUDA → MPS → CPU)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_realtime_model_path(model_path: Path) -> None:
    """Raise ``SystemExit`` if *model_path* does not exist."""
    if not model_path.exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        print("\n💡 Make sure you've downloaded the model first:")
        print("   uv run python scripts/download_model.py")
        sys.exit(1)


def apply_overrides(project_root: Path, vibevoice_dir: Path) -> None:
    """Copy project-local overrides into the vendored VibeVoice demo tree."""
    overrides_dir = project_root / "overrides"
    if not overrides_dir.exists():
        return

    # Single-file overrides
    file_map: dict[str, Path] = {
        "app.py": vibevoice_dir / "demo" / "web" / "app.py",
        "vibevoice_realtime_demo.py": vibevoice_dir / "demo" / "vibevoice_realtime_demo.py",
        "realtime_model_inference_from_file.py": (
            vibevoice_dir / "demo" / "realtime_model_inference_from_file.py"
        ),
        "text_processing.py": vibevoice_dir / "demo" / "web" / "text_processing.py",
        "index.html": vibevoice_dir / "demo" / "web" / "index.html",
    }
    for name, target in file_map.items():
        src = overrides_dir / name
        if src.exists():
            shutil.copy2(src, target)

    # Directory overrides
    for dirname in ("lavasr", "novasr"):
        src = overrides_dir / dirname
        target = vibevoice_dir / "demo" / "web" / dirname
        if src.exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)

    # Upsampler files
    for upsampler in ("lavasr_upsampler.py", "novasr_upsampler.py", "flashsr_upsampler.py"):
        src = overrides_dir / upsampler
        target = vibevoice_dir / "demo" / "web" / upsampler
        if src.exists():
            shutil.copy2(src, target)


def build_realtime_demo_cmd(
    *,
    demo_script: Path,
    host: str,
    port: int,
    model_path: Path,
    device: str,
    inference_steps: int,
    lazy_load: bool = False,
    startup_warmup: bool | None = None,
    reload: bool = False,
) -> list[str]:
    """Build the subprocess command list for the realtime demo."""
    cmd: list[str] = [
        sys.executable,
        str(demo_script),
        "--host", host,
        "--port", str(port),
        "--model_path", str(model_path),
        "--device", device,
        "--inference_steps", str(inference_steps),
    ]
    if lazy_load:
        cmd.append("--lazy-load")
    if startup_warmup is not None:
        cmd.append("--startup-warmup" if startup_warmup else "--no-startup-warmup")
    if reload:
        cmd.append("--reload")
    return cmd


def set_realtime_env(
    *,
    model_path: Path,
    device: str,
    lazy_load: bool = False,
    startup_warmup: bool | None = None,
) -> None:
    """Populate environment variables expected by the demo app."""
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["MODEL_DEVICE"] = device
    if lazy_load:
        os.environ["ENABLE_LAZY_LOAD"] = "true"
    if startup_warmup is not None:
        os.environ["ENABLE_STARTUP_WARMUP"] = "true" if startup_warmup else "false"
    elif lazy_load:
        os.environ["ENABLE_STARTUP_WARMUP"] = "false"


def run_realtime_demo_subprocess(
    *,
    project_root: Path,
    model_path: Path,
    device: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    inference_steps: int = 5,
    lazy_load: bool = False,
    startup_warmup: bool | None = None,
    reload: bool = False,
) -> None:
    """End-to-end: validate → apply overrides → set env → launch subprocess."""
    model_path = model_path.resolve()
    validate_realtime_model_path(model_path)

    vibevoice_dir = project_root / "third_party" / "VibeVoice"
    demo_script = vibevoice_dir / "demo" / "vibevoice_realtime_demo.py"
    if not demo_script.exists():
        print(f"❌ Error: VibeVoice demo script not found: {demo_script}")
        print("\n💡 Make sure you've run the bootstrap script first:")
        print("   ./scripts/bootstrap_uv.sh")
        sys.exit(1)

    apply_overrides(project_root, vibevoice_dir)
    set_realtime_env(
        model_path=model_path,
        device=device,
        lazy_load=lazy_load,
        startup_warmup=startup_warmup,
    )

    print("🚀 Starting VibeVoice realtime demo server...")
    print(f"   Model: {model_path}")
    print(f"   Device: {device}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Lazy load: {'enabled' if lazy_load else 'disabled'}")
    if startup_warmup is not None:
        print(f"   Startup warmup: {'enabled' if startup_warmup else 'disabled'}")
    print(f"\n🌐 Server running at: http://{host}:{port}")
    print(f"   Local access: http://127.0.0.1:{port}")
    print("   Press Ctrl+C to stop the server\n")

    cmd = build_realtime_demo_cmd(
        demo_script=demo_script,
        host=host,
        port=port,
        model_path=model_path,
        device=device,
        inference_steps=inference_steps,
        lazy_load=lazy_load,
        startup_warmup=startup_warmup,
        reload=reload,
    )

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running demo server: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# EngineAdapter implementation
# ---------------------------------------------------------------------------


class RealtimeDemoAdapter(EngineAdapter):
    """Adapter for the realtime 0.5B model using the subprocess demo."""

    def __init__(self, profile: ModelProfile, **kwargs: Any) -> None:
        super().__init__(profile, **kwargs)
        self.model_path: Path | None = kwargs.get("model_path")  # type: ignore[assignment]
        self.device: str | None = kwargs.get("device")

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> dict[str, Any]:
        return {
            "model": self.profile.key,
            "family": self.profile.family,
            "supports_stream": self.profile.supports_stream,
            "supports_multispeaker": self.profile.supports_multispeaker,
            "supports_voice_list": self.profile.supports_voice_list,
            "status": "available",
        }

    def list_voices(self) -> list[dict[str, Any]]:
        # Voice listing is handled by the running FastAPI app (StreamingTTSService)
        return []

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        # Synthesis is handled by the running FastAPI app endpoint
        raise NotImplementedError(
            "RealtimeDemoAdapter delegates synthesis to the FastAPI app."
        )

    def stream(self, request: SpeechRequest) -> Any:
        # Streaming is handled by the running FastAPI WebSocket endpoint
        raise NotImplementedError(
            "RealtimeDemoAdapter delegates streaming to the FastAPI app."
        )

    def health(self) -> dict[str, Any]:
        return {
            "adapter": "realtime_demo",
            "model": self.profile.key,
            "family": self.profile.family,
            "available": True,
        }

    # --- launch helper (used by run_server.py) ---

    def launch(
        self,
        *,
        project_root: Path,
        host: str = "0.0.0.0",
        port: int = 8000,
        inference_steps: int = 5,
        lazy_load: bool = False,
        startup_warmup: bool | None = None,
        reload: bool = False,
    ) -> None:
        """Launch the realtime demo subprocess (blocking)."""
        model_path = self.model_path or Path(self.profile.default_local_dir)
        device = self.device or detect_device()
        run_realtime_demo_subprocess(
            project_root=project_root,
            model_path=model_path,
            device=device,
            host=host,
            port=port,
            inference_steps=inference_steps,
            lazy_load=lazy_load,
            startup_warmup=startup_warmup,
            reload=reload,
        )
