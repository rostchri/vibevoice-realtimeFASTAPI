"""Native longform adapter for VibeVoice 1.5B / 7B models."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import numpy as np
import scipy.io.wavfile
import torch

from runner.adapters.base import EngineAdapter
from runner.errors import BackendUnavailableError, CapabilityError, InvalidRequestForModelError
from runner.model_registry import ModelProfile
from runner.types import SpeakerTurn, SpeechRequest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUPPORTED_VOICE_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


def _candidate_vibevoice_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("LONGFORM_VIBEVOICE_SOURCE")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend(
        [
            Path("/home/op/vibevoice-community-VibeVoice"),
            _PROJECT_ROOT / "third_party" / "VibeVoice",
        ]
    )
    return roots


def _resolve_vibevoice_root() -> Path:
    for root in _candidate_vibevoice_roots():
        marker = root / "vibevoice" / "modular" / "modeling_vibevoice_inference.py"
        if marker.is_file():
            return root
    raise ImportError("No VibeVoice longform source tree with modeling_vibevoice_inference.py was found")


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LongformNativeAdapter(EngineAdapter):
    """Adapter for VibeVoice 1.5B / 7B long-form models."""

    _backend_status_cache: ClassVar[dict[str, tuple[bool, str | None]]] = {}
    _backend_status_cache_lock: ClassVar[Lock] = Lock()

    def __init__(self, profile: ModelProfile, **kwargs: Any) -> None:
        super().__init__(profile, **kwargs)
        self.device = str(kwargs.get("device") or os.environ.get("MODEL_DEVICE") or _detect_device())
        if self.device == "mpx":
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"

        model_path = kwargs.get("model_path")
        if model_path is None:
            model_path = _PROJECT_ROOT / profile.default_local_dir
        self.model_path = Path(model_path).expanduser().resolve()
        self.inference_steps = int(kwargs.get("inference_steps") or os.environ.get("LONGFORM_INFERENCE_STEPS", "10"))
        self.cfg_scale = float(os.environ.get("LONGFORM_CFG_SCALE", "1.3"))
        self.sample_rate = int(os.environ.get("LONGFORM_SAMPLE_RATE", "24000"))

        self._backend_loaded = False
        self._backend_error: str | None = None
        self._runtime_loaded = False
        self._runtime_lock = Lock()

        self._processor_cls: Any = None
        self._model_cls: Any = None
        self._processor: Any = None
        self._model: Any = None

    # ------------------------------------------------------------------
    # Backend detection and loading
    # ------------------------------------------------------------------

    def _voice_search_dirs(self) -> list[Path]:
        dirs = [_PROJECT_ROOT / "voices"]
        extra_dirs = os.environ.get("LONGFORM_VOICE_DIRS", "")
        for raw_dir in extra_dirs.split(","):
            if raw_dir.strip():
                dirs.append(Path(raw_dir.strip()).expanduser())
        deduped: list[Path] = []
        seen: set[Path] = set()
        for directory in dirs:
            resolved = directory.resolve() if directory.exists() else directory
            if resolved not in seen:
                seen.add(resolved)
                deduped.append(resolved)
        return deduped

    def _load_backend(self) -> bool:
        """Check whether the code and local model assets are present."""
        try:
            vibevoice_root = _resolve_vibevoice_root()
            loaded_vibevoice = sys.modules.get("vibevoice")
            if loaded_vibevoice is not None:
                loaded_path = Path(getattr(loaded_vibevoice, "__file__", "")).resolve()
                if vibevoice_root not in loaded_path.parents:
                    for module_name in list(sys.modules):
                        if module_name == "vibevoice" or module_name.startswith("vibevoice."):
                            sys.modules.pop(module_name, None)
            if str(vibevoice_root) not in sys.path:
                sys.path.insert(0, str(vibevoice_root))

            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

            if not self.model_path.exists():
                raise ImportError(f"Model path does not exist: {self.model_path}")

            self._model_cls = VibeVoiceForConditionalGenerationInference
            self._processor_cls = VibeVoiceProcessor
            return True
        except Exception as exc:
            self._backend_error = str(exc)
            return False

    def _ensure_backend(self) -> None:
        with self._backend_status_cache_lock:
            if self._backend_loaded:
                return

            cached = self._backend_status_cache.get(self.profile.key)
            if cached is not None:
                available, error = cached
                self._backend_error = error
                if available:
                    if self._processor_cls is None or self._model_cls is None:
                        if not self._load_backend():
                            raise BackendUnavailableError(self.profile.key, detail=self._backend_error)
                    self._backend_loaded = True
                    return
                raise BackendUnavailableError(self.profile.key, detail=self._backend_error)

            available = self._load_backend()
            backend_error = self._backend_error
            self._backend_status_cache[self.profile.key] = (
                available,
                None if available else backend_error,
            )

            if not available:
                raise BackendUnavailableError(self.profile.key, detail=backend_error)

            self._backend_loaded = True

    def _ensure_runtime_loaded(self) -> None:
        self._ensure_backend()
        if self._runtime_loaded:
            return

        with self._runtime_lock:
            if self._runtime_loaded:
                return

            try:
                self._processor = self._processor_cls.from_pretrained(str(self.model_path))

                if self.device == "mps":
                    load_dtype = torch.float32
                    load_attempts = [
                        {
                            "attn": "sdpa",
                            "device_map": None,
                            "move_to": "mps",
                        }
                    ]
                elif self.device == "cuda":
                    # Enable TF32 for matmul and cuDNN kernels — no accuracy impact on bfloat16 models
                    # and significantly reduces GEMM latency on Ampere+ GPUs.
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    # Let cuDNN auto-select the fastest convolution kernel for the current input shapes.
                    torch.backends.cudnn.benchmark = True

                    load_dtype = torch.bfloat16
                    load_attempts = [
                        {
                            "attn": "flash_attention_2",
                            "device_map": "cuda",
                            "move_to": None,
                        },
                        {
                            "attn": "sdpa",
                            "device_map": None,
                            "move_to": "cuda",
                        },
                    ]
                else:
                    load_dtype = torch.float32
                    load_attempts = [
                        {
                            "attn": "sdpa",
                            "device_map": "cpu",
                            "move_to": None,
                        }
                    ]

                model = None
                last_error: Exception | None = None
                for attempt in load_attempts:
                    kwargs: dict[str, Any] = {
                        "torch_dtype": load_dtype,
                        "attn_implementation": attempt["attn"],
                    }
                    if attempt["device_map"] is not None:
                        kwargs["device_map"] = attempt["device_map"]
                    try:
                        candidate = self._model_cls.from_pretrained(str(self.model_path), **kwargs)
                        move_to = attempt["move_to"]
                        if move_to:
                            candidate.to(move_to)
                        model = candidate
                        break
                    except Exception as exc:
                        last_error = exc

                if model is None:
                    raise RuntimeError("Failed to load longform VibeVoice model") from last_error

                model.eval()
                model.set_ddpm_inference_steps(num_steps=self.inference_steps)
                self._model = model

                # Compile the model with torch.compile for faster repeat inference.
                # reduce-overhead mode fuses CuDNN/CUDA kernels to cut graph launch overhead.
                # Only enabled on CUDA; compile is a no-op on MPS/CPU or older PyTorch builds.
                if self.device == "cuda":
                    try:
                        self._model = torch.compile(self._model, mode="reduce-overhead", fullgraph=False)
                    except Exception:
                        pass  # torch.compile unavailable — continue without it

                self._runtime_loaded = True
                self._backend_error = None
            except Exception as exc:
                self._backend_error = str(exc)
                raise BackendUnavailableError(self.profile.key, detail=self._backend_error) from exc

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    def _build_script_from_input(self, request: SpeechRequest) -> tuple[str, int]:
        if request.speakers:
            speaker_ids: dict[str, int] = {}
            lines: list[str] = []
            for turn in request.speakers:
                if not isinstance(turn, SpeakerTurn):
                    turn = SpeakerTurn.model_validate(turn)
                speaker_name = turn.speaker.strip()
                line_text = turn.text.strip()
                if not speaker_name or not line_text:
                    continue
                if speaker_name not in speaker_ids:
                    speaker_ids[speaker_name] = len(speaker_ids)
                lines.append(f"Speaker {speaker_ids[speaker_name]}: {line_text}")
            if not lines:
                raise InvalidRequestForModelError("At least one non-empty speaker turn is required.")
            return "\n".join(lines), len(speaker_ids)

        if request.input is None or request.input.strip() == "":
            raise InvalidRequestForModelError("Field 'input' is required when 'speakers' is absent.")
        return f"Speaker 0: {request.input.strip()}", 1

    def _resolve_voice_candidate(self, voice_name: str) -> str:
        candidate = Path(voice_name).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())

        for directory in self._voice_search_dirs():
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in _SUPPORTED_VOICE_EXTENSIONS:
                    if path.stem == voice_name or path.name == voice_name:
                        return str(path.resolve())

        raise InvalidRequestForModelError(
            f"Voice reference '{voice_name}' was not found. "
            "Pass an absolute file path or configure LONGFORM_VOICE_DIRS."
        )

    def _resolve_voice_samples(self, request: SpeechRequest, speaker_count: int) -> list[str] | None:
        if request.voice is None or request.voice.strip() == "":
            return None

        voices = [item.strip() for item in request.voice.split(",") if item.strip()]
        if not voices:
            return None

        if len(voices) not in {1, speaker_count}:
            raise InvalidRequestForModelError(
                f"Expected 1 or {speaker_count} voice references, got {len(voices)}."
            )

        resolved = [self._resolve_voice_candidate(item) for item in voices]
        if len(resolved) == 1 and speaker_count > 1:
            resolved = resolved * speaker_count
        return resolved

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        target_device = self.device if self.device in ("cuda", "mps") else "cpu"
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(target_device, non_blocking=True)
        return inputs

    # ------------------------------------------------------------------
    # EngineAdapter interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            self._ensure_backend()
            return True
        except BackendUnavailableError:
            return False

    def get_backend_error(self) -> str | None:
        return self._backend_error

    def capabilities(self) -> dict[str, Any]:
        available = self.is_available()
        return {
            "model": self.profile.key,
            "family": self.profile.family,
            "supports_stream": self.profile.supports_stream,
            "supports_multispeaker": self.profile.supports_multispeaker,
            "supports_voice_list": True,
            "status": "available" if available else "backend_unavailable",
        }

    def list_voices(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = []
        for directory in self._voice_search_dirs():
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in _SUPPORTED_VOICE_EXTENSIONS:
                    continue
                voices.append(
                    {
                        "id": str(path.resolve()),
                        "name": path.stem,
                        "object": "voice",
                        "category": "reference_audio",
                    }
                )
        return voices

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        self._ensure_runtime_loaded()
        assert self._processor is not None
        assert self._model is not None

        script, speaker_count = self._build_script_from_input(request)
        voice_samples = self._resolve_voice_samples(request, speaker_count)

        processor_kwargs = {
            "text": [script],
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        if voice_samples is not None:
            processor_kwargs["voice_samples"] = [voice_samples]

        try:
            inputs = self._processor(**processor_kwargs)
            inputs = self._to_device(inputs)

            temperature = float(request.temp) if request.temp is not None else 1.0
            do_sample = request.temp is not None and temperature > 0

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self._processor.tokenizer,
                    generation_config={
                        "do_sample": do_sample,
                        "temperature": temperature if do_sample else 1.0,
                    },
                    verbose=False,
                    is_prefill=voice_samples is not None,
                )
        except InvalidRequestForModelError:
            raise
        except Exception as exc:
            self._backend_error = str(exc)
            raise BackendUnavailableError(self.profile.key, detail=self._backend_error) from exc

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise BackendUnavailableError(self.profile.key, detail="Longform backend returned no audio.")

        audio = outputs.speech_outputs[0]
        if torch.is_tensor(audio):
            audio_np = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)
        audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)

        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, self.sample_rate, audio_np)
        return buffer.getvalue(), "audio/wav"

    def stream(self, request: SpeechRequest) -> Any:
        raise CapabilityError(self.profile.key, "stream")

    def health(self) -> dict[str, Any]:
        available = self.is_available()
        return {
            "adapter": "longform_native",
            "model": self.profile.key,
            "family": self.profile.family,
            "available": available,
            "loaded": self._runtime_loaded,
            **({"error": self._backend_error} if not available else {}),
        }
