import asyncio
import copy
import datetime
import io
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

import numpy as np
import scipy.io.wavfile
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from pydub import AudioSegment
from starlette.websockets import WebSocketDisconnect, WebSocketState
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

from .lavasr_upsampler import LavaSRUpsampler
from .text_processing import normalize_text, split_text_into_sentences

# ---------------------------------------------------------------------------
# Runner-package imports (multi-model support)
# ---------------------------------------------------------------------------
# When app.py is copied into the vendored tree the runner package may not be on
# sys.path.  Walk upwards and add the first parent that contains the runner
# package so both local overrides/ and vendored demo/web/ copies work.
_APP_DIR = Path(__file__).resolve().parent
for _possible_root in (_APP_DIR, *_APP_DIR.parents):
    if (_possible_root / "runner" / "__init__.py").is_file():
        if str(_possible_root) not in sys.path:
            sys.path.insert(0, str(_possible_root))
        break

try:
    from runner.adapter_factory import make_adapter
    from runner.errors import UnknownModelError
    from runner.model_registry import (
        DEFAULT_MODEL_KEY,
        get_model_profile,
        list_aliases,
        list_model_keys,
        resolve_model_key,
    )

    _RUNNER_AVAILABLE = True
except ImportError:
    _RUNNER_AVAILABLE = False

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000
UPSAMPLED_RATE = 48_000
DEFAULT_TEMPERATURE = 0.9
MAX_INPUT_LENGTH = 100_000


def get_timestamp():
    timestamp = (
        datetime.datetime.now(datetime.timezone.utc)
        .astimezone(datetime.timezone(datetime.timedelta(hours=8)))
        .strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    )
    return timestamp


def parse_temperature(temp_value: Optional[Any]) -> Tuple[float, bool]:
    try:
        temperature = float(temp_value) if temp_value is not None else DEFAULT_TEMPERATURE
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE, False
    return temperature, temp_value is not None and temperature > 0


class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
        enable_flashsr: bool = True,
    ) -> None:
        # Keep model_path as string for HuggingFace repo IDs (Path() converts / to \ on Windows)
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE
        self.enable_flashsr = enable_flashsr

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, object] = {}
        self._voice_cache_lock = threading.Lock()
        self.flashsr: Optional[LavaSRUpsampler] = None

        self._compute_stream: Optional[torch.cuda.Stream] = None
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        self._active_inference_steps: Optional[int] = None

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)
        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)

    def is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    def load(self) -> None:
        if self.is_loaded():
            print("[startup] StreamingTTSService already loaded")
            return

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self._compute_stream = torch.cuda.Stream()
            self._transfer_stream = torch.cuda.Stream()
            print("[startup] CUDA optimizations enabled (TF32, cudnn benchmark, streams)")

        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Decide dtype and load strategy (with hard fallback when FlashAttention is unavailable)
        if self.device == "cuda":
            load_dtype = torch.bfloat16
            load_attempts = [
                {
                    "attn": "flash_attention_2",
                    "device_map": "cuda",
                    "move_to": None,
                    "label": "cuda + flash_attention_2",
                },
                {
                    "attn": "sdpa",
                    "device_map": None,
                    "move_to": "cuda",
                    "label": "cuda + sdpa (no device_map)",
                },
            ]
        elif self.device == "mps":
            load_dtype = torch.float32
            load_attempts = [
                {
                    "attn": "sdpa",
                    "device_map": None,
                    "move_to": "mps",
                    "label": "mps + sdpa",
                }
            ]
        else:
            load_dtype = torch.float32
            load_attempts = [
                {
                    "attn": "sdpa",
                    "device_map": None,
                    "move_to": None,
                    "label": "cpu + sdpa",
                }
            ]

        print(f"Using device: {self.device}, torch_dtype: {load_dtype}")

        self.model = None
        last_error: Optional[Exception] = None
        for attempt in load_attempts:
            kwargs = {
                "torch_dtype": load_dtype,
                "attn_implementation": attempt["attn"],
                "low_cpu_mem_usage": False,
            }
            if attempt["device_map"] is not None:
                kwargs["device_map"] = attempt["device_map"]

            try:
                print(f"[startup] Loading model with {attempt['label']}")
                candidate = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    **kwargs,
                )
                move_target = attempt["move_to"]
                if move_target:
                    candidate.to(move_target)
                self.model = candidate
                print(f"[startup] Model load succeeded with {attempt['label']}")
                break
            except Exception as exc:
                last_error = exc
                print(f"[startup] Model load failed with {attempt['label']}: {exc}")

        if self.model is None:
            raise RuntimeError("Failed to load model after all fallback attempts") from last_error

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        self._active_inference_steps = self.inference_steps

        if self.device == "cuda" and hasattr(torch, "compile"):
            print("[startup] Compiling model with torch.compile(mode='reduce-overhead')...")
            # Using fullgraph=False to allow for dynamic splitting logic compatibility
            # and reduce-overhead to leverage CUDA graphs
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
                print("[startup] Model compiled successfully")
            except Exception as e:
                print(f"[startup] torch.compile failed: {e}, falling back to eager mode")

        if self.default_voice_key is not None:
            self._ensure_voice_cached(self.default_voice_key)

        # Initialize FlashSR upsampler
        if self.enable_flashsr:
            print("[startup] Initializing LavaSR upsampler for 24kHz -> 48kHz super-resolution")
            self.flashsr = LavaSRUpsampler(device=self.device, enable=True)
            self.flashsr.load()
        else:
            print("[startup] LavaSR disabled, audio will remain at 24kHz")
            self.flashsr = LavaSRUpsampler(device=self.device, enable=False)

    def _set_inference_steps(self, steps: int) -> None:
        if self.model is None:
            raise RuntimeError("StreamingTTSService not initialized")

        if self._active_inference_steps != steps:
            self.model.set_ddpm_inference_steps(num_steps=steps)
            self._active_inference_steps = steps

        self.inference_steps = steps

    def warmup(self, text: str = "Warmup run.") -> None:
        if self.processor is None or self.model is None:
            print("[startup] Warmup skipped: service not initialized")
            return

        started_at = time.perf_counter()
        try:
            _, prefilled_outputs = self._get_voice_resources(self.default_voice_key)
            inputs = self._prepare_inputs(text, prefilled_outputs)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
                verbose=False,
                refresh_negative=True,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )

            if self.device == "cuda":
                torch.cuda.synchronize()

            if self.flashsr and self.flashsr.enabled and outputs.speech_outputs:
                warmup_audio = outputs.speech_outputs[0]
                if warmup_audio is not None:
                    if torch.is_tensor(warmup_audio):
                        warmup_audio = warmup_audio.reshape(-1)[: self.sample_rate // 4]
                    else:
                        warmup_audio = np.asarray(warmup_audio).reshape(-1)[: self.sample_rate // 4]
                    if len(warmup_audio) > 0:
                        self.flashsr.upsample_chunks(warmup_audio, sample_rate=self.sample_rate)

            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            print(f"[startup] Warmup complete in {elapsed_ms:.1f} ms")
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            print(f"[startup] Warmup failed after {elapsed_ms:.1f} ms: {exc}")

    def _load_voice_presets(self) -> Dict[str, Path]:
        voices_dir = BASE.parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: Dict[str, Path] = {}
        for pt_path in voices_dir.rglob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        default_key = "en-WHTest_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> object:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key in self._voice_cache:
            return self._voice_cache[key]

        with self._voice_cache_lock:
            if key not in self._voice_cache:
                preset_path = self.voice_presets[key]
                print(f"[startup] Loading voice preset {key} from {preset_path}")
                print(f"[startup] Loading prefilled prompt from {preset_path}")
                prefilled_outputs = torch.load(
                    preset_path,
                    map_location=self._torch_device,
                    weights_only=False,
                )
                self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object]:
        key = (
            requested_key
            if requested_key and requested_key in self.voice_presets
            else self.default_voice_key
        )
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if self.processor is None or self.model is None:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {}
        non_blocking = self.device == "cuda"
        for key, value in processed.items():
            if torch.is_tensor(value):
                if value.device != self._torch_device:
                    prepared[key] = value.to(self._torch_device, non_blocking=non_blocking)
                else:
                    prepared[key] = value
            else:
                prepared[key] = value
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        assert self.model is not None
        assert self.processor is not None
        try:
            if self._compute_stream:
                with torch.cuda.stream(self._compute_stream):
                    self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={
                            "do_sample": do_sample,
                            "temperature": temperature if do_sample else 1.0,
                            "top_p": top_p if do_sample else 1.0,
                        },
                        audio_streamer=audio_streamer,
                        stop_check_fn=stop_event.is_set,
                        verbose=False,
                        refresh_negative=refresh_negative,
                        all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                    )
            else:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={
                        "do_sample": do_sample,
                        "temperature": temperature if do_sample else 1.0,
                        "top_p": top_p if do_sample else 1.0,
                    },
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    refresh_negative=refresh_negative,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[..., None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        # 1. Clean and normalize
        text = normalize_text(text)
        if not text.strip():
            return

        # 2. Split into sentences
        sentences = split_text_into_sentences(text)
        if not sentences:
            return

        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass

        self._set_inference_steps(steps_to_use)

        # Global stop signal for the entire request
        stop_signal = stop_event or threading.Event()

        # 3. Stream each sentence sequentially
        for sentence in sentences:
            if stop_signal.is_set():
                break

            print(f"[Streaming] Processing sentence: {sentence[:50]}...")

            inputs = self._prepare_inputs(sentence, prefilled_outputs)
            audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            errors: list = []

            # Create a dedicated worker logic for this sentence
            def _worker():
                self._run_generation(
                    inputs=inputs,
                    audio_streamer=audio_streamer,
                    errors=errors,
                    cfg_scale=cfg_scale,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    refresh_negative=refresh_negative,
                    prefilled_outputs=prefilled_outputs,
                    stop_event=stop_signal,
                )

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            try:
                # Yield chunks from this sentence's streamer
                stream = audio_streamer.get_stream(0)
                for audio_chunk in stream:
                    if stop_signal.is_set():
                        break

                    if torch.is_tensor(audio_chunk):
                        tensor_chunk = audio_chunk.detach()
                        if tensor_chunk.ndim > 1:
                            tensor_chunk = tensor_chunk.reshape(-1)

                        if self.flashsr and self.flashsr.enabled:
                            audio_chunk = self.flashsr.upsample_chunks(
                                tensor_chunk, sample_rate=self.sample_rate
                            )
                        elif self._transfer_stream and tensor_chunk.device.type == "cuda":
                            with torch.cuda.stream(self._transfer_stream):
                                cpu_chunk = tensor_chunk.to(
                                    device="cpu",
                                    dtype=torch.float32,
                                    non_blocking=True,
                                )
                            self._transfer_stream.synchronize()
                            audio_chunk = cpu_chunk.numpy()
                        else:
                            audio_chunk = tensor_chunk.to(device="cpu", dtype=torch.float32).numpy()
                    else:
                        if self.flashsr and self.flashsr.enabled:
                            audio_chunk = self.flashsr.upsample_chunks(
                                audio_chunk, sample_rate=self.sample_rate
                            )

                    chunk_to_yield = np.asarray(audio_chunk, dtype=np.float32)
                    if chunk_to_yield.ndim > 1:
                        chunk_to_yield = chunk_to_yield.reshape(-1)

                    peak = np.max(np.abs(chunk_to_yield)) if chunk_to_yield.size else 0.0
                    if peak > 1.0:
                        chunk_to_yield = chunk_to_yield / peak

                    yield chunk_to_yield

            except Exception as e:
                emit("generation_error", message=str(e))
                errors.append(e)
            finally:
                # Ensure this sentence's stream is closed
                audio_streamer.end()
                thread.join(timeout=30)

                if errors:
                    # Decide if we want to stop strictly on error or continue to next sentence
                    # For now, let's log and maybe continue? Or stop?
                    # The original code raised logic, let's stop.
                    raise errors[0]

    def get_output_sample_rate(self) -> int:
        """Get the actual output sample rate (48kHz if LavaSR enabled, otherwise 24kHz)."""
        if self.flashsr and self.flashsr.enabled:
            return UPSAMPLED_RATE
        return self.sample_rate

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


app = FastAPI()


async def ensure_service_loaded() -> StreamingTTSService:
    service: StreamingTTSService = app.state.tts_service
    if service.is_loaded():
        return service

    init_lock: asyncio.Lock = app.state.service_init_lock
    async with init_lock:
        if service.is_loaded():
            return service

        started_at = time.perf_counter()
        print("[startup] Lazy load triggered; loading model on demand")
        await asyncio.to_thread(service.load)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        print(f"[startup] Lazy load complete in {elapsed_ms:.1f} ms")
        return service


@app.on_event("startup")
async def _startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH not set in environment")

    device = os.environ.get("MODEL_DEVICE", "cuda")

    inference_steps = int(os.environ.get("INFERENCE_STEPS", "5"))

    # LavaSR enabled by default
    enable_flashsr_str = os.environ.get(
        "ENABLE_LAVASR", os.environ.get("ENABLE_FLASHSR", "true")
    ).lower()
    enable_flashsr = enable_flashsr_str in ("true", "1", "yes", "on")
    enable_lazy_load = os.environ.get("ENABLE_LAZY_LOAD", "false").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
    enable_warmup = os.environ.get("ENABLE_STARTUP_WARMUP", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    service = StreamingTTSService(
        model_path=model_path,
        device=device,
        inference_steps=inference_steps,
        enable_flashsr=enable_flashsr,
    )

    app.state.tts_service = service
    app.state.model_path = model_path
    app.state.device = device
    app.state.lazy_load_enabled = enable_lazy_load
    app.state.websocket_lock = asyncio.Lock()
    app.state.service_init_lock = asyncio.Lock()
    app.state.active_model_key = os.environ.get("ACTIVE_MODEL_KEY", "realtime-0.5b")

    if enable_lazy_load:
        if enable_warmup:
            print("[startup] ENABLE_STARTUP_WARMUP ignored because ENABLE_LAZY_LOAD is enabled")
        print("[startup] Lazy load enabled; model will initialize on first speech request.")
        return

    service.load()
    if enable_warmup:
        service.warmup()
    else:
        print("[startup] Warmup disabled")

    print("[startup] Model ready.")


def streaming_tts(text: str, **kwargs) -> Iterator[np.ndarray]:
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream(text, **kwargs)


@app.websocket("/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    text = ws.query_params.get("text", "")
    model_param = ws.query_params.get("model")

    # --- Reject non-realtime models on /stream ---
    if _RUNNER_AVAILABLE and model_param:
        try:
            ws_model_key = resolve_model_key(model_param)
            ws_profile = get_model_profile(ws_model_key)
            if ws_profile.family != "realtime":
                error_msg = {
                    "type": "error",
                    "error": {
                        "message": (
                            f"Model '{ws_model_key}' does not support WebSocket streaming. "
                            "Only realtime models are supported on /stream."
                        ),
                        "type": "capability_error",
                    },
                }
                await ws.send_text(json.dumps(error_msg))
                await ws.close(code=1008, reason="Model does not support streaming")
                return
        except (UnknownModelError, Exception):
            pass  # fall through to default realtime behaviour

    print(f"Client connected, text={text!r}")
    cfg_param = ws.query_params.get("cfg")
    steps_param = ws.query_params.get("steps")
    voice_param = ws.query_params.get("voice")
    temp_param = ws.query_params.get("temp")

    try:
        cfg_scale = float(cfg_param) if cfg_param is not None else 1.5
    except ValueError:
        cfg_scale = 1.5
    if cfg_scale <= 0:
        cfg_scale = 1.5
    try:
        inference_steps = int(steps_param) if steps_param is not None else None
        if inference_steps is not None and inference_steps <= 0:
            inference_steps = None
    except ValueError:
        inference_steps = None
    temperature, do_sample = parse_temperature(temp_param)

    service = await ensure_service_loaded()
    lock: asyncio.Lock = app.state.websocket_lock

    if lock.locked():
        busy_message = {
            "type": "log",
            "event": "backend_busy",
            "data": {"message": "Please wait for the other requests to complete."},
            "timestamp": get_timestamp(),
        }
        print("Please wait for the other requests to complete.")
        try:
            await ws.send_text(json.dumps(busy_message))
        except Exception:
            pass
        await ws.close(code=1013, reason="Service busy")
        return

    acquired = False
    try:
        await lock.acquire()
        acquired = True

        log_queue: "Queue[Dict[str, Any]]" = Queue()

        def enqueue_log(event: str, **data: Any) -> None:
            log_queue.put({"event": event, "data": data})

        async def flush_logs() -> None:
            while True:
                try:
                    entry = log_queue.get_nowait()
                except Empty:
                    break
                message = {
                    "type": "log",
                    "event": entry.get("event"),
                    "data": entry.get("data", {}),
                    "timestamp": get_timestamp(),
                }
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    break

        enqueue_log(
            "backend_request_received",
            text_length=len(text or ""),
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice=voice_param,
            temperature=temperature if do_sample else None,
        )

        stop_signal = threading.Event()

        iterator = streaming_tts(
            text,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice_key=voice_param,
            do_sample=do_sample,
            temperature=temperature,
            log_callback=enqueue_log,
            stop_event=stop_signal,
        )
        sentinel = object()
        first_ws_send_logged = False

        await flush_logs()

        try:
            while ws.client_state == WebSocketState.CONNECTED:
                await flush_logs()
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                chunk = cast(np.ndarray, chunk)
                payload = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(payload)
                if not first_ws_send_logged:
                    first_ws_send_logged = True
                    enqueue_log("backend_first_chunk_sent")
                await flush_logs()
        except WebSocketDisconnect:
            print("Client disconnected (WebSocketDisconnect)")
            enqueue_log("client_disconnected")
            stop_signal.set()
        finally:
            stop_signal.set()
            enqueue_log("backend_stream_complete")
            await flush_logs()
            try:
                iterator_close = getattr(iterator, "close", None)
                if callable(iterator_close):
                    iterator_close()
            except Exception:
                pass
            # clear the log queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
            print("WS handler exit")
    finally:
        if acquired:
            lock.release()


class OpenAISpeechRequest(BaseModel):
    model: str = "tts-1"
    # Optional at the schema level so longform models can accept speakers-only
    # requests; family-specific validation below rejects blank input where needed.
    input: Optional[str] = Field(
        None,
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        description=(
            "Required for realtime models. Optional for longform models when "
            "'speakers' is provided."
        ),
    )
    voice: Optional[str] = None
    response_format: Optional[str] = Field("opus", pattern=r"^(opus|wav|mp3)$")
    speed: Optional[float] = 1.0
    temp: Optional[float] = None
    speakers: Optional[list] = None
    stream: bool = False


def _has_non_whitespace_text(value: Optional[str]) -> bool:
    return value is not None and value.strip() != ""


def _has_speakers(value: Optional[list]) -> bool:
    return value is not None and len(value) > 0


def _invalid_request(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": {"message": message, "type": "invalid_request"}},
    )


def _resolve_and_validate_model(request: OpenAISpeechRequest) -> Tuple[str, Optional[JSONResponse]]:
    """Resolve the requested model and validate compatibility.

    Returns ``(model_key, error_response)``.  If *error_response* is not
    ``None`` the caller should return it immediately.
    """
    if not _RUNNER_AVAILABLE:
        # Runner package not loaded – fall through to realtime behaviour
        return "realtime-0.5b", None

    try:
        model_key = resolve_model_key(request.model)
    except UnknownModelError as exc:
        return "", JSONResponse(
            status_code=404,
            content={"error": {"message": str(exc), "type": "unknown_model"}},
        )

    profile = get_model_profile(model_key)

    if profile.family == "realtime" and not _has_non_whitespace_text(request.input):
        return model_key, _invalid_request(
            f"Model '{model_key}' requires a non-empty 'input' field."
        )

    # Reject speakers for realtime models
    if profile.family == "realtime" and _has_speakers(request.speakers):
        return model_key, _invalid_request(
            f"Model '{model_key}' does not support the 'speakers' field. "
            "Use a longform model (tts-1.5b, tts-7b) for multi-speaker dialogue."
        )

    # For longform models, check backend availability
    if profile.family == "longform":
        if not _has_non_whitespace_text(request.input) and not _has_speakers(request.speakers):
            return model_key, _invalid_request(
                f"Model '{model_key}' requires either a non-empty "
                "'input' field or at least one speaker."
            )
        if request.stream:
            return model_key, _invalid_request(
                f"Model '{model_key}' does not support streaming."
            )
        adapter = make_adapter(model_key)
        if not adapter.is_available():
            return model_key, JSONResponse(
                status_code=501,
                content={
                    "error": {
                        "message": (
                            f"Model '{model_key}' is registered but no compatible "
                            "long-form backend is installed/configured."
                        ),
                        "type": "backend_unavailable",
                    }
                },
            )

    return model_key, None


@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAISpeechRequest):
    # --- Model resolution & capability check ---
    model_key, error_resp = _resolve_and_validate_model(request)
    if error_resp is not None:
        return error_resp

    service = await ensure_service_loaded()
    lock: asyncio.Lock = app.state.websocket_lock

    async with lock:
        # Determine voice and sampling controls
        voice_key = request.voice
        temperature, do_sample = parse_temperature(request.temp)

        # Collect audio
        stop_event = threading.Event()

        def generate_all():
            chunks = []
            iterator = service.stream(
                request.input,
                voice_key=voice_key,
                do_sample=do_sample,
                temperature=temperature,
                stop_event=stop_event,
            )
            for chunk in iterator:
                chunks.append(chunk)
            return chunks

        try:
            chunks = await asyncio.to_thread(generate_all)
        except Exception as e:
            traceback.print_exc()
            return Response(status_code=500, content=str(e))

        if not chunks:
            return Response(status_code=500, content="No audio generated")

        full_audio = np.concatenate(chunks)

        # Get the actual output sample rate (48kHz if FlashSR enabled, 24kHz otherwise)
        output_sample_rate = service.get_output_sample_rate()

        # Convert to WAV first
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, output_sample_rate, full_audio)
        wav_data = buffer.getvalue()

        # Convert to requested format
        if request.response_format == "mp3":
            buffer.seek(0)
            audio = AudioSegment.from_wav(buffer)
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format="mp3")
            return Response(content=mp3_buffer.getvalue(), media_type="audio/mpeg")
        elif request.response_format == "opus":
            try:
                buffer.seek(0)
                audio = AudioSegment.from_wav(buffer)
                opus_buffer = io.BytesIO()
                # Export as opus using the actual output sample rate
                audio.export(
                    opus_buffer,
                    format="opus",
                    codec="libopus",
                    parameters=["-ar", str(output_sample_rate)],
                )
                return Response(content=opus_buffer.getvalue(), media_type="audio/opus")
            except Exception as e:
                # Fallback to WAV if opus encoding fails (e.g., ffmpeg not available)
                print(f"[Warning] Opus encoding failed: {e}. Falling back to WAV format.")
                return Response(content=wav_data, media_type="audio/wav")
        else:  # wav or default
            return Response(content=wav_data, media_type="audio/wav")


@app.get("/v1/audio/voices")
def get_voices():
    service: StreamingTTSService = app.state.tts_service
    voices = []
    for voice_id in sorted(service.voice_presets.keys()):
        voices.append(
            {
                "id": voice_id,
                "name": voice_id,
                "object": "voice",
                "created": int(datetime.datetime.now().timestamp()),
                "category": "vibe_voice",
            }
        )
    return {"voices": voices}


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/web")
@app.get("/web/")
def web_index():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    service: StreamingTTSService = app.state.tts_service
    result: Dict[str, Any] = {
        "status": "ok",
        "lazy_load": app.state.lazy_load_enabled,
        "model_loaded": service.is_loaded(),
        "default_voice": service.default_voice_key,
    }
    if _RUNNER_AVAILABLE:
        active_key = getattr(app.state, "active_model_key", DEFAULT_MODEL_KEY)
        try:
            profile = get_model_profile(active_key)
            result.update({
                "active_model": active_key,
                "family": profile.family,
                "adapter": profile.loader_mode,
            })
        except Exception:
            result["active_model"] = active_key
    return result


@app.get("/config")
def get_config():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    result: Dict[str, Any] = {
        "voices": voices,
        "default_voice": service.default_voice_key,
    }
    if _RUNNER_AVAILABLE:
        models_info: list[Dict[str, Any]] = []
        for key in list_model_keys():
            try:
                adapter = make_adapter(key)
                models_info.append(adapter.capabilities())
            except Exception:
                models_info.append({"model": key, "status": "error"})
        result.update({
            "available_models": list_model_keys(),
            "default_model": DEFAULT_MODEL_KEY,
            "aliases": list_aliases(),
            "models": models_info,
        })
    return result
