"""Benchmark script for VibeVoice 7B longform model - measures RTF (Real-Time Factor).

RTF < 1.0 means faster than real-time. Lower is better.

Usage:
    python scripts/benchmark_7b.py --device cuda:1 --model-path models/VibeVoice-7B
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_7b.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Resolve VibeVoice longform source tree
_VIBEVOICE_ROOTS = [
    Path("/home/op/vibevoice-community-VibeVoice"),
    _ROOT / "third_party" / "VibeVoice",
]
for _root in _VIBEVOICE_ROOTS:
    _marker = _root / "vibevoice" / "modular" / "modeling_vibevoice_inference.py"
    if _marker.is_file():
        sys.path.insert(0, str(_root))
        break

import numpy as np
import torch

# ------------------------------------------------------------------
# Test texts (English, varied lengths to measure RTF across durations)
# ------------------------------------------------------------------
TEST_TEXTS = [
    # ~3s audio
    "The quick brown fox jumps over the lazy dog.",
    # ~8s audio
    "Artificial intelligence is transforming the way we interact with technology. "
    "From voice assistants to autonomous vehicles, AI systems are becoming an "
    "integral part of everyday life.",
    # ~15s audio
    "Text-to-speech synthesis has advanced remarkably over the past decade. "
    "Modern neural TTS systems can produce speech that is nearly indistinguishable "
    "from human voice recordings. These systems learn from large datasets of human "
    "speech, capturing nuances of prosody, rhythm, and intonation that were "
    "previously impossible to replicate with rule-based approaches.",
]


def load_model(model_path: str, device: str, compiled: bool = False) -> tuple:
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    print(f"Loading processor from {model_path} ...")
    processor = VibeVoiceProcessor.from_pretrained(model_path)

    if "cuda" in device:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"Loading model on {device} (bfloat16) ...")
    t0 = time.perf_counter()
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
    except Exception as e:
        print(f"  flash_attention_2 failed ({e}), falling back to sdpa")
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)  # DO NOT CHANGE — quality requirement

    if compiled and "cuda" in device:
        print("  Applying torch.compile(reduce-overhead) ...")
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print(f"  torch.compile unavailable: {e}")

    load_elapsed = time.perf_counter() - t0
    print(f"Model loaded in {load_elapsed:.1f}s")
    return processor, model


def run_inference(processor, model, text: str, device: str, sample_rate: int = 24000) -> tuple[float, float]:
    """Run inference for one text sample, return (gen_time_sec, audio_duration_sec)."""
    script = f"Speaker 0: {text.strip()}"
    inputs = processor(
        text=[script],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    # Move to device
    inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}

    if "cuda" in device:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={
                "do_sample": False,
                "temperature": 1.0,
            },
            verbose=False,
            is_prefill=False,
        )
    if "cuda" in device:
        torch.cuda.synchronize()
    gen_time = time.perf_counter() - t0

    audio = outputs.speech_outputs[0]
    if torch.is_tensor(audio):
        audio_np = audio.detach().cpu().float().numpy().reshape(-1)
    else:
        audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)

    audio_duration = len(audio_np) / sample_rate
    return gen_time, audio_duration


def benchmark(
    model_path: str,
    device: str,
    n_runs: int = 3,
    warmup_runs: int = 1,
    sample_rate: int = 24000,
    compiled: bool = False,
) -> None:
    print(f"\n{'='*60}")
    print(f"VibeVoice 7B Benchmark")
    print(f"  Model:    {model_path}")
    print(f"  Device:   {device}")
    print(f"  Steps:    10 (fixed, quality requirement)")
    print(f"  Compiled: {compiled}")
    print(f"  Warmup:   {warmup_runs} run(s), Timed: {n_runs} run(s)")
    print(f"{'='*60}\n")

    processor, model = load_model(model_path, device, compiled=compiled)

    # VRAM usage
    if "cuda" in device:
        dev_idx = device.split(":")[-1] if ":" in device else 0
        mem_alloc = torch.cuda.memory_allocated(device) / 1e9
        mem_reserv = torch.cuda.memory_reserved(device) / 1e9
        print(f"\nVRAM after load: {mem_alloc:.2f} GB allocated / {mem_reserv:.2f} GB reserved")

    # Warmup
    print(f"\n--- Warmup ({warmup_runs} run) ---")
    for i in range(warmup_runs):
        text = TEST_TEXTS[1]  # medium text for warmup
        gen_t, audio_dur = run_inference(processor, model, text, device, sample_rate)
        rtf = gen_t / audio_dur
        print(f"  warmup {i+1}: audio={audio_dur:.2f}s gen={gen_t:.2f}s RTF={rtf:.3f}")

    # Timed runs across all test texts
    print(f"\n--- Timed Runs ---")
    rtfs: list[float] = []
    for run_i in range(n_runs):
        text = TEST_TEXTS[run_i % len(TEST_TEXTS)]
        gen_t, audio_dur = run_inference(processor, model, text, device, sample_rate)
        rtf = gen_t / audio_dur
        rtfs.append(rtf)
        status = "✓ realtime" if rtf < 1.0 else "✗ slower-than-realtime"
        print(f"  run {run_i+1}: text_len={len(text):3d}ch audio={audio_dur:.2f}s gen={gen_t:.2f}s RTF={rtf:.3f} {status}")

    # Summary
    avg_rtf = sum(rtfs) / len(rtfs)
    min_rtf = min(rtfs)
    max_rtf = max(rtfs)
    print(f"\n{'='*60}")
    print(f"RTF Summary (lower=faster, <1.0 = realtime capable)")
    print(f"  Average: {avg_rtf:.3f}")
    print(f"  Min:     {min_rtf:.3f}")
    print(f"  Max:     {max_rtf:.3f}")
    print(f"  Realtime capable: {'YES' if avg_rtf < 1.0 else 'NO'}")
    print(f"{'='*60}\n")

    if "cuda" in device:
        mem_alloc = torch.cuda.memory_allocated(device) / 1e9
        mem_reserv = torch.cuda.memory_reserved(device) / 1e9
        print(f"VRAM peak: {mem_alloc:.2f} GB allocated / {mem_reserv:.2f} GB reserved")

    return avg_rtf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VibeVoice 7B longform model RTF")
    parser.add_argument("--model-path", default=str(_ROOT / "models" / "VibeVoice-7B"), help="Path to model")
    parser.add_argument("--device", default="cuda", help="Torch device (e.g. cuda, cuda:1, cpu)")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Model sample rate")
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile(reduce-overhead)")
    args = parser.parse_args()

    benchmark(
        model_path=args.model_path,
        device=args.device,
        n_runs=args.n_runs,
        warmup_runs=args.warmup_runs,
        sample_rate=args.sample_rate,
        compiled=args.compile,
    )
