#!/usr/bin/env python3
"""
Benchmark VibeVoice TTS performance - measures RTF (Real-Time Factor)
RTF < 1.0 means faster than realtime
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import copy
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party/VibeVoice"))

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

SAMPLE_RATE = 24000


def benchmark_tts(
    model_path: str,
    device: str = "cuda",
    inference_steps: int = 15,
    text: str = "Hello, this is a performance benchmark test for VibeVoice text to speech synthesis. We are measuring the real time factor to determine how fast the model can generate audio compared to real time playback.",
    num_runs: int = 3,
    warmup: bool = True,
):
    print(f"\n{'=' * 60}")
    print(f"VibeVoice TTS Benchmark")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Inference steps: {inference_steps}")
    print(f"Text length: {len(text)} chars, {len(text.split())} words")
    print(f"{'=' * 60}\n")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("✓ TF32 and cuDNN benchmark enabled for Ampere GPU")

    print("[1/4] Loading processor...")
    t0 = time.time()
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
    print(f"      Done in {time.time() - t0:.2f}s")

    print("[2/4] Loading model...")
    t0 = time.time()

    if device == "cuda":
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
    else:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device,
            attn_implementation="sdpa",
        )

    model.eval()
    model.model.noise_scheduler = model.model.noise_scheduler.from_config(
        model.model.noise_scheduler.config,
        algorithm_type="sde-dpmsolver++",
        beta_schedule="squaredcos_cap_v2",
    )
    model.set_ddpm_inference_steps(num_steps=inference_steps)

    if device == "cuda" and hasattr(torch, "compile"):
        print("[startup] Compiling model with torch.compile(mode='reduce-overhead')...")
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[startup] Model compiled successfully")
        except Exception as e:
            print(f"[startup] torch.compile failed: {e}, falling back to eager mode")

    print(f"      Done in {time.time() - t0:.2f}s")

    voices_dir = os.path.join(
        os.path.dirname(__file__), "third_party/VibeVoice/demo/voices/streaming_model"
    )
    voice_path = os.path.join(voices_dir, "en-Carter_man.pt")

    print(f"[3/4] Loading voice preset...")
    t0 = time.time()
    prefilled_outputs = torch.load(voice_path, map_location=device, weights_only=False)
    print(f"      Done in {time.time() - t0:.2f}s")

    print(f"[4/4] Preparing inputs...")
    t0 = time.time()
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    print(f"      Done in {time.time() - t0:.2f}s")

    print(f"\n{'=' * 60}")
    print(f"Running Benchmark ({num_runs} runs)")
    print(f"{'=' * 60}\n")

    if warmup:
        print("[Warmup] Running 1 warmup iteration...")
        _ = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
        )
        if device == "cuda":
            torch.cuda.synchronize()
        print("[Warmup] Complete\n")

    results = []

    for run in range(num_runs):
        print(f"[Run {run + 1}/{num_runs}] Generating audio...")

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
        )

        if device == "cuda":
            torch.cuda.synchronize()

        gen_time = time.time() - start_time

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio_samples = (
                outputs.speech_outputs[0].shape[-1]
                if len(outputs.speech_outputs[0].shape) > 0
                else len(outputs.speech_outputs[0])
            )
            audio_duration = audio_samples / SAMPLE_RATE
            rtf = gen_time / audio_duration if audio_duration > 0 else float("inf")

            results.append(
                {
                    "gen_time": gen_time,
                    "audio_duration": audio_duration,
                    "rtf": rtf,
                    "audio_samples": audio_samples,
                }
            )

            print(f"          Gen time: {gen_time:.3f}s")
            print(
                f"          Audio:    {audio_duration:.3f}s ({audio_samples} samples)"
            )
            print(
                f"          RTF:      {rtf:.3f}x {'✓ FASTER THAN REALTIME' if rtf < 1 else ''}"
            )
        else:
            print(f"          ERROR: No audio generated")

        print()

    if results:
        avg_rtf = np.mean([r["rtf"] for r in results])
        avg_gen_time = np.mean([r["gen_time"] for r in results])
        avg_audio_dur = np.mean([r["audio_duration"] for r in results])
        std_rtf = np.std([r["rtf"] for r in results])

        print(f"{'=' * 60}")
        print(f"BENCHMARK RESULTS")
        print(f"{'=' * 60}")
        print(
            f"Average Generation Time: {avg_gen_time:.3f}s ± {np.std([r['gen_time'] for r in results]):.3f}s"
        )
        print(f"Average Audio Duration:  {avg_audio_dur:.3f}s")
        print(f"Average RTF:             {avg_rtf:.3f}x ± {std_rtf:.3f}")
        print(f"Speed:                   {1 / avg_rtf:.1f}x realtime")
        print(f"{'=' * 60}")

        if avg_rtf < 1:
            print(f"✓ GENERATION IS {1 / avg_rtf:.1f}x FASTER THAN REALTIME")
        else:
            print(f"⚠ Generation is {avg_rtf:.1f}x SLOWER than realtime")
        print(f"{'=' * 60}\n")

        return {
            "avg_rtf": avg_rtf,
            "avg_gen_time": avg_gen_time,
            "avg_audio_duration": avg_audio_dur,
            "device": device,
            "inference_steps": inference_steps,
        }

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VibeVoice TTS")
    parser.add_argument(
        "--model-path", type=str, default="models/VibeVoice-Realtime-0.5B"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    )
    parser.add_argument("--inference-steps", type=int, default=15)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    text = (
        args.text
        or "Hello, this is a performance benchmark test for VibeVoice text to speech synthesis. We are measuring the real time factor to determine how fast the model can generate audio compared to real time playback."
    )

    device = args.device
    if device.startswith("cuda:"):
        device_id = int(device.split(":")[1])
        torch.cuda.set_device(device_id)
        device = "cuda"

    benchmark_tts(
        model_path=args.model_path,
        device=device,
        inference_steps=args.inference_steps,
        text=text,
        num_runs=args.num_runs,
    )
