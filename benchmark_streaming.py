#!/usr/bin/env python3
"""
Benchmark LavaSR with realistic TTS chunk sizes (streaming scenario)
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from overrides.lavasr_upsampler import LavaSRUpsampler

SAMPLE_RATE = 24000
OUTPUT_RATE = 48000


def benchmark_streaming_chunks():
    """Benchmark with realistic TTS chunk sizes"""
    print("=" * 70)
    print("LavaSR Streaming Chunk Benchmark")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[1/2] Initializing LavaSR on {device}...")
    t0 = time.time()
    upsampler = LavaSRUpsampler(device=device, enable=True)
    upsampler.load()
    print(f"      Done in {time.time() - t0:.2f}s")

    # Realistic TTS chunk sizes: 0.1s to 2.0s
    chunk_durations = [0.1, 0.25, 0.5, 1.0, 2.0]

    print("\n[2/2] Benchmarking realistic TTS chunk sizes...")
    print(
        f"\n{'Duration':>10} | {'Samples':>10} | {'Time':>10} | {'RTF':>10} | {'Speed':>12}"
    )
    print("-" * 70)

    results = []
    for duration in chunk_durations:
        samples = int(SAMPLE_RATE * duration)
        audio = np.random.randn(samples).astype(np.float32)

        # Warmup
        _ = upsampler.upsample(audio)
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark (10 runs)
        times = []
        for _ in range(10):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            upsampled = upsampler.upsample(audio)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time = np.mean(times)
        output_duration = len(upsampled) / OUTPUT_RATE
        rtf = avg_time / output_duration
        speed = 1 / rtf if rtf > 0 else float("inf")

        results.append(
            {
                "duration": duration,
                "samples": samples,
                "time_ms": avg_time * 1000,
                "rtf": rtf,
                "speed": speed,
            }
        )

        print(
            f"{duration:>10.2f}s | {samples:>10} | {avg_time * 1000:>9.2f}ms | {rtf:>10.4f}x | {speed:>11.0f}x"
        )

    print("=" * 70)
    print("\nSummary:")
    print(f"  Average RTF: {np.mean([r['rtf'] for r in results]):.4f}x")
    print(f"  Average Speed: {np.mean([r['speed'] for r in results]):.0f}x realtime")

    return results


if __name__ == "__main__":
    benchmark_streaming_chunks()
