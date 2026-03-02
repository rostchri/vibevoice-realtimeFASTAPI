#!/usr/bin/env python3
"""
Test LavaSR upsampler integration and benchmark performance.
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


def test_lavasr():
    print("=" * 60)
    print("LavaSR Upsampler Test (24kHz -> 48kHz direct)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n[1/3] Initializing LavaSR...")
    t0 = time.time()
    upsampler = LavaSRUpsampler(device=device, enable=True)
    upsampler.load()
    print(f"      Done in {time.time() - t0:.2f}s")

    print("\n[2/3] Creating test audio (24kHz sine wave)...")
    duration = 5.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    print(f"      Input: {len(audio)} samples, {duration}s duration")

    print("\n[3/3] Benchmarking upsampling (10 runs)...")
    times = []

    for i in range(10):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        upsampled = upsampler.upsample(audio)
        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)

        if i == 0:
            print(
                f"      Output: {len(upsampled)} samples, {len(upsampled) / OUTPUT_RATE:.3f}s duration"
            )

    avg_time = np.mean(times[1:])  # Skip first run (warmup)
    output_duration = len(upsampled) / OUTPUT_RATE
    rtf = avg_time / output_duration

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Avg upsampling time: {avg_time * 1000:.2f}ms")
    print(f"Output audio duration: {output_duration:.3f}s")
    print(f"Upsampling RTF: {rtf:.4f}x")
    print(f"Speed: {1 / rtf:.0f}x realtime")
    print("=" * 60)


if __name__ == "__main__":
    test_lavasr()
