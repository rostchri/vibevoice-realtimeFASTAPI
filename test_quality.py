#!/usr/bin/env python3
"""
Test LavaSR quality: load audio -> upsample -> transcribe with Parakeet -> compare
"""

import os
import sys
import time
import tempfile
import wave
import numpy as np
import torch
import torchaudio
import requests

sys.path.insert(0, os.path.dirname(__file__))

from overrides.lavasr_upsampler import LavaSRUpsampler

PARAKEET_URL = "http://localhost:5092/v1/audio/transcriptions"


def load_audio(path: str, target_sr: int = None) -> tuple:
    """Load audio file"""
    waveform, sr = torchaudio.load(path)
    if target_sr and sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze().numpy(), sr


def save_wav(audio: np.ndarray, path: str, sample_rate: int):
    """Save audio as WAV file"""
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def transcribe_with_parakeet(wav_path: str) -> dict:
    """Transcribe audio using Parakeet API"""
    with open(wav_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"model": "parakeet-tdt-0.6b-v3"}

        start = time.time()
        response = requests.post(PARAKEET_URL, files=files, data=data, timeout=60)
        elapsed = time.time() - start

    if response.status_code == 200:
        result = response.json()
        result["latency"] = elapsed
        return result
    else:
        return {"error": f"{response.status_code} - {response.text}"}


def test_lavasr_quality(audio_path: str):
    """Test LavaSR quality on a given audio file"""
    print("=" * 70)
    print("LavaSR Quality Test")
    print("=" * 70)

    # Load audio
    print(f"\n[1/4] Loading audio from {audio_path}...")
    audio, sr = load_audio(audio_path)
    print(f"      Original: {len(audio)} samples at {sr}Hz ({len(audio) / sr:.2f}s)")

    # If not 24kHz, resample to 24kHz first (simulating TTS output)
    if sr != 24000:
        print(f"      Resampling to 24kHz...")
        audio, sr = load_audio(audio_path, target_sr=24000)
        print(f"      Resampled: {len(audio)} samples at {sr}Hz")

    # Transcribe original (resampled to 48kHz with simple interpolation)
    print("\n[2/4] Transcribing original (simple 24kHz -> 48kHz resampling)...")
    simple_resampler = torchaudio.transforms.Resample(24000, 48000)
    audio_48k_simple = (
        simple_resampler(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        simple_path = f.name
    save_wav(audio_48k_simple, simple_path, 48000)

    result_simple = transcribe_with_parakeet(simple_path)
    text_simple = result_simple.get("text", result_simple.get("error", ""))
    print(
        f"      Text: {text_simple[:100]}..."
        if len(text_simple) > 100
        else f"      Text: {text_simple}"
    )
    print(f"      Latency: {result_simple.get('latency', 0):.2f}s")

    # Initialize LavaSR
    print("\n[3/4] Initializing LavaSR...")
    t0 = time.time()
    upsampler = LavaSRUpsampler(device="cuda", enable=True)
    upsampler.load()
    print(f"      Done in {time.time() - t0:.2f}s")

    # Upsample with LavaSR
    print("\n[4/4] Upsampling with LavaSR and transcribing...")
    t0 = time.time()
    audio_48k_lavasr = upsampler.upsample(audio)
    upsample_time = time.time() - t0
    print(
        f"      Upsampled: {len(audio_48k_lavasr)} samples ({len(audio_48k_lavasr) / 48000:.2f}s)"
    )
    print(f"      Upsampling time: {upsample_time * 1000:.1f}ms")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        lavasr_path = f.name
    save_wav(audio_48k_lavasr, lavasr_path, 48000)

    result_lavasr = transcribe_with_parakeet(lavasr_path)
    text_lavasr = result_lavasr.get("text", result_lavasr.get("error", ""))
    print(
        f"      Text: {text_lavasr[:100]}..."
        if len(text_lavasr) > 100
        else f"      Text: {text_lavasr}"
    )
    print(f"      Latency: {result_lavasr.get('latency', 0):.2f}s")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nSimple resampling transcription:")
    print(f"  {text_simple}")
    print(f"\nLavaSR transcription:")
    print(f"  {text_lavasr}")

    # Cleanup
    os.unlink(simple_path)
    os.unlink(lavasr_path)

    return {
        "simple": text_simple,
        "lavasr": text_lavasr,
        "upsample_time_ms": upsample_time * 1000,
    }


if __name__ == "__main__":
    # Use test audio file
    test_audio = "/home/op/testargentina.wav"

    if os.path.exists(test_audio):
        results = test_lavasr_quality(test_audio)
    else:
        print(f"Test audio not found: {test_audio}")
        print("Please provide a valid audio file path")
