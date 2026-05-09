"""Quality validation for VibeVoice 7B via Parakeet ASR (WER check).

Generates audio with the 7B model, sends to Parakeet on port 5092,
computes WER and reports pass/fail.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python scripts/quality_check_7b.py --device cuda
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

for _root in [
    Path("/home/op/vibevoice-community-VibeVoice"),
    _ROOT / "third_party" / "VibeVoice",
]:
    if (_root / "vibevoice" / "modular" / "modeling_vibevoice_inference.py").is_file():
        sys.path.insert(0, str(_root))
        break

import numpy as np
import requests
import scipy.io.wavfile
import torch

# English test sentences (clear, unambiguous — good for ASR WER measurement)
TEST_CASES = [
    "The weather today is sunny with a high of seventy five degrees.",
    "Machine learning models require large amounts of training data.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Text to speech technology has improved dramatically in recent years.",
    "Please call the office at nine o'clock tomorrow morning.",
]

WER_THRESHOLD = 0.20  # 20% WER — acceptable for longform TTS


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / max(n, 1)


def load_model(model_path: str, device: str):
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    print(f"Loading model from {model_path} on {device} ...")
    if "cuda" in device:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    processor = VibeVoiceProcessor.from_pretrained(model_path)
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device
        )
    except Exception:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device
        )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    return processor, model


def generate_audio(processor, model, text: str, device: str, sample_rate: int = 24000) -> bytes:
    script = f"Speaker 0: {text.strip()}"
    inputs = processor(text=[script], padding=True, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False, "temperature": 1.0},
            verbose=False,
            is_prefill=False,
        )

    audio = outputs.speech_outputs[0]
    if torch.is_tensor(audio):
        audio_np = audio.detach().cpu().float().numpy().reshape(-1)
    else:
        audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, audio_np)
    return buf.getvalue()


def transcribe(audio_bytes: bytes, parakeet_url: str) -> str:
    resp = requests.post(
        f"{parakeet_url}/v1/audio/transcriptions",
        files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
        data={"model": "parakeet-tdt-0.6b-v3", "language": "en"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality check VibeVoice 7B via Parakeet WER")
    parser.add_argument("--model-path", default=str(_ROOT / "models" / "VibeVoice-7B"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parakeet-url", default="http://localhost:5092")
    parser.add_argument("--wer-threshold", type=float, default=WER_THRESHOLD)
    args = parser.parse_args()

    # Verify Parakeet is reachable
    try:
        r = requests.get(f"{args.parakeet_url}/health", timeout=5)
        print(f"Parakeet health: {r.status_code}")
    except Exception as e:
        print(f"ERROR: Parakeet not reachable at {args.parakeet_url}: {e}")
        sys.exit(1)

    processor, model = load_model(args.model_path, args.device)

    # Warmup
    print("\nWarming up model...")
    generate_audio(processor, model, "Hello world.", args.device)

    print(f"\n{'='*60}")
    print(f"Quality Check — WER threshold: {args.wer_threshold:.0%}")
    print(f"{'='*60}")

    wers: list[float] = []
    all_passed = True

    for i, ref_text in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        audio_bytes = generate_audio(processor, model, ref_text, args.device)
        gen_time = time.perf_counter() - t0
        audio_dur = len(audio_bytes) / (24000 * 2 + 44)  # rough estimate
        hypothesis = transcribe(audio_bytes, args.parakeet_url)
        wer = _word_error_rate(ref_text, hypothesis)
        wers.append(wer)
        passed = wer <= args.wer_threshold
        all_passed = all_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n[{i}/{len(TEST_CASES)}] {status}  WER={wer:.1%}  gen={gen_time:.1f}s")
        print(f"  REF: {ref_text}")
        print(f"  HYP: {hypothesis}")

    avg_wer = sum(wers) / len(wers)
    print(f"\n{'='*60}")
    print(f"Average WER: {avg_wer:.1%}  (threshold: {args.wer_threshold:.0%})")
    print(f"Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    print(f"{'='*60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
