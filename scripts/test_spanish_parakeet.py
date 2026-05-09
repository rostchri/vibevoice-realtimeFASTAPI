#!/usr/bin/env python3
"""
End-to-end Spanish TTS quality test.

Generates Spanish audio from VibeVoice, transcribes with Parakeet ASR on
port 5092, and reports word-error-rate and pass/fail per test case.

Usage:
    python scripts/test_spanish_parakeet.py [--host HOST] [--port PORT]
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Spanish test sentences — short, medium, and long utterances
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": "sp_short_1",
        "voice": "sp-Spk0_woman",
        "text": "Hola, buenos días.",
        "description": "Simple greeting (woman)",
    },
    {
        "id": "sp_short_2",
        "voice": "sp-Spk1_man",
        "text": "Gracias por tu ayuda.",
        "description": "Simple thanks (man)",
    },
    {
        "id": "sp_medium_1",
        "voice": "sp-Spk0_woman",
        "text": "El cielo está despejado hoy y hace un calor agradable.",
        "description": "Weather sentence (woman)",
    },
    {
        "id": "sp_medium_2",
        "voice": "sp-Spk1_man",
        "text": "Quiero una mesa para dos personas esta noche.",
        "description": "Restaurant reservation (man)",
    },
    {
        "id": "sp_long_1",
        "voice": "sp-Spk0_woman",
        "text": (
            "La inteligencia artificial está transformando muchas industrias, "
            "desde la medicina hasta la educación y el entretenimiento."
        ),
        "description": "Multi-topic sentence (woman)",
    },
    {
        "id": "sp_long_2",
        "voice": "sp-Spk1_man",
        "text": (
            "Mañana tengo una reunión importante con el equipo de desarrollo "
            "para revisar el progreso del proyecto."
        ),
        "description": "Work meeting sentence (man)",
    },
    {
        "id": "sp_numbers",
        "voice": "sp-Spk1_man",
        "text": "El precio total es treinta y cinco euros con cincuenta céntimos.",
        "description": "Numbers in Spanish (man)",
    },
    {
        "id": "sp_multi_sentence",
        "voice": "sp-Spk0_woman",
        "text": (
            "Hola y bienvenido. "
            "El sistema ya está listo. "
            "Gracias por su paciencia."
        ),
        "description": "Multi-sentence pipeline test (woman)",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPANISH_NUMBER_WORDS = {
    "cero": 0,
    "uno": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}
SPANISH_TENS = {
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}
SPANISH_HUNDREDS = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "trescientos": 300,
    "cuatrocientos": 400,
    "quinientos": 500,
    "seiscientos": 600,
    "setecientos": 700,
    "ochocientos": 800,
    "novecientos": 900,
}
SPANISH_SINGLE_DIGITS = {
    "uno": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
}


def strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def parse_spanish_number_tokens(tokens: list[str], start: int) -> tuple[int, int] | None:
    total = 0
    current = 0
    consumed = 0
    index = start
    seen = False

    while index < len(tokens):
        token = tokens[index]
        if token in SPANISH_HUNDREDS:
            current += SPANISH_HUNDREDS[token]
            seen = True
            index += 1
            consumed += 1
            continue
        if token in SPANISH_NUMBER_WORDS:
            current += SPANISH_NUMBER_WORDS[token]
            seen = True
            index += 1
            consumed += 1
            continue
        if token in SPANISH_TENS:
            current += SPANISH_TENS[token]
            seen = True
            index += 1
            consumed += 1
            if (
                index + 1 < len(tokens)
                and tokens[index] == "y"
                and tokens[index + 1] in SPANISH_SINGLE_DIGITS
            ):
                current += SPANISH_SINGLE_DIGITS[tokens[index + 1]]
                index += 2
                consumed += 2
            continue
        if token == "mil":
            current = max(current, 1)
            total += current * 1000
            current = 0
            seen = True
            index += 1
            consumed += 1
            continue
        break

    if not seen:
        return None
    return total + current, consumed


def canonicalize_spanish_numbers(text: str) -> str:
    tokens = text.split()
    normalized_tokens: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token.isdigit():
            normalized_tokens.append(str(int(token)))
            index += 1
            continue

        parsed = parse_spanish_number_tokens(tokens, index)
        if parsed is None:
            normalized_tokens.append(token)
            index += 1
            continue

        value, consumed = parsed
        normalized_tokens.append(str(value))
        index += consumed

    return " ".join(normalized_tokens)


def normalize_for_compare(text: str) -> str:
    """Lowercase, remove accents, canonicalize Spanish numbers, collapse whitespace."""
    text = strip_accents(text.lower().strip())
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return canonicalize_spanish_numbers(text)


def word_error_rate(ref: str, hyp: str) -> float:
    """Compute word-error-rate between reference and hypothesis strings."""
    ref_words = normalize_for_compare(ref).split()
    hyp_words = normalize_for_compare(hyp).split()
    if not ref_words:
        return 0.0

    # Dynamic programming edit distance
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
    return dp[m] / len(ref_words)


def generate_wav(tts_base: str, text: str, voice: str) -> bytes:
    """Call VibeVoice /v1/audio/speech and return raw WAV bytes."""
    resp = requests.post(
        f"{tts_base}/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "wav",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


def transcribe_wav(parakeet_base: str, wav_bytes: bytes, language: str = "es") -> str:
    """Send WAV to Parakeet /v1/audio/transcriptions and return transcript."""
    resp = requests.post(
        f"{parakeet_base}/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={"model": "parakeet-tdt-0.6b-v3", "language": language},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    # OpenAI-compatible: {"text": "..."}
    if isinstance(result, dict):
        return result.get("text", "")
    return str(result)


def save_test_audio(wav_bytes: bytes, test_id: str, output_dir: Path) -> Path:
    out_path = output_dir / f"{test_id}.wav"
    out_path.write_bytes(wav_bytes)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tests(tts_base: str, parakeet_base: str, output_dir: Path, wer_threshold: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = 0

    print(f"\n{'='*70}")
    print("  VibeVoice Spanish TTS + Parakeet ASR Verification")
    print(f"  TTS:      {tts_base}")
    print(f"  Parakeet: {parakeet_base}")
    print(f"  WER pass threshold: {wer_threshold:.0%}")
    print(f"{'='*70}\n")

    for tc in TEST_CASES:
        test_id = tc["id"]
        text = tc["text"]
        voice = tc["voice"]
        desc = tc["description"]

        print(f"[{test_id}] {desc}")
        print(f"  Voice: {voice}")
        print(f"  Input: {text!r}")

        try:
            t0 = time.perf_counter()
            wav_bytes = generate_wav(tts_base, text, voice)
            tts_ms = (time.perf_counter() - t0) * 1000

            audio_path = save_test_audio(wav_bytes, test_id, output_dir)
            print(f"  TTS:   {tts_ms:.0f} ms  →  {audio_path.name}  ({len(wav_bytes)//1024} KB)")

            t1 = time.perf_counter()
            transcript = transcribe_wav(parakeet_base, wav_bytes, language="es")
            asr_ms = (time.perf_counter() - t1) * 1000

            wer = word_error_rate(text, transcript)
            passed = wer <= wer_threshold

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  ASR:   {asr_ms:.0f} ms  →  {transcript!r}")
            print(f"  WER:   {wer:.1%}  {status}\n")

            results.append({
                "id": test_id,
                "description": desc,
                "voice": voice,
                "input": text,
                "transcript": transcript,
                "wer": wer,
                "passed": passed,
                "tts_ms": tts_ms,
                "asr_ms": asr_ms,
            })
            if not passed:
                failures += 1

        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            results.append({
                "id": test_id,
                "description": desc,
                "voice": voice,
                "input": text,
                "transcript": None,
                "wer": None,
                "passed": False,
                "error": str(exc),
            })
            failures += 1

    # Summary
    total = len(TEST_CASES)
    passed = total - failures
    print(f"{'='*70}")
    print(f"  Results: {passed}/{total} passed  ({failures} failures)")
    avg_wer = sum(r["wer"] for r in results if r["wer"] is not None) / max(
        1, sum(1 for r in results if r["wer"] is not None)
    )
    print(f"  Average WER: {avg_wer:.1%}")
    print(f"{'='*70}\n")

    # Save JSON report
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Report saved to: {report_path}")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Spanish TTS + ASR verification test")
    parser.add_argument("--tts-host", default="http://localhost:8001",
                        help="VibeVoice TTS base URL (default: http://localhost:8001)")
    parser.add_argument("--parakeet-host", default="http://localhost:5092",
                        help="Parakeet ASR base URL (default: http://localhost:5092)")
    parser.add_argument("--output-dir", default="/tmp/vibevoice-spanish-test",
                        help="Directory to save WAV files and report")
    parser.add_argument("--wer-threshold", type=float, default=0.30,
                        help="WER threshold for pass (default: 0.30 = 30%%)")
    parser.add_argument("--wait-ready", action="store_true",
                        help="Wait for TTS server to become ready before testing")
    args = parser.parse_args()

    if args.wait_ready:
        print(f"Waiting for TTS server at {args.tts_host}/health ...")
        for _ in range(60):
            try:
                r = requests.get(f"{args.tts_host}/health", timeout=5)
                if r.status_code == 200:
                    d = r.json()
                    if d.get("model_loaded"):
                        print("  TTS server ready.\n")
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            print("ERROR: TTS server did not become ready in 5 minutes.")
            sys.exit(1)

    failures = run_tests(
        tts_base=args.tts_host,
        parakeet_base=args.parakeet_host,
        output_dir=Path(args.output_dir),
        wer_threshold=args.wer_threshold,
    )
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
