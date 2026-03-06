#!/usr/bin/env python3
"""
Benchmark realtime websocket endpoint (/stream) for TTFA, chunk pacing, and RTF.

Usage example:
  uv run python scripts/benchmark_stream_endpoint.py \
    --ws-url ws://127.0.0.1:8000/stream \
    --voice en-Carter_man \
    --runs 10
"""

import argparse
import asyncio
import datetime
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, cast
from urllib.parse import urlencode, urlparse, urlunparse

import websockets
from websockets.exceptions import ConnectionClosed

DEFAULT_TEXTS = [
    "Hello, this is a deterministic realtime streaming benchmark.",
    "We measure time to first audio, chunk pacing, and overall real-time factor.",
    "Quality is locked by default settings while we optimize throughput.",
]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def append_query(url: str, params: Dict[str, str]) -> str:
    parsed = urlparse(url)
    query = urlencode(params)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            query,
            parsed.fragment,
        )
    )


async def run_single_benchmark(
    ws_url: str,
    text: str,
    voice: Optional[str],
    cfg_scale: float,
    steps: int,
    temp: float,
    sample_rate: int,
    timeout_s: float,
) -> Dict[str, object]:
    params = {
        "text": text,
        "cfg": str(cfg_scale),
        "steps": str(steps),
        "temp": str(temp),
    }
    if voice:
        params["voice"] = voice

    url = append_query(ws_url, params)

    start = asyncio.get_running_loop().time()
    end = start
    first_audio_at = None
    previous_audio_at = None
    chunk_intervals_ms: List[float] = []
    audio_bytes = 0
    audio_chunks = 0
    events: Dict[str, int] = {}

    try:
        async with websockets.connect(
            url,
            max_size=None,
            ping_timeout=timeout_s,
            close_timeout=5,
        ) as ws:
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"timed out waiting for websocket messages ({timeout_s}s)"
                    ) from exc
                except ConnectionClosed as exc:
                    if exc.code == 1013:
                        raise RuntimeError("backend reported busy (close code 1013)")
                    end = asyncio.get_running_loop().time()
                    break

                now = asyncio.get_running_loop().time()
                end = now
                if isinstance(message, bytes):
                    audio_chunks += 1
                    audio_bytes += len(message)
                    if first_audio_at is None:
                        first_audio_at = now
                    if previous_audio_at is not None:
                        chunk_intervals_ms.append((now - previous_audio_at) * 1000.0)
                    previous_audio_at = now
                else:
                    try:
                        payload = json.loads(message)
                        event_name = payload.get("event")
                        if event_name:
                            events[event_name] = events.get(event_name, 0) + 1
                    except json.JSONDecodeError:
                        events["invalid_log_payload"] = events.get("invalid_log_payload", 0) + 1
    except Exception:
        raise

    if first_audio_at is None or audio_chunks == 0:
        raise RuntimeError("no audio chunks were received from /stream")

    total_s = end - start
    audio_duration_s = (audio_bytes / 2.0) / float(sample_rate)
    rtf = total_s / audio_duration_s if audio_duration_s > 0 else math.inf

    return {
        "ttfa_ms": (first_audio_at - start) * 1000.0,
        "total_time_s": total_s,
        "audio_duration_s": audio_duration_s,
        "rtf": rtf,
        "audio_chunks": audio_chunks,
        "audio_bytes": audio_bytes,
        "avg_chunk_interval_ms": statistics.mean(chunk_intervals_ms)
        if chunk_intervals_ms
        else math.nan,
        "p95_chunk_interval_ms": percentile(chunk_intervals_ms, 95),
        "chunk_intervals_ms": chunk_intervals_ms,
        "events": events,
    }


def print_summary(summary: Dict[str, float]) -> None:
    print("\n=== /stream benchmark summary ===")
    print(f"runs: {int(summary['runs'])}")
    print(f"ttfa avg/p95: {summary['ttfa_avg_ms']:.2f} / {summary['ttfa_p95_ms']:.2f} ms")
    print(
        f"chunk interval avg/p95: {summary['chunk_interval_avg_ms']:.2f} / {summary['chunk_interval_p95_ms']:.2f} ms"
    )
    print(f"rtf avg/p95: {summary['rtf_avg']:.4f} / {summary['rtf_p95']:.4f}")
    print(f"audio duration avg: {summary['audio_duration_avg_s']:.3f} s")


def compare_to_baseline(
    baseline_summary: Dict[str, float], current_summary: Dict[str, float]
) -> None:
    def improvement_pct(new_value: float, old_value: float) -> float:
        if old_value == 0:
            return math.nan
        return ((old_value - new_value) / old_value) * 100.0

    print("\n=== baseline comparison (positive is better) ===")
    print(
        f"ttfa improvement: {improvement_pct(current_summary['ttfa_avg_ms'], baseline_summary['ttfa_avg_ms']):.2f}%"
    )
    print(
        f"chunk interval improvement: {improvement_pct(current_summary['chunk_interval_avg_ms'], baseline_summary['chunk_interval_avg_ms']):.2f}%"
    )
    print(
        f"rtf improvement: {improvement_pct(current_summary['rtf_avg'], baseline_summary['rtf_avg']):.2f}%"
    )


async def run(args: argparse.Namespace) -> int:
    texts: List[str]
    if args.text:
        texts = [args.text.strip()]
    elif args.text_file:
        lines = Path(args.text_file).read_text(encoding="utf-8").splitlines()
        texts = [line.strip() for line in lines if line.strip()]
        if not texts:
            raise RuntimeError(f"No valid text lines found in {args.text_file}")
    else:
        texts = DEFAULT_TEXTS

    runs: List[Dict[str, object]] = []
    failures: List[str] = []

    for index in range(args.runs):
        text = texts[index % len(texts)]
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await run_single_benchmark(
                    ws_url=args.ws_url,
                    text=text,
                    voice=args.voice,
                    cfg_scale=args.cfg_scale,
                    steps=args.steps,
                    temp=args.temp,
                    sample_rate=args.sample_rate,
                    timeout_s=args.timeout_s,
                )
                result["run"] = index + 1
                result["text"] = text
                runs.append(result)
                print(
                    f"run {index + 1:02d}: ttfa={result['ttfa_ms']:.2f}ms, "
                    f"avg_chunk={result['avg_chunk_interval_ms']:.2f}ms, "
                    f"rtf={result['rtf']:.4f}, chunks={result['audio_chunks']}"
                )
                break
            except Exception as exc:
                message = f"run {index + 1:02d} attempt {attempt}: {exc}"
                if attempt <= args.max_retries:
                    print(f"{message}; retrying...")
                    await asyncio.sleep(args.retry_delay_s)
                    continue
                failures.append(message)
                print(f"{message}; giving up")
                break

    if not runs:
        raise RuntimeError("No successful benchmark runs")

    ttfa_values = [float(cast(float, entry["ttfa_ms"])) for entry in runs]
    rtf_values = [float(cast(float, entry["rtf"])) for entry in runs]
    audio_duration_values = [float(cast(float, entry["audio_duration_s"])) for entry in runs]
    all_intervals: List[float] = []
    for entry in runs:
        all_intervals.extend(cast(List[float], entry["chunk_intervals_ms"]))

    summary = {
        "runs": float(len(runs)),
        "ttfa_avg_ms": statistics.mean(ttfa_values),
        "ttfa_p95_ms": percentile(ttfa_values, 95),
        "chunk_interval_avg_ms": statistics.mean(all_intervals) if all_intervals else math.nan,
        "chunk_interval_p95_ms": percentile(all_intervals, 95),
        "rtf_avg": statistics.mean(rtf_values),
        "rtf_p95": percentile(rtf_values, 95),
        "audio_duration_avg_s": statistics.mean(audio_duration_values),
    }

    print_summary(summary)

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    output = {
        "timestamp_utc": now_utc.isoformat(),
        "config": {
            "ws_url": args.ws_url,
            "voice": args.voice,
            "cfg_scale": args.cfg_scale,
            "steps": args.steps,
            "temp": args.temp,
            "sample_rate": args.sample_rate,
            "runs_requested": args.runs,
            "max_retries": args.max_retries,
        },
        "summary": summary,
        "runs": runs,
        "failures": failures,
    }

    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        baseline_summary = baseline.get("summary", {})
        if baseline_summary:
            compare_to_baseline(baseline_summary, summary)

    output_path = args.out_json
    if not output_path:
        stamp = now_utc.strftime("%Y%m%d_%H%M%S")
        output_path = f"/tmp/stream_benchmark_{stamp}.json"

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark output: {output_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark /stream websocket endpoint")
    parser.add_argument(
        "--ws-url",
        type=str,
        default="ws://127.0.0.1:8000/stream",
        help="WebSocket URL for /stream endpoint",
    )
    parser.add_argument("--voice", type=str, default=None, help="Voice preset ID")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="CFG scale query parameter",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Inference steps query parameter",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature query parameter (0 for deterministic)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Expected output sample rate for RTF calculation",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text prompt for all runs",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to text file (one prompt per line)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Per-message websocket timeout",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count per failed run",
    )
    parser.add_argument(
        "--retry-delay-s",
        type=float,
        default=1.0,
        help="Delay before retry after a failed run",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Path to save JSON output (defaults to /tmp)",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Previous benchmark JSON to compare against",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be > 0")
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be > 0")
    if args.text and args.text_file:
        raise SystemExit("Use either --text or --text-file, not both")

    exit_code = asyncio.run(run(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
