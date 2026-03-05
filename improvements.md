# Realtime Performance Improvements (Quality-Locked)

## Reviewed Plan Summary

- Keep the current Hugging Face/VibeVoice realtime inference path for Realtime-0.5B TTS.
- Do not switch to vLLM for now: this repo relies on VibeVoice streaming diffusion classes (`VibeVoiceStreamingForConditionalGenerationInference`, `AudioStreamer`) and has no equivalent vLLM-backed TTS path.
- Prioritize optimizations that improve latency/throughput **without any quality degradation**.

## Goals

- Improve latency and throughput on RTX 3060 while preserving output quality.
- Maintain compatibility with existing endpoints:
  - `GET /config`
  - `POST /v1/audio/speech`
  - `WS /stream`

## Quality Lock (Non-Negotiable)

- Keep generation settings fixed during optimization runs:
  - `INFERENCE_STEPS=5`
  - `cfg_scale=1.5`
  - deterministic decode (`do_sample=false`, equivalent to `temp` unset or `0`)
- Keep `ENABLE_LAVASR=true` in the quality-locked profile.
- Use identical voices, text sets, model weights, and hardware for baseline vs candidate comparisons.
- Reject any optimization that violates quality gates, even if latency improves.

## Success Criteria

- Performance: at least one metric improves by >=15% versus baseline:
  - time-to-first-audio (TTFA)
  - average and p95 chunk interval
  - end-to-end RTF
- Quality: all quality gates pass:
  - EN 10-phrase average similarity to normalized text >= 0.95
  - ES 10-phrase average similarity to source text >= 0.88
  - no increase in audible artifacts, clipping, or dropouts in manual A/B spot checks
- Reliability: no API/WebSocket functional regressions.

## Optimization Roadmap

### 1) Baseline and Instrumentation

- Establish a reproducible baseline with current defaults and save outputs to `/tmp`.
- Capture per-request metrics:
  - TTFA
  - chunk interval distribution (avg, p95)
  - RTF
  - error/timeout rate
- Add a concise baseline table in commit or PR notes for every optimization batch.

### 2) Startup Warmup (No Quality Risk)

- Run one short warmup generation after model and voice cache load.
- Prime CUDA execution path and optional post-processing path during startup.
- Ensure warmup failures are non-fatal and clearly logged with elapsed time.

### 3) Streaming Hot-Path Optimizations (Quality-Safe)

- Reduce repeated per-request overhead:
  - avoid resetting DDPM steps when value is unchanged
  - avoid redundant setup across sentence boundaries where safe
  - minimize unnecessary tensor/device transfers and sync points
- Audit expensive copies in the generation path (for example deep copies of cached prompt state) and replace with safer, cheaper alternatives only if output parity is verified.
- Keep changes small, isolated, and measurable.

### 4) Concurrency and Transfer Efficiency (Quality-Safe)

- Improve overlap between generation and host transfer while preserving chunk order.
- Use stream/event synchronization only where required for correctness.
- Avoid introducing jitter spikes in chunk delivery under concurrent requests.

### 5) Validation on Every Change

- For each optimization commit, run:
  1. realtime benchmark for `/stream`
  2. `/v1/audio/speech` smoke test
  3. transcription similarity checks (EN + ES)
  4. short manual A/B listening pass on representative phrases
- Promote only commits that pass both performance and quality gates.

### 6) Rollout and Documentation

- Document two explicit run profiles:
  - **quality-locked** (default): performance optimizations with quality guarantees
  - **latency-experimental** (optional): clearly labeled, not used for quality-guaranteed claims
- Update README with exact launch commands, expected behavior, and known tradeoffs.

## Risk and Rollback

- Risks:
  - hidden quality regressions from aggressive memory/copy optimizations
  - instability from threading/stream synchronization changes
- Rollback strategy:
  - keep commits atomic and benchmarked
  - revert only the offending optimization commit
  - restore last known quality-locked baseline immediately if any quality gate fails
