# Fork notes — longform (1.5B) HTTP serving + SSE

This fork adds two things the upstream groxaxo/vibevoice-realtimeFASTAPI lacked,
built + validated on an AMD RX 7900 XTX (ROCm PyTorch):

## 1. Longform (1.5B / 7B) HTTP serving — voice cloning over the API

Upstream `scripts/run_server.py` refused to serve longform models
(`🚧 Long-form serving … not yet implemented; sys.exit(1)`). This fork adds
**`runner/longform_app.py`**, an OpenAI-compatible FastAPI app that wraps the
existing `LongformNativeAdapter` (which already does reference-audio cloning +
multi-speaker), and wires `run_server.py` to launch it for `--model tts-1.5b`.

Endpoints:
- `POST /v1/audio/speech` — `{model,input,voice,response_format,speakers}`.
  `voice` = a reference name (from `LONGFORM_VOICE_DIRS`) **or** an uploaded WAV
  path → cloning (`is_prefill=True`).
- `GET  /v1/audio/voices` — list reference voices.
- `POST /v1/audio/voices` — upload a reference WAV (multipart `file`) → returns an
  `id` (path) usable as `voice`. This is the **custom voice cloning** path.
- `GET /health` · `GET /config`.

Validated: upload `en-Alice_woman.wav` → `POST /v1/audio/speech` with that voice →
HTTP 200, **~1.75× realtime** (`is_prefill` cloning), 24 kHz.

Longform is single-shot (whole clip at once), so it does **not** stream/SSE.

## 2. SSE streaming for realtime-0.5B

`overrides/app.py` `POST /v1/audio/speech` now honours `stream: true` and returns
`text/event-stream` (SSE) of base64 PCM16 chunks — so a controller can consume
streaming audio like the old `vibevoice-rs` (SSE) without the raw WebSocket.

Events:
```
event: start  data: {"sample_rate":24000,"channels":1,"format":"pcm_s16le"}
event: audio  data: <base64(pcm_s16le chunk)>
event: done   data: [DONE]
```
Implementation note: the model's `stream()` is a **blocking** generator, so the
SSE endpoint uses an **async generator + threaded queue** (`call_soon_threadsafe`)
— a plain sync StreamingResponse generator deadlocks the event loop under load.

Validated: `text/event-stream`, 27 `audio` events + `start`/`done`.

## Applying

The complete, exact change set is `patches/0001-longform-http-serving-and-sse.patch`
(also mirrored in the `/container/vibevoice-fastapi` build repo, applied at
`docker build` on top of the pinned upstream commit). It touches
`overrides/app.py`, `runner/longform_app.py`, `scripts/run_server.py`.

Set `LONGFORM_VOICE_DIRS=<VibeVoice>/demo/voices` to expose built-in reference
voices in `GET /v1/audio/voices`.
