# Fork notes — longform (1.5B) HTTP serving + SSE (both models)

Adds to upstream groxaxo/vibevoice-realtimeFASTAPI, built + validated on an AMD
RX 7900 XTX (ROCm PyTorch).

## 1. Longform (1.5B / 7B) HTTP serving — voice cloning over the API

Upstream `scripts/run_server.py` refused to serve longform models
(`🚧 not yet implemented; sys.exit(1)`). This adds **`runner/longform_app.py`**,
an OpenAI-compatible FastAPI app wrapping the existing `LongformNativeAdapter`
(reference-audio cloning + multi-speaker), and wires `run_server.py` to launch it
for `--model tts-1.5b`.

- `POST /v1/audio/speech` — `{model,input,voice,response_format,speakers,stream}`.
  `voice` = a reference name (`LONGFORM_VOICE_DIRS`) **or** an uploaded WAV path.
- `GET  /v1/audio/voices` — list reference voices.
- `POST /v1/audio/voices` — upload a reference WAV (multipart `file`) → `id`
  usable as `voice` (**custom cloning**).
- `GET /health` · `GET /config`.

Validated: upload `en-Alice_woman.wav` → clone = HTTP 200, **~1.75× realtime**
(`is_prefill=True`), 24 kHz.

## 2. SSE streaming on BOTH models

`POST /v1/audio/speech` with `"stream": true` returns `text/event-stream`:
```
event: start  data: {"sample_rate":24000,"channels":1,"format":"pcm_s16le"}
event: audio  data: <base64(pcm_s16le chunk)>
event: done   data: [DONE]
```
- **realtime-0.5B** — true incremental streaming (chunks as generated). Uses an
  **async generator + threaded queue** (`call_soon_threadsafe`); a plain sync
  StreamingResponse generator deadlocks the event loop under the model's own
  threading. Validated: 27 `audio` events.
- **tts-1.5B** — single-shot model, so the whole clip is generated then **framed**
  as the same SSE events (uniform consumer; not low-latency). Validated: 11 events.

Omit `stream` (or `false`) for a buffered complete-audio response.

## Security

Deployed behind a mandatory Bearer proxy (see `/container/vibevoice-fastapi`):
every HTTP path (buffered + SSE speech, voice upload, voices, config) requires
`Authorization: Bearer $VIBEVOICE_API_KEY`; SSE is streamed through unbuffered.
The only un-proxied surface is the raw WebSocket `/stream` — **prefer SSE**.

## Applying

Complete change set: `patches/0001-longform-http-serving-and-sse.patch` (mirrored
in the `/container/vibevoice-fastapi` build repo, `git apply`-ed at docker build on
the pinned upstream commit). Touches `overrides/app.py`, `runner/longform_app.py`,
`scripts/run_server.py`. Set `LONGFORM_VOICE_DIRS=<VibeVoice>/demo/voices` for
built-in reference voices.
