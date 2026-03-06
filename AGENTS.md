# AI Agent Instructions

This file contains instructions for AI agents working on the VibeVoice Realtime FastAPI project.

## Project Overview

This is a high-performance local TTS (Text-to-Speech) runner for Microsoft's VibeVoice Realtime model with OpenAI-compatible API endpoints.

## Key Commands

### Development
```bash
# Install dependencies
uv sync

# Install with dev tools (linting, testing)
uv sync --extra dev

# Run the server
uv run python scripts/run_realtime_demo.py --port 8000

# Download model (first time setup)
uv run python scripts/download_model.py
```

### Testing & Quality
```bash
# Run linting
uv run --extra dev ruff check .

# Run type checking
uv run --extra dev mypy overrides/ scripts/

# Run tests
uv run --extra dev pytest test_lavasr.py test_quality.py

# Format code
uv run --extra dev ruff format .
```

### Docker
```bash
# Build image
docker-compose build

# Run container
docker-compose up
```

## Code Quality Standards

1. **Run linter before committing**: `uv run --extra dev ruff check .`
2. **Format code**: `uv run --extra dev ruff format .`
3. **Check types**: `uv run --extra dev mypy overrides/ scripts/`
4. **Test changes**: Run relevant test files

## Project Structure

- `overrides/` - Custom implementations that override VibeVoice defaults
  - `app.py` - Main FastAPI application
  - `lavasr_upsampler.py` - Neural audio super-resolution (24kHz → 48kHz)
  - `text_processing.py` - Text normalization and sentence splitting
- `scripts/` - Utility scripts for running and managing the service
- `third_party/VibeVoice/` - Microsoft's VibeVoice repository (git submodule)
- `models/` - Downloaded model weights (not tracked in git)

## Important Notes

1. **Python Version**: Project requires Python 3.11 (pinned in .python-version)
2. **Model Path**: Set `MODEL_PATH` environment variable or use default `models/VibeVoice-Realtime-0.5B`
3. **LavaSR**: Enabled by default for 48kHz output, can be disabled with `ENABLE_LAVASR=false`
4. **CUDA**: Optimized for NVIDIA GPUs with Flash Attention 2 support
5. **Dependencies**: Managed via `uv` - do not use pip directly

## Configuration

Key environment variables:
- `MODEL_PATH` - Path to model weights
- `MODEL_DEVICE` - Device to use (cuda/mps/cpu)
- `INFERENCE_STEPS` - Number of DDPM steps (default: 5)
- `ENABLE_LAVASR` - Enable neural upsampling (default: true)
- `HF_TOKEN` - Hugging Face token for gated models

## API Endpoints

- `GET /` - Web UI
- `GET /config` - Get available voices
- `GET /v1/audio/voices` - List voices (OpenAI-compatible)
- `POST /v1/audio/speech` - Generate speech (OpenAI-compatible)
- `WS /stream` - WebSocket streaming endpoint

## When Making Changes

1. Update imports if modifying module structure
2. Test with both CUDA and CPU devices if changing inference code
3. Verify audio quality hasn't degraded
4. Update README.md if changing user-facing features
5. Keep commits atomic and well-documented
