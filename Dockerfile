FROM python:3.11-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ENABLE_FLASHSR=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv python install 3.11

COPY pyproject.toml ./
COPY .python-version ./
COPY README.md ./
COPY main.py ./

COPY third_party/VibeVoice ./third_party/VibeVoice
COPY overrides ./overrides
COPY scripts ./scripts

RUN mkdir -p models

RUN uv venv --python 3.11
RUN . .venv/bin/activate && uv sync

RUN if [ -f "overrides/app.py" ]; then \
        cp overrides/app.py third_party/VibeVoice/demo/web/app.py; \
    fi
RUN if [ -f "overrides/index.html" ]; then \
        cp overrides/index.html third_party/VibeVoice/demo/web/index.html; \
    fi

EXPOSE 8000

ENV HF_HOME=/app/models

CMD [".venv/bin/python", "scripts/run_realtime_demo.py", "--port", "8000", "--host", "0.0.0.0"]
