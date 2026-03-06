#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="${SERVICE_NAME:-vibevoice-realtime}"
SERVICE_USER="${SERVICE_USER:-$(id -un)}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8881}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/VibeVoice-Realtime-0.5B}"
LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/vibevoice_server.log}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-}"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
UV_BIN="${UV_BIN:-$(command -v uv)}"

if [[ -z "${UV_BIN}" ]]; then
    echo "uv not found in PATH" >&2
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "Project root not found: ${PROJECT_ROOT}" >&2
    exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
    exit 1
fi

cuda_env_line=""
if [[ -n "${CUDA_DEVICE}" ]]; then
    cuda_env_line="Environment=CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"
fi

sudo tee "${UNIT_PATH}" >/dev/null <<EOF
[Unit]
Description=VibeVoice Realtime FastAPI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment=PATH=$(dirname "${UV_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=MODEL_PATH=${MODEL_PATH}
${cuda_env_line}
Environment=ENABLE_LAZY_LOAD=true
Environment=ENABLE_STARTUP_WARMUP=false
ExecStart=${UV_BIN} run python scripts/run_realtime_demo.py --host ${HOST} --port ${PORT} --model-path ${MODEL_PATH} --lazy-load --no-startup-warmup
Restart=always
RestartSec=5
StandardOutput=append:${LOG_PATH}
StandardError=append:${LOG_PATH}

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"

echo "Installed ${SERVICE_NAME}.service at ${UNIT_PATH}"
echo "Start it with: sudo systemctl start ${SERVICE_NAME}.service"
echo "Check it with: sudo systemctl status ${SERVICE_NAME}.service --no-pager"
