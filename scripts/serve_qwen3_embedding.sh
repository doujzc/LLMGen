#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Embedding-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM="${VLLM:-vllm}"

if ! command -v "$VLLM" >/dev/null 2>&1; then
  echo "vllm is required to serve the embedding model" >&2
  exit 127
fi

ARGS=(
  "$VLLM" serve "$MODEL"
  --runner pooling
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
)
if [[ -n "${REVISION:-}" ]]; then
  ARGS+=(--revision "$REVISION")
fi

exec "${ARGS[@]}"
