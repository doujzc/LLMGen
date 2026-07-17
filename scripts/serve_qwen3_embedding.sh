#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/skillret/common.sh
source "$ROOT/scripts/skillret/common.sh"

MODEL="${MODEL:-$EMBEDDING_MODEL}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
HOST="${HOST:-$VLLM_HOST}"
PORT="${PORT:-$VLLM_PORT}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
DTYPE="${DTYPE:-$VLLM_DTYPE}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$VLLM_MAX_MODEL_LEN}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$VLLM_TENSOR_PARALLEL_SIZE}"
VLLM="${VLLM:-vllm}"

if ! command -v "$VLLM" >/dev/null 2>&1; then
  echo "vllm is required to serve the embedding model" >&2
  exit 127
fi

ARGS=(
  "$VLLM" serve "$MODEL"
  --task embed
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
