#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ARGS=(
  --dataset-dir "$DATASET_DIR"
  --processed-dir "$PROCESSED_DIR"
  --embedding-dir "$EMBEDDING_DIR"
  --embedding-provider "$EMBEDDING_PROVIDER"
  --embedding-model "$EMBEDDING_MODEL"
  --batch-size "$EMBEDDING_BATCH_SIZE"
)
if [[ -n "${DATASET_NAME:-}" ]]; then
  ARGS+=(--dataset-name "$DATASET_NAME")
fi
if [[ "$EMBEDDING_PROVIDER" == "openai" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  ARGS+=(
    --embedding-base-url "$EMBEDDING_BASE_URL"
    --embedding-timeout "$EMBEDDING_TIMEOUT"
    --embedding-max-retries "$EMBEDDING_MAX_RETRIES"
  )
fi
if [[ -n "$EMBEDDING_DIMENSIONS" ]]; then
  ARGS+=(--embedding-dimensions "$EMBEDDING_DIMENSIONS")
fi
if [[ -n "${EMBEDDING_MAX_BATCH_CHARS:-}" ]]; then
  ARGS+=(--embedding-max-batch-chars "$EMBEDDING_MAX_BATCH_CHARS")
fi
if [[ -n "${EMBEDDING_MAX_SKILL_CHARS:-}" ]]; then
  ARGS+=(--max-skill-chars "$EMBEDDING_MAX_SKILL_CHARS")
fi

"$PYTHON" "${PREPARE_SCRIPT:-scripts/prepare_skillret.py}" "${ARGS[@]}" "$@"
