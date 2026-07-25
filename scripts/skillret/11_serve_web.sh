#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/skillret/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_DIR="${WEB_MODEL_DIR:-$ROUTER_OUTPUT_DIR/retrieval}"
if (( $# > 0 )) && [[ "$1" != -* ]]; then
  MODEL_DIR="$1"
  shift
fi
skillret_require_dir "$MODEL_DIR"

WEB_DEVICE="${WEB_DEVICE:-$DEVICE}"
if [[ "$WEB_DEVICE" == "cuda" ]]; then
  WEB_DEVICE="cuda:0"
fi

ARGS=(
  --model-dir "$MODEL_DIR"
  --host "${WEB_HOST:-127.0.0.1}"
  --port "${WEB_PORT:-8080}"
  --device "$WEB_DEVICE"
  --dtype "${WEB_DTYPE:-$EVAL_DTYPE}"
  --max-code-paths "${WEB_MAX_CODE_PATHS:-$EVAL_MAX_CODE_PATHS}"
  --max-num-beams "${WEB_MAX_NUM_BEAMS:-8}"
  --max-batch-queries "${WEB_MAX_BATCH_QUERIES:-1000}"
  --max-batch-size "${WEB_MAX_BATCH_SIZE:-8}"
)
if [[ -n "${WEB_BASE_MODEL:-}" ]]; then
  ARGS+=(--base-model-name-or-path "$WEB_BASE_MODEL")
fi
if [[ -n "${WEB_MAX_INPUT_LENGTH:-}" ]]; then
  ARGS+=(--max-input-length "$WEB_MAX_INPUT_LENGTH")
fi
case "$ROUTER_TRUST_REMOTE_CODE" in
  1|true|yes) ARGS+=(--trust-remote-code) ;;
  0|false|no) ;;
  *)
    echo "ROUTER_TRUST_REMOTE_CODE must be 0 or 1" >&2
    exit 2
    ;;
esac

exec "$PYTHON" -m web_server.server "${ARGS[@]}" "$@"
