#!/usr/bin/env bash
set -euo pipefail

PROMPTGEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROMPTGEN_CONFIG="${PROMPTGEN_CONFIG:-configs/promptgen.env}"
if [[ "$PROMPTGEN_CONFIG" != /* ]]; then
  PROMPTGEN_CONFIG="$PROMPTGEN_ROOT/$PROMPTGEN_CONFIG"
fi
if [[ ! -f "$PROMPTGEN_CONFIG" ]]; then
  echo "PromptGen config does not exist: $PROMPTGEN_CONFIG" >&2
  return 2
fi

# shellcheck source=/dev/null
source "$PROMPTGEN_CONFIG"
cd "$PROMPTGEN_ROOT"

promptgen_require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required input does not exist: $1" >&2
    return 2
  fi
}

promptgen_positive_integer() {
  if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "$1 must be a positive integer, got: $2" >&2
    return 2
  fi
}

promptgen_train_launch() {
  promptgen_positive_integer ROUTER_NUM_GPUS "$ROUTER_NUM_GPUS"
  case "$ROUTER_FINETUNE_MODE" in
    full|lora) ;;
    *)
      echo "ROUTER_FINETUNE_MODE must be full or lora" >&2
      return 2
      ;;
  esac
  PROMPTGEN_LAUNCH=("$PYTHON")
  if (( ROUTER_NUM_GPUS > 1 )); then
    PROMPTGEN_LAUNCH=(
      "$PYTHON" -m torch.distributed.run --standalone
      --nproc-per-node "$ROUTER_NUM_GPUS"
    )
  fi
}

promptgen_step() {
  printf '\n[%s] %s\n' "$1" "$2"
}
