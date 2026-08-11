#!/usr/bin/env bash
set -euo pipefail

TOP1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOP1_CONFIG="${TOP1_CONFIG:-configs/top1.env}"
if [[ "$TOP1_CONFIG" != /* ]]; then
  TOP1_CONFIG="$TOP1_ROOT/$TOP1_CONFIG"
fi
if [[ ! -f "$TOP1_CONFIG" ]]; then
  echo "Top1 config does not exist: $TOP1_CONFIG" >&2
  return 2
fi

# shellcheck source=/dev/null
source "$TOP1_CONFIG"
cd "$TOP1_ROOT"

top1_require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required input does not exist: $1" >&2
    return 2
  fi
}

top1_positive_integer() {
  if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "$1 must be a positive integer, got: $2" >&2
    return 2
  fi
}

top1_train_launch() {
  top1_positive_integer ROUTER_NUM_GPUS "$ROUTER_NUM_GPUS"
  case "$ROUTER_FINETUNE_MODE" in
    full|lora) ;;
    *)
      echo "ROUTER_FINETUNE_MODE must be full or lora" >&2
      return 2
      ;;
  esac
  TOP1_LAUNCH=("$PYTHON")
  if (( ROUTER_NUM_GPUS > 1 )); then
    TOP1_LAUNCH=(
      "$PYTHON" -m torch.distributed.run --standalone
      --nproc-per-node "$ROUTER_NUM_GPUS"
    )
  fi
}

top1_step() {
  printf '\n[%s] %s\n' "$1" "$2"
}
