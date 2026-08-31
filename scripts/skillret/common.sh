#!/usr/bin/env bash

if [[ -n "${SKILLRET_COMMON_LOADED:-}" ]]; then
  return 0
fi
SKILLRET_COMMON_LOADED=1

SKILLRET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKILLRET_CONFIG="${SKILLRET_CONFIG:-configs/skillret.env}"
if [[ "$SKILLRET_CONFIG" != /* ]]; then
  SKILLRET_CONFIG="$SKILLRET_ROOT/$SKILLRET_CONFIG"
fi
if [[ ! -f "$SKILLRET_CONFIG" ]]; then
  echo "SkillRet config does not exist: $SKILLRET_CONFIG" >&2
  return 2
fi

# shellcheck source=/dev/null
source "$SKILLRET_CONFIG"
cd "$SKILLRET_ROOT"

skillret_require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required input does not exist: $path" >&2
    return 2
  fi
}

skillret_require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required input directory does not exist: $path" >&2
    return 2
  fi
}

skillret_require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    return 2
  fi
}

skillret_configure_router() {
  skillret_require_positive_integer ROUTER_NUM_GPUS "$ROUTER_NUM_GPUS"
  skillret_require_positive_integer ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE \
    "$ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE"
  skillret_require_positive_integer ROUTER_GRADIENT_ACCUMULATION_STEPS \
    "$ROUTER_GRADIENT_ACCUMULATION_STEPS"
  case "$ROUTER_FINETUNE_MODE" in
    lora|full) ;;
    *)
      echo "ROUTER_FINETUNE_MODE must be 'lora' or 'full'" >&2
      return 2
      ;;
  esac

  ROUTER_LAUNCH=("$PYTHON")
  if (( ROUTER_NUM_GPUS > 1 )); then
    ROUTER_LAUNCH=(
      "$PYTHON" -m torch.distributed.run
      --standalone
      --nproc-per-node "$ROUTER_NUM_GPUS"
    )
  fi

  ROUTER_DEEPSPEED_ARGS=()
  if [[ -n "$ROUTER_DEEPSPEED_CONFIG" && "$ROUTER_DEEPSPEED_CONFIG" != "none" ]]; then
    skillret_require_file "$ROUTER_DEEPSPEED_CONFIG"
    ROUTER_DEEPSPEED_ARGS=(--deepspeed "$ROUTER_DEEPSPEED_CONFIG")
  fi

  ROUTER_PRECISION_ARGS=()
  case "$ROUTER_PRECISION" in
    bf16) ROUTER_PRECISION_ARGS=(--bf16) ;;
    fp16) ROUTER_PRECISION_ARGS=(--fp16) ;;
    fp32|none) ;;
    *)
      echo "ROUTER_PRECISION must be bf16, fp16, or fp32" >&2
      return 2
      ;;
  esac

  ROUTER_CHECKPOINT_ARGS=()
  case "$ROUTER_GRADIENT_CHECKPOINTING" in
    1|true|yes)
      ROUTER_CHECKPOINT_ARGS=(
        --gradient-checkpointing
        --gradient-checkpointing-mode "$ROUTER_GRADIENT_CHECKPOINTING_MODE"
      )
      ;;
    0|false|no) ;;
    *)
      echo "ROUTER_GRADIENT_CHECKPOINTING must be 0 or 1" >&2
      return 2
      ;;
  esac

  ROUTER_TRUST_ARGS=()
  case "$ROUTER_TRUST_REMOTE_CODE" in
    1|true|yes) ROUTER_TRUST_ARGS=(--trust-remote-code) ;;
    0|false|no) ;;
    *)
      echo "ROUTER_TRUST_REMOTE_CODE must be 0 or 1" >&2
      return 2
      ;;
  esac

  ROUTER_COMMON_ARGS=(
    --virtual-tokens "$INDEX_DIR/virtual_tokens.txt"
    --skill-catalog "$PROCESSED_DIR/catalog_train.jsonl"
    --skill-codes "$INDEX_DIR/train_codes.jsonl"
    --skill-registry "$INDEX_DIR/train_registry.json"
    --output-dir "$ROUTER_OUTPUT_DIR"
    --num-levels "$NUM_LEVELS"
    --max-length "$ROUTER_MAX_LENGTH"
    --per-device-train-batch-size "$ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE"
    --per-device-eval-batch-size "$ROUTER_PER_DEVICE_EVAL_BATCH_SIZE"
    --gradient-accumulation-steps "$ROUTER_GRADIENT_ACCUMULATION_STEPS"
    --weight-decay "$ROUTER_WEIGHT_DECAY"
    --warmup-ratio "$ROUTER_WARMUP_RATIO"
    --logging-steps "$ROUTER_LOGGING_STEPS"
    --save-steps "$ROUTER_SAVE_STEPS"
    --eval-steps "$ROUTER_EVAL_STEPS"
    --save-total-limit "$ROUTER_SAVE_TOTAL_LIMIT"
    --dataloader-num-workers "$ROUTER_DATALOADER_NUM_WORKERS"
    --seed "$ROUTER_SEED"
    ${ROUTER_DEEPSPEED_ARGS[@]+"${ROUTER_DEEPSPEED_ARGS[@]}"}
    ${ROUTER_PRECISION_ARGS[@]+"${ROUTER_PRECISION_ARGS[@]}"}
    ${ROUTER_CHECKPOINT_ARGS[@]+"${ROUTER_CHECKPOINT_ARGS[@]}"}
    ${ROUTER_TRUST_ARGS[@]+"${ROUTER_TRUST_ARGS[@]}"}
  )

  ROUTER_COMPAT_EXTRA_ARGS=()
  if [[ -n "$ROUTER_EXTRA_ARGS" ]]; then
    # Backward compatibility with the old full script's word-split string.
    # shellcheck disable=SC2206
    ROUTER_COMPAT_EXTRA_ARGS=($ROUTER_EXTRA_ARGS)
  fi

  if [[ "$DEVICE" == cuda* ]]; then
    local available_gpus
    available_gpus=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
    if (( available_gpus < ROUTER_NUM_GPUS )); then
      echo "Router requested $ROUTER_NUM_GPUS GPUs, but only $available_gpus are visible" >&2
      echo "Set CUDA_VISIBLE_DEVICES or override ROUTER_NUM_GPUS" >&2
      return 2
    fi
  fi
}

skillret_print_step() {
  printf '\n[%s] %s\n' "$1" "$2"
}
