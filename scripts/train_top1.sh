#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOP1_CONFIG="${TOP1_CONFIG:-$ROOT_DIR/configs/top1.env}"
if [[ "$TOP1_CONFIG" != /* ]]; then
  TOP1_CONFIG="$ROOT_DIR/$TOP1_CONFIG"
fi
if [[ ! -f "$TOP1_CONFIG" ]]; then
  echo "Top1 config does not exist: $TOP1_CONFIG" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$TOP1_CONFIG"
cd "$ROOT_DIR"

if [[ "$PYTHON" == */* && ! -x "$PYTHON" ]]; then
  echo "Python executable does not exist: $PYTHON" >&2
  echo "Create it with: uv venv --python 3.12" >&2
  exit 2
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required input does not exist: $1" >&2
    exit 2
  fi
}

require_file "$TOP1_TRAIN_DATA"
require_file "$TOP1_CANDIDATE_REGISTRY"
require_file "$TOP1_SYSTEM_PROMPT"
if [[ -n "$TOP1_VALIDATION_DATA" ]]; then
  require_file "$TOP1_VALIDATION_DATA"
fi
if [[ ! "$TOP1_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOP1_NUM_GPUS must be a positive integer" >&2
  exit 2
fi

LAUNCH=("$PYTHON")
if (( TOP1_NUM_GPUS > 1 )); then
  LAUNCH=(
    "$PYTHON" -m torch.distributed.run --standalone
    --nproc-per-node "$TOP1_NUM_GPUS"
  )
fi

OPTIONAL_ARGS=()
if [[ -n "$TOP1_VALIDATION_DATA" ]]; then
  OPTIONAL_ARGS+=(--validation-data "$TOP1_VALIDATION_DATA")
fi
if [[ -n "$TOP1_DEEPSPEED_CONFIG" && "$TOP1_DEEPSPEED_CONFIG" != "none" ]]; then
  require_file "$TOP1_DEEPSPEED_CONFIG"
  OPTIONAL_ARGS+=(--deepspeed "$TOP1_DEEPSPEED_CONFIG")
fi
if [[ "$TOP1_GRADIENT_CHECKPOINTING" == "1" ]]; then
  OPTIONAL_ARGS+=(
    --gradient-checkpointing
    --gradient-checkpointing-mode "$TOP1_GRADIENT_CHECKPOINTING_MODE"
  )
elif [[ "$TOP1_GRADIENT_CHECKPOINTING" != "0" ]]; then
  echo "TOP1_GRADIENT_CHECKPOINTING must be 0 or 1" >&2
  exit 2
fi
if [[ "$TOP1_TRUST_REMOTE_CODE" == "1" ]]; then
  OPTIONAL_ARGS+=(--trust-remote-code)
elif [[ "$TOP1_TRUST_REMOTE_CODE" != "0" ]]; then
  echo "TOP1_TRUST_REMOTE_CODE must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$TOP1_RESUME" ]]; then
  OPTIONAL_ARGS+=(--resume-from-checkpoint "$TOP1_RESUME")
fi

LEARNING_RATE="$TOP1_FULL_LEARNING_RATE"
if [[ "$TOP1_FINETUNE_MODE" == "lora" ]]; then
  LEARNING_RATE="$TOP1_LORA_LEARNING_RATE"
elif [[ "$TOP1_FINETUNE_MODE" != "full" ]]; then
  echo "TOP1_FINETUNE_MODE must be full or lora" >&2
  exit 2
fi

echo "[top1] train direct candidate-name router"
exec "${LAUNCH[@]}" scripts/train_top1.py \
  --model-name-or-path "$TOP1_MODEL" \
  --train-data "$TOP1_TRAIN_DATA" \
  --candidate-registry "$TOP1_CANDIDATE_REGISTRY" \
  --system-prompt-file "$TOP1_SYSTEM_PROMPT" \
  --output-dir "$TOP1_OUTPUT_DIR" \
  --finetune-mode "$TOP1_FINETUNE_MODE" \
  --precision "$TOP1_PRECISION" \
  --epochs "$TOP1_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --max-length "$TOP1_MAX_LENGTH" \
  --per-device-train-batch-size "$TOP1_PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$TOP1_PER_DEVICE_EVAL_BATCH_SIZE" \
  --eval-accumulation-steps "$TOP1_EVAL_ACCUMULATION_STEPS" \
  --gradient-accumulation-steps "$TOP1_GRADIENT_ACCUMULATION_STEPS" \
  --weight-decay "$TOP1_WEIGHT_DECAY" \
  --warmup-ratio "$TOP1_WARMUP_RATIO" \
  --logging-steps "$TOP1_LOGGING_STEPS" \
  --save-steps "$TOP1_SAVE_STEPS" \
  --eval-steps "$TOP1_EVAL_STEPS" \
  --save-total-limit "$TOP1_SAVE_TOTAL_LIMIT" \
  --dataloader-num-workers "$TOP1_DATALOADER_NUM_WORKERS" \
  --seed "$TOP1_SEED" \
  "${OPTIONAL_ARGS[@]}" \
  "$@"
