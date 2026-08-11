#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

top1_require_file "$TOP1_TRAIN_DATA"
top1_require_file "$TOP1_CANDIDATE_REGISTRY"
top1_require_file "$TOP1_SYSTEM_PROMPT"
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_validate.sh"
top1_train_launch

VALIDATION_ARGS=()
if [[ -n "$TOP1_VALIDATION_DATA" ]]; then
  top1_require_file "$TOP1_VALIDATION_DATA"
  VALIDATION_ARGS=(--retrieval-validation "$TOP1_VALIDATION_DATA")
fi

PRECISION_ARGS=()
case "$ROUTER_PRECISION" in
  bf16) PRECISION_ARGS=(--bf16) ;;
  fp16) PRECISION_ARGS=(--fp16) ;;
  fp32|none) ;;
  *) echo "ROUTER_PRECISION must be bf16, fp16, or fp32" >&2; exit 2 ;;
esac

DEEPSPEED_ARGS=()
if [[ -n "$ROUTER_DEEPSPEED_CONFIG" && "$ROUTER_DEEPSPEED_CONFIG" != "none" ]]; then
  top1_require_file "$ROUTER_DEEPSPEED_CONFIG"
  DEEPSPEED_ARGS=(--deepspeed "$ROUTER_DEEPSPEED_CONFIG")
fi

CHECKPOINT_ARGS=()
case "$ROUTER_GRADIENT_CHECKPOINTING" in
  1|true|yes)
    CHECKPOINT_ARGS=(
      --gradient-checkpointing
      --gradient-checkpointing-mode "$ROUTER_GRADIENT_CHECKPOINTING_MODE"
    )
    ;;
  0|false|no) ;;
  *) echo "ROUTER_GRADIENT_CHECKPOINTING must be 0 or 1" >&2; exit 2 ;;
esac

TRUST_ARGS=()
case "$ROUTER_TRUST_REMOTE_CODE" in
  1|true|yes) TRUST_ARGS=(--trust-remote-code) ;;
  0|false|no) ;;
  *) echo "ROUTER_TRUST_REMOTE_CODE must be 0 or 1" >&2; exit 2 ;;
esac

FINETUNE_ARGS=()
LEARNING_RATE="$ROUTER_FULL_LEARNING_RATE"
if [[ "$ROUTER_FINETUNE_MODE" == "lora" ]]; then
  FINETUNE_ARGS=(--lora)
  LEARNING_RATE="$ROUTER_LORA_LEARNING_RATE"
fi

RESUME_ARGS=()
if [[ -n "$ROUTER_RESUME" ]]; then
  RESUME_ARGS=(--resume-retrieval-from-checkpoint "$ROUTER_RESUME")
fi

EXTRA_ARGS=()
if [[ -n "$ROUTER_EXTRA_ARGS" ]]; then
  # Backward-compatible escape hatch for experimental Trainer flags.
  # shellcheck disable=SC2206
  EXTRA_ARGS=($ROUTER_EXTRA_ARGS)
fi

top1_step "01" "train multi-turn candidate-name Top1 router"
"${TOP1_LAUNCH[@]}" scripts/train_router.py \
  --routing-mode candidate_name_top1 \
  --model-name-or-path "$ROUTER_MODEL" \
  --candidate-registry "$TOP1_CANDIDATE_REGISTRY" \
  --retrieval-system-prompt-file "$TOP1_SYSTEM_PROMPT" \
  --output-dir "$TOP1_RUN_DIR/router" \
  --stage retrieval \
  --retrieval-train "$TOP1_TRAIN_DATA" \
  "${VALIDATION_ARGS[@]}" \
  --retrieval-epochs "$ROUTER_EPOCHS" \
  --retrieval-learning-rate "$LEARNING_RATE" \
  --max-length "$ROUTER_MAX_LENGTH" \
  --per-device-train-batch-size "$ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$ROUTER_PER_DEVICE_EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$ROUTER_GRADIENT_ACCUMULATION_STEPS" \
  --weight-decay "$ROUTER_WEIGHT_DECAY" \
  --warmup-ratio "$ROUTER_WARMUP_RATIO" \
  --logging-steps "$ROUTER_LOGGING_STEPS" \
  --save-steps "$ROUTER_SAVE_STEPS" \
  --eval-steps "$ROUTER_EVAL_STEPS" \
  --save-total-limit "$ROUTER_SAVE_TOTAL_LIMIT" \
  --dataloader-num-workers "$ROUTER_DATALOADER_NUM_WORKERS" \
  "${FINETUNE_ARGS[@]}" \
  "${DEEPSPEED_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${TRUST_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
