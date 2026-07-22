#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"
skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/train_registry.json"
skillret_require_file "$ROUTER_DATA_DIR/memorization_train.jsonl"
skillret_require_file "$ROUTER_DATA_DIR/memorization_validation.jsonl"
skillret_configure_router

FINETUNE_ARGS=()
case "$ROUTER_FINETUNE_MODE" in
  lora)
    FINETUNE_ARGS=(
      --lora
      --lora-r "$ROUTER_LORA_R"
      --lora-alpha "$ROUTER_LORA_ALPHA"
      --lora-dropout "$ROUTER_LORA_DROPOUT"
      --lora-target-modules "$ROUTER_LORA_TARGET_MODULES"
      --lora-modules-to-save "$ROUTER_LORA_MODULES_TO_SAVE"
    )
    ;;
  full) ;;
esac

RESUME_ARGS=()
if [[ -n "$ROUTER_RESUME_MEMORIZATION" ]]; then
  RESUME_ARGS=(--resume-memorization-from-checkpoint "$ROUTER_RESUME_MEMORIZATION")
fi

"${ROUTER_LAUNCH[@]}" scripts/train_router.py \
  --model-name-or-path "$ROUTER_MODEL" \
  "${ROUTER_COMMON_ARGS[@]}" \
  --stage memorization \
  --memorization-train "$ROUTER_DATA_DIR/memorization_train.jsonl" \
  --memorization-validation "$ROUTER_DATA_DIR/memorization_validation.jsonl" \
  --memorization-epochs "$ROUTER_MEMORIZATION_EPOCHS" \
  --memorization-learning-rate "$ROUTER_MEMORIZATION_LR" \
  "${FINETUNE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${ROUTER_COMPAT_EXTRA_ARGS[@]}" \
  "$@"
