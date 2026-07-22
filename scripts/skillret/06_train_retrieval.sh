#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"
skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/train_registry.json"
skillret_require_file "$ROUTER_DATA_DIR/retrieval_train.jsonl"
skillret_require_file "$ROUTER_DATA_DIR/retrieval_validation.jsonl"
skillret_require_dir "$ROUTER_OUTPUT_DIR/memorization"
skillret_configure_router

MODEL_ARGS=()
case "$ROUTER_FINETUNE_MODE" in
  lora)
    MODEL_ARGS=(
      --model-name-or-path "$ROUTER_MODEL"
      --adapter-name-or-path "$ROUTER_OUTPUT_DIR/memorization"
    )
    ;;
  full)
    MODEL_ARGS=(--model-name-or-path "$ROUTER_OUTPUT_DIR/memorization")
    ;;
esac

RESUME_ARGS=()
if [[ -n "$ROUTER_RESUME_RETRIEVAL" ]]; then
  RESUME_ARGS=(--resume-retrieval-from-checkpoint "$ROUTER_RESUME_RETRIEVAL")
fi

REPLAY_ARGS=()
if [[ "$ROUTER_RETRIEVAL_REPLAY_FRACTION" != "0" && "$ROUTER_RETRIEVAL_REPLAY_FRACTION" != "0.0" ]]; then
  skillret_require_file "$ROUTER_DATA_DIR/memorization_train.jsonl"
  REPLAY_ARGS=(
    --retrieval-replay-data "$ROUTER_DATA_DIR/memorization_train.jsonl"
    --retrieval-replay-fraction "$ROUTER_RETRIEVAL_REPLAY_FRACTION"
  )
fi

"${ROUTER_LAUNCH[@]}" scripts/train_router.py \
  "${MODEL_ARGS[@]}" \
  "${ROUTER_COMMON_ARGS[@]}" \
  --stage retrieval \
  --retrieval-train "$ROUTER_DATA_DIR/retrieval_train.jsonl" \
  --retrieval-validation "$ROUTER_DATA_DIR/retrieval_validation.jsonl" \
  --retrieval-epochs "$ROUTER_RETRIEVAL_EPOCHS" \
  --retrieval-learning-rate "$ROUTER_RETRIEVAL_LR" \
  "${REPLAY_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${ROUTER_COMPAT_EXTRA_ARGS[@]}" \
  "$@"
