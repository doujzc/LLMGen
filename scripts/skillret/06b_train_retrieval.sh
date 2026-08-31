#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"
skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/train_registry.json"
skillret_require_file "$ROUTER_DATA_DIR/retrieval_train.jsonl"
skillret_require_file "$ROUTER_DATA_DIR/retrieval_validation.jsonl"

RETRIEVAL_INIT_DIR="$ROUTER_OUTPUT_DIR/memorization"
if [[ "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0" && "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0.0" ]]; then
  RETRIEVAL_INIT_DIR="$ROUTER_OUTPUT_DIR/retrieval_alignment"
fi
skillret_require_dir "$RETRIEVAL_INIT_DIR"
skillret_configure_router

MODEL_ARGS=()
case "$ROUTER_FINETUNE_MODE" in
  lora)
    MODEL_ARGS=(
      --model-name-or-path "$ROUTER_MODEL"
      --adapter-name-or-path "$RETRIEVAL_INIT_DIR"
    )
    ;;
  full)
    MODEL_ARGS=(--model-name-or-path "$RETRIEVAL_INIT_DIR")
    ;;
esac

RESUME_ARGS=()
if [[ -n "$ROUTER_RESUME_RETRIEVAL" ]]; then
  RESUME_ARGS=(--resume-retrieval-from-checkpoint "$ROUTER_RESUME_RETRIEVAL")
fi

REPLAY_ARGS=()
if [[ "$ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION" != "0" && "$ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION" != "0.0" ]]; then
  skillret_require_file "$ROUTER_DATA_DIR/retrieval_alignment_train.jsonl"
  REPLAY_ARGS+=(
    --retrieval-alignment-replay-data "$ROUTER_DATA_DIR/retrieval_alignment_train.jsonl"
    --retrieval-alignment-replay-fraction "$ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION"
  )
fi
if [[ "$ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION" != "0" && "$ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION" != "0.0" ]]; then
  skillret_require_file "$ROUTER_DATA_DIR/memorization_train.jsonl"
  REPLAY_ARGS+=(
    --retrieval-memorization-replay-data "$ROUTER_DATA_DIR/memorization_train.jsonl"
    --retrieval-memorization-replay-fraction "$ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION"
  )
fi

skillret_print_step "06b" \
  "multi-Skill retrieval with alignment/memorization replay"
"${ROUTER_LAUNCH[@]}" scripts/train_router.py \
  "${MODEL_ARGS[@]}" \
  "${ROUTER_COMMON_ARGS[@]}" \
  --stage retrieval \
  --retrieval-train "$ROUTER_DATA_DIR/retrieval_train.jsonl" \
  --retrieval-validation "$ROUTER_DATA_DIR/retrieval_validation.jsonl" \
  --retrieval-epochs "$ROUTER_RETRIEVAL_EPOCHS" \
  --retrieval-learning-rate "$ROUTER_RETRIEVAL_LR" \
  ${REPLAY_ARGS[@]+"${REPLAY_ARGS[@]}"} \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  ${ROUTER_COMPAT_EXTRA_ARGS[@]+"${ROUTER_COMPAT_EXTRA_ARGS[@]}"} \
  "$@"
