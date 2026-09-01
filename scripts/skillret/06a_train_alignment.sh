#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"
skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/train_registry.json"
skillret_require_file "$ROUTER_DATA_DIR/retrieval_alignment_train.jsonl"
MEMORIZATION_MODEL_DIR="${ROUTER_MEMORIZATION_MODEL_DIR:-$ROUTER_OUTPUT_DIR/memorization}"
skillret_require_dir "$MEMORIZATION_MODEL_DIR"
skillret_configure_router

ALIGNMENT_MODEL_ARGS=()
case "$ROUTER_FINETUNE_MODE" in
  lora)
    ALIGNMENT_MODEL_ARGS=(
      --model-name-or-path "$ROUTER_MODEL"
      --adapter-name-or-path "$MEMORIZATION_MODEL_DIR"
    )
    ;;
  full)
    ALIGNMENT_MODEL_ARGS=(--model-name-or-path "$MEMORIZATION_MODEL_DIR")
    ;;
esac

ALIGNMENT_RESUME_ARGS=()
if [[ -n "${ROUTER_RESUME_ALIGNMENT:-}" ]]; then
  ALIGNMENT_RESUME_ARGS=(--resume-retrieval-from-checkpoint "$ROUTER_RESUME_ALIGNMENT")
fi

skillret_print_step "06a" "single-skill retrieval alignment curriculum"
"${ROUTER_LAUNCH[@]}" scripts/train_router.py \
  "${ALIGNMENT_MODEL_ARGS[@]}" \
  "${ROUTER_COMMON_ARGS[@]}" \
  --stage retrieval \
  --phase-output-subdir retrieval_alignment \
  --retrieval-train "$ROUTER_DATA_DIR/retrieval_alignment_train.jsonl" \
  --retrieval-epochs "$ROUTER_ALIGNMENT_EPOCHS" \
  --retrieval-learning-rate "$ROUTER_ALIGNMENT_LR" \
  ${ALIGNMENT_RESUME_ARGS[@]+"${ALIGNMENT_RESUME_ARGS[@]}"} \
  ${ROUTER_COMPAT_EXTRA_ARGS[@]+"${ROUTER_COMPAT_EXTRA_ARGS[@]}"} \
  "$@"
