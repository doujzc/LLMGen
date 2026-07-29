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

RETRIEVAL_INIT_DIR="$ROUTER_OUTPUT_DIR/memorization"
if [[ "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0" && "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0.0" ]]; then
  skillret_require_file "$ROUTER_DATA_DIR/retrieval_alignment_train.jsonl"
  ALIGNMENT_MODEL_ARGS=()
  case "$ROUTER_FINETUNE_MODE" in
    lora)
      ALIGNMENT_MODEL_ARGS=(
        --model-name-or-path "$ROUTER_MODEL"
        --adapter-name-or-path "$ROUTER_OUTPUT_DIR/memorization"
      )
      ;;
    full)
      ALIGNMENT_MODEL_ARGS=(--model-name-or-path "$ROUTER_OUTPUT_DIR/memorization")
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
    "${ALIGNMENT_RESUME_ARGS[@]}" \
    "${ROUTER_COMPAT_EXTRA_ARGS[@]}"
  RETRIEVAL_INIT_DIR="$ROUTER_OUTPUT_DIR/retrieval_alignment"
fi

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
  "${REPLAY_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${ROUTER_COMPAT_EXTRA_ARGS[@]}" \
  "$@"
