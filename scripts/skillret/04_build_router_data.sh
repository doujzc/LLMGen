#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$PROCESSED_DIR/queries_train.jsonl"
skillret_require_file "$PROCESSED_DIR/qrels_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"

ALIGNMENT_ARGS=()
if [[ "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0" && "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0.0" ]]; then
  skillret_require_file "$PROCESSED_DIR/queries_alignment.jsonl"
  skillret_require_file "$PROCESSED_DIR/qrels_alignment.jsonl"
  ALIGNMENT_ARGS=(
    --alignment-queries "$PROCESSED_DIR/queries_alignment.jsonl"
    --alignment-qrels "$PROCESSED_DIR/qrels_alignment.jsonl"
  )
fi

"$PYTHON" scripts/build_router_data.py \
  --catalog "$PROCESSED_DIR/catalog_train.jsonl" \
  --queries "$PROCESSED_DIR/queries_train.jsonl" \
  --qrels "$PROCESSED_DIR/qrels_train.jsonl" \
  "${ALIGNMENT_ARGS[@]}" \
  --codes "$INDEX_DIR/train_codes.jsonl" \
  --virtual-tokens "$INDEX_DIR/virtual_tokens.txt" \
  --output-dir "$ROUTER_DATA_DIR" \
  --memorization-validation-fraction "$MEMORIZATION_VALIDATION_FRACTION" \
  --retrieval-validation-fraction "$ROUTER_VALIDATION_FRACTION" \
  --seed "$ROUTER_DATA_SEED" \
  "$@"
