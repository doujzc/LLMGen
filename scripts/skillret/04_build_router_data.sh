#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$PROCESSED_DIR/queries_train.jsonl"
skillret_require_file "$PROCESSED_DIR/qrels_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"

ALIGNMENT_ARGS=()
ALIGNMENT_ONLY_ARGS=()
case "${ROUTER_ALIGNMENT_ONLY:-0}" in
  1|true|yes)
    ALIGNMENT_ONLY_ARGS=(--skip-multiskill-retrieval)
    ;;
  0|false|no|"")
    ;;
  *)
    echo "ROUTER_ALIGNMENT_ONLY must be 0 or 1" >&2
    exit 2
    ;;
esac

if (( ${#ALIGNMENT_ONLY_ARGS[@]} > 0 )) || {
  [[ "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0" && "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0.0" ]]
} || {
  [[ "${ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION:-0}" != "0" &&
    "${ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION:-0}" != "0.0" ]]
}; then
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
  ${ALIGNMENT_ARGS[@]+"${ALIGNMENT_ARGS[@]}"} \
  ${ALIGNMENT_ONLY_ARGS[@]+"${ALIGNMENT_ONLY_ARGS[@]}"} \
  --codes "$INDEX_DIR/train_codes.jsonl" \
  --virtual-tokens "$INDEX_DIR/virtual_tokens.txt" \
  --output-dir "$ROUTER_DATA_DIR" \
  --memorization-validation-fraction "$MEMORIZATION_VALIDATION_FRACTION" \
  --retrieval-validation-fraction "$ROUTER_VALIDATION_FRACTION" \
  --seed "$ROUTER_DATA_SEED" \
  "$@"
