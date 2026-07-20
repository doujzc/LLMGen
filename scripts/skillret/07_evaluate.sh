#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

case "$EVAL_PROTOCOL" in
  closedset|unseen|both) ;;
  *)
    echo "EVAL_PROTOCOL must be 'closedset', 'unseen', or 'both'" >&2
    exit 2
    ;;
esac

skillret_require_dir "$ROUTER_OUTPUT_DIR/retrieval"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"

read -r -a CUTOFF_ARGS <<< "$EVAL_CUTOFFS"

run_inference() {
  local codes="$1"
  local registry="$2"
  local queries="$3"
  local qrels="$4"
  local output_dir="$5"
  shift 5
  mkdir -p "$output_dir"
  "$PYTHON" scripts/infer_router.py \
    --model-name-or-path "$ROUTER_OUTPUT_DIR/retrieval" \
    --base-model-name-or-path "$ROUTER_MODEL" \
    --virtual-tokens "$INDEX_DIR/virtual_tokens.txt" \
    --codes "$codes" \
    --registry "$registry" \
    --queries "$queries" \
    --qrels "$qrels" \
    --output-jsonl "$output_dir/predictions.jsonl" \
    --metrics-output "$output_dir/metrics.json" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --beam-size "$EVAL_BEAM_SIZE" \
    --num-code-paths "$EVAL_NUM_CODE_PATHS" \
    --top-k "$EVAL_TOP_K" \
    --cutoffs "${CUTOFF_ARGS[@]}" \
    --device "$DEVICE" \
    --dtype "$EVAL_DTYPE" \
    "$@"
}

run_closedset() {
  local output_dir="$1"
  local queries
  local qrels
  skillret_require_file "$INDEX_DIR/train_codes.jsonl"
  skillret_require_file "$INDEX_DIR/train_registry.json"
  case "$QUERY_SET" in
    validation)
      local data_dir="$ROUTER_DATA_DIR/closedset_validation"
      "$PYTHON" scripts/export_closedset_validation.py \
        --retrieval-validation "$ROUTER_DATA_DIR/retrieval_validation.jsonl" \
        --codes "$INDEX_DIR/train_codes.jsonl" \
        --output-dir "$data_dir"
      queries="$data_dir/queries.jsonl"
      qrels="$data_dir/qrels.jsonl"
      ;;
    dataset-validation)
      skillret_require_file "$PROCESSED_DIR/queries_validation.jsonl"
      skillret_require_file "$PROCESSED_DIR/qrels_validation.jsonl"
      queries="$PROCESSED_DIR/queries_validation.jsonl"
      qrels="$PROCESSED_DIR/qrels_validation.jsonl"
      ;;
    test)
      skillret_require_file "$PROCESSED_DIR/queries_test.jsonl"
      skillret_require_file "$PROCESSED_DIR/qrels_test.jsonl"
      queries="$PROCESSED_DIR/queries_test.jsonl"
      qrels="$PROCESSED_DIR/qrels_test.jsonl"
      ;;
    train)
      queries="$PROCESSED_DIR/queries_train.jsonl"
      qrels="$PROCESSED_DIR/qrels_train.jsonl"
      ;;
    *)
      echo "QUERY_SET must be 'validation', 'dataset-validation', 'test', or 'train'" >&2
      exit 2
      ;;
  esac
  run_inference \
    "$INDEX_DIR/train_codes.jsonl" \
    "$INDEX_DIR/train_registry.json" \
    "$queries" "$qrels" "$output_dir" \
    "${EVAL_FORWARD_ARGS[@]}"
}

run_unseen() {
  local output_dir="$1"
  skillret_require_file "$INDEX_DIR/test_codes.jsonl"
  skillret_require_file "$INDEX_DIR/test_registry.json"
  skillret_require_file "$PROCESSED_DIR/queries_test.jsonl"
  skillret_require_file "$PROCESSED_DIR/qrels_test.jsonl"
  run_inference \
    "$INDEX_DIR/test_codes.jsonl" \
    "$INDEX_DIR/test_registry.json" \
    "$PROCESSED_DIR/queries_test.jsonl" \
    "$PROCESSED_DIR/qrels_test.jsonl" \
    "$output_dir" \
    "${EVAL_FORWARD_ARGS[@]}"
}

EVAL_FORWARD_ARGS=("$@")
case "$EVAL_PROTOCOL" in
  closedset)
    run_closedset "${EVAL_DIR:-$RUN_DIR/evaluation}"
    ;;
  unseen)
    run_unseen "${EVAL_DIR:-$RUN_DIR/evaluation}"
    ;;
  both)
    if [[ -n "${EVAL_DIR:-}" ]]; then
      run_closedset "$EVAL_DIR/closedset-validation"
      run_unseen "$EVAL_DIR/unseen"
    else
      run_closedset "$RUN_DIR/evaluation/closedset-validation"
      run_unseen "$RUN_DIR/evaluation/unseen"
    fi
    ;;
esac
