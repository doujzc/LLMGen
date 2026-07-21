#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SKILLRET_CONFIG="${SKILLRET_CONFIG:-configs/clawhub.env}"
# shellcheck source=scripts/skillret/common.sh
source "$ROOT/scripts/skillret/common.sh"

DIAG_SPLIT="${DIAG_SPLIT:-test}"
case "$DIAG_SPLIT" in
  train|validation|test) ;;
  *)
    echo "DIAG_SPLIT must be train, validation, or test" >&2
    exit 2
    ;;
esac

skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/train_registry.json"
skillret_require_file "$PROCESSED_DIR/qrels_train.jsonl"
skillret_require_file "$PROCESSED_DIR/qrels_${DIAG_SPLIT}.jsonl"
skillret_require_file "$PROCESSED_DIR/queries_${DIAG_SPLIT}.jsonl"
skillret_require_file "$ROUTER_DATA_DIR/retrieval_train.jsonl"

DIAG_OUTPUT="${DIAG_OUTPUT:-$RUN_DIR/diagnostics/${DIAG_SPLIT}.json}"
DIAG_PREDICTIONS="${DIAG_PREDICTIONS:-$RUN_DIR/evaluation/predictions.jsonl}"
DIAG_SAMPLE_SIZE="${DIAG_SAMPLE_SIZE:-128}"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-1}"
DIAG_WITH_MODEL="${DIAG_WITH_MODEL:-0}"

ARGS=(
  --codes "$INDEX_DIR/train_codes.jsonl"
  --registry "$INDEX_DIR/train_registry.json"
  --train-qrels "$PROCESSED_DIR/qrels_train.jsonl"
  --eval-qrels "$PROCESSED_DIR/qrels_${DIAG_SPLIT}.jsonl"
  --eval-queries "$PROCESSED_DIR/queries_${DIAG_SPLIT}.jsonl"
  --router-train "$ROUTER_DATA_DIR/retrieval_train.jsonl"
  --output "$DIAG_OUTPUT"
  --cutoffs 1 5 10
)

if [[ -f "$DIAG_PREDICTIONS" ]]; then
  ARGS+=(--predictions "$DIAG_PREDICTIONS")
else
  echo "Prediction file not found; continuing with data/code diagnostics: $DIAG_PREDICTIONS" >&2
fi
if [[ -f "$ROUTER_OUTPUT_DIR/retrieval/router_manifest.json" ]]; then
  ARGS+=(--router-manifest "$ROUTER_OUTPUT_DIR/retrieval/router_manifest.json")
fi
if [[ -f "$STAGE1_DIR/history.jsonl" ]]; then
  ARGS+=(--stage1-history "$STAGE1_DIR/history.jsonl")
fi
if [[ -f "$DATASET_DIR/queries_train.jsonl" ]]; then
  ARGS+=(--source-train-queries "$DATASET_DIR/queries_train.jsonl")
fi

case "$DIAG_WITH_MODEL" in
  1|true|yes)
    skillret_require_dir "$ROUTER_OUTPUT_DIR/retrieval"
    skillret_require_file "$INDEX_DIR/virtual_tokens.txt"
    ARGS+=(
      --model-name-or-path "$ROUTER_OUTPUT_DIR/retrieval"
      --base-model-name-or-path "$ROUTER_MODEL"
      --virtual-tokens "$INDEX_DIR/virtual_tokens.txt"
      --device "$DEVICE"
      --dtype "$EVAL_DTYPE"
      --batch-size "$DIAG_BATCH_SIZE"
      --sample-size "$DIAG_SAMPLE_SIZE"
      --max-length "$ROUTER_MAX_LENGTH"
    )
    case "$ROUTER_TRUST_REMOTE_CODE" in
      1|true|yes) ARGS+=(--trust-remote-code) ;;
    esac
    ;;
  0|false|no) ;;
  *)
    echo "DIAG_WITH_MODEL must be 0 or 1" >&2
    exit 2
    ;;
esac

"$PYTHON" scripts/diagnose_router.py "${ARGS[@]}" "$@"
