#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SKILLRET_CONFIG="${SKILLRET_CONFIG:-configs/clawhub.env}"
source "$ROOT/scripts/skillret/common.sh"

MODEL_DIR="${1:-$ROUTER_OUTPUT_DIR/retrieval}"
if (( $# > 0 )); then
  shift
fi
skillret_require_dir "$MODEL_DIR"
skillret_require_file "$MODEL_DIR/router_manifest.json"
skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/train_registry.json"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"

MODEL_PHASE=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' \
    "$MODEL_DIR/router_manifest.json"
)
TRAINING_DATA="$ROUTER_DATA_DIR/${MODEL_PHASE}_train.jsonl"
skillret_require_file "$TRAINING_DATA"

"$PYTHON" scripts/export_router_bundle.py \
  --model-dir "$MODEL_DIR" \
  --catalog "$PROCESSED_DIR/catalog_train.jsonl" \
  --codes "$INDEX_DIR/train_codes.jsonl" \
  --registry "$INDEX_DIR/train_registry.json" \
  --virtual-tokens "$INDEX_DIR/virtual_tokens.txt" \
  --training-data "$TRAINING_DATA" \
  --phase "$MODEL_PHASE" \
  "$@"
