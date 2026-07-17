#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-runs/skillret}"
ROUTER_MODEL="${ROUTER_MODEL:-Qwen/Qwen3-1.7B}"
QUERY_SET="${QUERY_SET:-validation}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BEAM_SIZE="${BEAM_SIZE:-20}"
NUM_CODE_PATHS="${NUM_CODE_PATHS:-20}"
TOP_K="${TOP_K:-20}"
CUTOFFS="${CUTOFFS:-1 5 10}"

CODES="$RUN_DIR/index/train_codes.jsonl"
REGISTRY="$RUN_DIR/index/train_registry.json"
EVAL_DIR="${EVAL_DIR:-$RUN_DIR/evaluation/closedset-$QUERY_SET}"
mkdir -p "$EVAL_DIR"

case "$QUERY_SET" in
  validation)
    DATA_DIR="$RUN_DIR/router_data/closedset_validation"
    "$PYTHON" scripts/export_closedset_validation.py \
      --retrieval-validation "$RUN_DIR/router_data/retrieval_validation.jsonl" \
      --codes "$CODES" \
      --output-dir "$DATA_DIR"
    QUERIES="$DATA_DIR/queries.jsonl"
    QRELS="$DATA_DIR/qrels.jsonl"
    ;;
  train)
    QUERIES="data/skillret/processed/queries_train.jsonl"
    QRELS="data/skillret/processed/qrels_train.jsonl"
    ;;
  *)
    echo "QUERY_SET must be 'validation' or 'train'" >&2
    exit 2
    ;;
esac

# Supplying the base model is required for LoRA and safely ignored for a local
# full-parameter checkpoint that contains config.json.
# shellcheck disable=SC2086
"$PYTHON" scripts/infer_router.py \
  --model-name-or-path "$RUN_DIR/router/retrieval" \
  --base-model-name-or-path "$ROUTER_MODEL" \
  --virtual-tokens "$RUN_DIR/index/virtual_tokens.txt" \
  --codes "$CODES" \
  --registry "$REGISTRY" \
  --queries "$QUERIES" \
  --qrels "$QRELS" \
  --output-jsonl "$EVAL_DIR/predictions.jsonl" \
  --metrics-output "$EVAL_DIR/metrics.json" \
  --batch-size "$BATCH_SIZE" \
  --beam-size "$BEAM_SIZE" \
  --num-code-paths "$NUM_CODE_PATHS" \
  --top-k "$TOP_K" \
  --cutoffs $CUTOFFS \
  --device "$DEVICE" \
  --dtype "$DTYPE"
