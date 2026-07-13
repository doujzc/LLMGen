#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-runs/skillret}"
DEVICE="${DEVICE:-cuda}"
NUM_LEVELS="${NUM_LEVELS:-3}"
BRANCHING_FACTORS="${BRANCHING_FACTORS:-64 64 64}"
SK_EPSILONS="${SK_EPSILONS:-0 0 0.01}"
RQ_LAYERS="${RQ_LAYERS:-512 256 128}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-ThakiCloud/SkillRet-Embedding-0.6B}"
EMBEDDING_REVISION="${EMBEDDING_REVISION:-}"
if [[ -z "$EMBEDDING_REVISION" && "$EMBEDDING_MODEL" == "ThakiCloud/SkillRet-Embedding-0.6B" ]]; then
  EMBEDDING_REVISION="0e10886e80a0aacc9efddc28282a258e2ab7eae1"
fi
EMBEDDING_REVISION_ARGS=()
if [[ -n "$EMBEDDING_REVISION" ]]; then
  EMBEDDING_REVISION_ARGS=(--embedding-revision "$EMBEDDING_REVISION")
fi
ROUTER_MODEL="${ROUTER_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ROUTER_EXTRA_ARGS="${ROUTER_EXTRA_ARGS:---bf16 --gradient-checkpointing --lora}"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  "$PYTHON" scripts/download_skillret.py
fi
"$PYTHON" scripts/prepare_skillret.py \
  --embedding-model "$EMBEDDING_MODEL" "${EMBEDDING_REVISION_ARGS[@]}" \
  --batch-size 1 --device "$DEVICE"

# Word splitting is intentional for the configurable per-level lists.
# shellcheck disable=SC2086
"$PYTHON" scripts/train_tokenizer.py \
  --data-root data/skillret --output-dir "$RUN_DIR/stage1" \
  --device "$DEVICE" \
  --num-levels "$NUM_LEVELS" --branching-factors $BRANCHING_FACTORS \
  --sk-epsilons $SK_EPSILONS --layers $RQ_LAYERS \
  --e-dim 64 --epochs 100 --batch-size 512 --lr 1e-4 \
  --graph-lambda 0.001 --amp-dtype bf16

"$PYTHON" scripts/export_skill_codes.py \
  --checkpoint "$RUN_DIR/stage1/best.pt" \
  --output-dir "$RUN_DIR/index" --device "$DEVICE"

"$PYTHON" scripts/build_router_data.py \
  --catalog data/skillret/processed/catalog_train.jsonl \
  --queries data/skillret/processed/queries_train.jsonl \
  --qrels data/skillret/processed/qrels_train.jsonl \
  --codes "$RUN_DIR/index/train_codes.jsonl" \
  --virtual-tokens "$RUN_DIR/index/virtual_tokens.txt" \
  --output-dir "$RUN_DIR/router_data"

# Set ROUTER_EXTRA_ARGS="--bf16 --gradient-checkpointing" for full-parameter SFT.
# shellcheck disable=SC2086
"$PYTHON" scripts/train_router.py \
  --model-name-or-path "$ROUTER_MODEL" \
  --virtual-tokens "$RUN_DIR/index/virtual_tokens.txt" \
  --output-dir "$RUN_DIR/router" --stage both --num-levels "$NUM_LEVELS" \
  --memorization-train "$RUN_DIR/router_data/memorization_train.jsonl" \
  --memorization-validation "$RUN_DIR/router_data/memorization_validation.jsonl" \
  --retrieval-train "$RUN_DIR/router_data/retrieval_train.jsonl" \
  --retrieval-validation "$RUN_DIR/router_data/retrieval_validation.jsonl" \
  --max-length 1024 --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --memorization-epochs 1 --retrieval-epochs 3 $ROUTER_EXTRA_ARGS

"$PYTHON" scripts/infer_router.py \
  --model-name-or-path "$RUN_DIR/router/retrieval" \
  --base-model-name-or-path "$ROUTER_MODEL" \
  --virtual-tokens "$RUN_DIR/index/virtual_tokens.txt" \
  --codes "$RUN_DIR/index/test_codes.jsonl" \
  --registry "$RUN_DIR/index/test_registry.json" \
  --queries data/skillret/processed/queries_test.jsonl \
  --qrels data/skillret/processed/qrels_test.jsonl \
  --output-jsonl "$RUN_DIR/evaluation/predictions.jsonl" \
  --metrics-output "$RUN_DIR/evaluation/metrics.json" \
  --batch-size 1 \
  --beam-size 20 --num-code-paths 20 --top-k 20 \
  --device "$DEVICE" --dtype bfloat16
