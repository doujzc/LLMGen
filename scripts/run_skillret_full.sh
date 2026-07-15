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
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
EMBEDDING_DIMENSIONS="${EMBEDDING_DIMENSIONS:-}"
EMBEDDING_DIMENSION_ARGS=()
if [[ -n "$EMBEDDING_DIMENSIONS" ]]; then
  EMBEDDING_DIMENSION_ARGS=(--embedding-dimensions "$EMBEDDING_DIMENSIONS")
fi
ROUTER_MODEL="${ROUTER_MODEL:-Qwen/Qwen3-1.7B}"
ROUTER_FINETUNE_MODE="${ROUTER_FINETUNE_MODE:-lora}"
ROUTER_NUM_GPUS="${ROUTER_NUM_GPUS:-4}"
ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE="${ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
ROUTER_GRADIENT_ACCUMULATION_STEPS="${ROUTER_GRADIENT_ACCUMULATION_STEPS:-4}"
for value in \
  "$ROUTER_NUM_GPUS" \
  "$ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE" \
  "$ROUTER_GRADIENT_ACCUMULATION_STEPS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Router GPU, batch-size, and accumulation values must be positive integers" >&2
    exit 2
  fi
done
case "$ROUTER_FINETUNE_MODE" in
  lora) ROUTER_FINETUNE_ARGS=(--lora) ;;
  full) ROUTER_FINETUNE_ARGS=() ;;
  *)
    echo "ROUTER_FINETUNE_MODE must be 'lora' or 'full'" >&2
    exit 2
    ;;
esac
ROUTER_EXTRA_ARGS="${ROUTER_EXTRA_ARGS:---bf16 --gradient-checkpointing}"
ROUTER_LAUNCH=("$PYTHON")
if (( ROUTER_NUM_GPUS > 1 )); then
  ROUTER_LAUNCH=(
    "$PYTHON" -m torch.distributed.run
    --standalone
    --nproc-per-node "$ROUTER_NUM_GPUS"
  )
fi
if [[ "$DEVICE" == cuda* ]]; then
  AVAILABLE_GPUS=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
  if (( AVAILABLE_GPUS < ROUTER_NUM_GPUS )); then
    echo "Stage 2 requested $ROUTER_NUM_GPUS GPUs, but only $AVAILABLE_GPUS are visible" >&2
    echo "Set CUDA_VISIBLE_DEVICES or override ROUTER_NUM_GPUS" >&2
    exit 2
  fi
fi

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  "$PYTHON" scripts/download_skillret.py
fi
if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  "$PYTHON" scripts/prepare_skillret.py \
    --embedding-provider openai \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-base-url "$EMBEDDING_BASE_URL" \
    "${EMBEDDING_DIMENSION_ARGS[@]}" \
    --batch-size "$EMBEDDING_BATCH_SIZE"
fi

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

# ROUTER_FINETUNE_MODE selects LoRA or full-parameter SFT.
# shellcheck disable=SC2086
"${ROUTER_LAUNCH[@]}" scripts/train_router.py \
  --model-name-or-path "$ROUTER_MODEL" \
  --virtual-tokens "$RUN_DIR/index/virtual_tokens.txt" \
  --output-dir "$RUN_DIR/router" --stage both --num-levels "$NUM_LEVELS" \
  --memorization-train "$RUN_DIR/router_data/memorization_train.jsonl" \
  --memorization-validation "$RUN_DIR/router_data/memorization_validation.jsonl" \
  --retrieval-train "$RUN_DIR/router_data/retrieval_train.jsonl" \
  --retrieval-validation "$RUN_DIR/router_data/retrieval_validation.jsonl" \
  --max-length 1024 \
  --per-device-train-batch-size "$ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$ROUTER_GRADIENT_ACCUMULATION_STEPS" \
  --memorization-epochs 1 --retrieval-epochs 3 \
  "${ROUTER_FINETUNE_ARGS[@]}" $ROUTER_EXTRA_ARGS

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
