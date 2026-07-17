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
ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE="${ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
ROUTER_GRADIENT_ACCUMULATION_STEPS="${ROUTER_GRADIENT_ACCUMULATION_STEPS:-8}"
ROUTER_DEEPSPEED_CONFIG="${ROUTER_DEEPSPEED_CONFIG-configs/deepspeed_zero3.json}"
ROUTER_VALIDATION_FRACTION="${ROUTER_VALIDATION_FRACTION:-0.02}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-closedset}"
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
case "$EVAL_PROTOCOL" in
  closedset|unseen|both) ;;
  *)
    echo "EVAL_PROTOCOL must be 'closedset', 'unseen', or 'both'" >&2
    exit 2
    ;;
esac
ROUTER_EXTRA_ARGS="${ROUTER_EXTRA_ARGS:---bf16 --gradient-checkpointing}"
ROUTER_DEEPSPEED_ARGS=()
if [[ -n "$ROUTER_DEEPSPEED_CONFIG" && "$ROUTER_DEEPSPEED_CONFIG" != "none" ]]; then
  if [[ ! -f "$ROUTER_DEEPSPEED_CONFIG" ]]; then
    echo "DeepSpeed config does not exist: $ROUTER_DEEPSPEED_CONFIG" >&2
    exit 2
  fi
  ROUTER_DEEPSPEED_ARGS=(--deepspeed "$ROUTER_DEEPSPEED_CONFIG")
fi
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
  --output-dir "$RUN_DIR/router_data" \
  --memorization-validation-fraction 0 \
  --retrieval-validation-fraction "$ROUTER_VALIDATION_FRACTION"

# DeepSpeed phases run in separate processes so the retrieval engine starts
# from the consolidated memorization artifact rather than a live ZeRO engine.
ROUTER_COMMON_ARGS=(
  --virtual-tokens "$RUN_DIR/index/virtual_tokens.txt"
  --output-dir "$RUN_DIR/router"
  --num-levels "$NUM_LEVELS"
  --max-length 1024
  --per-device-train-batch-size "$ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$ROUTER_GRADIENT_ACCUMULATION_STEPS"
)

# ROUTER_FINETUNE_MODE selects LoRA or full-parameter SFT. Word splitting of
# ROUTER_EXTRA_ARGS is intentional so callers can append Trainer CLI flags.
# shellcheck disable=SC2086
"${ROUTER_LAUNCH[@]}" scripts/train_router.py \
  --model-name-or-path "$ROUTER_MODEL" \
  "${ROUTER_COMMON_ARGS[@]}" \
  --stage memorization \
  --memorization-train "$RUN_DIR/router_data/memorization_train.jsonl" \
  --memorization-validation "$RUN_DIR/router_data/memorization_validation.jsonl" \
  --memorization-epochs 1 \
  "${ROUTER_FINETUNE_ARGS[@]}" \
  "${ROUTER_DEEPSPEED_ARGS[@]}" $ROUTER_EXTRA_ARGS

if [[ "$ROUTER_FINETUNE_MODE" == "lora" ]]; then
  ROUTER_RETRIEVAL_MODEL_ARGS=(
    --model-name-or-path "$ROUTER_MODEL"
    --adapter-name-or-path "$RUN_DIR/router/memorization"
  )
else
  ROUTER_RETRIEVAL_MODEL_ARGS=(
    --model-name-or-path "$RUN_DIR/router/memorization"
  )
fi

# shellcheck disable=SC2086
"${ROUTER_LAUNCH[@]}" scripts/train_router.py \
  "${ROUTER_RETRIEVAL_MODEL_ARGS[@]}" \
  "${ROUTER_COMMON_ARGS[@]}" \
  --stage retrieval \
  --retrieval-train "$RUN_DIR/router_data/retrieval_train.jsonl" \
  --retrieval-validation "$RUN_DIR/router_data/retrieval_validation.jsonl" \
  --retrieval-epochs 3 \
  "${ROUTER_DEEPSPEED_ARGS[@]}" $ROUTER_EXTRA_ARGS

run_closedset_eval() {
  local output_dir="$1"
  PYTHON="$PYTHON" RUN_DIR="$RUN_DIR" ROUTER_MODEL="$ROUTER_MODEL" \
    DEVICE="$DEVICE" DTYPE=bfloat16 EVAL_DIR="$output_dir" QUERY_SET=validation \
    bash scripts/eval_skillret_closedset.sh
}

run_unseen_eval() {
  local output_dir="$1"
  mkdir -p "$output_dir"
  "$PYTHON" scripts/infer_router.py \
    --model-name-or-path "$RUN_DIR/router/retrieval" \
    --base-model-name-or-path "$ROUTER_MODEL" \
    --virtual-tokens "$RUN_DIR/index/virtual_tokens.txt" \
    --codes "$RUN_DIR/index/test_codes.jsonl" \
    --registry "$RUN_DIR/index/test_registry.json" \
    --queries data/skillret/processed/queries_test.jsonl \
    --qrels data/skillret/processed/qrels_test.jsonl \
    --output-jsonl "$output_dir/predictions.jsonl" \
    --metrics-output "$output_dir/metrics.json" \
    --batch-size 1 \
    --beam-size 20 --num-code-paths 20 --top-k 20 \
    --device "$DEVICE" --dtype bfloat16
}

case "$EVAL_PROTOCOL" in
  closedset)
    run_closedset_eval "$RUN_DIR/evaluation"
    ;;
  unseen)
    run_unseen_eval "$RUN_DIR/evaluation"
    ;;
  both)
    run_closedset_eval "$RUN_DIR/evaluation/closedset-validation"
    run_unseen_eval "$RUN_DIR/evaluation/unseen"
    ;;
esac
