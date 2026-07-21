#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-.venv/bin/python}"
export DATASET_DIR="${DATASET_DIR:-data/skillret}"
export RUN_DIR="${OUT:-outputs/skillret-smoke}"
export PROCESSED_DIR="$RUN_DIR/processed"
export EMBEDDING_DIR="$RUN_DIR/embeddings"
export STAGE1_DIR="$RUN_DIR/stage1"
export INDEX_DIR="$RUN_DIR/index"
export ROUTER_DATA_DIR="$RUN_DIR/router_data"
export ROUTER_OUTPUT_DIR="$RUN_DIR/router"
export DEVICE=cpu

export EMBEDDING_PROVIDER=sentence-transformers
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export EMBEDDING_BATCH_SIZE=8
export NUM_LEVELS=2
export BRANCHING_FACTORS="4 4"
export SK_EPSILONS="0 0.05"
export RQ_LAYERS="32 16"
export TOKENIZER_E_DIM=8
export TOKENIZER_EPOCHS=1
export TOKENIZER_BATCH_SIZE=16
export TOKENIZER_AMP_DTYPE=none
export ROUTER_MODEL=hf-internal-testing/tiny-random-gpt2
export ROUTER_FINETUNE_MODE=full
export ROUTER_NUM_GPUS=1
export ROUTER_DEEPSPEED_CONFIG=none
export ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE=8
export ROUTER_GRADIENT_ACCUMULATION_STEPS=1
export ROUTER_MAX_LENGTH=64
export ROUTER_MEMORIZATION_EPOCHS=1
export ROUTER_RETRIEVAL_EPOCHS=1
export ROUTER_SAVE_STEPS=1000
export ROUTER_EVAL_STEPS=1000
export ROUTER_PRECISION=fp32
export ROUTER_GRADIENT_CHECKPOINTING=0
export ROUTER_VALIDATION_FRACTION=0.25
export EVAL_DTYPE=float32
export EVAL_BATCH_SIZE=4
export EVAL_MAX_CODE_PATHS=8

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  bash scripts/skillret/00_download.sh
fi
bash scripts/skillret/01_prepare.sh \
  --max-train-skills 32 --max-test-skills 16 --max-skill-chars 2048
bash scripts/skillret/02_train_tokenizer.sh \
  --kmeans-iters 3 --sk-iters 5 --scheduler constant
bash scripts/skillret/03_export_codes.sh
bash scripts/skillret/04_build_router_data.sh
bash scripts/skillret/05_train_memorization.sh
bash scripts/skillret/06_train_retrieval.sh
EVAL_DIR="$RUN_DIR/evaluation" bash scripts/skillret/07_evaluate.sh
