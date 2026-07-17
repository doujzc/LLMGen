#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
OUT="${OUT:-outputs/skillret-smoke}"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  "$PYTHON" scripts/download_skillret.py
fi
"$PYTHON" scripts/prepare_skillret.py \
  --processed-dir "$OUT/processed" --embedding-dir "$OUT/embeddings" \
  --embedding-provider sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu --batch-size 8 --max-train-skills 32 --max-test-skills 16 \
  --max-skill-chars 2048
"$PYTHON" scripts/train_tokenizer.py \
  --data-root "$OUT" --output-dir "$OUT/stage1" \
  --device cpu \
  --num-levels 2 --branching-factors 4 4 --sk-epsilons 0 0.05 \
  --layers 32 16 --e-dim 8 --epochs 1 --batch-size 16 \
  --kmeans-iters 3 --sk-iters 5 --scheduler constant
"$PYTHON" scripts/export_skill_codes.py \
  --checkpoint "$OUT/stage1/best.pt" --processed-dir "$OUT/processed" \
  --embedding-dir "$OUT/embeddings" --output-dir "$OUT/index" \
  --device cpu
"$PYTHON" scripts/build_router_data.py \
  --catalog "$OUT/processed/catalog_train.jsonl" \
  --queries "$OUT/processed/queries_train.jsonl" \
  --qrels "$OUT/processed/qrels_train.jsonl" \
  --codes "$OUT/index/train_codes.jsonl" \
  --virtual-tokens "$OUT/index/virtual_tokens.txt" \
  --output-dir "$OUT/router_data" \
  --memorization-validation-fraction 0 \
  --retrieval-validation-fraction 0.25
"$PYTHON" scripts/train_router.py \
  --model-name-or-path hf-internal-testing/tiny-random-gpt2 \
  --virtual-tokens "$OUT/index/virtual_tokens.txt" \
  --output-dir "$OUT/router" --stage both --num-levels 2 --max-length 64 \
  --memorization-train "$OUT/router_data/memorization_train.jsonl" \
  --retrieval-train "$OUT/router_data/retrieval_train.jsonl" \
  --retrieval-validation "$OUT/router_data/retrieval_validation.jsonl" \
  --per-device-train-batch-size 8 --gradient-accumulation-steps 1 \
  --memorization-epochs 1 --retrieval-epochs 1 --save-steps 1000 --eval-steps 1000
PYTHON="$PYTHON" RUN_DIR="$OUT" ROUTER_MODEL=hf-internal-testing/tiny-random-gpt2 \
  DEVICE=cpu DTYPE=float32 BATCH_SIZE=4 BEAM_SIZE=8 NUM_CODE_PATHS=8 \
  EVAL_DIR="$OUT/evaluation" bash scripts/eval_skillret_closedset.sh
