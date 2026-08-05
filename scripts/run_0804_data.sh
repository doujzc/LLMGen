#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

stage() {
  echo
  echo "[$(date '+%F %T')] $*"
}

export CANDIDATES_PATH="${CANDIDATES_PATH:-candidates_0804.jsonl}"
export DATASET_ROOT="${DATASET_ROOT:-data_0804}"
export CATALOG_PATH="${CATALOG_PATH:-$DATASET_ROOT/catalog.jsonl}"
export CATALOG_REPORT="${CATALOG_REPORT:-$DATASET_ROOT/catalog_report.json}"

stage "0804 Stage 00: validate and import the candidate catalog"
"$PYTHON" scripts/light_data/00_build_catalog.py \
  --candidates "$CANDIDATES_PATH" \
  --output "$CATALOG_PATH" \
  --report "$CATALOG_REPORT"

export API_CONFIG="${API_CONFIG:-$HOME/llm_api.txt}"
export PROFILE_MODEL="${PROFILE_MODEL:-Qwen3.7-Plus}"
export GENERATION_MODEL="${GENERATION_MODEL:-Qwen3.7-Plus}"
export REVIEW_MODEL="${REVIEW_MODEL:-GLM-5.2}"
export API_WORKERS="${API_WORKERS:-12}"

export DATA_WORK_DIR="${DATA_WORK_DIR:-$DATASET_ROOT/work/qwen37-plus-glm52-v4-routing}"
export FINAL_DIR="${FINAL_DIR:-$DATASET_ROOT/final}"
export EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-$(wc -l < "$CATALOG_PATH")}"

# Match the current Light dataset's generation and one-epoch scale gates.
ONE_EPOCH_TRAIN_FLOOR=$((
  (3353 * 15 * EXPECTED_CANDIDATES + 567) / 568
))
export WORKFLOWS_PER_SKILL="${WORKFLOWS_PER_SKILL:-24}"
export QUERY_VARIANTS="${QUERY_VARIANTS:-3}"
export QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-1}"
export IMPLICIT_VARIANTS="${IMPLICIT_VARIANTS:-1}"
export TARGET_ORDER_VARIANTS="${TARGET_ORDER_VARIANTS:-4}"
export ALIGNMENT_VARIANTS="${ALIGNMENT_VARIANTS:-16}"
export MIN_ALIGNMENT_QUERIES_PER_SKILL="${MIN_ALIGNMENT_QUERIES_PER_SKILL:-15}"
export MIN_TRAIN_POSITIVES_PER_SKILL="${MIN_TRAIN_POSITIVES_PER_SKILL:-100}"
export MIN_AUGMENTED_TRAIN_QUERIES="${MIN_AUGMENTED_TRAIN_QUERIES:-$ONE_EPOCH_TRAIN_FLOOR}"
export COVERAGE_ROUNDS="${COVERAGE_ROUNDS:-5}"
export ALIGNMENT_COVERAGE_ROUNDS="${ALIGNMENT_COVERAGE_ROUNDS:-5}"
export FINAL_ALIGNMENT_COVERAGE_ROUNDS="${FINAL_ALIGNMENT_COVERAGE_ROUNDS:-5}"
export COVERAGE_OVERSAMPLE_FACTOR="${COVERAGE_OVERSAMPLE_FACTOR:-3.0}"
export APPLY_RECOVERY=0
export MANUAL_ALIGNMENT_PATH=""

stage "0804 Stages 01-05: generate, review, export, and validate"
bash scripts/run_clawhub_data.sh
