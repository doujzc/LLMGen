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

stage "Light Stage 00: validate and import the self-contained candidate catalog"
"$PYTHON" scripts/light_data/00_build_catalog.py \
  --candidates "${CANDIDATES_PATH:-data_light/candidates.jsonl}" \
  --output "${CATALOG_PATH:-data_light/catalog.jsonl}" \
  --report "${CATALOG_REPORT:-data_light/catalog_report.json}"

export API_CONFIG="${API_CONFIG:-$HOME/llm_api.txt}"
export PROFILE_MODEL="${PROFILE_MODEL:-Qwen3.7-Plus}"
export GENERATION_MODEL="${GENERATION_MODEL:-Qwen3.7-Plus}"
export REVIEW_MODEL="${REVIEW_MODEL:-GLM-5.2}"
export API_WORKERS="${API_WORKERS:-12}"

export CATALOG_PATH="${CATALOG_PATH:-data_light/catalog.jsonl}"
export DATA_WORK_DIR="${DATA_WORK_DIR:-data_light/work/qwen37-plus-glm52-v4-routing}"
export FINAL_DIR="${FINAL_DIR:-data_light/final}"
export EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-$(wc -l < "$CATALOG_PATH")}"

# One epoch exposes each light candidate to approximately as many ordered
# retrieval sequences as the historical 568-candidate ClawHub run did in
# fifteen epochs. Recompute the floor if the light catalog size changes.
ONE_EPOCH_TRAIN_FLOOR=$(( (3353 * 15 * EXPECTED_CANDIDATES + 567) / 568 ))
export WORKFLOWS_PER_SKILL="${WORKFLOWS_PER_SKILL:-24}"
export QUERY_VARIANTS="${QUERY_VARIANTS:-3}"
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
export MANUAL_ALIGNMENT_PATH="${MANUAL_ALIGNMENT_PATH:-data_light/manual_alignment.jsonl}"

stage "Light Stages 01-05: generate, review, export, and validate"
bash scripts/run_clawhub_data.sh
