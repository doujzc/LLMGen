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

export API_CONFIG="${API_CONFIG:-$HOME/deepseek_api_key.txt}"
export API_BASE_URL="${API_BASE_URL:-https://api.deepseek.com}"
export PROFILE_MODEL="${PROFILE_MODEL:-deepseek-v4-flash}"
export GENERATION_MODEL="${GENERATION_MODEL:-deepseek-v4-flash}"
export REVIEW_MODEL="${REVIEW_MODEL:-deepseek-v4-flash}"
export API_WORKERS="${API_WORKERS:-12}"

export CATALOG_PATH="${CATALOG_PATH:-data_light/catalog.jsonl}"
export DATA_WORK_DIR="${DATA_WORK_DIR:-data_light/work}"
export FINAL_DIR="${FINAL_DIR:-data_light/final}"
export EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-$(wc -l < "$CATALOG_PATH")}"

# Lightweight, but still complete: three direct examples per candidate plus
# reviewed 2/3-skill explicit/implicit workflows and deterministic order swaps.
export WORKFLOWS_PER_SKILL="${WORKFLOWS_PER_SKILL:-2}"
export QUERY_VARIANTS="${QUERY_VARIANTS:-2}"
export IMPLICIT_VARIANTS="${IMPLICIT_VARIANTS:-1}"
export TARGET_ORDER_VARIANTS="${TARGET_ORDER_VARIANTS:-2}"
export ALIGNMENT_VARIANTS="${ALIGNMENT_VARIANTS:-3}"
export MIN_ALIGNMENT_QUERIES_PER_SKILL="${MIN_ALIGNMENT_QUERIES_PER_SKILL:-3}"
export MIN_TRAIN_POSITIVES_PER_SKILL="${MIN_TRAIN_POSITIVES_PER_SKILL:-4}"
export COVERAGE_ROUNDS="${COVERAGE_ROUNDS:-3}"
export ALIGNMENT_COVERAGE_ROUNDS="${ALIGNMENT_COVERAGE_ROUNDS:-3}"
export FINAL_ALIGNMENT_COVERAGE_ROUNDS="${FINAL_ALIGNMENT_COVERAGE_ROUNDS:-3}"
export COVERAGE_OVERSAMPLE_FACTOR="${COVERAGE_OVERSAMPLE_FACTOR:-2.0}"
export APPLY_RECOVERY=0

stage "Light Stages 01-05: generate, review, export, and validate"
bash scripts/run_clawhub_data.sh
