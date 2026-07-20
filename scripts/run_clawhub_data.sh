#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

API_CONFIG="${API_CONFIG:-$HOME/llm_api.txt}"
PROFILE_MODEL="${PROFILE_MODEL:-Qwen3.6-Plus}"
GENERATION_MODEL="${GENERATION_MODEL:-Qwen3.6-Plus}"
REVIEW_MODEL="${REVIEW_MODEL:-GLM-5.1}"
API_WORKERS="${API_WORKERS:-16}"

"$PYTHON" scripts/clawhub_data/00_profile_skills.py \
  --api-config "$API_CONFIG" --model "$PROFILE_MODEL" --workers "$API_WORKERS"
"$PYTHON" scripts/clawhub_data/01_build_workflows.py
"$PYTHON" scripts/clawhub_data/01b_apply_recovery_workflows.py
"$PYTHON" scripts/clawhub_data/02_generate_queries.py \
  --api-config "$API_CONFIG" --model "$GENERATION_MODEL" --workers "$API_WORKERS"
"$PYTHON" scripts/clawhub_data/03_review_queries.py \
  --api-config "$API_CONFIG" --model "$REVIEW_MODEL" --workers "$API_WORKERS"
"$PYTHON" scripts/clawhub_data/04_export_dataset.py
