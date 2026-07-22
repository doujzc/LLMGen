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
WORKFLOWS_PER_SKILL="${WORKFLOWS_PER_SKILL:-3}"
MIN_TRAIN_POSITIVES_PER_SKILL="${MIN_TRAIN_POSITIVES_PER_SKILL:-10}"
COVERAGE_ROUNDS="${COVERAGE_ROUNDS:-5}"
COVERAGE_OVERSAMPLE_FACTOR="${COVERAGE_OVERSAMPLE_FACTOR:-3.0}"
SKIP_BASE_WORKFLOWS="${SKIP_BASE_WORKFLOWS:-0}"
QUERY_VARIANTS="${QUERY_VARIANTS:-3}"
IMPLICIT_VARIANTS="${IMPLICIT_VARIANTS:-1}"
TARGET_ORDER_VARIANTS="${TARGET_ORDER_VARIANTS:-3}"
ALIGNMENT_VARIANTS="${ALIGNMENT_VARIANTS:-3}"
MIN_ALIGNMENT_QUERIES_PER_SKILL="${MIN_ALIGNMENT_QUERIES_PER_SKILL:-5}"
ALIGNMENT_COVERAGE_ROUNDS="${ALIGNMENT_COVERAGE_ROUNDS:-5}"
FINAL_ALIGNMENT_COVERAGE_ROUNDS="${FINAL_ALIGNMENT_COVERAGE_ROUNDS:-5}"
APPLY_RECOVERY="${APPLY_RECOVERY:-1}"

CATALOG_PATH="${CATALOG_PATH:-data/clawhub/catalog.jsonl}"
DATA_WORK_DIR="${DATA_WORK_DIR:-data/clawhub_training}"
PROFILES_PATH="${PROFILES_PATH:-$DATA_WORK_DIR/skill_profiles.jsonl}"
WORKFLOWS_PATH="${WORKFLOWS_PATH:-$DATA_WORK_DIR/workflows.jsonl}"
QUERIES_PATH="${QUERIES_PATH:-$DATA_WORK_DIR/queries.generated.jsonl}"
REVIEWS_PATH="${REVIEWS_PATH:-$DATA_WORK_DIR/query_reviews.jsonl}"
ALIGNMENT_QUERIES_PATH="${ALIGNMENT_QUERIES_PATH:-$DATA_WORK_DIR/queries.alignment.generated.jsonl}"
ALIGNMENT_REVIEWS_PATH="${ALIGNMENT_REVIEWS_PATH:-$DATA_WORK_DIR/query_alignment_reviews.jsonl}"
FINAL_DIR="${FINAL_DIR:-$DATA_WORK_DIR/final}"
RECOVERY_CONFIG="${RECOVERY_CONFIG:-configs/clawhub_recovery.json}"

stage() {
  echo
  echo "[$(date '+%F %T')] $*"
}

stage "Stage 00: profile candidate skills"
"$PYTHON" scripts/clawhub_data/00_profile_skills.py \
  --catalog "$CATALOG_PATH" --output "$PROFILES_PATH" \
  --api-config "$API_CONFIG" --model "$PROFILE_MODEL" --workers "$API_WORKERS"
if [[ "$SKIP_BASE_WORKFLOWS" != "1" ]]; then
  stage "Stage 01: build multi-skill workflows"
  "$PYTHON" scripts/clawhub_data/01_build_workflows.py \
    --profiles "$PROFILES_PATH" --output "$WORKFLOWS_PATH" \
    --workflows-per-skill "$WORKFLOWS_PER_SKILL"
fi
if [[ "$APPLY_RECOVERY" == "1" ]]; then
  stage "Stage 01b: append configured recovery workflows"
  "$PYTHON" scripts/clawhub_data/01b_apply_recovery_workflows.py \
    --profiles "$PROFILES_PATH" --workflows "$WORKFLOWS_PATH" \
    --config "$RECOVERY_CONFIG"
fi
stage "Stage 02a: generate direct single-skill curriculum queries"
"$PYTHON" scripts/clawhub_data/02a_generate_alignment_queries.py \
  --profiles "$PROFILES_PATH" --output "$ALIGNMENT_QUERIES_PATH" \
  --variants "$ALIGNMENT_VARIANTS" \
  --api-config "$API_CONFIG" --model "$GENERATION_MODEL" --workers "$API_WORKERS"
stage "Stage 03a: independently review single-skill curriculum queries"
"$PYTHON" scripts/clawhub_data/03a_review_alignment_queries.py \
  --queries "$ALIGNMENT_QUERIES_PATH" --profiles "$PROFILES_PATH" \
  --output "$ALIGNMENT_REVIEWS_PATH" \
  --api-config "$API_CONFIG" --model "$REVIEW_MODEL" --workers "$API_WORKERS"
for ((round = 1; round <= ALIGNMENT_COVERAGE_ROUNDS; round++)); do
  stage "Stage 03a2: single-skill coverage backfill $round/$ALIGNMENT_COVERAGE_ROUNDS"
  "$PYTHON" scripts/clawhub_data/03a2_backfill_alignment.py \
    --profiles "$PROFILES_PATH" --queries "$ALIGNMENT_QUERIES_PATH" \
    --reviews "$ALIGNMENT_REVIEWS_PATH" --round "$round" \
    --variants "$ALIGNMENT_VARIANTS" \
    --min-passed-per-skill "$MIN_ALIGNMENT_QUERIES_PER_SKILL" \
    --api-config "$API_CONFIG" --model "$GENERATION_MODEL" --workers "$API_WORKERS"
  "$PYTHON" scripts/clawhub_data/03a_review_alignment_queries.py \
    --queries "$ALIGNMENT_QUERIES_PATH" --profiles "$PROFILES_PATH" \
    --output "$ALIGNMENT_REVIEWS_PATH" \
    --api-config "$API_CONFIG" --model "$REVIEW_MODEL" --workers "$API_WORKERS"
done
stage "Stage 02: generate explicit and implicit queries"
"$PYTHON" scripts/clawhub_data/02_generate_queries.py \
  --workflows "$WORKFLOWS_PATH" --output "$QUERIES_PATH" \
  --variants "$QUERY_VARIANTS" --implicit-variants "$IMPLICIT_VARIANTS" \
  --api-config "$API_CONFIG" --model "$GENERATION_MODEL" --workers "$API_WORKERS"
stage "Stage 03: independently review generated queries"
"$PYTHON" scripts/clawhub_data/03_review_queries.py \
  --queries "$QUERIES_PATH" --workflows "$WORKFLOWS_PATH" --output "$REVIEWS_PATH" \
  --api-config "$API_CONFIG" --model "$REVIEW_MODEL" --workers "$API_WORKERS"
for ((round = 1; round <= COVERAGE_ROUNDS; round++)); do
  stage "Stage 03b: coverage backfill round $round/$COVERAGE_ROUNDS"
  "$PYTHON" scripts/clawhub_data/03b_build_coverage_workflows.py \
    --profiles "$PROFILES_PATH" --workflows "$WORKFLOWS_PATH" \
    --queries "$QUERIES_PATH" --reviews "$REVIEWS_PATH" \
    --alignment-queries "$ALIGNMENT_QUERIES_PATH" \
    --alignment-reviews "$ALIGNMENT_REVIEWS_PATH" \
    --round "$round" \
    --min-train-positives-per-skill "$MIN_TRAIN_POSITIVES_PER_SKILL" \
    --variants-per-workflow "$QUERY_VARIANTS" \
    --oversample-factor "$COVERAGE_OVERSAMPLE_FACTOR"
  stage "Stage 02: generate new coverage queries for round $round"
  "$PYTHON" scripts/clawhub_data/02_generate_queries.py \
    --workflows "$WORKFLOWS_PATH" --output "$QUERIES_PATH" \
    --variants "$QUERY_VARIANTS" --implicit-variants "$IMPLICIT_VARIANTS" \
    --api-config "$API_CONFIG" --model "$GENERATION_MODEL" --workers "$API_WORKERS"
  stage "Stage 03: review new coverage queries for round $round"
  "$PYTHON" scripts/clawhub_data/03_review_queries.py \
    --queries "$QUERIES_PATH" --workflows "$WORKFLOWS_PATH" --output "$REVIEWS_PATH" \
    --api-config "$API_CONFIG" --model "$REVIEW_MODEL" --workers "$API_WORKERS"
done
for ((round = 1; round <= FINAL_ALIGNMENT_COVERAGE_ROUNDS; round++)); do
  backfill_round=$((ALIGNMENT_COVERAGE_ROUNDS + round))
  stage "Stage 03c: final combined-coverage alignment rescue $round/$FINAL_ALIGNMENT_COVERAGE_ROUNDS"
  "$PYTHON" scripts/clawhub_data/03a2_backfill_alignment.py \
    --profiles "$PROFILES_PATH" --queries "$ALIGNMENT_QUERIES_PATH" \
    --reviews "$ALIGNMENT_REVIEWS_PATH" --round "$backfill_round" \
    --variants "$ALIGNMENT_VARIANTS" \
    --min-passed-per-skill "$MIN_ALIGNMENT_QUERIES_PER_SKILL" \
    --multiskill-queries "$QUERIES_PATH" \
    --multiskill-reviews "$REVIEWS_PATH" --workflows "$WORKFLOWS_PATH" \
    --min-combined-per-skill "$MIN_TRAIN_POSITIVES_PER_SKILL" \
    --api-config "$API_CONFIG" --model "$GENERATION_MODEL" --workers "$API_WORKERS"
  "$PYTHON" scripts/clawhub_data/03a_review_alignment_queries.py \
    --queries "$ALIGNMENT_QUERIES_PATH" --profiles "$PROFILES_PATH" \
    --output "$ALIGNMENT_REVIEWS_PATH" \
    --api-config "$API_CONFIG" --model "$REVIEW_MODEL" --workers "$API_WORKERS"
done
stage "Stage 04: enforce coverage gate and export target-order variants"
"$PYTHON" scripts/clawhub_data/04_export_dataset.py \
  --catalog "$CATALOG_PATH" --profiles "$PROFILES_PATH" \
  --workflows "$WORKFLOWS_PATH" --queries "$QUERIES_PATH" \
  --reviews "$REVIEWS_PATH" --output-dir "$FINAL_DIR" \
  --alignment-queries "$ALIGNMENT_QUERIES_PATH" \
  --alignment-reviews "$ALIGNMENT_REVIEWS_PATH" \
  --min-train-positives-per-skill "$MIN_TRAIN_POSITIVES_PER_SKILL" \
  --target-order-variants "$TARGET_ORDER_VARIANTS"
stage "Stage 04a: export single-skill curriculum data"
"$PYTHON" scripts/clawhub_data/04a_export_alignment.py \
  --catalog "$CATALOG_PATH" --queries "$ALIGNMENT_QUERIES_PATH" \
  --reviews "$ALIGNMENT_REVIEWS_PATH" --output-dir "$FINAL_DIR" \
  --min-queries-per-skill "$MIN_ALIGNMENT_QUERIES_PER_SKILL"

stage "Stage 05: validate implicit intent, order augmentation, and coverage"
AUDIT_ARGS=(--dataset-dir "$FINAL_DIR")
if [[ -n "${EXPECTED_CANDIDATES:-}" ]]; then
  AUDIT_ARGS+=(--expected-candidates "$EXPECTED_CANDIDATES")
fi
"$PYTHON" scripts/clawhub_data/05_validate_dataset.py "${AUDIT_ARGS[@]}"
stage "Dataset ready: $FINAL_DIR"
