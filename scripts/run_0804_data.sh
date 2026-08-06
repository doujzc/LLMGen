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
export HELDOUT_CSV="${HELDOUT_CSV:-result.csv}"
export DATASET_ROOT="${DATASET_ROOT:-data_0804}"
export WORK_DIR="${WORK_DIR:-$DATASET_ROOT/work/deepseek-v4-flash-teststyle-v1}"
export FINAL_DIR="${FINAL_DIR:-$DATASET_ROOT/final}"
export API_CONFIG="${API_CONFIG:-$HOME/deepseek_api_key.txt}"
export API_BASE_URL="${API_BASE_URL:-https://api.deepseek.com}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
export API_WORKERS="${API_WORKERS:-12}"
export STRICT_REVIEW_MODEL="${STRICT_REVIEW_MODEL:-deepseek-reasoner}"
export STRICT_REVIEW_WORKERS="${STRICT_REVIEW_WORKERS:-8}"

export PATCHES_PATH="${PATCHES_PATH:-$DATASET_ROOT/metadata_patches.jsonl}"
export RAW_CATALOG="${RAW_CATALOG:-$WORK_DIR/catalog.raw.jsonl}"
export SOURCE_PROFILES="${SOURCE_PROFILES:-$FINAL_DIR/skills.jsonl}"
export SOURCE_ROUTING_QUERIES="${SOURCE_ROUTING_QUERIES:-$FINAL_DIR/queries.jsonl}"
export CATALOG_PATH="${CATALOG_PATH:-$WORK_DIR/catalog.jsonl}"
export PROFILES_PATH="${PROFILES_PATH:-$WORK_DIR/skill_profiles.jsonl}"
export DISTRIBUTION_PROFILE="${DISTRIBUTION_PROFILE:-$DATASET_ROOT/distribution_profile.json}"
export WORKFLOWS_PATH="${WORKFLOWS_PATH:-$WORK_DIR/workflows.jsonl}"
export QUERIES_PATH="${QUERIES_PATH:-$WORK_DIR/queries.generated.jsonl}"
export REVIEWS_PATH="${REVIEWS_PATH:-$WORK_DIR/query_reviews.strict.jsonl}"
export ALIGNMENT_QUERIES_PATH="${ALIGNMENT_QUERIES_PATH:-$WORK_DIR/queries.alignment.generated.jsonl}"
export ALIGNMENT_REVIEWS_PATH="${ALIGNMENT_REVIEWS_PATH:-$WORK_DIR/query_alignment_reviews.jsonl}"

export MIN_WORKFLOWS_PER_SKILL="${MIN_WORKFLOWS_PER_SKILL:-45}"
export MIN_SEMANTIC_TRAIN_PER_SKILL="${MIN_SEMANTIC_TRAIN_PER_SKILL:-100}"
export MIN_ALIGNMENT_PER_SKILL="${MIN_ALIGNMENT_PER_SKILL:-15}"
export MIN_COMBINED_TRAIN_PER_SKILL="${MIN_COMBINED_TRAIN_PER_SKILL:-115}"
export ALIGNMENT_VARIANTS="${ALIGNMENT_VARIANTS:-18}"
export ALIGNMENT_COVERAGE_ROUNDS="${ALIGNMENT_COVERAGE_ROUNDS:-3}"
export QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-4}"
export REVIEW_BATCH_SIZE="${REVIEW_BATCH_SIZE:-20}"
export REVIEW_CHECKPOINT_BATCHES="${REVIEW_CHECKPOINT_BATCHES:-25}"
export STRICT_COVERAGE_ROUNDS="${STRICT_COVERAGE_ROUNDS:-3}"
export STRICT_COVERAGE_OVERSAMPLE="${STRICT_COVERAGE_OVERSAMPLE:-2.0}"
export MIN_AUGMENTED_TRAIN_QUERIES="${MIN_AUGMENTED_TRAIN_QUERIES:-15000}"
export REBUILD_STATIC="${REBUILD_STATIC:-0}"

mkdir -p "$WORK_DIR"
if [[ ! -f "$HELDOUT_CSV" ]]; then
  echo "Missing held-out CSV: $HELDOUT_CSV" >&2
  exit 2
fi

stage "0804 Stage 00: import and prepare the unchanged candidate registry"
"$PYTHON" scripts/light_data/00_build_catalog.py \
  --candidates "$CANDIDATES_PATH" --output "$RAW_CATALOG" \
  --report "$WORK_DIR/catalog.raw.report.json"
"$PYTHON" scripts/0804_data/00_prepare_registry.py \
  --catalog "$RAW_CATALOG" --profiles "$SOURCE_PROFILES" \
  --patches "$PATCHES_PATH" --output-catalog "$CATALOG_PATH" \
  --output-profiles "$PROFILES_PATH"

stage "0804 Stage 01: extract aggregate test distribution (no held-out rows retained)"
"$PYTHON" scripts/0804_data/01_build_distribution_profile.py \
  --heldout "$HELDOUT_CSV" --profiles "$PROFILES_PATH" \
  --output "$DISTRIBUTION_PROFILE"

if [[ "$REBUILD_STATIC" == "1" || ! -s "$WORKFLOWS_PATH" ]]; then
  stage "0804 Stage 02: build balanced two-candidate workflows"
  "$PYTHON" scripts/0804_data/02_build_workflows.py \
    --profiles "$PROFILES_PATH" --distribution-profile "$DISTRIBUTION_PROFILE" \
    --patches "$PATCHES_PATH" \
    --source-routing-queries "$SOURCE_ROUTING_QUERIES" \
    --source-light-queries data_light/final/queries.jsonl \
    --heldout "$HELDOUT_CSV" \
    --output "$WORKFLOWS_PATH" \
    --min-workflows-per-skill "$MIN_WORKFLOWS_PER_SKILL"
else
  stage "0804 Stage 02: reuse workflows for API resume ($WORKFLOWS_PATH)"
fi

stage "0804 Stage 03a: generate single-Skill alignment with DeepSeek Flash"
"$PYTHON" scripts/clawhub_data/02a_generate_alignment_queries.py \
  --profiles "$PROFILES_PATH" --output "$ALIGNMENT_QUERIES_PATH" \
  --variants "$ALIGNMENT_VARIANTS" --batch-size 1 \
  --api-config "$API_CONFIG" --model "$DEEPSEEK_MODEL" --workers "$API_WORKERS"

stage "0804 Stage 03b: review single-Skill alignment with DeepSeek Flash"
"$PYTHON" scripts/clawhub_data/03a_review_alignment_queries.py \
  --queries "$ALIGNMENT_QUERIES_PATH" --profiles "$PROFILES_PATH" \
  --output "$ALIGNMENT_REVIEWS_PATH" --batch-size 10 \
  --api-config "$API_CONFIG" --model "$DEEPSEEK_MODEL" --workers "$API_WORKERS"

for ((round = 1; round <= ALIGNMENT_COVERAGE_ROUNDS; round++)); do
  stage "0804 Stage 03c: alignment coverage backfill $round/$ALIGNMENT_COVERAGE_ROUNDS"
  "$PYTHON" scripts/clawhub_data/03a2_backfill_alignment.py \
    --profiles "$PROFILES_PATH" --queries "$ALIGNMENT_QUERIES_PATH" \
    --reviews "$ALIGNMENT_REVIEWS_PATH" --round "$round" \
    --variants "$ALIGNMENT_VARIANTS" --batch-size 1 \
    --min-passed-per-skill "$MIN_ALIGNMENT_PER_SKILL" \
    --api-config "$API_CONFIG" --model "$DEEPSEEK_MODEL" --workers "$API_WORKERS"
  "$PYTHON" scripts/clawhub_data/03a_review_alignment_queries.py \
    --queries "$ALIGNMENT_QUERIES_PATH" --profiles "$PROFILES_PATH" \
    --output "$ALIGNMENT_REVIEWS_PATH" --batch-size 10 \
    --api-config "$API_CONFIG" --model "$DEEPSEEK_MODEL" --workers "$API_WORKERS"
done

stage "0804 Stage 04: generate compact two-Skill queries with DeepSeek Flash"
"$PYTHON" scripts/0804_data/03_generate_queries.py \
  --workflows "$WORKFLOWS_PATH" --distribution-profile "$DISTRIBUTION_PROFILE" \
  --output "$QUERIES_PATH" --heldout "$HELDOUT_CSV" \
  --batch-size "$QUERY_BATCH_SIZE" --api-config "$API_CONFIG" \
  --model "$DEEPSEEK_MODEL" --workers "$API_WORKERS"

stage "0804 Stage 05: independently reject unrelated Skill conjunctions"
"$PYTHON" scripts/0804_data/04b_strict_review_queries.py \
  --queries "$QUERIES_PATH" --workflows "$WORKFLOWS_PATH" \
  --output "$REVIEWS_PATH" --batch-size "$REVIEW_BATCH_SIZE" \
  --checkpoint-batches "$REVIEW_CHECKPOINT_BATCHES" \
  --api-config "$API_CONFIG" --model "$STRICT_REVIEW_MODEL" \
  --workers "$STRICT_REVIEW_WORKERS"

for ((round = 1; round <= STRICT_COVERAGE_ROUNDS; round++)); do
  stage "0804 Stage 05b: strict-review coverage backfill $round/$STRICT_COVERAGE_ROUNDS"
  "$PYTHON" scripts/0804_data/04c_backfill_strict_coverage.py \
    --profiles "$PROFILES_PATH" --workflows "$WORKFLOWS_PATH" \
    --queries "$QUERIES_PATH" --reviews "$REVIEWS_PATH" \
    --min-train-per-skill "$MIN_SEMANTIC_TRAIN_PER_SKILL" \
    --oversample-factor "$STRICT_COVERAGE_OVERSAMPLE" --round "$round"
  "$PYTHON" scripts/0804_data/03_generate_queries.py \
    --workflows "$WORKFLOWS_PATH" --distribution-profile "$DISTRIBUTION_PROFILE" \
    --output "$QUERIES_PATH" --heldout "$HELDOUT_CSV" \
    --batch-size "$QUERY_BATCH_SIZE" --api-config "$API_CONFIG" \
    --model "$DEEPSEEK_MODEL" --workers "$API_WORKERS"
  "$PYTHON" scripts/0804_data/04b_strict_review_queries.py \
    --queries "$QUERIES_PATH" --workflows "$WORKFLOWS_PATH" \
    --output "$REVIEWS_PATH" --batch-size "$REVIEW_BATCH_SIZE" \
    --checkpoint-batches "$REVIEW_CHECKPOINT_BATCHES" \
    --api-config "$API_CONFIG" --model "$STRICT_REVIEW_MODEL" \
    --workers "$STRICT_REVIEW_WORKERS"
done

stage "0804 Stage 06: export the train/validation/test files"
"$PYTHON" scripts/clawhub_data/04_export_dataset.py \
  --catalog "$CATALOG_PATH" --profiles "$PROFILES_PATH" \
  --workflows "$WORKFLOWS_PATH" --queries "$QUERIES_PATH" \
  --reviews "$REVIEWS_PATH" --output-dir "$FINAL_DIR" \
  --alignment-queries "$ALIGNMENT_QUERIES_PATH" \
  --alignment-reviews "$ALIGNMENT_REVIEWS_PATH" \
  --min-train-positives-per-skill "$MIN_COMBINED_TRAIN_PER_SKILL" \
  --min-augmented-train-queries "$MIN_AUGMENTED_TRAIN_QUERIES" \
  --target-order-variants 2
"$PYTHON" scripts/clawhub_data/04a_export_alignment.py \
  --catalog "$CATALOG_PATH" --queries "$ALIGNMENT_QUERIES_PATH" \
  --reviews "$ALIGNMENT_REVIEWS_PATH" --output-dir "$FINAL_DIR" \
  --min-queries-per-skill "$MIN_ALIGNMENT_PER_SKILL"

stage "0804 Stage 07: audit candidate consistency, style, coverage, and leakage"
EXPECTED_CANDIDATES="$(awk 'NF { count += 1 } END { print count }' "$CANDIDATES_PATH")"
"$PYTHON" scripts/clawhub_data/05_validate_dataset.py \
  --dataset-dir "$FINAL_DIR" --expected-candidates "$EXPECTED_CANDIDATES"
"$PYTHON" scripts/0804_data/05_audit_final.py \
  --dataset-dir "$FINAL_DIR" --candidates "$CANDIDATES_PATH" \
  --heldout "$HELDOUT_CSV" --distribution-profile "$DISTRIBUTION_PROFILE" \
  --min-semantic-train-per-skill "$MIN_SEMANTIC_TRAIN_PER_SKILL"

stage "Dataset ready: $FINAL_DIR"
