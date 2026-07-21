#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SKILLRET_CONFIG="${SKILLRET_CONFIG:-configs/clawhub.env}"
# shellcheck source=scripts/skillret/common.sh
source "$ROOT/scripts/skillret/common.sh"

skillret_require_file "$ROUTER_DATA_DIR/memorization_train.jsonl"
skillret_require_dir "$ROUTER_OUTPUT_DIR/memorization"
skillret_require_dir "$ROUTER_OUTPUT_DIR/retrieval"

MEMORIZATION_PROMPT="Map the Agent Skill document to its fixed-length hierarchical skill code. Answer with code tokens only."

run_check() {
  local label="$1"
  local model_path="$2"
  DIAG_WITH_MODEL=1 \
  DIAG_MODEL_PATH="$model_path" \
  DIAG_ROUTER_MANIFEST="$model_path/router_manifest.json" \
  DIAG_ROUTER_TRAIN="$ROUTER_DATA_DIR/memorization_train.jsonl" \
  DIAG_TEACHER_EVAL_DATA="$ROUTER_DATA_DIR/memorization_train.jsonl" \
  DIAG_PREDICTIONS="$RUN_DIR/diagnostics/no-predictions.jsonl" \
  DIAG_OUTPUT="$RUN_DIR/diagnostics/${label}-on-memorization.json" \
  DIAG_SYSTEM_PROMPT="$MEMORIZATION_PROMPT" \
  DIAG_MIN_TRAIN_CODE_ACCURACY=0.95 \
    bash "$ROOT/scripts/clawhub_train/08_diagnose.sh"
}

run_check memorization-checkpoint "$ROUTER_OUTPUT_DIR/memorization"
run_check retrieval-checkpoint "$ROUTER_OUTPUT_DIR/retrieval"
