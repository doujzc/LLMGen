#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

top1_require_file "$TOP1_TRAIN_DATA"
top1_require_file "$TOP1_CANDIDATE_REGISTRY"

OPTIONAL_ARGS=()
if [[ -n "$TOP1_VALIDATION_DATA" ]]; then
  top1_require_file "$TOP1_VALIDATION_DATA"
  OPTIONAL_ARGS+=(--validation "$TOP1_VALIDATION_DATA")
fi
if [[ -n "$TOP1_TEST_DATA" && -f "$TOP1_TEST_DATA" ]]; then
  OPTIONAL_ARGS+=(--test "$TOP1_TEST_DATA")
fi

top1_step "00" "validate user-provided multi-turn Top1 JSONL"
"$PYTHON" scripts/top1/validate_data.py \
  --candidate-registry "$TOP1_CANDIDATE_REGISTRY" \
  --train "$TOP1_TRAIN_DATA" \
  "${OPTIONAL_ARGS[@]}" \
  "$@"
