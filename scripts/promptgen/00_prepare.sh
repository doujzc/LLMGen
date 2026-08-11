#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

promptgen_require_file "$PROMPTGEN_SOURCE"
promptgen_require_file "$PROMPTGEN_CANDIDATE_REGISTRY"
promptgen_step "00" "collapse PromptGen labels into seven direct candidate names"
"$PYTHON" scripts/promptgen/00_prepare.py \
  --source "$PROMPTGEN_SOURCE" \
  --candidate-registry "$PROMPTGEN_CANDIDATE_REGISTRY" \
  --output-dir "$PROMPTGEN_DATA_DIR" \
  "$@"
