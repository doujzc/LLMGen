#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

promptgen_require_file "$PROMPTGEN_DATA_DIR/test.jsonl"
promptgen_require_file "$INFERENCE_MODEL/candidate_registry.json"
promptgen_require_file "$INFERENCE_MODEL/router_system_prompt.md"
mkdir -p "$PROMPTGEN_RUN_DIR/evaluation"

promptgen_step "02" "evaluate direct-name Top1 routing"
"$PYTHON" scripts/infer_candidate_router.py \
  --model-name-or-path "$INFERENCE_MODEL" \
  --queries "$PROMPTGEN_DATA_DIR/test.jsonl" \
  --output-jsonl "$PROMPTGEN_RUN_DIR/evaluation/predictions.jsonl" \
  --metrics-output "$PROMPTGEN_RUN_DIR/evaluation/metrics.json" \
  --batch-size "$INFERENCE_BATCH_SIZE" \
  --decoding-mode "$INFERENCE_DECODING_MODE" \
  --num-beams "$INFERENCE_NUM_BEAMS" \
  --device "$INFERENCE_DEVICE" \
  --dtype "$INFERENCE_DTYPE" \
  "$@"
