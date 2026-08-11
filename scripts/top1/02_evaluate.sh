#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -z "$TOP1_TEST_DATA" ]]; then
  echo "TOP1_TEST_DATA is empty; set it to a labeled or unlabeled JSONL file" >&2
  exit 2
fi
top1_require_file "$TOP1_TEST_DATA"
top1_require_file "$INFERENCE_MODEL/candidate_registry.json"
top1_require_file "$INFERENCE_MODEL/router_system_prompt.md"
mkdir -p "$TOP1_RUN_DIR/evaluation"

top1_step "02" "evaluate direct-name Top1 routing"
"$PYTHON" scripts/infer_candidate_router.py \
  --model-name-or-path "$INFERENCE_MODEL" \
  --queries "$TOP1_TEST_DATA" \
  --output-jsonl "$TOP1_RUN_DIR/evaluation/predictions.jsonl" \
  --metrics-output "$TOP1_RUN_DIR/evaluation/metrics.json" \
  --batch-size "$INFERENCE_BATCH_SIZE" \
  --decoding-mode "$INFERENCE_DECODING_MODE" \
  --num-beams "$INFERENCE_NUM_BEAMS" \
  --device "$INFERENCE_DEVICE" \
  --dtype "$INFERENCE_DTYPE" \
  "$@"
