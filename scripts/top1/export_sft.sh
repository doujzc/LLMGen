#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

top1_require_file "$TOP1_TRAIN_DATA"
top1_require_file "$TOP1_CANDIDATE_REGISTRY"
top1_require_file "$TOP1_SYSTEM_PROMPT"

TRUST_ARGS=()
case "$ROUTER_TRUST_REMOTE_CODE" in
  1|true|yes) TRUST_ARGS=(--trust-remote-code) ;;
  0|false|no) ;;
  *) echo "ROUTER_TRUST_REMOTE_CODE must be 0 or 1" >&2; exit 2 ;;
esac

top1_step "SFT" "export standard messages JSONL"
"$PYTHON" scripts/top1/export_sft_data.py \
  --input-jsonl "$TOP1_TRAIN_DATA" \
  --output-jsonl "$TOP1_RUN_DIR/router/retrieval/sft_input.jsonl" \
  --candidate-registry "$TOP1_CANDIDATE_REGISTRY" \
  --system-prompt-file "$TOP1_SYSTEM_PROMPT" \
  "${TRUST_ARGS[@]}" \
  "$@"
