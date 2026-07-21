#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

skillret_require_file "$STAGE1_DIR/best.pt"
read -r -a CODE_SPLIT_ARGS <<< "${CODE_SPLITS:-train test}"
ARGS=(
  --checkpoint "$STAGE1_DIR/best.pt"
  --processed-dir "$PROCESSED_DIR"
  --embedding-dir "$EMBEDDING_DIR"
  --output-dir "$INDEX_DIR"
  --device "$DEVICE"
  --batch-size "$CODE_EXPORT_BATCH_SIZE"
  --splits "${CODE_SPLIT_ARGS[@]}"
  --assignment-mode "$CODE_ASSIGNMENT_MODE"
  --assignment-exact-group-size "$CODE_ASSIGNMENT_EXACT_GROUP_SIZE"
  --max-collision-rate "$CODE_MAX_COLLISION_RATE"
  --max-raw-collision-rate "$CODE_MAX_RAW_COLLISION_RATE"
  --max-bucket-size "$CODE_MAX_BUCKET_SIZE"
  --min-level-utilization "$CODE_MIN_LEVEL_UTILIZATION"
  --min-normalized-entropy "$CODE_MIN_NORMALIZED_ENTROPY"
  --min-raw-level-utilization "$CODE_MIN_RAW_LEVEL_UTILIZATION"
  --min-raw-normalized-entropy "$CODE_MIN_RAW_NORMALIZED_ENTROPY"
)
if [[ -n "$CODE_QUALITY_GATE_SPLIT" ]]; then
  ARGS+=(--quality-gate-split "$CODE_QUALITY_GATE_SPLIT")
fi
"$PYTHON" scripts/export_skill_codes.py \
  "${ARGS[@]}" \
  "$@"
