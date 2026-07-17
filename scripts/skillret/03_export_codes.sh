#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

skillret_require_file "$STAGE1_DIR/best.pt"
"$PYTHON" scripts/export_skill_codes.py \
  --checkpoint "$STAGE1_DIR/best.pt" \
  --processed-dir "$PROCESSED_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --output-dir "$INDEX_DIR" \
  --device "$DEVICE" \
  --batch-size "$CODE_EXPORT_BATCH_SIZE" \
  "$@"
