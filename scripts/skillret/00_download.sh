#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" scripts/download_skillret.py \
  --output-dir "$DATASET_DIR" \
  "$@"
