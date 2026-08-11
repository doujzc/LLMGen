#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The prepared 5,000-row dataset is tracked in this repository. Set
# REBUILD_PROMPTGEN_DATA=1 only when importing a newer PromptGen source file.
if [[ "${REBUILD_PROMPTGEN_DATA:-0}" == "1" ]]; then
  "$SCRIPT_DIR/00_prepare.sh"
fi
"$SCRIPT_DIR/01_train.sh" "$@"
"$SCRIPT_DIR/02_evaluate.sh"
