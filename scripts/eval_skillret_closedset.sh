#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/skillret/common.sh
source "$ROOT/scripts/skillret/common.sh"

export EVAL_PROTOCOL=closedset
export QUERY_SET="${QUERY_SET:-validation}"
export EVAL_DIR="${EVAL_DIR:-$RUN_DIR/evaluation/closedset-$QUERY_SET}"
exec bash scripts/skillret/07_evaluate.sh "$@"
