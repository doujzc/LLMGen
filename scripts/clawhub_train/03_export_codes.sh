#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SKILLRET_CONFIG="${SKILLRET_CONFIG:-configs/clawhub.env}"
exec bash "$ROOT/scripts/skillret/03_export_codes.sh" "$@"
