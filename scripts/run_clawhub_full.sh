#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SKILLRET_CONFIG="${SKILLRET_CONFIG:-configs/clawhub.env}"
export SKIP_DOWNLOAD=1
exec bash "$ROOT/scripts/run_skillret_full.sh" "$@"
