#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/mnt/c/Users/T/Documents/Codex/2026-07-22/clawhub-1000-skill/outputs/clawhub-top-1000}"
ARCHIVES_DIR="${ARCHIVES_DIR:-$SNAPSHOT_ROOT/archives}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$SNAPSHOT_ROOT/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SNAPSHOT_ROOT/llmgen-dataset-v2}"

mkdir -p "$OUTPUT_ROOT/work"
echo "[$(date '+%F %T')] Importing and verifying the 1,000-skill archive snapshot"
"$PYTHON" scripts/clawhub_data/00_import_archives.py \
  --archives-dir "$ARCHIVES_DIR" \
  --manifest "$SOURCE_MANIFEST" \
  --output "$OUTPUT_ROOT/catalog.jsonl" \
  --expected-count 1000

export PYTHON
export CATALOG_PATH="$OUTPUT_ROOT/catalog.jsonl"
export DATA_WORK_DIR="$OUTPUT_ROOT/work"
export FINAL_DIR="$OUTPUT_ROOT/final"
export APPLY_RECOVERY=0
export QUERY_VARIANTS="${QUERY_VARIANTS:-3}"
export IMPLICIT_VARIANTS="${IMPLICIT_VARIANTS:-1}"
export TARGET_ORDER_VARIANTS="${TARGET_ORDER_VARIANTS:-3}"
export EXPECTED_CANDIDATES=1000

bash scripts/run_clawhub_data.sh
