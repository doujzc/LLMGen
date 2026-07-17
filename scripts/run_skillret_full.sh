#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/skillret/common.sh
source "$ROOT/scripts/skillret/common.sh"

case "$EVAL_PROTOCOL" in
  closedset|unseen|both) ;;
  *)
    echo "EVAL_PROTOCOL must be 'closedset', 'unseen', or 'both'" >&2
    exit 2
    ;;
esac
# Fail before the long preprocessing/tokenizer stages if Stage 2 cannot launch.
skillret_configure_router

run_step() {
  local number="$1"
  local description="$2"
  local script="$3"
  skillret_print_step "$number" "$description"
  bash "$script"
}

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  run_step 00 "download and validate SkillRet" scripts/skillret/00_download.sh
fi
if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  run_step 01 "normalize data and embed skills" scripts/skillret/01_prepare.sh
fi
run_step 02 "train hierarchical skill tokenizer" scripts/skillret/02_train_tokenizer.sh
run_step 03 "export train/test skill codes" scripts/skillret/03_export_codes.sh
run_step 04 "build memorization and retrieval SFT data" scripts/skillret/04_build_router_data.sh
run_step 05 "train router memorization phase" scripts/skillret/05_train_memorization.sh
run_step 06 "train router retrieval phase" scripts/skillret/06_train_retrieval.sh
run_step 07 "run constrained retrieval evaluation" scripts/skillret/07_evaluate.sh
