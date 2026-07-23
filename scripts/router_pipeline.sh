#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/router_pipeline.sh <dataset> <command> [arguments...]

Datasets:
  clawhub    1000-candidate dataset, configs/clawhub.env
  light      301-candidate dataset, configs/light.env

Commands:
  full                         Run stages 01-07
  prepare | 01                 Validate/embed the dataset
  train-tokenizer | 02         Train the hierarchical Skill tokenizer
  export-codes | 03            Export Skill codes and run quality gates
  build-router-data | 04       Build memorization/retrieval SFT data
  train-memorization | 05      Train the memorization phase
  train-retrieval | 06         Train alignment and retrieval phases
  evaluate | 07                Run constrained closed-set evaluation
  diagnose | 08                Diagnose data, codes, and router predictions
  diagnose-memorization | 09   Compare memorization/retrieval code accuracy
  export-web | 10 [MODEL_DIR]  Export a completed model/checkpoint bundle
  web | 11 [MODEL_DIR]         Start the manual-testing Web UI
  paths                        Print the selected effective paths

Environment variables still override values inside the selected config.
EOF
}

if (( $# == 0 )) || [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 2 )); then
  usage >&2
  exit 2
fi

DATASET="$1"
COMMAND="$2"
shift 2

case "$DATASET" in
  clawhub|clawhub1000|1000)
    DATASET="clawhub"
    CONFIG="configs/clawhub.env"
    ;;
  light|light301|301)
    DATASET="light"
    CONFIG="configs/light.env"
    ;;
  *)
    echo "Unknown dataset: $DATASET (expected clawhub or light)" >&2
    exit 2
    ;;
esac

# The positional dataset is authoritative; inherited SKILLRET_CONFIG must not
# accidentally route a command to the other candidate set.
export SKILLRET_CONFIG="$CONFIG"
export SKIP_DOWNLOAD=1

run_stage() {
  local script="$1"
  shift
  exec bash "$ROOT/$script" "$@"
}

case "$COMMAND" in
  full|train)
    run_stage scripts/skillret/full.sh "$@"
    ;;
  prepare|01)
    run_stage scripts/skillret/01_prepare.sh "$@"
    ;;
  train-tokenizer|tokenizer|02)
    run_stage scripts/skillret/02_train_tokenizer.sh "$@"
    ;;
  export-codes|codes|03)
    run_stage scripts/skillret/03_export_codes.sh "$@"
    ;;
  build-router-data|router-data|04)
    run_stage scripts/skillret/04_build_router_data.sh "$@"
    ;;
  train-memorization|memorization|05)
    run_stage scripts/skillret/05_train_memorization.sh "$@"
    ;;
  train-retrieval|retrieval|06)
    run_stage scripts/skillret/06_train_retrieval.sh "$@"
    ;;
  evaluate|eval|07)
    run_stage scripts/skillret/07_evaluate.sh "$@"
    ;;
  diagnose|08)
    run_stage scripts/skillret/08_diagnose.sh "$@"
    ;;
  diagnose-memorization|09)
    run_stage scripts/skillret/09_diagnose_memorization.sh "$@"
    ;;
  export-web|export-bundle|10)
    run_stage scripts/skillret/10_export_web_bundle.sh "$@"
    ;;
  web|serve-web|11)
    run_stage scripts/skillret/11_serve_web.sh "$@"
    ;;
  paths|config)
    # shellcheck source=scripts/skillret/common.sh
    source "$ROOT/scripts/skillret/common.sh"
    printf 'dataset=%s\n' "$DATASET"
    printf 'dataset_name=%s\n' "$DATASET_NAME"
    printf 'config=%s\n' "$SKILLRET_CONFIG"
    printf 'dataset_dir=%s\n' "$DATASET_DIR"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'processed_dir=%s\n' "$PROCESSED_DIR"
    printf 'embedding_dir=%s\n' "$EMBEDDING_DIR"
    printf 'stage1_dir=%s\n' "$STAGE1_DIR"
    printf 'index_dir=%s\n' "$INDEX_DIR"
    printf 'router_data_dir=%s\n' "$ROUTER_DATA_DIR"
    printf 'router_output_dir=%s\n' "$ROUTER_OUTPUT_DIR"
    printf 'num_levels=%s\n' "$NUM_LEVELS"
    printf 'branching_factors=%s\n' "$BRANCHING_FACTORS"
    printf 'router_finetune_mode=%s\n' "$ROUTER_FINETUNE_MODE"
    printf 'router_num_gpus=%s\n' "$ROUTER_NUM_GPUS"
    printf 'router_deepspeed_config=%s\n' "$ROUTER_DEEPSPEED_CONFIG"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 2
    ;;
esac
