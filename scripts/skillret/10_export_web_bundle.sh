#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/skillret/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_DIR="${1:-${ROUTER_EXPORT_MODEL_DIR:-$ROUTER_OUTPUT_DIR/retrieval}}"
if (( $# > 0 )); then
  shift
fi
skillret_print_step "10" "export Web bundle from $MODEL_DIR"
skillret_require_dir "$MODEL_DIR"
skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/train_registry.json"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"

COMMON_ARGS=(
  --model-dir "$MODEL_DIR"
  --catalog "$PROCESSED_DIR/catalog_train.jsonl"
  --codes "$INDEX_DIR/train_codes.jsonl"
  --registry "$INDEX_DIR/train_registry.json"
  --virtual-tokens "$INDEX_DIR/virtual_tokens.txt"
)

if [[ ! -f "$MODEL_DIR/router_manifest.json" ]]; then
  CHECKPOINT_NAME="$(basename "$MODEL_DIR")"
  if [[ ! "$CHECKPOINT_NAME" =~ ^checkpoint-[0-9]+$ ]]; then
    echo "Model directory has no router_manifest.json and is not checkpoint-N: $MODEL_DIR" >&2
    exit 2
  fi
  skillret_require_file "$MODEL_DIR/trainer_state.json"
  skillret_require_file "$ROUTER_DATA_DIR/retrieval_train.jsonl"
  skillret_require_file "$ROUTER_DATA_DIR/retrieval_validation.jsonl"

  TOKENIZER_SOURCE="${ROUTER_CHECKPOINT_TOKENIZER_SOURCE:-}"
  if [[ -z "$TOKENIZER_SOURCE" ]]; then
    if [[ -f "$ROUTER_OUTPUT_DIR/retrieval_alignment/tokenizer_config.json" ]]; then
      TOKENIZER_SOURCE="$ROUTER_OUTPUT_DIR/retrieval_alignment"
    elif [[ -f "$ROUTER_OUTPUT_DIR/memorization/tokenizer_config.json" ]]; then
      TOKENIZER_SOURCE="$ROUTER_OUTPUT_DIR/memorization"
    else
      echo "No completed preceding-phase tokenizer was found." >&2
      echo "Set ROUTER_CHECKPOINT_TOKENIZER_SOURCE explicitly." >&2
      exit 2
    fi
  fi
  skillret_require_dir "$TOKENIZER_SOURCE"

  TEMPLATE_MANIFEST="${ROUTER_CHECKPOINT_TEMPLATE_MANIFEST:-$TOKENIZER_SOURCE/router_manifest.json}"
  skillret_require_file "$TEMPLATE_MANIFEST"
  CHECKPOINT_PHASE="$(basename "$(dirname "$MODEL_DIR")")"
  if [[ "$CHECKPOINT_PHASE" != "retrieval" ]]; then
    echo "Only completed 06b retrieval/checkpoint-N directories are supported: $MODEL_DIR" >&2
    exit 2
  fi
  CHECKPOINT_EXPORT_DIR="${ROUTER_CHECKPOINT_EXPORT_DIR:-$RUN_DIR/exports/${CHECKPOINT_PHASE}-${CHECKPOINT_NAME}}"

  REPLAY_ARGS=()
  if [[ "$ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION" != "0" && "$ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION" != "0.0" ]]; then
    skillret_require_file "$ROUTER_DATA_DIR/retrieval_alignment_train.jsonl"
    REPLAY_ARGS+=(
      --alignment-replay-data "$ROUTER_DATA_DIR/retrieval_alignment_train.jsonl"
      --alignment-replay-fraction "$ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION"
    )
  fi
  if [[ "$ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION" != "0" && "$ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION" != "0.0" ]]; then
    skillret_require_file "$ROUTER_DATA_DIR/memorization_train.jsonl"
    REPLAY_ARGS+=(
      --memorization-replay-data "$ROUTER_DATA_DIR/memorization_train.jsonl"
      --memorization-replay-fraction "$ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION"
    )
  fi

  TRUST_ARGS=()
  case "$ROUTER_TRUST_REMOTE_CODE" in
    1|true|yes) TRUST_ARGS=(--trust-remote-code) ;;
    0|false|no) ;;
    *)
      echo "ROUTER_TRUST_REMOTE_CODE must be 0 or 1" >&2
      exit 2
      ;;
  esac

  "$PYTHON" scripts/export_router_bundle.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$CHECKPOINT_EXPORT_DIR" \
    --tokenizer-source "$TOKENIZER_SOURCE" \
    --template-manifest "$TEMPLATE_MANIFEST" \
    --training-data "$ROUTER_DATA_DIR/retrieval_train.jsonl" \
    --validation-data "$ROUTER_DATA_DIR/retrieval_validation.jsonl" \
    --phase retrieval \
    --num-levels "$NUM_LEVELS" \
    --max-length "$ROUTER_MAX_LENGTH" \
    --seed "$ROUTER_SEED" \
    --base-model-name-or-path "$ROUTER_MODEL" \
    ${REPLAY_ARGS[@]+"${REPLAY_ARGS[@]}"} \
    ${TRUST_ARGS[@]+"${TRUST_ARGS[@]}"} \
    "$@"
  exit 0
fi

MODEL_PHASE=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' \
    "$MODEL_DIR/router_manifest.json"
)
TRAINING_DATA="$ROUTER_DATA_DIR/${MODEL_PHASE}_train.jsonl"
skillret_require_file "$TRAINING_DATA"

"$PYTHON" scripts/export_router_bundle.py \
  "${COMMON_ARGS[@]}" \
  --training-data "$TRAINING_DATA" \
  --phase "$MODEL_PHASE" \
  "$@"
