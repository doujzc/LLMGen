#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ $# -lt 4 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/incremental/03_train_lora.sh \
    SOURCE_ROUTER_DIR CANDIDATE_STATE_DIR TRAINING_DATA_DIR OUTPUT_DIR \
    [extra train_router.py arguments]
EOF
  exit 2
fi

SOURCE_ROUTER_DIR="$1"
CANDIDATE_STATE_DIR="$2"
TRAINING_DATA_DIR="$3"
OUTPUT_DIR="$4"
shift 4

if [[ -x .venv/bin/python ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required input does not exist: $1" >&2
    exit 2
  fi
}

require_file "$CANDIDATE_STATE_DIR/skill_decode_map.json"
require_file "$CANDIDATE_STATE_DIR/virtual_tokens.txt"
require_file "$TRAINING_DATA_DIR/memorization_train.jsonl"
require_file "$TRAINING_DATA_DIR/retrieval_train.jsonl"
require_file "$TRAINING_DATA_DIR/manifest.json"

NUM_LEVELS="$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["num_levels"])' \
  "$CANDIDATE_STATE_DIR/skill_decode_map.json")"

MODEL_ARGS=()
LORA_ARGS=()
if [[ -f "$SOURCE_ROUTER_DIR/adapter_config.json" ]]; then
  BASE_MODEL="${INCREMENTAL_BASE_MODEL:-}"
  if [[ -z "$BASE_MODEL" ]]; then
    BASE_MODEL="$("$PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["base_model_name_or_path"])' \
      "$SOURCE_ROUTER_DIR/adapter_config.json")"
  fi
  MODEL_ARGS=(
    --model-name-or-path "$BASE_MODEL"
    --adapter-name-or-path "$SOURCE_ROUTER_DIR"
  )
else
  require_file "$SOURCE_ROUTER_DIR/config.json"
  MODEL_ARGS=(--model-name-or-path "$SOURCE_ROUTER_DIR")
  LORA_ARGS=(
    --lora
    --lora-r "${INCREMENTAL_LORA_R:-16}"
    --lora-alpha "${INCREMENTAL_LORA_ALPHA:-32}"
    --lora-dropout "${INCREMENTAL_LORA_DROPOUT:-0.05}"
    --lora-target-modules \
      "${INCREMENTAL_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
    # The trained source router already owns all virtual-token embeddings.
    --lora-modules-to-save none
  )
fi

PRECISION_ARGS=()
case "${INCREMENTAL_PRECISION:-bf16}" in
  bf16) PRECISION_ARGS=(--bf16) ;;
  fp16) PRECISION_ARGS=(--fp16) ;;
  fp32|none) ;;
  *)
    echo "INCREMENTAL_PRECISION must be bf16, fp16, or fp32" >&2
    exit 2
    ;;
esac

TRUST_ARGS=()
case "${INCREMENTAL_TRUST_REMOTE_CODE:-0}" in
  1|true|yes) TRUST_ARGS=(--trust-remote-code) ;;
  0|false|no) ;;
  *)
    echo "INCREMENTAL_TRUST_REMOTE_CODE must be 0 or 1" >&2
    exit 2
    ;;
esac

echo "[incremental 03a] memorize the new skill document"
echo "[incremental 03b] fit ${INCREMENTAL_NUM_QUERIES_HINT:-~10} direct retrieval queries"
"$PYTHON" scripts/train_router.py \
  "${MODEL_ARGS[@]}" \
  --virtual-tokens "$CANDIDATE_STATE_DIR/virtual_tokens.txt" \
  --output-dir "$OUTPUT_DIR" \
  --stage both \
  --memorization-train "$TRAINING_DATA_DIR/memorization_train.jsonl" \
  --retrieval-train "$TRAINING_DATA_DIR/retrieval_train.jsonl" \
  --num-levels "$NUM_LEVELS" \
  --max-length "${INCREMENTAL_MAX_LENGTH:-1024}" \
  --per-device-train-batch-size "${INCREMENTAL_BATCH_SIZE:-1}" \
  --per-device-eval-batch-size "${INCREMENTAL_EVAL_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${INCREMENTAL_GRADIENT_ACCUMULATION_STEPS:-1}" \
  --memorization-epochs "${INCREMENTAL_MEMORIZATION_EPOCHS:-20}" \
  --retrieval-epochs "${INCREMENTAL_RETRIEVAL_EPOCHS:-10}" \
  --memorization-learning-rate "${INCREMENTAL_MEMORIZATION_LR:-1e-4}" \
  --retrieval-learning-rate "${INCREMENTAL_RETRIEVAL_LR:-1e-4}" \
  --weight-decay "${INCREMENTAL_WEIGHT_DECAY:-0}" \
  --warmup-ratio "${INCREMENTAL_WARMUP_RATIO:-0}" \
  --logging-steps "${INCREMENTAL_LOGGING_STEPS:-1}" \
  --save-steps "${INCREMENTAL_SAVE_STEPS:-1000}" \
  --eval-steps "${INCREMENTAL_EVAL_STEPS:-1000}" \
  --save-total-limit "${INCREMENTAL_SAVE_TOTAL_LIMIT:-1}" \
  "${LORA_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  "${TRUST_ARGS[@]}" \
  "$@"

echo "[incremental 04] attach the active candidate state to the final adapter"
"$PYTHON" scripts/incremental/04_finalize_adapter.py \
  --model-dir "$OUTPUT_DIR/retrieval" \
  --candidate-state-dir "$CANDIDATE_STATE_DIR" \
  --training-data-dir "$TRAINING_DATA_DIR"

echo "Incremental router ready: $OUTPUT_DIR/retrieval"
