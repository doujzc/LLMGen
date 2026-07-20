#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

skillret_require_file "$PROCESSED_DIR/manifest.json"
skillret_require_file "$PROCESSED_DIR/collab_graph_train.npz"
skillret_require_file "$EMBEDDING_DIR/train.npy"
skillret_require_file "$EMBEDDING_DIR/manifest.json"

read -r -a BRANCHING_FACTOR_ARGS <<< "$BRANCHING_FACTORS"
read -r -a SK_EPSILON_ARGS <<< "$SK_EPSILONS"
read -r -a RQ_LAYER_ARGS <<< "$RQ_LAYERS"
ARGS=(
  --data-root "$DATASET_DIR"
  --manifest-path "$PROCESSED_DIR/manifest.json"
  --embedding-path "$EMBEDDING_DIR/train.npy"
  --embedding-manifest-path "$EMBEDDING_DIR/manifest.json"
  --graph-path "$PROCESSED_DIR/collab_graph_train.npz"
  --output-dir "$STAGE1_DIR"
  --device "$DEVICE"
  --num-levels "$NUM_LEVELS"
  --branching-factors "${BRANCHING_FACTOR_ARGS[@]}"
  --sk-epsilons "${SK_EPSILON_ARGS[@]}"
  --layers "${RQ_LAYER_ARGS[@]}"
  --e-dim "$TOKENIZER_E_DIM"
  --epochs "$TOKENIZER_EPOCHS"
  --batch-size "$TOKENIZER_BATCH_SIZE"
  --lr "$TOKENIZER_LR"
  --graph-lambda "$TOKENIZER_GRAPH_LAMBDA"
  --amp-dtype "$TOKENIZER_AMP_DTYPE"
  --codebook-version "$CODEBOOK_VERSION"
)
if [[ -n "$TOKENIZER_RESUME" ]]; then
  ARGS+=(--resume "$TOKENIZER_RESUME")
fi

"$PYTHON" scripts/train_tokenizer.py "${ARGS[@]}" "$@"
