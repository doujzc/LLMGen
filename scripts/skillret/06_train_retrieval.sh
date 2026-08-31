#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
skillret_require_file "$INDEX_DIR/virtual_tokens.txt"
skillret_require_file "$PROCESSED_DIR/catalog_train.jsonl"
skillret_require_file "$INDEX_DIR/train_codes.jsonl"
skillret_require_file "$INDEX_DIR/train_registry.json"
skillret_require_file "$ROUTER_DATA_DIR/retrieval_train.jsonl"
skillret_require_file "$ROUTER_DATA_DIR/retrieval_validation.jsonl"
skillret_require_dir "$ROUTER_OUTPUT_DIR/memorization"
skillret_configure_router

if [[ "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0" && "${ROUTER_ALIGNMENT_EPOCHS:-0}" != "0.0" ]]; then
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/06a_train_alignment.sh"
fi

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/06b_train_retrieval.sh" "$@"
