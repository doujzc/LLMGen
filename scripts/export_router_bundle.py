#!/usr/bin/env python3
"""Attach portable skill decoding artifacts to an existing router dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.router import RouterDataError
from llmgen.router_bundle import dump_router_decoder_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--virtual-tokens", required=True)
    parser.add_argument(
        "--training-data",
        help="Router phase training JSONL used to annotate target supervision.",
    )
    parser.add_argument("--phase", choices=("memorization", "retrieval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise RouterDataError(f"model directory does not exist: {model_dir}")
    manifest_path = model_dir / "router_manifest.json"
    if not manifest_path.is_file():
        raise RouterDataError(
            f"router manifest does not exist; not a complete router dump: {model_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RouterDataError("router_manifest.json must contain an object")
    training_data = args.training_data or manifest.get("train_data")
    if training_data and not Path(training_data).is_file():
        raise RouterDataError(
            f"router training data does not exist: {training_data}; "
            "pass --training-data with its current location"
        )
    phase = args.phase or manifest.get("phase")
    artifacts = dump_router_decoder_artifacts(
        output_dir=model_dir,
        catalog_path=args.catalog,
        codes_path=args.codes,
        registry_path=args.registry,
        virtual_tokens_path=args.virtual_tokens,
        training_data_path=training_data,
        supervision_phase=str(phase) if phase else None,
    )
    manifest["decoder_artifacts"] = artifacts
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
