#!/usr/bin/env python3
"""Attach an updated candidate state to the trained incremental LoRA adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from llmgen.incremental import load_candidate_state, utc_now
from llmgen.router import RouterDataError
from llmgen.router_bundle import (
    BUNDLED_VIRTUAL_TOKENS_FILENAME,
    DECODE_MAP_FILENAME,
)
from llmgen.skillret import sha256_file


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--candidate-state-dir", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    router_manifest_path = model_dir / "router_manifest.json"
    if not router_manifest_path.is_file():
        raise RouterDataError(
            f"incremental output has no router_manifest.json: {model_dir}"
        )
    manifest = json.loads(router_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "retrieval":
        raise RouterDataError("incremental final model must be a retrieval phase output")

    candidate_map, decode_source, tokens_source = load_candidate_state(
        args.candidate_state_dir
    )
    training_manifest_path = (
        args.training_data_dir.expanduser().resolve() / "manifest.json"
    )
    if not training_manifest_path.is_file():
        raise RouterDataError(
            f"incremental training data has no manifest: {training_manifest_path}"
        )
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    skill_id = str(training_manifest.get("skill_id") or "")
    if not skill_id or skill_id not in candidate_map["skills"]:
        raise RouterDataError(
            "incremental training manifest does not identify an active new skill"
        )
    recorded_state_hash = (
        training_manifest.get("sources", {})
        .get("candidate_decode_map", {})
        .get("sha256")
    )
    actual_state_hash = sha256_file(decode_source)
    if recorded_state_hash != actual_state_hash:
        raise RouterDataError(
            "candidate state changed after incremental training data was built"
        )
    expected_token_hash = manifest.get("virtual_tokens_sha256")
    actual_token_hash = sha256_file(tokens_source)
    if expected_token_hash and expected_token_hash != actual_token_hash:
        raise RouterDataError(
            "incremental adapter and candidate state use different virtual tokens"
        )

    decode_destination = model_dir / DECODE_MAP_FILENAME
    tokens_destination = model_dir / BUNDLED_VIRTUAL_TOKENS_FILENAME
    _atomic_copy(decode_source, decode_destination)
    _atomic_copy(tokens_source, tokens_destination)
    manifest["decoder_artifacts"] = {
        "decode_map": DECODE_MAP_FILENAME,
        "decode_map_sha256": sha256_file(decode_destination),
        "virtual_tokens": BUNDLED_VIRTUAL_TOKENS_FILENAME,
        "virtual_tokens_sha256": sha256_file(tokens_destination),
        "num_skills": int(candidate_map["num_skills"]),
        "num_paths": int(candidate_map["num_paths"]),
        "num_levels": int(candidate_map["num_levels"]),
        "supervision": None,
        "provenance": candidate_map.get("provenance", {}),
    }
    manifest["stage1_checkpoint_sha256"] = (
        manifest.get("stage1_checkpoint_sha256")
        or candidate_map.get("provenance", {}).get("stage1_checkpoint_sha256")
    )
    manifest["incremental_update"] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "mode": "frozen_codebook_incremental_lora",
        "skill_id": skill_id,
        "candidate_state_revision": candidate_map.get(
            "incremental_state", {}
        ).get("revision"),
        "candidate_decode_map_sha256": actual_state_hash,
        "training_data_manifest": str(training_manifest_path),
        "training_data_manifest_sha256": sha256_file(training_manifest_path),
        "training_examples": training_manifest.get("examples"),
    }
    _atomic_json(router_manifest_path, manifest)
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "skill_id": skill_id,
                "num_skills": candidate_map["num_skills"],
                "num_paths": candidate_map["num_paths"],
                "decode_map_sha256": sha256_file(decode_destination),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
