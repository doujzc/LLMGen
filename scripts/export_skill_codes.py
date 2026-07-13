#!/usr/bin/env python3
"""Encode train and unseen test skills with a frozen ToolWeaver RQ-VAE."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llmgen.neural.toolweaver import load_toolweaver_rqvae
from llmgen.skillret import (
    all_code_tokens,
    code_token,
    ordered_ids_sha256,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/skillret/processed"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/skillret/embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/skillret/index"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--token-format",
        default=None,
        help="Must match the checkpoint; defaults to its model_config token_format.",
    )
    return parser.parse_args()


def _collision_metrics(buckets: dict[str, list[str]], num_skills: int) -> dict[str, Any]:
    collision_skills = sum(len(members) for members in buckets.values() if len(members) > 1)
    return {
        "num_skills": num_skills,
        "num_paths": len(buckets),
        "collision_skill_rate": collision_skills / max(num_skills, 1),
        "max_bucket_size": max((len(members) for members in buckets.values()), default=0),
        "mean_bucket_size": num_skills / max(len(buckets), 1),
    }


def _export_split(
    *,
    split: str,
    model,
    embeddings_path: Path,
    catalog_path: Path,
    output_dir: Path,
    branching_factors: list[int],
    token_format: str,
    device: str,
    batch_size: int,
    normalize_embeddings: bool,
    expected_order_hash: str | None,
    expected_embedding_sha256: str | None,
) -> dict[str, Any]:
    ids = [str(row["skill_id"]) for row in read_jsonl(catalog_path)]
    actual_order_hash = ordered_ids_sha256(ids)
    if expected_order_hash and actual_order_hash != expected_order_hash:
        raise ValueError(
            f"{split} catalog ordering does not match the processed/embedding manifest"
        )
    actual_embedding_sha256 = sha256_file(embeddings_path)
    if expected_embedding_sha256 and actual_embedding_sha256 != expected_embedding_sha256:
        raise ValueError(f"{split} embedding SHA-256 does not match its manifest")
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError(
            f"{split} embedding/catalog mismatch: {embeddings.shape} versus {len(ids)} ids"
        )

    all_indices: list[list[int]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = torch.from_numpy(
                np.asarray(embeddings[start : start + batch_size], dtype=np.float32).copy()
            )
            if normalize_embeddings:
                batch = torch.nn.functional.normalize(batch, p=2, dim=1)
            batch = batch.to(device)
            indices = model.get_indices(batch, use_sk=False)
            if indices.ndim != 2 or indices.shape[1] != len(branching_factors):
                raise RuntimeError(
                    f"RQ-VAE returned {tuple(indices.shape)}, expected [batch,{len(branching_factors)}]"
                )
            all_indices.extend(indices.detach().cpu().to(torch.int64).tolist())

    buckets: dict[str, list[str]] = defaultdict(list)
    rows = []
    for skill_id, indices in zip(ids, all_indices, strict=True):
        for level, (index, size) in enumerate(zip(indices, branching_factors, strict=True), start=1):
            if index < 0 or index >= size:
                raise RuntimeError(f"invalid code index at level {level}: {index} not in [0,{size})")
        tokens = [code_token(level, index, token_format) for level, index in enumerate(indices, 1)]
        key = "/".join(str(index) for index in indices)
        buckets[key].append(skill_id)
        rows.append({"skill_id": skill_id, "indices": indices, "tokens": tokens})

    codes_path = output_dir / f"{split}_codes.jsonl"
    registry_path = output_dir / f"{split}_registry.json"
    write_jsonl(codes_path, rows)
    registry = {
        "schema_version": 1,
        "split": split,
        "num_levels": len(branching_factors),
        "branching_factors": branching_factors,
        "token_format": token_format,
        "ordered_skill_ids_sha256": actual_order_hash,
        "buckets": {key: sorted(value) for key, value in sorted(buckets.items())},
    }
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "codes": str(codes_path),
        "registry": str(registry_path),
        "codes_sha256": sha256_file(codes_path),
        "registry_sha256": sha256_file(registry_path),
        "ordered_skill_ids_sha256": registry["ordered_skill_ids_sha256"],
        "metrics": _collision_metrics(registry["buckets"], len(ids)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_toolweaver_rqvae(
        args.checkpoint,
        device=args.device,
    )
    model_config = checkpoint["model_config"]
    branching_factors = [int(value) for value in model_config["num_emb_list"]]
    if not branching_factors:
        raise ValueError("checkpoint has no RQ levels")
    checkpoint_token_format = str(
        model_config.get("token_format", "<SK_L{level}_{index}>")
    )
    if args.token_format is not None and args.token_format != checkpoint_token_format:
        raise ValueError("--token-format must match checkpoint model_config.token_format")
    token_format = checkpoint_token_format
    normalize_embeddings = bool(
        checkpoint.get("training_config", {}).get("normalize_embeddings", False)
    )

    processed_manifest_path = args.processed_dir / "manifest.json"
    embedding_manifest_path = args.embedding_dir / "manifest.json"
    processed_manifest = json.loads(processed_manifest_path.read_text(encoding="utf-8"))
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "test"):
        processed_order = processed_manifest["splits"][split]["hashes"][
            "ordered_skill_ids_sha256"
        ]
        embedded_order = embedding_manifest.get("ordered_skill_ids_sha256", {}).get(split)
        if embedded_order and embedded_order != processed_order:
            raise ValueError(f"{split} processed and embedding manifests disagree on row order")
    provenance = checkpoint.get("data_provenance", {})
    if provenance:
        train_order_hash = processed_manifest["splits"]["train"]["hashes"][
            "ordered_skill_ids_sha256"
        ]
        if provenance.get("ordered_skill_ids_sha256") != train_order_hash:
            raise ValueError("checkpoint was trained with a different ordered train catalog")
        if provenance.get("embedding_file_sha256") != sha256_file(
            args.embedding_dir / "train.npy"
        ):
            raise ValueError("checkpoint was trained with a different train embedding file")
        expected_embedding_manifest_sha256 = provenance.get(
            "embedding_manifest_file_sha256"
        )
        if (
            expected_embedding_manifest_sha256
            and expected_embedding_manifest_sha256
            != sha256_file(embedding_manifest_path)
        ):
            raise ValueError(
                "checkpoint was trained with a different embedding manifest"
            )
        if provenance.get("manifest_file_sha256") != sha256_file(processed_manifest_path):
            raise ValueError("checkpoint was trained with a different processed manifest")

    split_artifacts = {}
    for split in ("train", "test"):
        split_artifacts[split] = _export_split(
            split=split,
            model=model,
            embeddings_path=args.embedding_dir / f"{split}.npy",
            catalog_path=args.processed_dir / f"catalog_{split}.jsonl",
            output_dir=args.output_dir,
            branching_factors=branching_factors,
            token_format=token_format,
            device=args.device,
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            expected_order_hash=processed_manifest["splits"][split]["hashes"].get(
                "ordered_skill_ids_sha256"
            )
            or embedding_manifest.get("ordered_skill_ids_sha256", {}).get(split),
            expected_embedding_sha256=embedding_manifest.get("sha256", {}).get(split),
        )

    tokens = all_code_tokens(branching_factors, token_format)
    virtual_tokens_path = args.output_dir / "virtual_tokens.txt"
    virtual_tokens_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_config": model_config,
        "toolweaver_source": checkpoint.get("toolweaver_source"),
        "num_levels": len(branching_factors),
        "branching_factors": branching_factors,
        "token_format": token_format,
        "normalize_embeddings": normalize_embeddings,
        "num_virtual_tokens": len(tokens),
        "virtual_tokens": str(virtual_tokens_path),
        "splits": split_artifacts,
        "incremental_contract": "test skills encoded by frozen encoder and codebooks",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
