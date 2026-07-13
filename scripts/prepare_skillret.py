#!/usr/bin/env python3
"""Normalize SkillRet, embed skills, and build the train-only co-use graph."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from llmgen.skillret import (
    SKILLRET_REPO_ID,
    SKILLRET_REVISION,
    build_collaborative_edges,
    normalize_query,
    normalize_skill,
    ordered_ids_sha256,
    raw_path,
    read_jsonl,
    sha256_file,
    validate_raw_dataset,
    write_jsonl,
)


SKILLRET_EMBEDDING_MODEL = "ThakiCloud/SkillRet-Embedding-0.6B"
SKILLRET_EMBEDDING_REVISION = "0e10886e80a0aacc9efddc28282a258e2ab7eae1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/skillret"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/skillret/processed"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/skillret/embeddings"))
    parser.add_argument("--embedding-model", default=SKILLRET_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--embedding-max-seq-length", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-skills", type=int, default=None)
    parser.add_argument("--max-test-skills", type=int, default=None)
    parser.add_argument("--max-skill-chars", type=int, default=None)
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser.parse_args()


def _prepare_split(args: argparse.Namespace, split: str, limit: int | None) -> dict[str, Any]:
    catalog_path = args.processed_dir / f"catalog_{split}.jsonl"
    query_path = args.processed_dir / f"queries_{split}.jsonl"
    qrel_path = args.processed_dir / f"qrels_{split}.jsonl"

    skill_ids: list[str] = []

    def normalized_skills():
        for row_number, row in enumerate(read_jsonl(raw_path(args.dataset_dir, "skills", split))):
            if limit is not None and row_number >= limit:
                break
            skill = normalize_skill(row, max_chars=args.max_skill_chars)
            skill_ids.append(skill["skill_id"])
            yield skill

    write_jsonl(catalog_path, normalized_skills())
    allowed = set(skill_ids)

    qrels = [
        {
            "query_id": str(row["query_id"]),
            "skill_id": str(row["skill_id"]),
            "relevance": float(row.get("relevance", 1)),
        }
        for row in read_jsonl(raw_path(args.dataset_dir, "qrels", split))
        if str(row.get("skill_id", "")) in allowed and float(row.get("relevance", 1)) > 0
    ]
    valid_queries = {row["query_id"] for row in qrels}
    queries = []
    for row in read_jsonl(raw_path(args.dataset_dir, "queries", split)):
        normalized = normalize_query(row, allowed)
        if normalized is not None and normalized["query_id"] in valid_queries:
            queries.append(normalized)
    query_ids = {row["query_id"] for row in queries}
    qrels = [row for row in qrels if row["query_id"] in query_ids]
    write_jsonl(query_path, queries)
    write_jsonl(qrel_path, qrels)

    return {
        "skill_ids": skill_ids,
        "counts": {"skills": len(skill_ids), "queries": len(queries), "qrels": len(qrels)},
        "files": {
            "catalog": str(catalog_path),
            "queries": str(query_path),
            "qrels": str(qrel_path),
        },
        "hashes": {
            "ordered_skill_ids_sha256": ordered_ids_sha256(skill_ids),
            "catalog_sha256": sha256_file(catalog_path),
            "queries_sha256": sha256_file(query_path),
            "qrels_sha256": sha256_file(qrel_path),
        },
    }


def _embed_catalog(
    catalog_path: Path,
    output_path: Path,
    *,
    model: Any,
    batch_size: int,
) -> tuple[list[str], tuple[int, int], dict[str, Any]]:
    row_count = sum(1 for _ in read_jsonl(catalog_path))
    ids: list[str] = []
    token_lengths: list[int] = []
    max_seq_length = getattr(model, "max_seq_length", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = None
    batch: list[dict[str, Any]] = []
    written = 0
    for row in read_jsonl(catalog_path):
        batch.append(row)
        if len(batch) < batch_size:
            continue
        texts = [str(item["text"]) for item in batch]
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            untruncated = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                verbose=False,
            )
            token_lengths.extend(len(ids) for ids in untruncated["input_ids"])
        chunk = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        chunk = np.asarray(chunk, dtype=np.float32)
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.float32, shape=(row_count, chunk.shape[1])
            )
        embeddings[written : written + len(batch)] = chunk
        ids.extend(str(item["skill_id"]) for item in batch)
        written += len(batch)
        batch.clear()
        print(f"embedded {written}/{row_count} from {catalog_path.name}")

    if batch:
        texts = [str(item["text"]) for item in batch]
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            untruncated = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                verbose=False,
            )
            token_lengths.extend(len(ids) for ids in untruncated["input_ids"])
        chunk = np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.float32, shape=(row_count, chunk.shape[1])
            )
        embeddings[written : written + len(batch)] = chunk
        ids.extend(str(item["skill_id"]) for item in batch)
        written += len(batch)
        print(f"embedded {written}/{row_count} from {catalog_path.name}")

    if embeddings is None:
        np.save(output_path, np.empty((0, 0), dtype=np.float32), allow_pickle=False)
        shape = (0, 0)
    else:
        embeddings.flush()
        shape = tuple(int(value) for value in embeddings.shape)
        del embeddings
    if token_lengths:
        lengths = np.asarray(token_lengths, dtype=np.int64)
        truncation_count = (
            int(np.count_nonzero(lengths > int(max_seq_length)))
            if max_seq_length is not None
            else 0
        )
        token_statistics = {
            "effective_max_seq_length": (
                int(max_seq_length) if max_seq_length is not None else None
            ),
            "untruncated_tokens": {
                "median": float(np.median(lengths)),
                "p95": float(np.percentile(lengths, 95)),
                "max": int(lengths.max()),
            },
            "truncated_documents": truncation_count,
            "truncation_rate": truncation_count / max(len(lengths), 1),
        }
    else:
        token_statistics = {
            "effective_max_seq_length": max_seq_length,
            "untruncated_tokens": None,
            "truncated_documents": None,
            "truncation_rate": None,
        }
    return ids, shape, token_statistics


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.processed_dir = args.processed_dir.expanduser().resolve()
    args.embedding_dir = args.embedding_dir.expanduser().resolve()
    full_run = args.max_train_skills is None and args.max_test_skills is None
    source = validate_raw_dataset(args.dataset_dir, strict_counts=full_run)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.embedding_dir.mkdir(parents=True, exist_ok=True)

    train = _prepare_split(args, "train", args.max_train_skills)
    test = _prepare_split(args, "test", args.max_test_skills)
    overlap = set(train["skill_ids"]) & set(test["skill_ids"])
    if overlap:
        raise ValueError("processed train/test skill pools overlap")

    qrels = read_jsonl(args.processed_dir / "qrels_train.jsonl")
    src, dst, weight = build_collaborative_edges(train["skill_ids"], qrels)
    graph_path = args.processed_dir / "collab_graph_train.npz"
    np.savez_compressed(
        graph_path,
        src=src,
        dst=dst,
        weight=weight,
        num_nodes=np.asarray(len(train["skill_ids"]), dtype=np.int64),
        ordered_skill_ids_sha256=np.asarray(train["hashes"]["ordered_skill_ids_sha256"]),
    )

    embedding_manifest: dict[str, Any] | None = None
    if not args.skip_embeddings:
        from sentence_transformers import SentenceTransformer

        requested_revision = args.embedding_revision
        if args.embedding_model == SKILLRET_EMBEDDING_MODEL and requested_revision is None:
            requested_revision = SKILLRET_EMBEDDING_REVISION
        model = SentenceTransformer(
            args.embedding_model,
            device=args.device,
            revision=requested_revision,
            trust_remote_code=args.trust_remote_code,
        )
        if args.embedding_max_seq_length is not None:
            if args.embedding_max_seq_length < 1:
                raise ValueError("--embedding-max-seq-length must be positive")
            model.max_seq_length = args.embedding_max_seq_length
        resolved_revision = None
        for module in model.modules():
            config = getattr(getattr(module, "auto_model", None), "config", None)
            resolved_revision = getattr(config, "_commit_hash", None)
            if resolved_revision:
                break
        shapes: dict[str, list[int]] = {}
        hashes: dict[str, str] = {}
        truncation: dict[str, Any] = {}
        for split, details in (("train", train), ("test", test)):
            ids, shape, token_statistics = _embed_catalog(
                Path(details["files"]["catalog"]),
                args.embedding_dir / f"{split}.npy",
                model=model,
                batch_size=args.batch_size,
            )
            if ids != details["skill_ids"]:
                raise RuntimeError(f"{split} embedding order changed")
            shapes[split] = list(shape)
            hashes[split] = sha256_file(args.embedding_dir / f"{split}.npy")
            truncation[split] = token_statistics
        embedding_manifest = {
            "model": args.embedding_model,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "normalized": True,
            "max_skill_chars": args.max_skill_chars,
            "shapes": shapes,
            "sha256": hashes,
            "tokenization": truncation,
            "ordered_skill_ids_sha256": {
                "train": train["hashes"]["ordered_skill_ids_sha256"],
                "test": test["hashes"]["ordered_skill_ids_sha256"],
            },
        }
        (args.embedding_dir / "manifest.json").write_text(
            json.dumps(embedding_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": SKILLRET_REPO_ID,
            "revision": SKILLRET_REVISION,
            "validated_counts": source["counts"],
        },
        "full_run": full_run,
        "splits": {
            "train": {key: value for key, value in train.items() if key != "skill_ids"},
            "test": {key: value for key, value in test.items() if key != "skill_ids"},
        },
        "graph": {
            "path": str(graph_path),
            "num_nodes": len(train["skill_ids"]),
            "num_edges": int(weight.size),
            "normalization": "co_use_count/sqrt(skill_frequency_product)",
            "source_split": "train_qrels_only",
            "sha256": sha256_file(graph_path),
        },
        "embeddings": embedding_manifest,
    }
    (args.processed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
