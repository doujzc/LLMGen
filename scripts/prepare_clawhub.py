#!/usr/bin/env python3
"""Validate and prepare the versioned ClawHub closed-set routing dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from llmgen.embeddings import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
)
from llmgen.skillret import (
    build_collaborative_edges,
    ordered_ids_sha256,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/clawhub_training/final"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/clawhub_training/final/processed"),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/clawhub_training/final/embeddings"),
    )
    parser.add_argument("--embedding-provider", choices=("openai",), default="openai")
    parser.add_argument("--embedding-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
    )
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--embedding-timeout", type=float, default=600.0)
    parser.add_argument("--embedding-max-retries", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--embedding-max-batch-chars",
        type=int,
        default=12_000,
        help="Bound aggregate request size in addition to --batch-size.",
    )
    parser.add_argument("--max-skill-chars", type=int)
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required ClawHub training file is missing: {path}")
    return list(read_jsonl(path))


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("dataset manifest has no artifact integrity table")
    for name, expected in artifacts.items():
        path = dataset_dir / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        if int(expected.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"artifact size differs from manifest: {name}")
        if str(expected.get("sha256") or "") != sha256_file(path):
            raise ValueError(f"artifact SHA-256 differs from manifest: {name}")

    skills = _load_rows(dataset_dir / "skills.jsonl")
    skill_ids = [str(row.get("skill_id") or "") for row in skills]
    if any(not value for value in skill_ids) or len(set(skill_ids)) != len(skill_ids):
        raise ValueError("skills.jsonl contains missing or duplicate skill_id values")
    allowed = set(skill_ids)
    query_ids_by_split: dict[str, set[str]] = {}
    workflow_splits: dict[str, str] = {}
    counts: dict[str, int] = {"skills": len(skills)}

    for split in SPLITS:
        queries = _load_rows(dataset_dir / f"queries_{split}.jsonl")
        qrels = _load_rows(dataset_dir / f"qrels_{split}.jsonl")
        query_ids: set[str] = set()
        positives: dict[str, set[str]] = {}
        for row in queries:
            query_id = str(row.get("id") or row.get("query_id") or "")
            query = str(row.get("query") or "")
            raw_targets = row.get("skill_ids")
            if not query_id or query_id in query_ids or not query.strip():
                raise ValueError(f"invalid or duplicate {split} query: {query_id!r}")
            if not isinstance(raw_targets, list) or not 2 <= len(raw_targets) <= 4:
                raise ValueError(f"query {query_id} must have 2-4 target skills")
            targets = {str(value) for value in raw_targets}
            unknown = targets.difference(allowed)
            if unknown:
                raise ValueError(f"query {query_id} references unknown skill: {min(unknown)}")
            evidence = row.get("evidence")
            if not isinstance(evidence, dict) or set(map(str, evidence)) != targets:
                raise ValueError(f"query {query_id} has incomplete evidence")
            if any(str(span) not in query for span in evidence.values()):
                raise ValueError(f"query {query_id} has a non-verbatim evidence span")
            workflow_id = str(row.get("workflow_id") or "")
            previous = workflow_splits.setdefault(workflow_id, split)
            if not workflow_id or previous != split:
                raise ValueError(f"workflow crosses query splits: {workflow_id}")
            query_ids.add(query_id)
            positives[query_id] = targets

        qrel_targets: dict[str, set[str]] = defaultdict(set)
        pairs: set[tuple[str, str]] = set()
        for row in qrels:
            query_id = str(row.get("query_id") or "")
            skill_id = str(row.get("skill_id") or "")
            pair = (query_id, skill_id)
            if query_id not in query_ids or skill_id not in allowed or pair in pairs:
                raise ValueError(f"invalid or duplicate {split} qrel: {pair}")
            pairs.add(pair)
            if float(row.get("relevance", 1)) > 0:
                qrel_targets[query_id].add(skill_id)
        if dict(qrel_targets) != positives:
            raise ValueError(f"{split} qrels disagree with query skill_ids")
        query_ids_by_split[split] = query_ids
        counts[f"queries_{split}"] = len(queries)
        counts[f"qrels_{split}"] = len(qrels)

    for index, split in enumerate(SPLITS):
        for other in SPLITS[index + 1 :]:
            overlap = query_ids_by_split[split] & query_ids_by_split[other]
            if overlap:
                raise ValueError(f"query IDs overlap across {split}/{other}: {min(overlap)}")
    if int(manifest.get("candidate_count", -1)) != counts["skills"]:
        raise ValueError("manifest candidate_count disagrees with skills.jsonl")
    if manifest.get("split_query_counts") != {
        split: counts[f"queries_{split}"] for split in SPLITS
    }:
        raise ValueError("manifest query split counts disagree with data files")
    if manifest.get("split_qrel_counts") != {
        split: counts[f"qrels_{split}"] for split in SPLITS
    }:
        raise ValueError("manifest qrel split counts disagree with data files")
    return {"skills": skills, "skill_ids": skill_ids, "counts": counts}


def _catalog_row(row: Mapping[str, Any], max_chars: int | None) -> dict[str, Any]:
    name = str(row.get("name") or "").strip()
    capability = str(row.get("capability_zh") or "").strip()
    description = str(row.get("description") or "").strip()
    text = " | ".join(value for value in (name, capability, description) if value)
    if max_chars is not None and max_chars > 0:
        text = text[:max_chars]
    if not text:
        raise ValueError(f"skill {row.get('skill_id')} has no training text")
    return {
        "skill_id": str(row["skill_id"]),
        "name": name,
        "description": description,
        "text": text,
        "domain": row.get("domain"),
        "roles": row.get("roles") or [],
        "capability_zh": capability,
        "mobile_fit": row.get("mobile_fit"),
        "rank": row.get("rank"),
        "source_url": row.get("canonical_url"),
    }


def _normalize_queries(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": str(row.get("id") or row.get("query_id")),
            "query": str(row["query"]),
            "skill_ids": [str(value) for value in row["skill_ids"]],
            "k": len(row["skill_ids"]),
        }
        for row in rows
    ]


def _batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_batch_chars: int,
) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    characters = 0
    for row in rows:
        size = len(str(row["text"]))
        if batch and (len(batch) >= batch_size or characters + size > max_batch_chars):
            yield batch
            batch = []
            characters = 0
        batch.append(row)
        characters += size
    if batch:
        yield batch


def _embed_catalog(
    catalog: list[dict[str, Any]],
    output_path: Path,
    model: OpenAIEmbeddingModel,
    *,
    batch_size: int,
    max_batch_chars: int,
) -> tuple[int, int]:
    chunks: list[np.ndarray] = []
    written = 0
    for batch in _batches(catalog, batch_size=batch_size, max_batch_chars=max_batch_chars):
        chunk = model.encode(
            [str(row["text"]) for row in batch],
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        chunks.append(np.asarray(chunk, dtype=np.float32))
        written += len(batch)
        print(f"embedded {written}/{len(catalog)} skills", flush=True)
    if not chunks:
        raise ValueError("cannot embed an empty skill catalog")
    values = np.concatenate(chunks, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, values, allow_pickle=False)
    return int(values.shape[0]), int(values.shape[1])


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.embedding_max_batch_chars < 1:
        raise ValueError("embedding batch limits must be positive")
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.processed_dir = args.processed_dir.expanduser().resolve()
    args.embedding_dir = args.embedding_dir.expanduser().resolve()
    validated = validate_dataset(args.dataset_dir)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.embedding_dir.mkdir(parents=True, exist_ok=True)

    catalog = [_catalog_row(row, args.max_skill_chars) for row in validated["skills"]]
    catalog_path = args.processed_dir / "catalog_train.jsonl"
    write_jsonl(catalog_path, catalog)
    split_details: dict[str, Any] = {}
    for split in SPLITS:
        queries_path = args.processed_dir / f"queries_{split}.jsonl"
        qrels_path = args.processed_dir / f"qrels_{split}.jsonl"
        queries = _normalize_queries(_load_rows(args.dataset_dir / f"queries_{split}.jsonl"))
        qrels = [
            {
                "query_id": str(row["query_id"]),
                "skill_id": str(row["skill_id"]),
                "relevance": float(row.get("relevance", 1)),
            }
            for row in _load_rows(args.dataset_dir / f"qrels_{split}.jsonl")
        ]
        write_jsonl(queries_path, queries)
        write_jsonl(qrels_path, qrels)
        split_details[split] = {
            "counts": {"skills": len(catalog), "queries": len(queries), "qrels": len(qrels)},
            "files": {
                "catalog": str(catalog_path),
                "queries": str(queries_path),
                "qrels": str(qrels_path),
            },
            "hashes": {
                "ordered_skill_ids_sha256": ordered_ids_sha256(validated["skill_ids"]),
                "catalog_sha256": sha256_file(catalog_path),
                "queries_sha256": sha256_file(queries_path),
                "qrels_sha256": sha256_file(qrels_path),
            },
        }

    train_qrels = _load_rows(args.processed_dir / "qrels_train.jsonl")
    source, target, weight = build_collaborative_edges(validated["skill_ids"], train_qrels)
    graph_path = args.processed_dir / "collab_graph_train.npz"
    np.savez_compressed(
        graph_path,
        src=source,
        dst=target,
        weight=weight,
        num_nodes=np.asarray(len(catalog), dtype=np.int64),
        ordered_skill_ids_sha256=np.asarray(ordered_ids_sha256(validated["skill_ids"])),
    )

    embedding_manifest = None
    if not args.skip_embeddings:
        model = OpenAIEmbeddingModel(
            OpenAIEmbeddingConfig(
                model=args.embedding_model,
                base_url=args.embedding_base_url,
                api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                dimensions=args.embedding_dimensions,
                timeout=args.embedding_timeout,
                max_retries=args.embedding_max_retries,
            )
        )
        embedding_path = args.embedding_dir / "train.npy"
        try:
            shape = _embed_catalog(
                catalog,
                embedding_path,
                model,
                batch_size=args.batch_size,
                max_batch_chars=args.embedding_max_batch_chars,
            )
        finally:
            model.close()
        embedding_manifest = {
            "provider": "openai",
            "model": args.embedding_model,
            "base_url": args.embedding_base_url,
            "requested_dimensions": args.embedding_dimensions,
            "normalized": True,
            "max_skill_chars": args.max_skill_chars,
            "batch_size": args.batch_size,
            "max_batch_chars": args.embedding_max_batch_chars,
            "shapes": {"train": list(shape)},
            "sha256": {"train": sha256_file(embedding_path)},
            "ordered_skill_ids_sha256": {
                "train": ordered_ids_sha256(validated["skill_ids"])
            },
        }
        (args.embedding_dir / "manifest.json").write_text(
            json.dumps(embedding_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "clawhub-router-v1",
        "candidate_policy": "shared closed set across train/validation/test queries",
        "source": {
            "path": str(args.dataset_dir),
            "manifest": str(args.dataset_dir / "manifest.json"),
            "manifest_sha256": sha256_file(args.dataset_dir / "manifest.json"),
            "validated_counts": validated["counts"],
        },
        "splits": split_details,
        "graph": {
            "path": str(graph_path),
            "num_nodes": len(catalog),
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
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
