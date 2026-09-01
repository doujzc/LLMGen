#!/usr/bin/env python3
"""Validate and prepare a versioned closed-set Agent Skill routing dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from llmgen.pipeline.ledger import EmbeddingRecord, JsonlShardLedger
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
    parser.add_argument("--dataset-name", default="closedset-router-v1")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
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
        raise FileNotFoundError(f"required closed-set training file is missing: {path}")
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
    train_semantic_positives: dict[str, set[str]] = {}
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
            intent_mode = str(row.get("intent_mode") or "explicit")
            if intent_mode not in {"explicit", "implicit"}:
                raise ValueError(f"query {query_id} has invalid intent_mode")
            implicit_ids = list(map(str, row.get("implicit_skill_ids") or []))
            if not set(implicit_ids) <= targets:
                raise ValueError(f"query {query_id} has a non-target implicit skill")
            target_intents = row.get("target_intents") or {
                skill_id: "implicit" if skill_id in implicit_ids else "explicit"
                for skill_id in targets
            }
            if not isinstance(target_intents, dict) or set(map(str, target_intents)) != targets:
                raise ValueError(f"query {query_id} has incomplete target intents")
            expected_intents = {
                skill_id: "implicit" if skill_id in implicit_ids else "explicit"
                for skill_id in targets
            }
            if {str(key): str(value) for key, value in target_intents.items()} != expected_intents:
                raise ValueError(f"query {query_id} target intents disagree with implicit skills")
            rationales = row.get("implicit_rationales") or {}
            if not isinstance(rationales, dict) or set(map(str, rationales)) != set(implicit_ids):
                raise ValueError(f"query {query_id} has incomplete implicit rationales")
            if intent_mode == "explicit" and implicit_ids:
                raise ValueError(f"explicit query {query_id} declares implicit targets")
            if intent_mode == "implicit" and not 1 <= len(implicit_ids) < len(targets):
                raise ValueError(f"implicit query {query_id} needs explicit and implicit targets")
            workflow_id = str(row.get("workflow_id") or "")
            previous = workflow_splits.setdefault(workflow_id, split)
            if not workflow_id or previous != split:
                raise ValueError(f"workflow crosses query splits: {workflow_id}")
            query_ids.add(query_id)
            positives[query_id] = targets
            if split == "train":
                source_query_id = str(row.get("source_query_id") or query_id)
                previous_targets = train_semantic_positives.setdefault(
                    source_query_id, targets
                )
                if previous_targets != targets:
                    raise ValueError(
                        f"target-order variants disagree for {source_query_id}"
                    )

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
    train_positive_counts: Counter[str] = Counter(
        skill_id
        for targets in train_semantic_positives.values()
        for skill_id in targets
    )
    alignment_queries_path = dataset_dir / "queries_alignment.jsonl"
    alignment_qrels_path = dataset_dir / "qrels_alignment.jsonl"
    if alignment_queries_path.is_file() != alignment_qrels_path.is_file():
        raise FileNotFoundError(
            "alignment queries/qrels must either both exist or both be absent"
        )
    alignment_positive_counts: Counter[str] = Counter()
    if alignment_queries_path.is_file():
        alignment_queries = _load_rows(alignment_queries_path)
        alignment_qrels = _load_rows(alignment_qrels_path)
        alignment_targets: dict[str, str] = {}
        for row in alignment_queries:
            query_id = str(row.get("id") or row.get("query_id") or "")
            targets = list(map(str, row.get("skill_ids") or []))
            if not query_id or query_id in alignment_targets or len(targets) != 1:
                raise ValueError(f"invalid single-skill alignment query: {query_id!r}")
            if targets[0] not in allowed:
                raise ValueError(
                    f"alignment query {query_id} references unknown skill: {targets[0]}"
                )
            alignment_targets[query_id] = targets[0]
        alignment_pairs: set[tuple[str, str]] = set()
        for row in alignment_qrels:
            query_id = str(row.get("query_id") or "")
            skill_id = str(row.get("skill_id") or "")
            pair = (query_id, skill_id)
            if (
                pair in alignment_pairs
                or alignment_targets.get(query_id) != skill_id
                or float(row.get("relevance", 1)) <= 0
            ):
                raise ValueError(f"invalid or duplicate alignment qrel: {pair}")
            alignment_pairs.add(pair)
        if len(alignment_pairs) != len(alignment_targets):
            raise ValueError("alignment qrels disagree with alignment queries")
        alignment_positive_counts.update(alignment_targets.values())
        counts["queries_alignment"] = len(alignment_queries)
        counts["qrels_alignment"] = len(alignment_qrels)
    combined_positive_counts = train_positive_counts + alignment_positive_counts
    required_positives = int(
        manifest.get("min_train_positives_per_skill_required", 1)
    )
    undercovered = {
        skill_id: combined_positive_counts[skill_id]
        for skill_id in sorted(allowed)
        if combined_positive_counts[skill_id] < required_positives
    }
    if undercovered:
        raise ValueError(
            f"candidate set contains skills with fewer than {required_positives} "
            "train positives: "
            + ", ".join(
                f"{skill_id}={count}"
                for skill_id, count in list(undercovered.items())[:10]
            )
        )
    recorded_below_minimum = manifest.get("skills_below_min_train_positives_count")
    if recorded_below_minimum is not None and int(recorded_below_minimum) != 0:
        raise ValueError("manifest reports candidates below train-positive minimum")
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
    normalized = []
    for row in rows:
        value = {
            "query_id": str(row.get("id") or row.get("query_id")),
            "query": str(row["query"]),
            "skill_ids": [str(value) for value in row["skill_ids"]],
            "k": len(row["skill_ids"]),
        }
        if row.get("source_query_id"):
            value["source_query_id"] = str(row["source_query_id"])
        if row.get("target_order_variant") is not None:
            value["target_order_variant"] = int(row["target_order_variant"])
        normalized.append(value)
    return normalized


def _normalize_qrel(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one qrel without discarding its optional target-order position."""

    normalized = {
        "query_id": str(row["query_id"]),
        "skill_id": str(row["skill_id"]),
        "relevance": float(row.get("relevance", 1)),
    }
    if "position" in row:
        normalized["position"] = row["position"]
    return normalized


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
    ledger: JsonlShardLedger | None = None,
) -> tuple[int, int]:
    chunks: list[np.ndarray] = []
    written = 0
    for batch in _batches(catalog, batch_size=batch_size, max_batch_chars=max_batch_chars):
        if ledger is None:
            chunk = model.encode(
                [str(row["text"]) for row in batch],
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            pending = [
                EmbeddingRecord.from_text(
                    "candidate-catalog",
                    str(row["text"]),
                    status="failed",
                    item_key=str(row["skill_id"]),
                    model=model.config.model,
                    metadata={"skill_id": str(row["skill_id"])},
                )
                for row in batch
            ]
            while True:
                scheduled = ledger.schedule_embeddings(pending)
                if not scheduled.records:
                    break
                try:
                    values = model.encode(
                        [record.input_text for record in scheduled.records],
                        batch_size=len(scheduled.records),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                except Exception as error:
                    ledger.record_embeddings(
                        [
                            EmbeddingRecord.from_text(
                                record.namespace,
                                record.input_text,
                                status="failed",
                                item_key=record.item_key,
                                model=record.model,
                                error={"type": type(error).__name__},
                                metadata=record.metadata,
                            )
                            for record in scheduled.records
                        ]
                    )
                    raise
                ledger.record_embeddings(
                    [
                        EmbeddingRecord.from_text(
                            record.namespace,
                            record.input_text,
                            status="succeeded",
                            item_key=record.item_key,
                            model=record.model,
                            vector=values[index].tolist(),
                            metadata=record.metadata,
                        )
                        for index, record in enumerate(scheduled.records)
                    ]
                )
            cached_vectors = []
            for record in pending:
                cached = ledger.successful_embedding(record.embedding_id)
                if cached is None or cached.vector is None:
                    raise RuntimeError(
                        f"embedding ledger has no vector for {record.embedding_id}"
                    )
                cached_vectors.append(cached.vector)
            chunk = np.asarray(cached_vectors, dtype=np.float32)
        chunks.append(np.asarray(chunk, dtype=np.float32))
        written += len(batch)
        print(f"embedded {written}/{len(catalog)} skills", flush=True)
    if not chunks:
        raise ValueError("cannot embed an empty skill catalog")
    values = np.concatenate(chunks, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
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

    def manifest_path(path: Path) -> str:
        """Record a path relative to the movable processed artifact root."""

        return Path(os.path.relpath(path, args.processed_dir)).as_posix()

    catalog = [_catalog_row(row, args.max_skill_chars) for row in validated["skills"]]
    catalog_path = args.processed_dir / "catalog_train.jsonl"
    write_jsonl(catalog_path, catalog)
    split_details: dict[str, Any] = {}
    normalized_queries_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        queries_path = args.processed_dir / f"queries_{split}.jsonl"
        qrels_path = args.processed_dir / f"qrels_{split}.jsonl"
        queries = _normalize_queries(_load_rows(args.dataset_dir / f"queries_{split}.jsonl"))
        normalized_queries_by_split[split] = queries
        qrels = [
            _normalize_qrel(row)
            for row in _load_rows(args.dataset_dir / f"qrels_{split}.jsonl")
        ]
        write_jsonl(queries_path, queries)
        write_jsonl(qrels_path, qrels)
        split_details[split] = {
            "counts": {"skills": len(catalog), "queries": len(queries), "qrels": len(qrels)},
            "files": {
                "catalog": manifest_path(catalog_path),
                "queries": manifest_path(queries_path),
                "qrels": manifest_path(qrels_path),
            },
            "hashes": {
                "ordered_skill_ids_sha256": ordered_ids_sha256(validated["skill_ids"]),
                "catalog_sha256": sha256_file(catalog_path),
                "queries_sha256": sha256_file(queries_path),
                "qrels_sha256": sha256_file(qrels_path),
            },
        }

    alignment_source_queries = args.dataset_dir / "queries_alignment.jsonl"
    alignment_source_qrels = args.dataset_dir / "qrels_alignment.jsonl"
    if alignment_source_queries.is_file() != alignment_source_qrels.is_file():
        raise FileNotFoundError("alignment queries/qrels must either both exist or both be absent")
    if alignment_source_queries.is_file():
        alignment_queries = _normalize_queries(_load_rows(alignment_source_queries))
        alignment_qrels = [
            _normalize_qrel(row) for row in _load_rows(alignment_source_qrels)
        ]
        if any(len(row["skill_ids"]) != 1 for row in alignment_queries):
            raise ValueError("alignment query must target exactly one skill")
        alignment_query_path = args.processed_dir / "queries_alignment.jsonl"
        alignment_qrel_path = args.processed_dir / "qrels_alignment.jsonl"
        write_jsonl(alignment_query_path, alignment_queries)
        write_jsonl(alignment_qrel_path, alignment_qrels)
        split_details["alignment"] = {
            "counts": {
                "skills": len(catalog),
                "queries": len(alignment_queries),
                "qrels": len(alignment_qrels),
            },
            "files": {
                "catalog": manifest_path(catalog_path),
                "queries": manifest_path(alignment_query_path),
                "qrels": manifest_path(alignment_qrel_path),
            },
            "hashes": {
                "ordered_skill_ids_sha256": ordered_ids_sha256(validated["skill_ids"]),
                "queries_sha256": sha256_file(alignment_query_path),
                "qrels_sha256": sha256_file(alignment_qrel_path),
            },
        }

    train_qrels = _load_rows(args.processed_dir / "qrels_train.jsonl")
    query_to_source = {
        str(row["query_id"]): str(row.get("source_query_id") or row["query_id"])
        for row in normalized_queries_by_split["train"]
    }
    semantic_pairs: set[tuple[str, str]] = set()
    semantic_train_qrels = []
    for row in train_qrels:
        source_query_id = query_to_source[str(row["query_id"])]
        pair = (source_query_id, str(row["skill_id"]))
        if pair in semantic_pairs:
            continue
        semantic_pairs.add(pair)
        semantic_train_qrels.append(
            {
                "query_id": source_query_id,
                "skill_id": str(row["skill_id"]),
                "relevance": float(row.get("relevance", 1)),
            }
        )
    source, target, weight = build_collaborative_edges(
        validated["skill_ids"], semantic_train_qrels
    )
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
        ledger_root = os.environ.get("LLMGEN_EMBEDDING_LEDGER_ROOT", "").strip()
        embedding_ledger = (
            JsonlShardLedger(
                ledger_root,
                batch_size=int(
                    os.environ.get(
                        "LLMGEN_EMBEDDING_LEDGER_BATCH_RECORDS",
                        str(args.batch_size),
                    )
                ),
            )
            if ledger_root
            else None
        )
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
                ledger=embedding_ledger,
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
            "ledger": (
                {
                    "path": str(embedding_ledger.root),
                    "stats": embedding_ledger.verify()["stats"]["embeddings"],
                }
                if embedding_ledger is not None
                else None
            ),
        }
        (args.embedding_dir / "manifest.json").write_text(
            json.dumps(embedding_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset_name,
        "candidate_policy": "shared closed set across train/validation/test queries",
        "source": {
            "path": manifest_path(args.dataset_dir),
            "manifest": manifest_path(args.dataset_dir / "manifest.json"),
            "manifest_sha256": sha256_file(args.dataset_dir / "manifest.json"),
            "validated_counts": validated["counts"],
        },
        "splits": split_details,
        "graph": {
            "path": manifest_path(graph_path),
            "num_nodes": len(catalog),
            "num_edges": int(weight.size),
            "normalization": "co_use_count/sqrt(skill_frequency_product)",
            "source_split": "train_qrels_only",
            "source_query_policy": "deduplicate_target_order_variants_by_source_query_id",
            "semantic_qrel_count": len(semantic_train_qrels),
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
