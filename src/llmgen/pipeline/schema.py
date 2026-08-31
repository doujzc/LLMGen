"""Versioned row contracts used at pipeline boundaries."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .io import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)


class PipelineSchemaError(ValueError):
    """Raised for malformed candidate or dataset boundary artifacts."""


def _stable_id(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    if normalized:
        return normalized
    # Non-Latin names remain readable and are valid closed-set identifiers.
    return re.sub(r"\s+", "-", name.strip())


def normalize_candidate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    id_policy: str,
    preserve_metadata: bool,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for source_line, raw in enumerate(rows, start=1):
        name = " ".join(str(raw.get("name") or "").split())
        description = str(
            raw.get("description") or raw.get("desc") or ""
        ).strip()
        raw_id = str(raw.get("id") or raw.get("skill_id") or "").strip()
        if not raw_id and id_policy == "explicit_or_name":
            raw_id = _stable_id(name)
        missing = [
            field
            for field, value in (
                ("id", raw_id),
                ("name", name),
                ("description", description),
            )
            if not value
        ]
        if missing:
            errors.append(
                f"line {source_line}: empty {', '.join(missing)}"
            )
            continue
        metadata = raw.get("metadata") if preserve_metadata else {}
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            errors.append(f"line {source_line}: metadata must be an object")
            continue
        normalized.append(
            {
                "skill_id": raw_id,
                "name": name,
                "description": description,
                "metadata": dict(metadata),
                "source_line": source_line,
            }
        )
    duplicates = sorted(
        skill_id
        for skill_id, count in Counter(
            row["skill_id"] for row in normalized
        ).items()
        if count > 1
    )
    if duplicates:
        errors.append("duplicate candidate IDs: " + ", ".join(duplicates[:10]))
    if not normalized:
        errors.append("candidate list is empty")
    if errors:
        raise PipelineSchemaError("invalid candidates: " + "; ".join(errors[:20]))
    return normalized


def read_candidate_file(
    path: str | Path,
    *,
    id_policy: str,
    preserve_metadata: bool,
) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise PipelineSchemaError(
                    f"invalid candidate JSON at {source}:{number}"
                ) from error
            if not isinstance(value, Mapping):
                raise PipelineSchemaError(
                    f"candidate at {source}:{number} must be an object"
                )
            rows.append(dict(value))
    return normalize_candidate_rows(
        rows,
        id_policy=id_policy,
        preserve_metadata=preserve_metadata,
    )


def catalog_rows(candidates: Iterable[Mapping[str, Any]], *, source: str) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "skill_id": str(row["skill_id"]),
            "owner": "generic-candidates",
            "slug": str(row["skill_id"]),
            "display_name": str(row["name"]),
            "name": str(row["name"]),
            "summary": None,
            "description": str(row["description"]),
            "canonical_url": f"jsonl://{source}#{row['skill_id']}",
            "metadata": dict(row.get("metadata") or {}),
            "description_provenance": {
                "type": "generic_candidates_jsonl",
                "source_skill_id": str(row["skill_id"]),
                "source_url": f"jsonl://{source}#{row['skill_id']}",
            },
        }
        for rank, row in enumerate(candidates, start=1)
    ]


def ensure_ordered_qrels(dataset_dir: str | Path) -> dict[str, Any]:
    """Rewrite qrels deterministically with explicit position fields.

    Existing exporter order is preserved semantically by taking the ordered
    ``query.skill_ids`` field as authoritative.  The dataset manifest artifact
    table is updated atomically after the rewrite.
    """

    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise PipelineSchemaError("dataset manifest must be an object")
    details: dict[str, Any] = {}
    for split in ("train", "validation", "test", "alignment"):
        query_path = root / f"queries_{split}.jsonl"
        qrel_path = root / f"qrels_{split}.jsonl"
        if not query_path.exists() and not qrel_path.exists():
            continue
        if not query_path.is_file() or not qrel_path.is_file():
            raise PipelineSchemaError(
                f"queries/qrels must both exist for {split}"
            )
        queries = read_jsonl(query_path)
        source_qrels = read_jsonl(qrel_path)
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for row in source_qrels:
            query_id = str(row.get("query_id") or "")
            skill_id = str(row.get("skill_id") or "")
            if not query_id or not skill_id:
                raise PipelineSchemaError(f"invalid {split} qrel")
            if skill_id in grouped.setdefault(query_id, {}):
                raise PipelineSchemaError(
                    f"duplicate {split} qrel: {(query_id, skill_id)}"
                )
            grouped[query_id][skill_id] = row
        ordered: list[dict[str, Any]] = []
        seen_queries: set[str] = set()
        for query in queries:
            query_id = str(query.get("id") or query.get("query_id") or "")
            raw_targets = query.get("skill_ids")
            if not query_id or query_id in seen_queries:
                raise PipelineSchemaError(f"invalid or duplicate {split} query")
            if not isinstance(raw_targets, list) or not raw_targets:
                raise PipelineSchemaError(
                    f"query {query_id} has no ordered skill_ids"
                )
            targets = [str(value) for value in raw_targets]
            if len(set(targets)) != len(targets):
                raise PipelineSchemaError(
                    f"query {query_id} repeats a target skill"
                )
            source = grouped.get(query_id, {})
            if set(source) != set(targets):
                raise PipelineSchemaError(
                    f"{split} qrels disagree with ordered targets for {query_id}"
                )
            for position, skill_id in enumerate(targets):
                row = dict(source[skill_id])
                row["query_id"] = query_id
                row["skill_id"] = skill_id
                row["relevance"] = row.get("relevance", 1)
                row["position"] = position
                ordered.append(row)
            seen_queries.add(query_id)
        orphan = sorted(set(grouped).difference(seen_queries))
        if orphan:
            raise PipelineSchemaError(
                f"orphan {split} qrels: {orphan[0]}"
            )
        atomic_write_jsonl(qrel_path, ordered)
        details[split] = {
            "query_count": len(queries),
            "qrel_count": len(ordered),
            "ordered": True,
        }

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PipelineSchemaError("dataset manifest has no artifact table")
    for name in list(artifacts):
        path = root / name
        if path.is_file():
            artifacts[name] = {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest["ordered_qrels_schema_version"] = 1
    manifest["ordered_qrels"] = details
    manifest["ordered_qrels_updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    return details


def validate_ordered_qrels(dataset_dir: str | Path) -> None:
    root = Path(dataset_dir)
    for split in ("train", "validation", "test", "alignment"):
        queries_path = root / f"queries_{split}.jsonl"
        qrels_path = root / f"qrels_{split}.jsonl"
        if not queries_path.exists() and not qrels_path.exists():
            continue
        queries = read_jsonl(queries_path)
        qrels = read_jsonl(qrels_path)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in qrels:
            grouped.setdefault(str(row.get("query_id") or ""), []).append(row)
        for query in queries:
            query_id = str(query.get("id") or query.get("query_id") or "")
            expected = [str(value) for value in query.get("skill_ids") or []]
            rows = sorted(
                grouped.get(query_id, []),
                key=lambda row: int(row.get("position", -1)),
            )
            positions = [int(row.get("position", -1)) for row in rows]
            actual = [str(row.get("skill_id") or "") for row in rows]
            if positions != list(range(len(expected))) or actual != expected:
                raise PipelineSchemaError(
                    f"ordered qrels disagree for {split} query {query_id}"
                )
