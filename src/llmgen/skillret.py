"""SkillRet dataset preparation utilities.

The functions in this module intentionally keep the heavy ML dependencies out of
the import path.  Dataset validation and collaborative graph construction can be
tested with only NumPy; Sentence Transformers is imported by the preparation
script only when embeddings are requested.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


SKILLRET_REPO_ID = "ThakiCloud/SKILLRET"
SKILLRET_REVISION = "7cae7cfbad2b0e1ebc9170892f568993aae543b0"
SKILLRET_EXPECTED_COUNTS = {
    "skills/full": 17_810,
    "skills/train": 10_123,
    "skills/test": 6_660,
    "queries/train": 63_259,
    "queries/test": 4_997,
    "qrels/train": 127_190,
    "qrels/test": 8_347,
}


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from *path* with useful line-level errors."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows atomically and return their count."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    temporary.replace(path)
    return count


def count_jsonl(path: str | Path) -> int:
    with Path(path).open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(ids: Sequence[str]) -> str:
    digest = sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def raw_path(dataset_dir: str | Path, kind: str, split: str) -> Path:
    path = Path(dataset_dir) / "data" / kind / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"SkillRet file is missing: {path}")
    return path


def validate_raw_dataset(
    dataset_dir: str | Path,
    *,
    strict_counts: bool = True,
) -> dict[str, Any]:
    """Validate the official six split files and leakage-sensitive invariants."""

    dataset_dir = Path(dataset_dir)
    counts: dict[str, int] = {}
    skill_ids: dict[str, set[str]] = {}
    query_ids: dict[str, set[str]] = {}
    query_positives: dict[str, dict[str, set[str]]] = {}

    for split in ("train", "test"):
        skills_file = raw_path(dataset_dir, "skills", split)
        queries_file = raw_path(dataset_dir, "queries", split)
        qrels_file = raw_path(dataset_dir, "qrels", split)

        counts[f"skills/{split}"] = count_jsonl(skills_file)
        counts[f"queries/{split}"] = count_jsonl(queries_file)
        counts[f"qrels/{split}"] = count_jsonl(qrels_file)

        skill_ids[split] = set()
        for row in read_jsonl(skills_file):
            value = str(row.get("id", "")).strip()
            if not value or value in skill_ids[split]:
                raise ValueError(f"missing or duplicate skill id in {skills_file}: {value!r}")
            skill_ids[split].add(value)

        query_ids[split] = set()
        query_positives[split] = {}
        for row in read_jsonl(queries_file):
            value = str(row.get("id", "")).strip()
            if not value or value in query_ids[split]:
                raise ValueError(f"missing or duplicate query id in {queries_file}: {value!r}")
            query_ids[split].add(value)
            positives = row.get("skill_ids")
            if not isinstance(positives, list) or not positives:
                raise ValueError(f"query {value} has no positive skill_ids")
            normalized_positives = {str(skill_id) for skill_id in positives}
            unknown = normalized_positives.difference(skill_ids[split])
            if unknown:
                raise ValueError(
                    f"query {value} references unknown {split} skill: {next(iter(unknown))}"
                )
            query_positives[split][value] = normalized_positives

        qrel_positives: dict[str, set[str]] = defaultdict(set)
        qrel_pairs: set[tuple[str, str]] = set()
        for row in read_jsonl(qrels_file):
            query_id = str(row.get("query_id", ""))
            skill_id = str(row.get("skill_id", ""))
            if query_id not in query_ids[split]:
                raise ValueError(f"qrel references unknown {split} query: {query_id}")
            if skill_id not in skill_ids[split]:
                raise ValueError(f"qrel references unknown {split} skill: {skill_id}")
            pair = (query_id, skill_id)
            if pair in qrel_pairs:
                raise ValueError(f"duplicate qrel in {split}: {pair}")
            qrel_pairs.add(pair)
            if float(row.get("relevance", 1)) > 0:
                qrel_positives[query_id].add(skill_id)

        if qrel_positives != query_positives[split]:
            differing = sorted(set(qrel_positives) | set(query_positives[split]))
            example = next(
                query_id
                for query_id in differing
                if qrel_positives.get(query_id, set())
                != query_positives[split].get(query_id, set())
            )
            raise ValueError(
                f"{split} qrels disagree with query.skill_ids for {example}"
            )

    overlap = skill_ids["train"] & skill_ids["test"]
    if overlap:
        example = next(iter(overlap))
        raise ValueError(f"train/test skill pools overlap; example: {example}")

    full_skills_path = dataset_dir / "data" / "skills.jsonl"
    if not full_skills_path.is_file():
        raise FileNotFoundError(f"SkillRet full catalog is missing: {full_skills_path}")
    counts["skills/full"] = count_jsonl(full_skills_path)
    full_ids: set[str] = set()
    for row in read_jsonl(full_skills_path):
        skill_id = str(row.get("id", "")).strip()
        if not skill_id or skill_id in full_ids:
            raise ValueError(f"missing or duplicate skill id in {full_skills_path}: {skill_id!r}")
        full_ids.add(skill_id)
    missing_from_full = (skill_ids["train"] | skill_ids["test"]).difference(full_ids)
    if missing_from_full:
        raise ValueError(f"split skill is absent from full catalog: {next(iter(missing_from_full))}")

    taxonomy_path = dataset_dir / "data" / "taxonomy.json"
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"SkillRet taxonomy is missing: {taxonomy_path}")
    try:
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid SkillRet taxonomy JSON: {exc}") from exc
    if not isinstance(taxonomy, (dict, list)) or not taxonomy:
        raise ValueError("SkillRet taxonomy must be a non-empty JSON object or list")

    if strict_counts:
        mismatches = {
            key: (SKILLRET_EXPECTED_COUNTS[key], actual)
            for key, actual in counts.items()
            if actual != SKILLRET_EXPECTED_COUNTS[key]
        }
        if mismatches:
            raise ValueError(f"SkillRet split counts do not match the pinned revision: {mismatches}")

    return {
        "repo_id": SKILLRET_REPO_ID,
        "revision": SKILLRET_REVISION,
        "counts": counts,
        "train_test_skill_overlap": 0,
    }


def skill_text(row: Mapping[str, Any], max_chars: int | None = None) -> str:
    """Render the official SkillRet retrieval text."""

    body = row.get("skill_md", row.get("body", ""))
    text = " | ".join(
        str(value or "").strip()
        for value in (row.get("name", ""), row.get("description", ""), body)
    )
    if max_chars is not None and max_chars > 0:
        return text[:max_chars]
    return text


def normalize_skill(row: Mapping[str, Any], max_chars: int | None = None) -> dict[str, Any]:
    skill_id = str(row.get("id", "")).strip()
    if not skill_id:
        raise ValueError("skill is missing id")
    return {
        "skill_id": skill_id,
        "name": str(row.get("name", "")),
        "description": str(row.get("description", "")),
        "text": skill_text(row, max_chars=max_chars),
        "hierarchy": {
            key: row.get(key)
            for key in ("major", "sub", "domain", "primary_action", "primary_object")
        },
        "namespace": row.get("namespace"),
        "license": row.get("license"),
        "source_url": row.get("source_url"),
    }


def normalize_query(row: Mapping[str, Any], allowed_skill_ids: set[str]) -> dict[str, Any] | None:
    query_id = str(row.get("id", "")).strip()
    positives = [str(value) for value in row.get("skill_ids", []) if str(value) in allowed_skill_ids]
    positives = list(dict.fromkeys(positives))
    if not query_id or not positives:
        return None
    return {
        "query_id": query_id,
        "query": str(row.get("query", "")),
        "skill_ids": positives,
        "k": len(positives),
    }


def build_collaborative_edges(
    ordered_skill_ids: Sequence[str],
    qrels: Iterable[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cosine-normalized co-use edges from positive train qrels.

    Each unordered pair is stored exactly once. Repeated qrels for the same
    query/skill pair are ignored.
    """

    index = {skill_id: row for row, skill_id in enumerate(ordered_skill_ids)}
    if len(index) != len(ordered_skill_ids):
        raise ValueError("ordered_skill_ids contains duplicates")

    by_query: dict[str, set[int]] = defaultdict(set)
    for qrel in qrels:
        if float(qrel.get("relevance", 1)) <= 0:
            continue
        skill_id = str(qrel.get("skill_id", ""))
        if skill_id in index:
            by_query[str(qrel.get("query_id", ""))].add(index[skill_id])

    frequency = np.zeros(len(ordered_skill_ids), dtype=np.int64)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for positives in by_query.values():
        nodes = sorted(positives)
        frequency[nodes] += 1
        for offset, source in enumerate(nodes):
            for target in nodes[offset + 1 :]:
                pair_counts[(source, target)] += 1

    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for (source, target), count in sorted(pair_counts.items()):
        denominator = math.sqrt(float(frequency[source]) * float(frequency[target]))
        if denominator > 0:
            sources.append(source)
            targets.append(target)
            weights.append(count / denominator)

    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def code_token(level: int, index: int, token_format: str = "<SK_L{level}_{index}>") -> str:
    return token_format.format(level=level, index=index)


def all_code_tokens(
    branching_factors: Sequence[int],
    token_format: str = "<SK_L{level}_{index}>",
) -> list[str]:
    return [
        code_token(level, index, token_format)
        for level, size in enumerate(branching_factors, start=1)
        for index in range(int(size))
    ]
