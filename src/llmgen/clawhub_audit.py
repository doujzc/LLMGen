"""Quality audit for exported ClawHub routing datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from llmgen.clawhub import atomic_json, sha256_file, utc_now
from llmgen.clawhub_dataset import DatasetBuildError, load_jsonl


SPLITS = ("train", "validation", "test")


def _mean_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    score_rows = [row.get("quality_scores") for row in rows]
    score_rows = [row for row in score_rows if isinstance(row, dict)]
    keys = sorted({str(key) for row in score_rows for key in row})
    return {
        key: sum(float(row[key]) for row in score_rows if key in row)
        / sum(key in row for row in score_rows)
        for key in keys
    }


def audit_training_dataset(
    dataset_dir: Path,
    *,
    expected_candidates: int | None = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest.get("artifacts", {}).items():
        path = dataset_dir / name
        if not path.is_file():
            raise DatasetBuildError(f"missing exported artifact: {path}")
        if path.stat().st_size != int(expected["bytes"]):
            raise DatasetBuildError(f"artifact size mismatch: {name}")
        if sha256_file(path) != str(expected["sha256"]):
            raise DatasetBuildError(f"artifact SHA-256 mismatch: {name}")

    skills = load_jsonl(dataset_dir / "skills.jsonl")
    candidate_ids = {str(row["skill_id"]) for row in skills}
    if len(candidate_ids) != len(skills):
        raise DatasetBuildError("skills.jsonl has duplicate candidates")
    if expected_candidates is not None and len(skills) != expected_candidates:
        raise DatasetBuildError(
            f"dataset has {len(skills)} candidates, expected {expected_candidates}"
        )
    alignment_rows = load_jsonl(dataset_dir / "queries_alignment.jsonl")
    alignment_counts = Counter(
        str(row["skill_ids"][0])
        for row in alignment_rows
        if isinstance(row.get("skill_ids"), list) and len(row["skill_ids"]) == 1
    )
    if len(alignment_counts) != len(candidate_ids):
        raise DatasetBuildError(
            "single-skill curriculum does not cover the complete candidate set"
        )

    split_rows = {
        split: load_jsonl(dataset_dir / f"queries_{split}.jsonl")
        for split in SPLITS
    }
    all_rows = [row for split in SPLITS for row in split_rows[split]]
    implicit_rows = [row for row in all_rows if row.get("intent_mode") == "implicit"]
    explicit_rows = [row for row in all_rows if row.get("intent_mode") == "explicit"]
    for row in all_rows:
        targets = list(map(str, row.get("skill_ids") or []))
        if len(targets) != len(set(targets)) or not set(targets) <= candidate_ids:
            raise DatasetBuildError(f"invalid targets in query {row.get('id')}")
        implicit_ids = list(map(str, row.get("implicit_skill_ids") or []))
        if row.get("intent_mode") == "implicit":
            if not 1 <= len(implicit_ids) < len(targets):
                raise DatasetBuildError(
                    f"invalid implicit targets in query {row.get('id')}"
                )
        elif implicit_ids:
            raise DatasetBuildError(
                f"explicit query declares implicit targets: {row.get('id')}"
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows["train"]:
        groups[str(row.get("source_query_id") or row["id"])].append(row)
    semantic_positive_counts: Counter[str] = Counter()
    position_coverages: list[float] = []
    variant_count_distribution: Counter[int] = Counter()
    for source_id, rows in groups.items():
        first_targets = set(map(str, rows[0]["skill_ids"]))
        orders = [tuple(map(str, row["skill_ids"])) for row in rows]
        if len(set(orders)) != len(orders):
            raise DatasetBuildError(f"duplicate target orders for {source_id}")
        if any(set(order) != first_targets for order in orders):
            raise DatasetBuildError(f"target set changes across order variants: {source_id}")
        if any(str(row["query"]) != str(rows[0]["query"]) for row in rows):
            raise DatasetBuildError(f"query text changes across order variants: {source_id}")
        semantic_positive_counts.update(first_targets)
        variant_count_distribution[len(rows)] += 1
        denominator = min(len(first_targets), len(rows))
        for skill_id in first_targets:
            observed = {order.index(skill_id) for order in orders}
            position_coverages.append(len(observed) / denominator)

    required = int(manifest.get("min_train_positives_per_skill_required", 1))
    combined_positive_counts = semantic_positive_counts + alignment_counts
    undercovered = {
        skill_id: combined_positive_counts[skill_id]
        for skill_id in sorted(candidate_ids)
        if combined_positive_counts[skill_id] < required
    }
    if undercovered:
        preview = ", ".join(
            f"{skill_id}={count}" for skill_id, count in list(undercovered.items())[:10]
        )
        raise DatasetBuildError(
            f"semantic train coverage is below {required}: {preview}"
        )
    if not implicit_rows:
        raise DatasetBuildError("dataset has no accepted implicit-intent samples")

    report = {
        "stage": "dataset_quality_audit",
        "created_at": utc_now(),
        "dataset_dir": str(dataset_dir),
        "candidate_count": len(skills),
        "query_counts": {split: len(split_rows[split]) for split in SPLITS},
        "semantic_train_query_count": len(groups),
        "single_skill_alignment": {
            "query_count": len(alignment_rows),
            "covered_candidate_count": len(alignment_counts),
            "minimum_per_candidate": min(alignment_counts.values(), default=0),
            "mean_per_candidate": len(alignment_rows) / len(candidate_ids) if candidate_ids else 0.0,
        },
        "target_order": {
            "augmented_train_query_count": len(split_rows["train"]),
            "augmentation_factor": len(split_rows["train"]) / len(groups) if groups else 0.0,
            "variant_count_distribution": dict(sorted(variant_count_distribution.items())),
            "mean_target_position_coverage": (
                sum(position_coverages) / len(position_coverages)
                if position_coverages
                else 0.0
            ),
        },
        "implicit_intent": {
            "query_count": len(implicit_rows),
            "query_fraction": len(implicit_rows) / len(all_rows) if all_rows else 0.0,
            "implicit_target_count": sum(
                len(row.get("implicit_skill_ids") or []) for row in implicit_rows
            ),
            "mean_quality_scores": _mean_scores(implicit_rows),
        },
        "explicit_intent": {
            "query_count": len(explicit_rows),
            "mean_quality_scores": _mean_scores(explicit_rows),
        },
        "semantic_train_coverage": {
            "required_per_candidate": required,
            "minimum": min(combined_positive_counts.values(), default=0),
            "mean": (
                sum(combined_positive_counts.values()) / len(candidate_ids)
                if candidate_ids
                else 0.0
            ),
            "undercovered_candidate_count": 0,
        },
        "status": "pass",
    }
    atomic_json(dataset_dir / "quality_report.json", report)
    return report
