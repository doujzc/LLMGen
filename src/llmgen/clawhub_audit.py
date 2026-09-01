"""Quality audit for exported ClawHub routing datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from llmgen.clawhub import atomic_json, sha256_file, utc_now
from llmgen.clawhub_alignment import minimum_alignment_requirement_counts
from llmgen.clawhub_dataset import (
    ROUTING_MODES,
    DatasetBuildError,
    load_jsonl,
)


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
    format_version = int(manifest.get("format_version", 1))
    strict_routing_schema = format_version >= 2
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
    if strict_routing_schema:
        for skill in skills:
            skill_id = str(skill["skill_id"])
            for field in ("aliases", "capability_facets", "trigger_phrases"):
                values = skill.get(field)
                if not isinstance(values, list) or not values:
                    raise DatasetBuildError(
                        f"skill {skill_id} has no routing profile field {field}"
                    )
            if skill.get("routing_mode") not in ROUTING_MODES:
                raise DatasetBuildError(
                    f"skill {skill_id} has invalid routing_mode"
                )
    alignment_rows = load_jsonl(dataset_dir / "queries_alignment.jsonl")
    alignment_counts = Counter(
        str(row["skill_ids"][0])
        for row in alignment_rows
        if isinstance(row.get("skill_ids"), list) and len(row["skill_ids"]) == 1
    )
    skill_profiles = {str(row["skill_id"]): row for row in skills}
    alignment_requirement_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in alignment_rows:
        targets = list(map(str, row.get("skill_ids") or []))
        if len(targets) != 1:
            raise DatasetBuildError(
                f"invalid single-skill data in {row.get('id')}"
            )
        if strict_routing_schema:
            primary = list(map(str, row.get("primary_skill_ids") or []))
            support = list(map(str, row.get("support_skill_ids") or []))
            if primary != targets or support:
                raise DatasetBuildError(
                    "invalid single-skill primary/support data in "
                    f"{row.get('id')}"
                )
        alignment_requirement_counts[targets[0]].update(
            map(str, row.get("generation_requirements") or ["core"])
        )
    if len(alignment_counts) != len(candidate_ids):
        raise DatasetBuildError(
            "single-skill curriculum does not cover the complete candidate set"
        )
    alignment_requirement_deficits: dict[str, dict[str, int]] = {}
    if strict_routing_schema:
        for skill_id in sorted(candidate_ids):
            deficits = {
                requirement: minimum
                - alignment_requirement_counts[skill_id][requirement]
                for requirement, minimum in minimum_alignment_requirement_counts(
                    skill_profiles[skill_id]
                ).items()
                if alignment_requirement_counts[skill_id][requirement] < minimum
            }
            if deficits:
                alignment_requirement_deficits[skill_id] = deficits
        if alignment_requirement_deficits:
            preview = ", ".join(
                f"{skill_id}={deficits}"
                for skill_id, deficits in list(
                    alignment_requirement_deficits.items()
                )[:10]
            )
            raise DatasetBuildError(
                "single-skill curriculum misses required routing scenarios: "
                + preview
            )

    split_rows = {
        split: load_jsonl(dataset_dir / f"queries_{split}.jsonl")
        for split in SPLITS
    }
    all_rows = [row for split in SPLITS for row in split_rows[split]]
    alignment_only = len(candidate_ids) == 1 and not all_rows
    target_counts_by_split = {
        split: Counter(len(row.get("skill_ids") or []) for row in rows)
        for split, rows in split_rows.items()
    }
    unseen_target_counts = sorted(
        (
            set(target_counts_by_split["validation"])
            | set(target_counts_by_split["test"])
        )
        - set(target_counts_by_split["train"])
    )
    if unseen_target_counts:
        raise DatasetBuildError(
            "held-out splits contain target counts absent from train: "
            + ", ".join(map(str, unseen_target_counts))
        )
    implicit_rows = [row for row in all_rows if row.get("intent_mode") == "implicit"]
    explicit_rows = [row for row in all_rows if row.get("intent_mode") == "explicit"]
    for row in all_rows:
        targets = list(map(str, row.get("skill_ids") or []))
        if len(targets) != len(set(targets)) or not set(targets) <= candidate_ids:
            raise DatasetBuildError(f"invalid targets in query {row.get('id')}")
        if strict_routing_schema:
            primary_ids = set(map(str, row.get("primary_skill_ids") or []))
            support_ids = set(map(str, row.get("support_skill_ids") or []))
            if (
                not primary_ids
                or primary_ids & support_ids
                or primary_ids | support_ids != set(targets)
            ):
                raise DatasetBuildError(
                    f"invalid primary/support targets in query {row.get('id')}"
                )
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
    first_position_coverages: list[float] = []
    missing_first_position_pairs: list[tuple[str, str]] = []
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
        first_position_targets = {order[0] for order in orders}
        for skill_id in first_targets:
            observed = {order.index(skill_id) for order in orders}
            position_coverages.append(len(observed) / denominator)
            is_first = float(skill_id in first_position_targets)
            first_position_coverages.append(is_first)
            if not is_first:
                missing_first_position_pairs.append((source_id, skill_id))

    if strict_routing_schema and missing_first_position_pairs:
        preview = ", ".join(
            f"{source_id}:{skill_id}"
            for source_id, skill_id in missing_first_position_pairs[:10]
        )
        raise DatasetBuildError(
            "target-order augmentation leaves query-target pairs without "
            f"query-only first-position supervision: {preview}"
        )

    required = int(manifest.get("min_train_positives_per_skill_required", 1))
    required_augmented_train_queries = int(
        manifest.get("min_augmented_train_queries_required", 0)
    )
    if len(split_rows["train"]) < required_augmented_train_queries:
        raise DatasetBuildError(
            "augmented train data has "
            f"{len(split_rows['train'])} queries, fewer than the required "
            f"{required_augmented_train_queries}"
        )
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
    if not implicit_rows and not alignment_only:
        raise DatasetBuildError("dataset has no accepted implicit-intent samples")

    patch_details = manifest.get("targeted_alignment_patch")
    report_created_at = (
        str(patch_details.get("created_at"))
        if isinstance(patch_details, dict) and patch_details.get("created_at")
        else utc_now()
    )
    targeted_categories = Counter(
        str(row.get("targeted_category"))
        for row in alignment_rows
        if row.get("curation_source") == "targeted_alignment_v1"
    )
    report = {
        "stage": "dataset_quality_audit",
        "created_at": report_created_at,
        "dataset_dir": ".",
        "format_version": format_version,
        "export_status": str(manifest.get("export_status") or "ready"),
        "provisional_note": manifest.get("provisional_note"),
        "review_completion": manifest.get("review_completion"),
        "schema_policy": (
            "routing_profiles_v2"
            if strict_routing_schema
            else "previous_snapshot_v1_with_targeted_patch"
        ),
        "candidate_count": len(skills),
        "query_counts": {split: len(split_rows[split]) for split in SPLITS},
        "execution_mode": "alignment_only" if alignment_only else "multiskill",
        "semantic_train_query_count": len(groups),
        "target_count_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in target_counts_by_split.items()
        },
        "single_skill_alignment": {
            "query_count": len(alignment_rows),
            "covered_candidate_count": len(alignment_counts),
            "minimum_per_candidate": min(alignment_counts.values(), default=0),
            "mean_per_candidate": len(alignment_rows) / len(candidate_ids) if candidate_ids else 0.0,
            "requirement_counts": dict(
                sorted(
                    Counter(
                        requirement
                        for counts in alignment_requirement_counts.values()
                        for requirement, count in counts.items()
                        for _ in range(count)
                    ).items()
                )
            ),
            "requirement_deficit_candidate_count": (
                len(alignment_requirement_deficits)
            ),
            "targeted_category_counts": dict(
                sorted(targeted_categories.items())
            ),
            "targeted_query_count": sum(targeted_categories.values()),
        },
        "target_order": {
            "augmented_train_query_count": len(split_rows["train"]),
            "minimum_augmented_train_queries_required": (
                required_augmented_train_queries
            ),
            "augmentation_factor": len(split_rows["train"]) / len(groups) if groups else 0.0,
            "variant_count_distribution": dict(sorted(variant_count_distribution.items())),
            "mean_target_position_coverage": (
                sum(position_coverages) / len(position_coverages)
                if position_coverages
                else 0.0
            ),
            "query_target_first_position_coverage": (
                sum(first_position_coverages) / len(first_position_coverages)
                if first_position_coverages
                else 0.0
            ),
            "missing_first_position_pair_count": len(
                missing_first_position_pairs
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
    quality_path = dataset_dir / "quality_report.json"
    atomic_json(quality_path, report)
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict) and "quality_report.json" in artifacts:
        artifacts["quality_report.json"] = {
            "bytes": quality_path.stat().st_size,
            "path": "quality_report.json",
            "sha256": sha256_file(quality_path),
        }
        atomic_json(manifest_path, manifest)
    return report
