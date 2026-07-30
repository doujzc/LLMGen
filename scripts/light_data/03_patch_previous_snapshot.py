#!/usr/bin/env python3
"""Apply the reviewed Light-301 routing patch to the previous data snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from llmgen.clawhub import atomic_json, atomic_jsonl, sha256_file
from llmgen.clawhub_audit import audit_training_dataset
from llmgen.clawhub_dataset import DatasetBuildError, load_jsonl


CURATION_SOURCE = "targeted_alignment_v1"
ALLOWED_CATEGORIES = {
    "badcase_paraphrase",
    "boundary_disambiguation",
    "brand_explicit",
    "semantic_correction",
}


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _positive_counts(
    dataset_dir: Path,
    alignment_rows: Iterable[Mapping[str, Any]],
) -> Counter[str]:
    semantic_targets: dict[str, set[str]] = {}
    for row in load_jsonl(dataset_dir / "queries_train.jsonl"):
        source_id = str(row.get("source_query_id") or row["id"])
        semantic_targets.setdefault(source_id, set()).update(
            map(str, row["skill_ids"])
        )
    counts: Counter[str] = Counter(
        skill_id
        for targets in semantic_targets.values()
        for skill_id in targets
    )
    counts.update(
        str(row["skill_ids"][0])
        for row in alignment_rows
    )
    return counts


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.name,
        "sha256": sha256_file(path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data_light/final"),
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=Path("data_light/manual_alignment.jsonl"),
    )
    parser.add_argument(
        "--patch-manifest",
        type=Path,
        default=Path("data_light/manual_alignment.manifest.json"),
    )
    parser.add_argument(
        "--metadata-patches",
        type=Path,
        default=Path("data_light/skill_metadata_patches.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    patch_manifest = json.loads(
        args.patch_manifest.read_text(encoding="utf-8")
    )
    if int(patch_manifest.get("schema_version", 0)) != 1:
        raise DatasetBuildError("unsupported Light-301 patch schema")
    replace_skill_ids = set(
        map(str, patch_manifest.get("replace_alignment_skill_ids") or [])
    )
    expected_replaced_counts = Counter(
        {
            str(key): int(value)
            for key, value in (
                patch_manifest.get("replaced_baseline_query_counts") or {}
            ).items()
        }
    )
    if set(expected_replaced_counts) != replace_skill_ids:
        raise DatasetBuildError(
            "replacement counts disagree with replacement skill IDs"
        )
    created_at = str(patch_manifest.get("created_at") or "")
    if not created_at:
        raise DatasetBuildError("patch manifest has no created_at")
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("format_version", 0)) != 1:
        raise DatasetBuildError(
            "targeted patch must be applied to the previous format-v1 snapshot"
        )
    already_patched = isinstance(
        manifest.get("targeted_alignment_patch"), dict
    )
    if not already_patched:
        expected_base_hashes = {
            str(key): str(value)
            for key, value in (
                patch_manifest.get("base_artifact_sha256") or {}
            ).items()
        }
        for name, expected_hash in expected_base_hashes.items():
            path = dataset_dir / name
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise DatasetBuildError(
                    f"dataset is not the recorded previous snapshot: {name}"
                )

    skills = load_jsonl(dataset_dir / "skills.jsonl")
    skill_by_id = {str(row["skill_id"]): row for row in skills}
    if len(skill_by_id) != len(skills):
        raise DatasetBuildError("Light-301 skill registry contains duplicates")
    for patch in load_jsonl(args.metadata_patches):
        skill_id = str(patch.get("skill_id") or "")
        skill = skill_by_id.get(skill_id)
        if skill is None:
            raise DatasetBuildError(
                f"metadata patch references unknown skill: {skill_id}"
            )
        updates = patch.get("updates")
        if not isinstance(updates, dict) or not updates:
            raise DatasetBuildError(
                f"metadata patch has no updates: {skill_id}"
            )
        unknown_fields = set(updates).difference(
            {"name", "description", "capability_zh"}
        )
        if unknown_fields:
            raise DatasetBuildError(
                f"metadata patch has unsupported fields: {sorted(unknown_fields)}"
            )
        for field, value in updates.items():
            if not isinstance(value, str) or not value.strip():
                raise DatasetBuildError(
                    f"metadata patch has invalid {field}: {skill_id}"
                )
            skill[field] = value.strip()
    atomic_jsonl(dataset_dir / "skills.jsonl", skills)

    baseline_rows = load_jsonl(dataset_dir / "queries_alignment.jsonl")
    retained_rows = []
    replaced_counts: Counter[str] = Counter()
    for row in baseline_rows:
        targets = list(map(str, row.get("skill_ids") or []))
        if len(targets) != 1:
            raise DatasetBuildError(
                f"alignment row is not single-target: {row.get('id')}"
            )
        skill_id = targets[0]
        if row.get("curation_source") == CURATION_SOURCE:
            continue
        if skill_id in replace_skill_ids:
            replaced_counts[skill_id] += 1
            continue
        retained_rows.append(row)
    if not already_patched and replaced_counts != expected_replaced_counts:
        raise DatasetBuildError(
            "previous snapshot replacement counts disagree with patch manifest"
        )
    replaced_counts = expected_replaced_counts

    curated = load_jsonl(args.curated)
    expected_query_count = int(
        patch_manifest.get("targeted_query_count", -1)
    )
    if len(curated) != expected_query_count:
        raise DatasetBuildError(
            f"curated query count is {len(curated)}, expected "
            f"{expected_query_count}"
        )
    expected_category_counts = {
        str(key): int(value)
        for key, value in (
            patch_manifest.get("category_counts") or {}
        ).items()
    }
    category_counts = Counter(
        str(row.get("category") or "") for row in curated
    )
    if dict(sorted(category_counts.items())) != dict(
        sorted(expected_category_counts.items())
    ):
        raise DatasetBuildError(
            "curated category counts disagree with patch manifest"
        )
    if set(category_counts).difference(ALLOWED_CATEGORIES):
        raise DatasetBuildError("curated data contains an unknown category")

    seen_queries = {
        _normalized_text(str(row["query"])) for row in retained_rows
    }
    seen_ids = {str(row["id"]) for row in retained_rows}
    added_rows: list[dict[str, Any]] = []
    added_skill_counts: Counter[str] = Counter()
    for raw in curated:
        skill_id = str(raw.get("skill_id") or "")
        query = str(raw.get("query") or "").strip()
        category = str(raw.get("category") or "")
        if skill_id not in skill_by_id:
            raise DatasetBuildError(
                f"curated query references unknown skill: {skill_id}"
            )
        normalized = _normalized_text(query)
        if not normalized or normalized in seen_queries:
            raise DatasetBuildError(
                f"empty or duplicate curated query for {skill_id}: {query}"
            )
        evidence = str(raw.get("evidence") or query).strip()
        if evidence not in query:
            raise DatasetBuildError(
                f"curated evidence is not verbatim for {skill_id}: {evidence}"
            )
        digest = hashlib.sha256(
            f"{skill_id}\0{normalized}".encode("utf-8")
        ).hexdigest()[:20]
        query_id = f"ca-targeted-{digest}"
        if query_id in seen_ids:
            raise DatasetBuildError(
                f"curated query ID collision: {query_id}"
            )
        row = {
            "curriculum_phase": "single_skill_alignment",
            "curation_source": CURATION_SOURCE,
            "evidence": {skill_id: evidence},
            "id": query_id,
            "implicit_rationales": {},
            "implicit_skill_ids": [],
            "intent_mode": "explicit",
            "quality_scores": {
                "coherence": 5,
                "mobile_style": 5,
                "specificity": 5,
                "target_relevance": 5,
            },
            "query": query,
            "skill_ids": [skill_id],
            "target_intents": {skill_id: "explicit"},
            "targeted_category": category,
        }
        seen_queries.add(normalized)
        seen_ids.add(query_id)
        added_rows.append(row)
        added_skill_counts[skill_id] += 1

    missing_replacements = replace_skill_ids.difference(added_skill_counts)
    if missing_replacements:
        raise DatasetBuildError(
            "replacement skills have no curated queries: "
            + ", ".join(sorted(missing_replacements))
        )
    alignment_rows = sorted(
        [*retained_rows, *added_rows],
        key=lambda row: str(row["id"]),
    )
    alignment_qrels = [
        {
            "query_id": str(row["id"]),
            "relevance": 1,
            "skill_id": str(row["skill_ids"][0]),
        }
        for row in alignment_rows
    ]
    atomic_jsonl(dataset_dir / "queries_alignment.jsonl", alignment_rows)
    atomic_jsonl(dataset_dir / "qrels_alignment.jsonl", alignment_qrels)

    candidate_ids = set(skill_by_id)
    alignment_counts = Counter(
        str(row["skill_ids"][0]) for row in alignment_rows
    )
    if set(alignment_counts) != candidate_ids:
        raise DatasetBuildError(
            "targeted alignment patch broke full candidate coverage"
        )
    minimum_alignment = min(alignment_counts.values())
    required_alignment = int(
        patch_manifest.get("minimum_alignment_queries_per_skill", 15)
    )
    if minimum_alignment < required_alignment:
        raise DatasetBuildError(
            f"minimum alignment coverage is {minimum_alignment}, "
            f"required {required_alignment}"
        )

    positive_counts = _positive_counts(dataset_dir, alignment_rows)
    required_positives = int(
        manifest.get("min_train_positives_per_skill_required", 1)
    )
    below_minimum = {
        skill_id: positive_counts[skill_id]
        for skill_id in sorted(candidate_ids)
        if positive_counts[skill_id] < required_positives
    }
    manifest["created_at"] = created_at
    manifest["mean_train_positives_per_covered_skill"] = (
        sum(positive_counts.values()) / len(candidate_ids)
    )
    manifest["min_train_positives_per_covered_skill"] = min(
        positive_counts.values()
    )
    manifest["skills_below_min_train_positives"] = below_minimum
    manifest["skills_below_min_train_positives_count"] = len(
        below_minimum
    )
    manifest["single_skill_alignment"] = {
        **manifest.get("single_skill_alignment", {}),
        "accepted_candidate_count": len(alignment_counts),
        "accepted_query_count": len(alignment_rows),
        "base_accepted_query_count": int(
            patch_manifest["base_alignment_query_count"]
        ),
        "candidate_count": len(candidate_ids),
        "mean_queries_per_skill": len(alignment_rows) / len(candidate_ids),
        "min_queries_per_skill": minimum_alignment,
        "replaced_baseline_query_count": sum(replaced_counts.values()),
        "source_counts": {
            "previous_snapshot_model_review": len(retained_rows),
            "targeted_manual_curation": len(added_rows),
        },
        "targeted_query_count": len(added_rows),
    }
    manifest["targeted_alignment_patch"] = {
        "schema_version": 1,
        "base_revision": str(patch_manifest["base_revision"]),
        "created_at": created_at,
        "source": str(args.curated),
        "source_sha256": sha256_file(args.curated),
        "metadata_patch_source": str(args.metadata_patches),
        "metadata_patch_sha256": sha256_file(args.metadata_patches),
        "replace_alignment_skill_ids": sorted(replace_skill_ids),
        "replaced_baseline_query_counts": dict(
            sorted(replaced_counts.items())
        ),
        "targeted_query_count": len(added_rows),
        "targeted_skill_count": len(added_skill_counts),
        "targeted_skill_counts": dict(
            sorted(added_skill_counts.items())
        ),
        "category_counts": dict(sorted(category_counts.items())),
    }
    for name in manifest.get("artifacts", {}):
        if name == "quality_report.json":
            continue
        path = dataset_dir / name
        manifest["artifacts"][name] = _artifact(path)
    atomic_json(manifest_path, manifest)

    # The audit has an explicit format-v1 compatibility policy and refreshes
    # quality_report.json plus its integrity entry in manifest.json.
    report = audit_training_dataset(
        dataset_dir,
        expected_candidates=len(candidate_ids),
    )
    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "alignment_queries": len(alignment_rows),
                "replaced": dict(sorted(replaced_counts.items())),
                "added": dict(sorted(added_skill_counts.items())),
                "quality_status": report["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
