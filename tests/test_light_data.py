from __future__ import annotations

import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

from llmgen.clawhub_audit import audit_training_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_light_catalog_accepts_name_description_candidates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(
        source,
        [{"name": "weather", "description": "查询实时天气和预报"}],
    )
    module = runpy.run_path(
        str(REPOSITORY_ROOT / "scripts/light_data/00_build_catalog.py")
    )

    rows = module["read_candidates"](source)

    assert rows == [
        {
            "id": "weather",
            "name": "weather",
            "desc": "查询实时天气和预报",
        }
    ]


def test_light_targeted_patch_covers_badcases_and_direct_brand_names() -> None:
    candidates = _read_jsonl(REPOSITORY_ROOT / "data_light/candidates.jsonl")
    manual = _read_jsonl(
        REPOSITORY_ROOT / "data_light/manual_alignment.jsonl"
    )
    queries = _read_jsonl(
        REPOSITORY_ROOT / "data_light/regression/queries.jsonl"
    )
    qrels = _read_jsonl(
        REPOSITORY_ROOT / "data_light/regression/qrels.jsonl"
    )

    candidate_ids = {str(row["id"]) for row in candidates}
    query_ids = [str(row["id"]) for row in queries]
    qrel_query_ids = [str(row["query_id"]) for row in qrels]
    regression_targets = {str(row["skill_id"]) for row in qrels}
    manual_targets = {str(row["skill_id"]) for row in manual}

    assert len(candidates) == len(candidate_ids) == 301
    assert len(queries) == len(set(query_ids)) == 31
    assert set(qrel_query_ids) == set(query_ids)
    assert regression_targets <= manual_targets <= candidate_ids
    assert all(int(row["relevance"]) == 1 for row in qrels)
    assert len(manual) == 167
    assert len(manual_targets) == 26
    assert Counter(str(row["category"]) for row in manual) == {
        "badcase_paraphrase": 29,
        "boundary_disambiguation": 28,
        "brand_explicit": 65,
        "semantic_correction": 45,
    }

    zhangle_queries = [
        str(row["query"])
        for row in manual
        if row["skill_id"] == "ai-zhangle-skills"
    ]
    assert len(zhangle_queries) == 12
    assert all(
        "华泰" in query or "涨乐" in query
        for query in zhangle_queries
    )
    speech = next(
        row for row in candidates if row["id"] == "speech-to-text"
    )
    zhangle = next(
        row for row in candidates if row["id"] == "ai-zhangle-skills"
    )
    assert speech["name"] == "ElevenLabs语音转写"
    assert "华泰证券" in str(zhangle["name"])
    assert "涨乐财富通" in str(zhangle["desc"])

    # Bad cases remain held out: training uses deliberately different wording.
    regression_text = {
        " ".join(str(row["query"]).casefold().split())
        for row in queries
    }
    manual_text = {
        " ".join(str(row["query"]).casefold().split())
        for row in manual
    }
    assert regression_text.isdisjoint(manual_text)


def test_light_snapshot_restores_previous_multiskill_data() -> None:
    dataset_dir = REPOSITORY_ROOT / "data_light/final"
    manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (dataset_dir / "quality_report.json").read_text(encoding="utf-8")
    )
    alignment = _read_jsonl(dataset_dir / "queries_alignment.jsonl")

    assert manifest["format_version"] == 1
    assert manifest["split_query_counts"] == {
        "test": 631,
        "train": 33098,
        "validation": 530,
    }
    assert manifest["target_order_augmentation"][
        "requested_variants_per_query"
    ] == 3
    assert manifest["target_order_augmentation"][
        "actual_variants_distribution"
    ] == {"2": 5338, "3": 7474}
    assert manifest["targeted_alignment_patch"]["base_revision"].startswith(
        "f325809"
    )
    assert manifest["targeted_alignment_patch"][
        "targeted_query_count"
    ] == 167
    assert manifest["targeted_alignment_patch"][
        "replaced_baseline_query_counts"
    ] == {
        "brainhole-factory": 16,
        "pua": 16,
        "tencent-meeting-mcp": 15,
    }
    assert len(alignment) == 5576
    assert quality["status"] == "pass"
    assert quality["candidate_count"] == 301
    assert quality["schema_policy"] == (
        "previous_snapshot_v1_with_targeted_patch"
    )
    assert quality["single_skill_alignment"][
        "targeted_query_count"
    ] == 167

    replaced = {
        "brainhole-factory",
        "pua",
        "tencent-meeting-mcp",
    }
    for skill_id in replaced:
        rows = [
            row
            for row in alignment
            if row["skill_ids"] == [skill_id]
        ]
        assert rows
        assert all(
            row.get("curation_source") == "targeted_alignment_v1"
            for row in rows
        )
    pua_queries = [
        str(row["query"])
        for row in alignment
        if row["skill_ids"] == ["pua"]
    ]
    assert not any(
        keyword in query
        for query in pua_queries
        for keyword in ("更幽默", "调整语气", "模仿作家")
    )

    for name, details in manifest["artifacts"].items():
        path = dataset_dir / name
        assert path.stat().st_size == int(details["bytes"])
        assert _sha256(path) == details["sha256"]


def test_legacy_quality_audit_has_an_explicit_compatibility_policy(
    tmp_path: Path,
) -> None:
    skills = [
        {"skill_id": "a", "name": "A", "description": "能力A"},
        {"skill_id": "b", "name": "B", "description": "能力B"},
    ]
    alignment = [
        {"id": "a1", "query": "使用能力A", "skill_ids": ["a"]},
        {"id": "b1", "query": "使用能力B", "skill_ids": ["b"]},
    ]
    train = [
        {
            "id": "q1",
            "source_query_id": "q1",
            "query": "先A再B",
            "skill_ids": ["a", "b"],
            "intent_mode": "explicit",
            "implicit_skill_ids": [],
        },
        {
            "id": "q1-order",
            "source_query_id": "q1",
            "query": "先A再B",
            "skill_ids": ["b", "a"],
            "intent_mode": "explicit",
            "implicit_skill_ids": [],
        },
    ]
    validation = [
        {
            "id": "q2",
            "query": "完成A和B",
            "skill_ids": ["a", "b"],
            "intent_mode": "implicit",
            "implicit_skill_ids": ["b"],
        }
    ]
    test = [
        {
            "id": "q3",
            "query": "再次完成A和B",
            "skill_ids": ["a", "b"],
            "intent_mode": "explicit",
            "implicit_skill_ids": [],
        }
    ]
    files = {
        "skills.jsonl": skills,
        "queries_alignment.jsonl": alignment,
        "queries_train.jsonl": train,
        "queries_validation.jsonl": validation,
        "queries_test.jsonl": test,
    }
    for name, rows in files.items():
        _write_jsonl(tmp_path / name, rows)
    (tmp_path / "quality_report.json").write_text(
        "{}\n", encoding="utf-8"
    )
    artifacts = {}
    for name in [*files, "quality_report.json"]:
        path = tmp_path / name
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "path": name,
            "sha256": _sha256(path),
        }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "artifacts": artifacts,
                "min_augmented_train_queries_required": 2,
                "min_train_positives_per_skill_required": 2,
                "targeted_alignment_patch": {
                    "created_at": "2026-07-30T00:00:00Z"
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_training_dataset(tmp_path, expected_candidates=2)

    assert report["status"] == "pass"
    assert report["schema_policy"] == (
        "previous_snapshot_v1_with_targeted_patch"
    )
    refreshed_manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    quality = tmp_path / "quality_report.json"
    assert refreshed_manifest["artifacts"]["quality_report.json"][
        "sha256"
    ] == _sha256(quality)


def test_quality_audit_allows_empty_multiskill_splits_for_alignment_only(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "skills.jsonl",
        [{"skill_id": "only", "name": "Only", "description": "唯一能力"}],
    )
    _write_jsonl(
        tmp_path / "queries_alignment.jsonl",
        [{"id": "a1", "query": "使用唯一能力", "skill_ids": ["only"]}],
    )
    for split in ("train", "validation", "test"):
        _write_jsonl(tmp_path / f"queries_{split}.jsonl", [])
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "artifacts": {},
                "min_augmented_train_queries_required": 0,
                "min_train_positives_per_skill_required": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_training_dataset(tmp_path, expected_candidates=1)

    assert report["status"] == "pass"
    assert report["execution_mode"] == "alignment_only"
    assert report["query_counts"] == {
        "train": 0,
        "validation": 0,
        "test": 0,
    }
