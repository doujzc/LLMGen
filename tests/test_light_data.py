from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_light_badcase_regression_set_uses_the_candidate_registry() -> None:
    candidates = _read_jsonl(REPOSITORY_ROOT / "data_light/candidates.jsonl")
    manual = _read_jsonl(
        REPOSITORY_ROOT / "data_light/manual_alignment.jsonl"
    )
    queries = _read_jsonl(REPOSITORY_ROOT / "data_light/regression/queries.jsonl")
    qrels = _read_jsonl(REPOSITORY_ROOT / "data_light/regression/qrels.jsonl")

    candidate_ids = {str(row["id"]) for row in candidates}
    query_ids = [str(row["id"]) for row in queries]
    qrel_query_ids = [str(row["query_id"]) for row in qrels]
    target_ids = {str(row["skill_id"]) for row in qrels}

    assert len(candidates) == len(candidate_ids) == 301
    assert len(queries) == len(set(query_ids)) == 31
    assert set(qrel_query_ids) == set(query_ids)
    assert target_ids <= candidate_ids
    assert all(int(row["relevance"]) == 1 for row in qrels)
    assert len(manual) == 18
    assert {str(row["skill_id"]) for row in manual} <= candidate_ids
    speech = next(row for row in candidates if row["id"] == "speech-to-text")
    assert speech["name"] == "ElevenLabs语音转写"


def test_light_snapshot_records_quality_and_alignment_provenance() -> None:
    dataset_dir = REPOSITORY_ROOT / "data_light/final"
    manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (dataset_dir / "quality_report.json").read_text(encoding="utf-8")
    )

    assert quality["status"] == "pass"
    assert quality["candidate_count"] == 301
    assert quality["target_order"][
        "query_target_first_position_coverage"
    ] == 1.0
    assert quality["single_skill_alignment"][
        "requirement_deficit_candidate_count"
    ] == 0
    assert manifest["single_skill_alignment"]["review_source_counts"] == {
        "legacy_model_review": 160,
        "manual_curation": 18,
        "model_review": 5458,
    }
