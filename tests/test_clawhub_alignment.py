from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmgen.clawhub_alignment import _validate_variant, export_alignment_dataset
from llmgen.clawhub_dataset import DatasetBuildError


def _profile(skill_id: str) -> dict:
    return {
        "skill_id": skill_id,
        "domain": "weather_environment",
        "roles": ["retrieve"],
        "capability_zh": "查询指定地点的实时天气和预报",
        "mobile_fit": "high",
        "unsafe_action": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_single_skill_variant_uses_whole_query_when_evidence_is_paraphrased() -> None:
    row = _validate_variant(
        {
            "query": "周末准备带孩子去杭州玩，帮我看看那两天会不会下雨。",
            "evidence": "查询杭州周末天气",
        },
        _profile("@owner/weather"),
        0,
    )
    assert row["skill_ids"] == ["@owner/weather"]
    assert row["evidence"]["@owner/weather"] == row["query"]


def test_alignment_export_requires_every_candidate(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    queries = tmp_path / "queries.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    output = tmp_path / "final"
    output.mkdir()
    (output / "manifest.json").write_text('{"artifacts": {}}')
    _write_jsonl(
        catalog,
        [
            {"skill_id": "@owner/one"},
            {"skill_id": "@owner/two"},
        ],
    )
    query_rows = [
        {
            "query_id": "q1",
            "query": "帮我查一下明天杭州的天气。",
            "skill_ids": ["@owner/one"],
            "evidence": {"@owner/one": "查一下明天杭州的天气"},
            "target_intents": {"@owner/one": "explicit"},
        }
    ]
    _write_jsonl(queries, query_rows)
    _write_jsonl(
        reviews,
        [{"query_id": "q1", "pass": True, "scores": {"target_relevance": 5}}],
    )
    with pytest.raises(DatasetBuildError, match="1 candidates lack"):
        export_alignment_dataset(catalog, queries, reviews, output)

    query_rows.append(
        {
            "query_id": "q2",
            "query": "把这个页面保存到我的笔记里。",
            "skill_ids": ["@owner/two"],
            "evidence": {"@owner/two": "保存到我的笔记里"},
            "target_intents": {"@owner/two": "explicit"},
        }
    )
    _write_jsonl(queries, query_rows)
    _write_jsonl(
        reviews,
        [
            {"query_id": "q1", "pass": True, "scores": {"target_relevance": 5}},
            {"query_id": "q2", "pass": True, "scores": {"target_relevance": 5}},
        ],
    )
    result = export_alignment_dataset(catalog, queries, reviews, output)
    assert result["accepted_candidate_count"] == 2
    assert len((output / "queries_alignment.jsonl").read_text().splitlines()) == 2
