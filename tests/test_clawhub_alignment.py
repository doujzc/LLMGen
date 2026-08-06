from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmgen.clawhub_alignment import (
    _alignment_variant_requirements,
    _alignment_generation_max_tokens,
    _generation_prompt,
    _validate_review,
    _validate_variant,
    append_legacy_alignment_queries,
    append_manual_alignment_queries,
    export_alignment_dataset,
    minimum_alignment_requirement_counts,
)
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


def test_single_skill_variant_allows_short_mobile_query_and_file_format() -> None:
    weather = _validate_variant(
        {"query": "今天北京天气怎样", "evidence": "北京天气"},
        _profile("weather"),
        0,
    )
    document = _validate_variant(
        {"query": "把这个 DOCX 转成 PDF", "evidence": "DOCX 转成 PDF"},
        _profile("docx"),
        0,
    )
    assert weather["query"] == "今天北京天气怎样"
    assert document["query"] == "把这个 DOCX 转成 PDF"


def test_single_skill_variant_allows_concise_phone_command() -> None:
    row = _validate_variant(
        {"query": "打给爸爸", "evidence": "打给爸爸"},
        _profile("phone"),
        0,
    )
    assert row["query"] == "打给爸爸"


def test_single_skill_variant_allows_a_declared_slash_command() -> None:
    profile = {
        **_profile("smart-followups"),
        "trigger_phrases": ["/followups"],
    }
    row = _validate_variant(
        {"query": "/followups", "evidence": "/followups"},
        profile,
        0,
    )
    assert row["query"] == "/followups"


def test_single_skill_variant_ignores_numeric_payload_in_language_ratio() -> None:
    query = (
        "帮我精确计算 "
        "1234567890123456789012345678901234567890 "
        "乘以 9876543210987654321098765432109876543210"
    )
    row = _validate_variant(
        {"query": query, "evidence": "精确计算"},
        _profile("large-number-calculation"),
        0,
    )
    assert row["query"] == query


def test_single_skill_variant_rejects_opaque_slug() -> None:
    with pytest.raises(DatasetBuildError, match="leaks a candidate identifier"):
        _validate_variant(
            {
                "query": "帮我调用 daily-hot-news 看今天热点",
                "evidence": "今天热点",
            },
            _profile("daily-hot-news"),
            0,
        )


def test_alignment_prompt_uses_full_routing_context_for_meta_skill() -> None:
    profile = {
        **_profile("pua"),
        "display_name": "失败恢复",
        "description": "任务失败两次后继续读日志、换方法并逐项验证。",
        "aliases": ["失败恢复"],
        "capability_facets": ["读取日志定位原因", "更换方案继续验证"],
        "trigger_phrases": ["失败两轮", "别直接归因环境"],
        "negative_boundaries": ["不负责普通文案润色"],
        "routing_mode": "meta",
        "confusable_skill_ids": [],
    }
    prompt = _generation_prompt([profile], 4)
    assert profile["description"] in prompt
    assert "失败两轮" in prompt
    assert "底层任务是上下文" in prompt
    assert "confusable_alternatives" in prompt
    assert "meta_task_context" in prompt
    assert "native_followup" in prompt


def test_alignment_variant_plan_covers_composite_and_native_scenarios() -> None:
    profile = {
        "skill_id": "brainhole-factory",
        "routing_mode": "composite",
        "aliases": ["脑洞工厂"],
    }
    plans = [
        _alignment_variant_requirements(profile, index, 8)
        for index in range(8)
    ]
    assert sum("composite_bundle" in plan for plan in plans) == 3
    assert sum("native_followup" in plan for plan in plans) == 2
    assert plans[0] == ["composite_bundle", "identity_explicit"]
    assert minimum_alignment_requirement_counts(profile) == {
        "identity_explicit": 1,
        "native_followup": 1,
        "composite_bundle": 2,
    }
    assert _alignment_generation_max_tokens(6, 16) >= 11_000
    assert _alignment_variant_requirements(
        {
            "skill_id": "html-tool",
            "routing_mode": "atomic",
            "aliases": ["html-tool"],
        },
        0,
        8,
    ) == ["core"]


def test_alignment_review_rejects_unmet_generation_requirement() -> None:
    row = {
        "query_id": "q1",
        "query_hash": "hash",
        "skill_ids": ["brainhole-factory"],
    }
    review = _validate_review(
        {
            "query_id": "q1",
            "scores": {
                "mobile_style": 5,
                "target_relevance": 5,
                "specificity": 5,
                "coherence": 5,
            },
            "missing": False,
            "extra_capability_needed": False,
            "requirement_satisfied": False,
            "unsafe": False,
            "pass": True,
            "issues": ["没有覆盖组合能力"],
        },
        row,
    )
    assert review["requirement_satisfied"] is False
    assert review["pass"] is False


def test_manual_alignment_is_transparently_marked(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.jsonl"
    queries = tmp_path / "queries.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    curated = tmp_path / "manual.jsonl"
    _write_jsonl(
        profiles,
        [
            {
                **_profile("weather"),
                "rank": 1,
                "owner": "test",
                "slug": "weather",
                "display_name": "天气",
                "summary": None,
                "description": "查询天气",
                "aliases": ["天气"],
                "capability_facets": ["查询天气"],
                "trigger_phrases": ["查询天气", "天气预报"],
                "negative_boundaries": [],
                "routing_mode": "atomic",
            }
        ],
    )
    _write_jsonl(queries, [])
    _write_jsonl(reviews, [])
    _write_jsonl(
        curated,
        [
            {
                "skill_id": "weather",
                "query": "帮我查一下杭州明天会不会下雨。",
                "generation_requirements": ["core"],
            }
        ],
    )

    result = append_manual_alignment_queries(
        profiles,
        queries,
        reviews,
        curated,
    )
    query = json.loads(queries.read_text().strip())
    review = json.loads(reviews.read_text().strip())
    assert result["added_query_count"] == 1
    assert query["curation_source"] == "manual_alignment"
    assert review["review_source"] == "manual_curation"
    assert review["model_pass"] is False


def test_legacy_alignment_import_fills_declared_deficit(
    tmp_path: Path,
) -> None:
    profiles = tmp_path / "profiles.jsonl"
    queries = tmp_path / "queries.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    legacy_queries = tmp_path / "legacy_queries.jsonl"
    legacy_reviews = tmp_path / "legacy_reviews.jsonl"
    coverage_failure = tmp_path / "coverage_failure.json"
    profile = {
        **_profile("weather"),
        "routing_mode": "atomic",
    }
    _write_jsonl(profiles, [profile])
    _write_jsonl(queries, [])
    _write_jsonl(reviews, [])
    _write_jsonl(
        legacy_queries,
        [
            {
                "query_id": "legacy-q1",
                "query_hash": "legacy-hash",
                "query": "帮我看看杭州明天会不会下雨。",
                "skill_ids": ["weather"],
                "evidence": {"weather": "杭州明天会不会下雨"},
                "target_intents": {"weather": "explicit"},
            }
        ],
    )
    _write_jsonl(
        legacy_reviews,
        [
            {
                "query_id": "legacy-q1",
                "query_hash": "legacy-hash",
                "skill_id": "weather",
                "scores": {
                    "mobile_style": 5,
                    "target_relevance": 5,
                    "specificity": 5,
                    "coherence": 5,
                },
                "pass": True,
            }
        ],
    )
    coverage_failure.write_text(
        json.dumps(
            {
                "min_train_positives_per_skill_required": 10,
                "skills_below_min_train_positives": {"weather": 9},
            }
        )
    )

    result = append_legacy_alignment_queries(
        profiles,
        queries,
        reviews,
        legacy_queries,
        legacy_reviews,
        coverage_failure,
    )
    query = json.loads(queries.read_text().strip())
    review = json.loads(reviews.read_text().strip())
    assert result["added_counts"] == {"weather": 1}
    assert query["curation_source"] == "legacy_alignment_review"
    assert review["review_source"] == "legacy_model_review"


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
            "data_schema_version": 2,
            "query_id": "q1",
            "query": "帮我查一下明天杭州的天气。",
            "skill_ids": ["@owner/one"],
            "primary_skill_ids": ["@owner/one"],
            "support_skill_ids": [],
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
            "data_schema_version": 2,
            "query_id": "q2",
            "query": "把这个页面保存到我的笔记里。",
            "skill_ids": ["@owner/two"],
            "primary_skill_ids": ["@owner/two"],
            "support_skill_ids": [],
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
