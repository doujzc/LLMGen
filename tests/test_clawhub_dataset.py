from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmgen.clawhub_dataset import (
    DatasetBuildError,
    _deduplicate_near_queries,
    _validate_generated_variant,
    _validate_profile,
    _validate_review,
    apply_recovery_workflows,
    build_workflow_specs,
)


def _profile(rank: int, fit: str = "high") -> dict:
    roles = ["retrieve"] if rank % 2 else ["create"]
    return {
        "rank": rank,
        "skill_id": f"@owner/skill-{rank}",
        "owner": "owner",
        "slug": f"skill-{rank}",
        "display_name": f"skill {rank}",
        "summary": f"capability {rank}",
        "description": None,
        "domain": "news_research" if rank % 2 else "documents_office",
        "roles": roles,
        "capability_zh": f"完成第{rank}项独立用户任务",
        "mobile_fit": fit,
        "unsafe_action": False,
    }


def test_profile_accepts_common_capability_alias() -> None:
    skill = {
        "rank": 1,
        "skill_id": "@owner/weather",
        "owner": "owner",
        "slug": "weather",
        "display_name": "weather",
        "summary": "forecast",
        "description": None,
    }
    raw = {
        "skill_id": "@owner/weather",
        "domain": "weather_environment",
        "roles": ["retrieve", "perceive"],
        "capability_z": "查询实时天气和未来预报",
        "mobile_fit": "high",
        "unsafe_action": False,
    }
    assert _validate_profile(raw, skill)["capability_zh"] == "查询实时天气和未来预报"


def test_workflows_exclude_low_mobile_fit_targets(tmp_path: Path) -> None:
    profiles = [_profile(index, "low" if index == 6 else "high") for index in range(1, 7)]
    source = tmp_path / "profiles.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in profiles))
    output = tmp_path / "workflows.jsonl"
    manifest = build_workflow_specs(
        source,
        output,
        workflows_per_skill=2,
        min_mobile_fit="medium",
        seed=7,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert manifest["candidate_skill_count"] == 6
    assert manifest["targetable_skill_count"] == 5
    assert len(rows) == 10
    assert all("@owner/skill-6" not in row["skill_ids"] for row in rows)


def test_recovery_workflows_are_idempotent_and_exclude_candidate_only(tmp_path: Path) -> None:
    profiles = [_profile(index) for index in range(1, 5)]
    source = tmp_path / "profiles.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in profiles))
    workflows = tmp_path / "workflows.jsonl"
    build_workflow_specs(source, workflows, workflows_per_skill=2, seed=7)
    config = tmp_path / "recovery.json"
    config.write_text(
        json.dumps(
            {
                "candidate_only": [{"skill_id": "@owner/skill-4", "reason": "meta"}],
                "recovery_workflows": [
                    {
                        "anchor_skill_id": "@owner/skill-1",
                        "collaborator_skill_ids": ["@owner/skill-2"],
                        "note": "coherent recovery",
                    }
                ],
            }
        )
    )
    first = apply_recovery_workflows(source, workflows, config)
    second = apply_recovery_workflows(source, workflows, config)
    rows = [json.loads(line) for line in workflows.read_text().splitlines()]
    assert first["recovery_workflow_count"] == second["recovery_workflow_count"] == 1
    assert sum(bool(row.get("recovery")) for row in rows) == 1
    assert "@owner/skill-4" not in rows[-1]["skill_ids"]


def _workflow() -> dict:
    return {
        "workflow_id": "wf-test",
        "anchor_skill_id": "@owner/weather",
        "anchor_round": 0,
        "skill_ids": ["@owner/weather", "@owner/calendar"],
        "domains": ["weather_environment", "productivity_planning"],
        "cross_domain": True,
        "unsafe_action": False,
        "targets": [
            {"skill_id": "@owner/weather"},
            {"skill_id": "@owner/calendar"},
        ],
    }


def test_generated_variant_requires_exact_evidence_for_every_target() -> None:
    raw = {
        "query": "明天下午去公园前先查一下会不会下雨，再把三点出发这件事加到日历里提醒我。",
        "evidence": {
            "@owner/weather": "查一下会不会下雨",
            "@owner/calendar": "加到日历里提醒我",
        },
    }
    row = _validate_generated_variant(raw, _workflow(), 0)
    assert row["skill_ids"] == ["@owner/weather", "@owner/calendar"]
    raw["evidence"].pop("@owner/calendar")
    with pytest.raises(DatasetBuildError, match="evidence keys"):
        _validate_generated_variant(raw, _workflow(), 0)


def test_review_pass_is_recomputed_from_strict_thresholds() -> None:
    query = {"query_id": "q1", "workflow_id": "wf", "skill_ids": ["a", "b"]}
    raw = {
        "query_id": "q1",
        "scores": {
            "mobile_style": 4,
            "complexity": 3,
            "target_necessity": 5,
            "coherence": 5,
            "specificity": 4,
        },
        "missing_skill_ids": [],
        "redundant_skill_ids": [],
        "unsafe": False,
        "pass": True,
        "issues": [],
    }
    assert _validate_review(raw, query)["pass"] is False


def test_review_with_issue_is_rejected_even_when_scores_pass() -> None:
    query = {"query_id": "q1", "workflow_id": "wf", "skill_ids": ["a", "b"]}
    raw = {
        "query_id": "q1",
        "scores": {key: 5 for key in (
            "mobile_style", "complexity", "target_necessity", "coherence", "specificity"
        )},
        "missing_skill_ids": [],
        "redundant_skill_ids": [],
        "unsafe": False,
        "pass": True,
        "issues": ["target未表达"],
    }
    assert _validate_review(raw, query)["pass"] is False


def test_near_duplicate_filter_keeps_higher_review_score() -> None:
    rows = [
        {"query_id": "q-low", "query": "明天下午出门前查天气再把行程加到日历提醒我"},
        {"query_id": "q-high", "query": "明天下午出门前查天气，再把行程加到日历提醒我"},
    ]
    reviews = {
        "q-low": {"scores": {key: 4 for key in ("a", "b")}},
        "q-high": {"scores": {"a": 5, "b": 5}},
    }
    accepted, rejected = _deduplicate_near_queries(rows, reviews, threshold=0.8)
    assert [row["query_id"] for row in accepted] == ["q-high"]
    assert rejected[0]["query_id"] == "q-low"
