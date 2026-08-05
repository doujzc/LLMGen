from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmgen.clawhub_dataset import (
    DatasetBuildError,
    _augment_target_orders,
    _deduplicate_near_queries,
    _generation_repair_prompt,
    _normalized_profile_list,
    _parse_generation_payload_partial,
    _profile_prompt,
    _validate_generated_variant,
    _validate_profile,
    _validate_review,
    append_coverage_workflows,
    apply_recovery_workflows,
    build_workflow_specs,
    export_training_dataset,
    routing_profile_context,
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


def test_profile_lists_deterministically_keep_the_requested_limit() -> None:
    assert _normalized_profile_list(
        [f"能力切面{index}" for index in range(12)],
        field="capability_facets",
        minimum=1,
        maximum=10,
        max_chars=80,
    ) == [f"能力切面{index}" for index in range(10)]


def test_routing_context_replaces_opaque_id_with_real_product_alias() -> None:
    context = routing_profile_context(
        {
            **_profile(1),
            "skill_id": "polyv-live-cli",
            "display_name": "polyv-live-cli",
            "aliases": ["polyv-live-cli", "保利威直播CLI"],
            "trigger_phrases": ["用 polyv-live-cli 查询频道"],
            "description": "通过 polyv-live-cli 管理保利威直播",
        }
    )

    assert context["name"] == "保利威直播CLI"
    assert context["aliases"] == ["保利威直播CLI"]
    assert context["trigger_phrases"] == ["用 保利威直播CLI 查询频道"]
    assert context["original_description"] == (
        "通过 保利威直播CLI 管理保利威直播"
    )


def test_profile_preserves_routing_triggers_facets_and_display_name() -> None:
    skill = {
        "rank": 1,
        "skill_id": "pua",
        "owner": "data-light",
        "slug": "pua",
        "display_name": "调教AI人设",
        "summary": None,
        "description": "任务失败两次后继续读日志、换方法并逐项验证。",
    }
    raw = {
        "skill_id": "pua",
        "domain": "agent_system_automation",
        "roles": ["meta", "automate"],
        "capability_zh": "在任务反复失败时推动智能体换方法继续验证",
        "aliases": ["失败恢复助手"],
        "capability_facets": ["读取日志定位失败原因", "更换方案并逐项验证"],
        "trigger_phrases": ["已经失败两轮", "不要归因环境"],
        "negative_boundaries": ["不负责普通语气美化"],
        "routing_mode": "meta",
        "mobile_fit": "low",
        "unsafe_action": False,
    }
    profile = _validate_profile(raw, skill)
    assert profile["aliases"][0] == "调教AI人设"
    assert profile["routing_mode"] == "meta"
    assert profile["capability_facets"] == raw["capability_facets"]
    assert profile["trigger_phrases"] == raw["trigger_phrases"]


def test_profile_prompt_keeps_distinctive_description_beyond_old_limit() -> None:
    marker = "失败两轮后必须换方法继续验证"
    prompt = _profile_prompt(
        [
            {
                "skill_id": "meta",
                "display_name": "恢复助手",
                "description": "前置说明" * 100 + marker,
            }
        ]
    )
    assert marker in prompt
    assert "trigger_phrases" in prompt
    assert "routing_mode" in prompt


def test_workflows_keep_every_candidate_regardless_of_mobile_fit(tmp_path: Path) -> None:
    profiles = [_profile(index, "low" if index == 6 else "high") for index in range(1, 7)]
    source = tmp_path / "profiles.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in profiles))
    output = tmp_path / "workflows.jsonl"
    manifest = build_workflow_specs(
        source,
        output,
        workflows_per_skill=2,
        seed=7,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert manifest["profiled_skill_count"] == 6
    assert manifest["candidate_skill_count"] == 6
    assert manifest["candidate_filtering"] is False
    assert len(rows) == 12
    assert {row["anchor_skill_id"] for row in rows} == {
        f"@owner/skill-{index}" for index in range(1, 7)
    }


def test_recovery_workflows_are_idempotent_without_filtering_candidates(tmp_path: Path) -> None:
    profiles = [_profile(index) for index in range(1, 5)]
    source = tmp_path / "profiles.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in profiles))
    workflows = tmp_path / "workflows.jsonl"
    build_workflow_specs(source, workflows, workflows_per_skill=2, seed=7)
    config = tmp_path / "recovery.json"
    config.write_text(
        json.dumps(
            {
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
    assert first["candidate_skill_count"] == second["candidate_skill_count"] == 4
    assert first["candidate_filtering"] is False
    assert sum(bool(row.get("recovery")) for row in rows) == 1
    assert {row["anchor_skill_id"] for row in rows if not row.get("recovery")} == {
        f"@owner/skill-{index}" for index in range(1, 5)
    }


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
            {"skill_id": "@owner/weather", "unsafe_action": False},
            {"skill_id": "@owner/calendar", "unsafe_action": False},
        ],
    }


def test_generated_variant_requires_exact_evidence_for_every_target() -> None:
    raw = {
        "intent_mode": "explicit",
        "query": "明天下午去公园前先查一下会不会下雨，再把三点出发这件事加到日历里提醒我。",
        "evidence": {
            "@owner/weather": "查一下会不会下雨",
            "@owner/calendar": "加到日历里提醒我",
        },
        "implicit_skill_ids": [],
        "implicit_rationales": {},
    }
    row = _validate_generated_variant(
        raw, _workflow(), 0, variants=3, implicit_variants=1
    )
    assert row["skill_ids"] == ["@owner/weather", "@owner/calendar"]
    raw["evidence"].pop("@owner/calendar")
    with pytest.raises(DatasetBuildError, match="evidence keys"):
        _validate_generated_variant(
            raw, _workflow(), 0, variants=3, implicit_variants=1
        )


def test_generated_variant_allows_real_product_name_that_is_also_id() -> None:
    workflow = _workflow()
    workflow["skill_ids"] = ["polyv-live-cli", "@owner/calendar"]
    workflow["anchor_skill_id"] = "polyv-live-cli"
    workflow["targets"][0]["skill_id"] = "polyv-live-cli"
    query = (
        "用 polyv-live-cli 查今晚直播的推流状态，再把开播时间加到日历里提醒我。"
    )

    row = _validate_generated_variant(
        {
            "intent_mode": "explicit",
            "query": query,
            "evidence": {
                "polyv-live-cli": "查今晚直播的推流状态",
                "@owner/calendar": "加到日历里提醒我",
            },
            "implicit_skill_ids": [],
            "implicit_rationales": {},
        },
        workflow,
        0,
        variants=3,
        implicit_variants=1,
    )

    assert row["query"] == query


def test_generation_failure_retains_invalid_item_for_targeted_repair() -> None:
    workflow = _workflow()
    invalid_item = {
        "workflow_id": "wf-test",
        "variants": [
            {
                "intent_mode": "explicit",
                "query": "明天下午去公园前帮我查一下会不会下雨，回来以后告诉我结果。",
                "evidence": {"@owner/weather": "查一下会不会下雨"},
                "implicit_skill_ids": [],
                "implicit_rationales": {},
            }
        ]
        * 3,
    }

    rows, failures = _parse_generation_payload_partial(
        {"items": [invalid_item]},
        [workflow],
        variants=3,
        implicit_variants=1,
    )

    assert not rows
    assert failures[0]["invalid_item"] == invalid_item
    prompt = _generation_repair_prompt(
        workflow,
        invalid_item,
        failures[0]["error"],
        variants=3,
        implicit_variants=1,
    )
    assert "上一次输出没有通过" in prompt
    assert "evidence keys do not match target skills" in prompt
    assert invalid_item["variants"][0]["query"] in prompt


def test_generated_variant_ignores_numeric_payload_in_language_ratio() -> None:
    first_number = "1234567890" * 8
    second_number = "9876543210" * 8
    query = (
        f"帮我查一下编号{first_number}对应的活动明天会不会下雨，"
        f"再把订单号{second_number}写进下午三点的日历提醒里。"
    )
    row = _validate_generated_variant(
        {
            "intent_mode": "explicit",
            "query": query,
            "evidence": {
                "@owner/weather": "明天会不会下雨",
                "@owner/calendar": "写进下午三点的日历提醒里",
            },
            "implicit_skill_ids": [],
            "implicit_rationales": {},
        },
        _workflow(),
        0,
        variants=3,
        implicit_variants=1,
    )
    assert row["query"] == query


def test_generated_variant_supports_strong_implicit_intent() -> None:
    raw = {
        "intent_mode": "implicit",
        "query": "五一带孩子去杭州玩三天，怕下雨也不想爬山，今晚把合适的安排定下来。",
        "evidence": {
            "@owner/weather": "怕下雨",
            "@owner/calendar": "五一带孩子去杭州玩三天",
        },
        "implicit_skill_ids": ["@owner/weather"],
        "implicit_rationales": {
            "@owner/weather": "怕下雨这一限制要求结合当地天气筛选安排",
        },
    }
    row = _validate_generated_variant(
        raw, _workflow(), 2, variants=3, implicit_variants=1
    )
    assert row["intent_mode"] == "implicit"
    assert row["target_intents"] == {
        "@owner/weather": "implicit",
        "@owner/calendar": "explicit",
    }


def test_generated_variant_rejects_unsafe_implicit_target() -> None:
    workflow = _workflow()
    workflow["targets"][0]["unsafe_action"] = True
    raw = {
        "intent_mode": "implicit",
        "query": "五一带孩子去杭州玩三天，怕下雨也不想爬山，今晚把合适的安排定下来。",
        "evidence": {
            "@owner/weather": "怕下雨",
            "@owner/calendar": "五一带孩子去杭州玩三天",
        },
        "implicit_skill_ids": ["@owner/weather"],
        "implicit_rationales": {
            "@owner/weather": "怕下雨这一限制要求结合当地天气筛选安排",
        },
    }
    with pytest.raises(DatasetBuildError, match="unsafe actions"):
        _validate_generated_variant(
            raw, workflow, 2, variants=3, implicit_variants=1
        )


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


def _export_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    profiles = [_profile(index, "low" if index == 3 else "high") for index in range(1, 4)]
    catalog = [
        {
            **{key: row[key] for key in ("rank", "skill_id", "owner", "slug", "display_name", "summary")},
            "description": f"description {row['rank']}",
            "canonical_url": f"https://example.test/{row['rank']}",
        }
        for row in profiles
    ]
    workflows = [
        {"workflow_id": "wf-1", "split_hint": "train"},
        {"workflow_id": "wf-2", "split_hint": "train"},
    ]
    queries = [
        {
            "query_id": "q-1",
            "query": "先完成第一项任务，再继续处理第二项任务并告诉我结果",
            "query_hash": "hash-1",
            "workflow_id": "wf-1",
            "anchor_skill_id": "@owner/skill-1",
            "skill_ids": ["@owner/skill-1", "@owner/skill-2"],
            "domains": ["news_research", "documents_office"],
            "evidence": {
                "@owner/skill-1": "第一项任务",
                "@owner/skill-2": "第二项任务",
            },
        },
        {
            "query_id": "q-2",
            "query": "先处理第二项任务，完成后接着执行第三项任务并汇总",
            "query_hash": "hash-2",
            "workflow_id": "wf-2",
            "anchor_skill_id": "@owner/skill-3",
            "skill_ids": ["@owner/skill-2", "@owner/skill-3"],
            "domains": ["documents_office", "news_research"],
            "evidence": {
                "@owner/skill-2": "第二项任务",
                "@owner/skill-3": "第三项任务",
            },
        },
    ]
    reviews = [
        {
            "query_id": row["query_id"],
            "pass": True,
            "scores": {
                "mobile_style": 5,
                "complexity": 5,
                "target_necessity": 5,
                "coherence": 5,
                "specificity": 5,
            },
        }
        for row in queries
    ]
    paths = tuple(
        tmp_path / name
        for name in ("catalog.jsonl", "profiles.jsonl", "workflows.jsonl", "queries.jsonl", "reviews.jsonl")
    )
    for path, rows in zip(paths, (catalog, profiles, workflows, queries, reviews)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return paths


def test_export_keeps_all_catalog_skills_without_mobile_fit_filter(tmp_path: Path) -> None:
    paths = _export_fixture(tmp_path)
    output = tmp_path / "final"
    manifest = export_training_dataset(
        *paths,
        output,
        min_train_positives_per_skill=1,
    )
    skills = [json.loads(line) for line in (output / "skills.jsonl").read_text().splitlines()]
    assert manifest["candidate_count"] == 3
    assert manifest["candidate_policy"] == "retain_all_input_catalog_skills"
    assert {row["skill_id"] for row in skills} == {
        "@owner/skill-1",
        "@owner/skill-2",
        "@owner/skill-3",
    }
    train_rows = [
        json.loads(line)
        for line in (output / "queries_train.jsonl").read_text().splitlines()
    ]
    q1 = [row for row in train_rows if row["source_query_id"] == "q-1"]
    assert [row["skill_ids"] for row in q1] == [
        ["@owner/skill-1", "@owner/skill-2"],
        ["@owner/skill-2", "@owner/skill-1"],
    ]
    assert manifest["semantic_split_query_counts"]["train"] == 2
    assert manifest["split_query_counts"]["train"] == 4


def test_four_target_order_augmentation_exposes_every_target_first() -> None:
    rows, metrics = _augment_target_orders(
        [
            {
                "id": "q-four",
                "query": "完成一个由四项能力组成的连贯任务",
                "skill_ids": ["a", "b", "c", "d"],
            }
        ],
        variants=4,
        seed=7,
    )
    assert {row["skill_ids"][0] for row in rows} == {"a", "b", "c", "d"}
    assert metrics["augmented_train_query_count"] == 4


def test_export_does_not_replace_dataset_when_training_coverage_is_low(tmp_path: Path) -> None:
    paths = _export_fixture(tmp_path)
    output = tmp_path / "final"
    output.mkdir()
    sentinel = '{"skill_id":"existing"}\n'
    (output / "skills.jsonl").write_text(sentinel)
    with pytest.raises(DatasetBuildError, match="fewer than 2 train positives"):
        export_training_dataset(
            *paths,
            output,
            min_train_positives_per_skill=2,
        )
    assert (output / "skills.jsonl").read_text() == sentinel
    report = json.loads((output / "coverage_failure.json").read_text())
    assert report["skills_below_min_train_positives_count"] == 2


def test_export_does_not_replace_dataset_below_training_scale_gate(
    tmp_path: Path,
) -> None:
    paths = _export_fixture(tmp_path)
    output = tmp_path / "final"
    output.mkdir()
    sentinel = '{"skill_id":"existing"}\n'
    (output / "skills.jsonl").write_text(sentinel)
    with pytest.raises(DatasetBuildError, match="fewer than the required 5"):
        export_training_dataset(
            *paths,
            output,
            min_train_positives_per_skill=1,
            min_augmented_train_queries=5,
        )
    assert (output / "skills.jsonl").read_text() == sentinel
    report = json.loads((output / "training_scale_failure.json").read_text())
    assert report["augmented_train_query_count"] == 4
    assert report["min_augmented_train_queries_required"] == 5


def test_export_can_explicitly_exclude_unreviewed_queries(tmp_path: Path) -> None:
    paths = _export_fixture(tmp_path)
    queries_path = paths[3]
    queries = [json.loads(line) for line in queries_path.read_text().splitlines()]
    unreviewed = {
        **queries[0],
        "query_id": "q-unreviewed",
        "query_hash": "hash-unreviewed",
        "query": "先处理第一项，再把第二项结果整理成简短清单",
    }
    queries_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [*queries, unreviewed])
    )
    output = tmp_path / "final"

    with pytest.raises(DatasetBuildError, match="1 generated queries have no review"):
        export_training_dataset(
            *paths,
            output,
            min_train_positives_per_skill=1,
        )

    manifest = export_training_dataset(
        *paths,
        output,
        min_train_positives_per_skill=1,
        allow_missing_reviews=True,
        provisional_note="Test-only partial review export.",
    )
    assert manifest["export_status"] == "provisional"
    assert manifest["review_completion"] == {
        "generated_query_count": 3,
        "reviewed_query_count": 2,
        "missing_review_count": 1,
        "missing_review_query_ids": ["q-unreviewed"],
        "missing_review_policy": "exclude_unreviewed",
    }
    assert manifest["generated_query_count"] == 3
    assert manifest["reviewed_query_count"] == 2
    assert manifest["accepted_before_dedup"] == 2


def test_coverage_backfill_targets_undercovered_candidates_idempotently(tmp_path: Path) -> None:
    _, profiles, workflows, queries, reviews = _export_fixture(tmp_path)
    first = append_coverage_workflows(
        profiles,
        workflows,
        queries,
        reviews,
        min_train_positives_per_skill=2,
        variants_per_workflow=2,
        oversample_factor=1.0,
        round_index=1,
        seed=7,
    )
    second = append_coverage_workflows(
        profiles,
        workflows,
        queries,
        reviews,
        min_train_positives_per_skill=2,
        variants_per_workflow=2,
        oversample_factor=1.0,
        round_index=1,
        seed=7,
    )
    rows = [json.loads(line) for line in workflows.read_text().splitlines()]
    added = [row for row in rows if row.get("coverage_round") == 1]
    assert first["undercovered_skill_count"] == 2
    assert first["added_workflow_count"] == len(added) >= 1
    backfilled_ids = {skill_id for row in added for skill_id in row["skill_ids"]}
    assert {"@owner/skill-1", "@owner/skill-3"} <= backfilled_ids
    assert second["already_present"] is True
