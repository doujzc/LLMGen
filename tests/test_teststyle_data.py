from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from llmgen.teststyle_data import (
    DatasetBuildError,
    HeldoutLeakageGate,
    _source_occurrences,
    _validate_strict_review,
    _validate_variant,
    append_teststyle_coverage_workflows,
    build_distribution_profile,
    strict_review_teststyle_queries,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_heldout(path: Path, query: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Query", "expected技能1", "expected技能2"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Query": query,
                "expected技能1": "weather",
                "expected技能2": "travel",
            }
        )


def test_distribution_profile_persists_only_aggregate_heldout_data(tmp_path: Path) -> None:
    query = "查杭州五一天气并安排三天行程"
    heldout = tmp_path / "heldout.csv"
    profiles = tmp_path / "profiles.jsonl"
    output = tmp_path / "distribution.json"
    _write_heldout(heldout, query)
    _write_jsonl(
        profiles,
        [
            {"skill_id": "weather", "domain": "weather_environment"},
            {"skill_id": "travel", "domain": "travel_local"},
        ],
    )

    result = build_distribution_profile(heldout, profiles, output)
    serialized = output.read_text(encoding="utf-8")

    assert result["target_count_distribution"] == {"2": 1}
    assert result["privacy"]["contains_query_text"] is False
    assert query not in serialized
    assert '"weather"' not in serialized
    assert '"travel"' not in serialized


def test_heldout_gate_rejects_exact_and_near_matches(tmp_path: Path) -> None:
    heldout = tmp_path / "heldout.csv"
    _write_heldout(heldout, "查杭州五一天气并安排三天行程")
    gate = HeldoutLeakageGate.from_csv(heldout)

    assert gate.match("查杭州五一天气并安排三天行程")["kind"] == "exact"
    assert gate.match("查一下杭州五一天气并安排三天行程") is not None
    assert gate.match("把录音转成文字并保存为会议纪要") is None


def test_workflow_sources_exclude_heldout_exact_and_near_queries(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            {
                "id": "leaked",
                "query": "查杭州五一天气并安排三天行程",
                "skill_ids": ["weather", "travel"],
                "evidence": {"weather": "查杭州五一天气", "travel": "安排三天行程"},
            },
            {
                "id": "safe",
                "query": "把录音转成文字并保存为会议纪要",
                "skill_ids": ["asr", "docx"],
                "evidence": {"asr": "录音转成文字", "docx": "保存为会议纪要"},
            },
        ],
    )

    rows = _source_occurrences(
        (("fixture", source, None),),
        {"weather", "travel", "asr", "docx"},
        set(),
        HeldoutLeakageGate(["查杭州五一天气并安排三天行程"]),
    )

    assert len(rows) == 1
    assert rows[0]["source_query_id"] == "safe"


def test_teststyle_validator_normalizes_short_query_without_losing_evidence() -> None:
    workflow = {
        "workflow_id": "wf-ts-test",
        "anchor_skill_id": "weather",
        "skill_ids": ["weather", "travel"],
        "domains": ["weather_environment", "travel_local"],
        "cross_domain": True,
        "unsafe_action": False,
        "targets": [
            {"skill_id": "weather", "unsafe_action": False},
            {"skill_id": "travel", "unsafe_action": False},
        ],
    }
    row = _validate_variant(
        {
            "variant": 0,
            "intent_mode": "explicit",
            "query": "查杭州周末天气再安排西湖三日行程。",
            "evidence": {
                "weather": "查杭州周末天气",
                "travel": "安排西湖三日行程",
            },
            "implicit_skill_ids": [],
            "implicit_rationales": {},
        },
        workflow,
        0,
        {"minimum_characters": 20, "maximum_characters": 40},
        HeldoutLeakageGate([]),
    )

    assert 20 <= len(row["query"]) <= 40
    assert not row["query"].endswith("。")
    assert all(span in row["query"] for span in row["evidence"].values())


def test_strict_review_rejects_mere_conjunction_even_if_model_marks_pass() -> None:
    query = {
        "query_id": "q-1",
        "query_hash": "hash-1",
        "workflow_id": "wf-1",
        "intent_mode": "explicit",
        "skill_ids": ["fund", "tax"],
    }
    result = _validate_strict_review(
        {
            "query_id": "q-1",
            "target_support": {
                "fund": {"supported": True, "necessary": True},
                "tax": {"supported": True, "necessary": True},
            },
            "relationship": "mere_conjunction",
            "coherence_score": 5,
            "natural": True,
            "reason": "两个任务互不依赖，只是用并连接",
            "pass": True,
        },
        query,
    )

    assert result["pass"] is False
    assert result["relationship"] == "mere_conjunction"


class _CheckpointClient:
    def __init__(self, fail_after: int) -> None:
        self.config = SimpleNamespace(model="strict-test-model")
        self.fail_after = fail_after
        self.calls = 0

    def complete_json(self, prompt: str, *, max_tokens: int) -> dict:
        del max_tokens
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("provider unavailable")
        query_ids = re.findall(r'"query_id": "([^"]+)"', prompt)
        return {
            "items": [
                {
                    "query_id": query_id,
                    "target_support": {
                        "a": {"supported": True, "necessary": True},
                        "b": {"supported": True, "necessary": True},
                    },
                    "relationship": "dependency",
                    "coherence_score": 5,
                    "natural": True,
                    "reason": "前一步结果是后一步输入",
                    "pass": True,
                }
                for query_id in query_ids
            ]
        }

    def map(self, function, values, *, progress_label: str):
        del progress_label
        results = []
        errors = []
        for value in values:
            try:
                results.append(function(value))
            except Exception as error:
                errors.append({"input": value, "error": str(error)})
        return results, errors

    def usage_dict(self) -> dict[str, int]:
        return {"requests": self.calls}


def test_strict_review_checkpoints_and_resumes_after_provider_failure(
    tmp_path: Path,
) -> None:
    queries = tmp_path / "queries.jsonl"
    workflows = tmp_path / "workflows.jsonl"
    output = tmp_path / "reviews.jsonl"
    _write_jsonl(
        queries,
        [
            {
                "query_id": f"q-{index}",
                "query_hash": f"hash-{index}",
                "workflow_id": f"wf-{index}",
                "query": f"先完成任务{index}再处理后续结果",
                "intent_mode": "explicit",
                "implicit_skill_ids": [],
                "skill_ids": ["a", "b"],
            }
            for index in range(2)
        ],
    )
    _write_jsonl(
        workflows,
        [
            {
                "workflow_id": f"wf-{index}",
                "targets": [{"skill_id": "a"}, {"skill_id": "b"}],
            }
            for index in range(2)
        ],
    )

    with pytest.raises(DatasetBuildError, match="covered only 1/2"):
        strict_review_teststyle_queries(
            queries,
            workflows,
            output,
            _CheckpointClient(fail_after=1),
            batch_size=1,
            checkpoint_batches=1,
        )
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1

    strict_review_teststyle_queries(
        queries,
        workflows,
        output,
        _CheckpointClient(fail_after=10),
        batch_size=1,
        checkpoint_batches=1,
    )
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_strict_coverage_backfill_clones_only_passed_coherent_seed(
    tmp_path: Path,
) -> None:
    profiles = tmp_path / "profiles.jsonl"
    workflows = tmp_path / "workflows.jsonl"
    queries = tmp_path / "queries.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_jsonl(
        profiles,
        [
            {"skill_id": "a", "domain": "news_research", "unsafe_action": False},
            {"skill_id": "b", "domain": "documents_office", "unsafe_action": False},
        ],
    )
    _write_jsonl(
        workflows,
        [
            {
                "workflow_id": "wf-base",
                "anchor_skill_id": "a",
                "split_hint": "train",
                "skill_ids": ["a", "b"],
                "domains": ["news_research", "documents_office"],
                "targets": [{"skill_id": "a"}, {"skill_id": "b"}],
            }
        ],
    )
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q-base",
                "workflow_id": "wf-base",
                "skill_ids": ["a", "b"],
                "evidence": {"a": "搜索行业资料", "b": "生成分析文档"},
            }
        ],
    )
    _write_jsonl(
        reviews,
        [
            {
                "query_id": "q-base",
                "pass": True,
                "relationship": "dependency",
                "strict_reason": "搜索结果作为报告输入",
            }
        ],
    )

    result = append_teststyle_coverage_workflows(
        profiles,
        workflows,
        queries,
        reviews,
        minimum_train_per_skill=2,
        oversample_factor=1.0,
    )
    output = [json.loads(line) for line in workflows.read_text(encoding="utf-8").splitlines()]

    assert result["added_workflow_count"] == 2
    assert all(row["teststyle_coverage_backfill"] for row in output[1:])
    assert all(row["split_hint"] == "train" for row in output[1:])
    assert all(row["relationship_requirement"]["type"] == "dependency" for row in output[1:])
