from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_clawhub import validate_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_versioned_clawhub_dataset_is_directly_trainable() -> None:
    validated = validate_dataset(REPO_ROOT / "data/clawhub_training/final")
    assert validated["counts"] == {
        "skills": 568,
        "queries_train": 3353,
        "qrels_train": 9707,
        "queries_validation": 448,
        "qrels_validation": 1344,
        "queries_test": 399,
        "qrels_test": 1197,
    }
    assert len(validated["skill_ids"]) == len(set(validated["skill_ids"]))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation_combines_alignment_with_unaugmented_train_coverage(
    tmp_path: Path,
) -> None:
    query = "帮我查一下杭州天气，再把结果记到笔记里。"
    train_queries = [
        {
            "id": f"q1-o{index}",
            "source_query_id": "q1",
            "query": query,
            "skill_ids": order,
            "workflow_id": "w1",
            "evidence": {"s1": "查一下杭州天气", "s2": "记到笔记里"},
            "intent_mode": "explicit",
            "implicit_skill_ids": [],
            "implicit_rationales": {},
            "target_intents": {"s1": "explicit", "s2": "explicit"},
        }
        for index, order in enumerate((["s1", "s2"], ["s2", "s1"]))
    ]
    files = {
        "skills.jsonl": [{"skill_id": "s1"}, {"skill_id": "s2"}],
        "queries_train.jsonl": train_queries,
        "qrels_train.jsonl": [
            {"query_id": row["id"], "skill_id": skill_id, "relevance": 1}
            for row in train_queries
            for skill_id in row["skill_ids"]
        ],
        "queries_validation.jsonl": [],
        "qrels_validation.jsonl": [],
        "queries_test.jsonl": [],
        "qrels_test.jsonl": [],
        "queries_alignment.jsonl": [
            {"id": "a1", "query": "帮我看看杭州明天会不会下雨。", "skill_ids": ["s1"]},
            {"id": "a2", "query": "把这段话存到我的笔记里。", "skill_ids": ["s2"]},
        ],
        "qrels_alignment.jsonl": [
            {"query_id": "a1", "skill_id": "s1", "relevance": 1},
            {"query_id": "a2", "skill_id": "s2", "relevance": 1},
        ],
    }
    for name, rows in files.items():
        _write_jsonl(tmp_path / name, rows)
    manifest = {
        "candidate_count": 2,
        "split_query_counts": {"train": 2, "validation": 0, "test": 0},
        "split_qrel_counts": {"train": 4, "validation": 0, "test": 0},
        "min_train_positives_per_skill_required": 2,
        "skills_below_min_train_positives_count": 0,
        "artifacts": {
            name: {
                "bytes": (tmp_path / name).stat().st_size,
                "sha256": _sha256(tmp_path / name),
            }
            for name in files
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    validated = validate_dataset(tmp_path)
    assert validated["counts"]["queries_alignment"] == 2

    # The two order variants represent one semantic positive, not two.
    manifest["min_train_positives_per_skill_required"] = 3
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fewer than 3"):
        validate_dataset(tmp_path)
