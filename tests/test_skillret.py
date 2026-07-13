import json

import numpy as np
import pytest

from llmgen.skillret import (
    all_code_tokens,
    build_collaborative_edges,
    skill_text,
    validate_raw_dataset,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_collaborative_edges_are_train_couse_cosine_normalized():
    qrels = [
        {"query_id": "q1", "skill_id": "a"},
        {"query_id": "q1", "skill_id": "b"},
        {"query_id": "q2", "skill_id": "a"},
        {"query_id": "q2", "skill_id": "b"},
        {"query_id": "q2", "skill_id": "c"},
        {"query_id": "q2", "skill_id": "c"},  # duplicate is ignored
    ]

    src, dst, weight = build_collaborative_edges(["a", "b", "c"], qrels)

    np.testing.assert_array_equal(src, [0, 0, 1])
    np.testing.assert_array_equal(dst, [1, 2, 2])
    np.testing.assert_allclose(weight, [1.0, 1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_validate_raw_dataset_rejects_cross_split_skill_leakage(tmp_path):
    for split in ("train", "test"):
        _write_jsonl(tmp_path / "data/skills" / f"{split}.jsonl", [{"id": "same"}])
        _write_jsonl(
            tmp_path / "data/queries" / f"{split}.jsonl",
            [{"id": f"q-{split}", "query": "x", "skill_ids": ["same"]}],
        )
        _write_jsonl(
            tmp_path / "data/qrels" / f"{split}.jsonl",
            [{"query_id": f"q-{split}", "skill_id": "same", "relevance": 1}],
        )
    _write_jsonl(tmp_path / "data/skills.jsonl", [{"id": "same"}])
    (tmp_path / "data/taxonomy.json").write_text('{"root": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        validate_raw_dataset(tmp_path, strict_counts=False)


def test_skill_text_and_virtual_token_namespace():
    assert skill_text({"name": "N", "description": "D", "skill_md": "M"}) == "N | D | M"
    assert all_code_tokens([2, 1]) == ["<SK_L1_0>", "<SK_L1_1>", "<SK_L2_0>"]
