from __future__ import annotations

import json

import pytest

from llmgen.router import RouterDataError
from llmgen.router_bundle import (
    BUNDLED_VIRTUAL_TOKENS_FILENAME,
    DECODE_MAP_FILENAME,
    build_skill_decode_map,
    dump_router_decoder_artifacts,
    load_skill_decode_map,
)


CATALOG = [
    {
        "skill_id": "s1",
        "name": "天气查询",
        "domain": "生活",
        "text": "天气查询 | 获取实时天气和预报",
    },
    {"skill_id": "s2", "name": "行程规划", "domain": "出行"},
    {"skill_id": "s3", "name": "日历提醒", "domain": "效率"},
]
CODE_ROWS = [
    {"skill_id": "s1", "indices": [0, 0], "tokens": ["<L1_0>", "<L2_0>"]},
    {"skill_id": "s2", "indices": [0, 0], "tokens": ["<L1_0>", "<L2_0>"]},
    {"skill_id": "s3", "indices": [1, 1], "tokens": ["<L1_1>", "<L2_1>"]},
]
REGISTRY = {"num_levels": 2, "buckets": {"0/0": ["s1", "s2"], "1/1": ["s3"]}}
TOKENS = ("<L1_0>", "<L1_1>", "<L2_0>", "<L2_1>")


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_decode_map_exposes_token_and_exact_path_names() -> None:
    payload = build_skill_decode_map(
        catalog_rows=CATALOG,
        code_rows=CODE_ROWS,
        registry=REGISTRY,
        virtual_tokens=TOKENS,
        supervision_rows=[
            {"target_skill_ids": ["s1", "s2"]},
            {"target_skill_ids": ["s1", "s3"]},
        ],
        supervision_phase="retrieval",
    )

    assert payload["num_skills"] == 3
    assert payload["num_paths"] == 2
    assert payload["skills"]["s1"]["text"] == "天气查询 | 获取实时天气和预报"
    assert payload["skills"]["s1"]["train_target_count"] == 2
    assert payload["skills"]["s2"]["train_target_count"] == 1
    assert payload["supervision"]["num_candidates"] == 3
    assert payload["token_to_candidates"]["<L1_0>"] == [
        {"skill_id": "s1", "name": "天气查询"},
        {"skill_id": "s2", "name": "行程规划"},
    ]
    assert payload["paths"][0]["candidates"] == [
        {"skill_id": "s1", "name": "天气查询"},
        {"skill_id": "s2", "name": "行程规划"},
    ]


def test_retrieval_decode_map_rejects_candidate_without_train_positive() -> None:
    with pytest.raises(RouterDataError, match="without train positives"):
        build_skill_decode_map(
            catalog_rows=CATALOG,
            code_rows=CODE_ROWS,
            registry=REGISTRY,
            virtual_tokens=TOKENS,
            supervision_rows=[{"target_skill_ids": ["s1", "s3"]}],
            supervision_phase="retrieval",
        )


def test_dump_router_decoder_artifacts_is_self_contained(tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    index = tmp_path / "index"
    index.mkdir()
    codes = index / "train_codes.jsonl"
    registry = index / "train_registry.json"
    tokens = index / "virtual_tokens.txt"
    training_data = tmp_path / "retrieval_train.jsonl"
    output = tmp_path / "model"
    _write_jsonl(catalog, CATALOG)
    _write_jsonl(codes, CODE_ROWS)
    registry.write_text(json.dumps(REGISTRY), encoding="utf-8")
    tokens.write_text("\n".join(TOKENS) + "\n", encoding="utf-8")
    _write_jsonl(training_data, [{"target_skill_ids": ["s1", "s2", "s3"]}])
    (index / "manifest.json").write_text(
        json.dumps({"checkpoint_sha256": "stage-one-hash"}), encoding="utf-8"
    )

    metadata = dump_router_decoder_artifacts(
        output_dir=output,
        catalog_path=catalog,
        codes_path=codes,
        registry_path=registry,
        virtual_tokens_path=tokens,
        training_data_path=training_data,
        supervision_phase="retrieval",
    )

    assert metadata["decode_map"] == DECODE_MAP_FILENAME
    assert metadata["virtual_tokens"] == BUNDLED_VIRTUAL_TOKENS_FILENAME
    assert (output / BUNDLED_VIRTUAL_TOKENS_FILENAME).read_text() == tokens.read_text()
    restored = load_skill_decode_map(output / DECODE_MAP_FILENAME)
    assert restored["provenance"]["stage1_checkpoint_sha256"] == "stage-one-hash"
    assert restored["skills"]["s3"]["name"] == "日历提醒"
    assert restored["supervision"]["phase"] == "retrieval"
    assert restored["supervision"]["num_candidates"] == 3
