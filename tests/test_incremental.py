from __future__ import annotations

from argparse import Namespace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from llmgen.incremental import (
    add_candidate,
    build_incremental_training_rows,
    compute_frozen_skill_code,
    frozen_rq_beam_paths,
    incremental_ancestor_hashes,
    load_candidate_state,
    remove_candidate,
    select_frozen_code,
    write_candidate_state,
)
from llmgen.router import RouterDataError
from llmgen.router_bundle import build_skill_decode_map
from llmgen.skillret import sha256_file
from scripts.infer_router import _validate_training_contract


TOKENS = ("<L1_0>", "<L1_1>", "<L2_0>", "<L2_1>")


def _base_map(*, collision: bool = False) -> dict:
    code_rows = [
        {
            "skill_id": "s1",
            "indices": [0, 0],
            "tokens": ["<L1_0>", "<L2_0>"],
        },
        {
            "skill_id": "s2",
            "indices": [0, 0] if collision else [1, 1],
            "tokens": (
                ["<L1_0>", "<L2_0>"]
                if collision
                else ["<L1_1>", "<L2_1>"]
            ),
        },
    ]
    buckets = (
        {"0/0": ["s1", "s2"]}
        if collision
        else {"0/0": ["s1"], "1/1": ["s2"]}
    )
    return build_skill_decode_map(
        catalog_rows=[
            {"skill_id": "s1", "name": "天气", "text": "天气 | 查询天气"},
            {"skill_id": "s2", "name": "日历", "text": "日历 | 创建提醒"},
        ],
        code_rows=code_rows,
        registry={"num_levels": 2, "buckets": buckets},
        virtual_tokens=TOKENS,
        provenance={"stage1_checkpoint_sha256": "stage1"},
        supervision_rows=[{"target_skill_ids": ["s1", "s2"]}],
        supervision_phase="retrieval",
    )


def test_remove_candidate_disables_only_its_unique_trie_path() -> None:
    source = _base_map()
    source_sha = "a" * 64

    updated, operation = remove_candidate(
        source,
        skill_id="s1",
        source_sha256=source_sha,
    )

    assert operation["skill_ids"] == ["s1"]
    assert set(updated["skills"]) == {"s2"}
    assert updated["num_paths"] == 1
    assert updated["paths"][0]["tokens"] == ["<L1_1>", "<L2_1>"]
    assert updated["supervision"] is None
    assert source_sha in incremental_ancestor_hashes(updated)


def test_remove_candidate_preserves_other_shared_path_members_by_default() -> None:
    source = _base_map(collision=True)

    updated, operation = remove_candidate(
        source,
        skill_id="s1",
        source_sha256="b" * 64,
    )

    assert operation["skill_ids"] == ["s1"]
    assert operation["shared_path"] is True
    assert set(updated["skills"]) == {"s2"}
    assert updated["num_paths"] == 1
    assert updated["paths"][0]["skill_ids"] == ["s2"]


def test_remove_candidate_can_disable_an_entire_shared_path() -> None:
    source = _base_map(collision=True)

    updated, operation = remove_candidate(
        {
            **source,
            "skills": {
                **source["skills"],
                "s3": {"skill_id": "s3", "name": "文件", "text": "管理文件"},
            },
            "skill_to_code": {
                **source["skill_to_code"],
                "s3": {
                    "indices": [1, 1],
                    "tokens": ["<L1_1>", "<L2_1>"],
                    "code_text": "<L1_1><L2_1>",
                },
            },
            "paths": [
                *source["paths"],
                {
                    "tokens": ["<L1_1>", "<L2_1>"],
                    "code_text": "<L1_1><L2_1>",
                    "skill_ids": ["s3"],
                    "candidates": [{"skill_id": "s3", "name": "文件"}],
                },
            ],
            "num_skills": 3,
            "num_paths": 2,
        },
        skill_id="s1",
        source_sha256="b" * 64,
        disable_shared_path=True,
    )
    assert operation["skill_ids"] == ["s1", "s2"]
    assert set(updated["skills"]) == {"s3"}


def test_add_candidate_keeps_every_old_code_and_rebuilds_decode_fields() -> None:
    source = _base_map()
    original_codes = json.loads(json.dumps(source["skill_to_code"]))

    updated, operation = add_candidate(
        source,
        skill={
            "skill_id": "s3",
            "name": "路线规划",
            "description": "规划公交和步行路线",
        },
        indices=[0, 1],
        tokens=["<L1_0>", "<L2_1>"],
        source_sha256="c" * 64,
        assignment_mode="nearest_available",
        update_mode="lora_train",
    )

    assert operation["update_mode"] == "lora_train"
    assert updated["skill_to_code"]["s1"] == original_codes["s1"]
    assert updated["skill_to_code"]["s2"] == original_codes["s2"]
    assert updated["skill_to_code"]["s3"]["code_text"] == "<L1_0><L2_1>"
    assert updated["skills"]["s3"]["text"] == "路线规划 | 规划公交和步行路线"
    assert updated["num_skills"] == 3
    assert updated["num_paths"] == 3


def test_frozen_assignment_can_skip_an_occupied_nearest_path() -> None:
    codebooks = [
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([[0.0], [1.0]], dtype=np.float32),
    ]
    ranked = frozen_rq_beam_paths([0.0], codebooks, beam_size=4)
    assert ranked[0][0] == (0, 0)

    path, _, collision = select_frozen_code(
        [0.0],
        codebooks,
        occupied_paths={(0, 0)},
        assignment_mode="nearest_available",
        beam_size=4,
    )
    assert path == (0, 1)
    assert collision is False


def test_compute_skill_code_uses_frozen_encoder_and_namespace(
    tmp_path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"frozen-stage-one")

    class Encoder(torch.nn.Module):
        def forward(self, values):
            return values[:, :1]

    model = SimpleNamespace(
        encoder=Encoder(),
        rq=SimpleNamespace(
            vq_layers=[
                SimpleNamespace(
                    embedding=SimpleNamespace(
                        weight=torch.tensor([[0.0], [1.0]])
                    )
                ),
                SimpleNamespace(
                    embedding=SimpleNamespace(
                        weight=torch.tensor([[0.0], [1.0]])
                    )
                ),
            ]
        ),
    )
    checkpoint = {
        "model_config": {
            "in_dim": 2,
            "num_emb_list": [2, 2],
            "token_format": "<L{level}_{index}>",
        },
        "training_config": {"normalize_embeddings": False},
    }
    monkeypatch.setattr(
        "llmgen.neural.toolweaver.load_toolweaver_rqvae",
        lambda *args, **kwargs: (model, checkpoint),
    )
    source = _base_map()
    source["provenance"] = {}

    code = compute_frozen_skill_code(
        skill={"skill_id": "s3"},
        stage1_checkpoint=checkpoint_path,
        source_decode_map=source,
        embedding=[0.0, 1.0],
        assignment_mode="nearest_available",
        assignment_beam_size=4,
        device="cpu",
    )

    assert code["indices"] == [0, 1]
    assert code["tokens"] == ["<L1_0>", "<L2_1>"]
    assert code["collision"] is False


def test_incremental_dataset_has_one_memorization_and_ten_retrieval_rows() -> None:
    source = _base_map()
    queries = [f"帮我查一下第{index}个城市明天的天气" for index in range(10)]

    memorization, retrieval = build_incremental_training_rows(
        source,
        skill_id="s1",
        queries=queries,
    )

    assert len(memorization) == 1
    assert len(retrieval) == 10
    assert memorization[0]["target_skill_ids"] == ["s1"]
    assert all(row["target_skill_ids"] == ["s1"] for row in retrieval)
    assert all(
        row["target_paths"] == [["<L1_0>", "<L2_0>"]]
        for row in retrieval
    )


def test_candidate_state_round_trip_and_router_lineage(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    base = _base_map()
    base_decode_path = model_dir / "skill_decode_map.json"
    base_decode_path.write_text(
        json.dumps(base, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    token_path = model_dir / "virtual_tokens.txt"
    token_path.write_text("\n".join(TOKENS) + "\n", encoding="utf-8")
    base_hash = sha256_file(base_decode_path)
    (model_dir / "router_manifest.json").write_text(
        json.dumps(
            {
                "generation_contract": {
                    "mode": "autoregressive_multi_path",
                    "path_separator": "\n",
                },
                "virtual_tokens_sha256": sha256_file(token_path),
                "decoder_artifacts": {"decode_map_sha256": base_hash},
                "max_length": 512,
            }
        ),
        encoding="utf-8",
    )

    updated, operation = add_candidate(
        base,
        skill={"skill_id": "s3", "name": "路线", "description": "规划路线"},
        indices=[0, 1],
        tokens=["<L1_0>", "<L2_1>"],
        source_sha256=base_hash,
        assignment_mode="nearest_available",
        update_mode="index_only",
    )
    state_dir = tmp_path / "state"
    write_candidate_state(state_dir, updated, operation)
    restored, restored_path, restored_tokens = load_candidate_state(state_dir)
    assert restored["num_skills"] == 3
    assert restored_tokens.read_text() == token_path.read_text()

    max_length = _validate_training_contract(
        Namespace(
            model_name_or_path=str(model_dir),
            virtual_tokens=str(restored_tokens),
            decode_map=str(restored_path),
        )
    )
    assert max_length == 512

    unrelated = json.loads(json.dumps(restored))
    unrelated.pop("incremental_state")
    restored_path.write_text(
        json.dumps(unrelated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RouterDataError, match="incremental ancestor"):
        _validate_training_contract(
            Namespace(
                model_name_or_path=str(model_dir),
                virtual_tokens=str(restored_tokens),
                decode_map=str(restored_path),
            )
        )
