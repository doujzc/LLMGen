from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmgen.router import RouterDataError
from llmgen.router_bundle import (
    BUNDLED_VIRTUAL_TOKENS_FILENAME,
    DECODE_MAP_FILENAME,
    build_skill_decode_map,
    dump_router_decoder_artifacts,
    load_skill_decode_map,
)
from scripts import export_router_bundle


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


def test_materialize_completed_checkpoint_for_web(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "router" / "retrieval" / "checkpoint-50"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"vocab_size": 8}), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(b"consolidated-weights")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 50, "epoch": 1.25}), encoding="utf-8"
    )

    tokenizer_source = tmp_path / "router" / "retrieval_alignment"
    tokenizer_source.mkdir()
    template_manifest = tokenizer_source / "router_manifest.json"
    template_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "phase": "retrieval",
                "finetune_mode": "full",
                "distributed": {"backend": "deepspeed", "world_size": 4},
            }
        ),
        encoding="utf-8",
    )

    catalog = tmp_path / "catalog.jsonl"
    index = tmp_path / "index"
    index.mkdir()
    codes = index / "train_codes.jsonl"
    registry = index / "train_registry.json"
    tokens = index / "virtual_tokens.txt"
    training_data = tmp_path / "router_data" / "retrieval_train.jsonl"
    replay_data = tmp_path / "router_data" / "memorization_train.jsonl"
    alignment_replay_data = (
        tmp_path / "router_data" / "retrieval_alignment_train.jsonl"
    )
    training_data.parent.mkdir()
    _write_jsonl(catalog, CATALOG)
    _write_jsonl(codes, CODE_ROWS)
    registry.write_text(json.dumps(REGISTRY), encoding="utf-8")
    tokens.write_text("\n".join(TOKENS) + "\n", encoding="utf-8")
    _write_jsonl(
        training_data,
        [
            {
                "phase": "retrieval",
                "target_paths": [
                    ["<L1_0>", "<L2_0>"],
                ],
                "target_skill_ids": ["s1", "s2"],
            },
            {
                "phase": "retrieval",
                "target_paths": [["<L1_0>", "<L2_0>"]],
                "target_skill_ids": ["s1"],
            },
        ],
    )
    _write_jsonl(
        replay_data,
        [
            {
                "phase": "memorization",
                "target_paths": [["<L1_1>", "<L2_1>"]],
                "target_skill_ids": ["s3"],
            }
        ],
    )
    _write_jsonl(
        alignment_replay_data,
        [
            {
                "phase": "retrieval",
                "target_paths": [["<L1_0>", "<L2_0>"]],
                "target_skill_ids": ["s2"],
            }
        ],
    )

    def fake_save_tokenizer(**kwargs):
        output = Path(kwargs["output_dir"])
        (output / "tokenizer_config.json").write_text(
            json.dumps({"added_tokens_decoder": {}}), encoding="utf-8"
        )
        return {
            "source": str(Path(kwargs["tokenizer_source"]).resolve()),
            "vocab_size": 8,
            "num_virtual_tokens": len(TOKENS),
        }

    monkeypatch.setattr(
        export_router_bundle,
        "_save_checkpoint_tokenizer",
        fake_save_tokenizer,
    )
    output = tmp_path / "exports" / "retrieval-checkpoint-50"
    result = export_router_bundle.materialize_checkpoint_bundle(
        checkpoint_dir=checkpoint,
        output_dir=output,
        tokenizer_source=tokenizer_source,
        catalog_path=catalog,
        codes_path=codes,
        registry_path=registry,
        virtual_tokens_path=tokens,
        training_data_path=training_data,
        validation_data_path=None,
        replay_data_path=replay_data,
        replay_fraction=0.25,
        phase="retrieval",
        num_levels=2,
        max_length=1024,
        seed=42,
        template_manifest_path=template_manifest,
        base_model_name_or_path="Qwen/Qwen3-1.7B",
        trust_remote_code=False,
        alignment_replay_data_path=alignment_replay_data,
        alignment_replay_fraction=0.25,
    )

    assert result["global_step"] == 50
    assert result["output_dir"] == str(output.resolve())
    assert (output / "model.safetensors").read_bytes() == b"consolidated-weights"
    assert (output / "tokenizer_config.json").is_file()
    assert (output / BUNDLED_VIRTUAL_TOKENS_FILENAME).is_file()
    restored = load_skill_decode_map(output / DECODE_MAP_FILENAME)
    assert restored["num_skills"] == 3
    manifest = json.loads((output / "router_manifest.json").read_text())
    assert manifest["phase"] == "retrieval"
    assert manifest["checkpoint_export"]["global_step"] == 50
    assert manifest["checkpoint_export"]["inference_mode"] == "full"
    assert manifest["generation_contract"]["max_target_paths"] == 1
    assert manifest["examples"] == {
        "train": 4,
        "primary_train": 2,
        "replay": 2,
        "replay_by_source": {"alignment": 1, "memorization": 1},
        "validation": 0,
    }
    assert manifest["replay_fraction_actual"] == 0.5
    assert manifest["replay_sources"]["alignment"]["repeat_factor"] == 1.0
    assert manifest["replay_sources"]["memorization"]["repeat_factor"] == 1.0
    assert restored["skills"]["s3"]["train_target_count"] == 1


def test_materialize_rejects_checkpoint_that_is_still_saving(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(RouterDataError, match="may still be saving"):
        export_router_bundle._checkpoint_step(checkpoint)


def test_checkpoint_tokenizer_reconstructs_virtual_namespace(
    tmp_path, monkeypatch
) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer_source = tmp_path / "tokenizer-source"
    tokenizer_source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    tokens = tmp_path / "virtual_tokens.txt"
    tokens.write_text("\n".join(TOKENS) + "\n", encoding="utf-8")
    config = output / "config.json"
    config.write_text(json.dumps({"vocab_size": 8}), encoding="utf-8")

    class FakeTokenizer:
        additional_special_tokens = []

        def __init__(self):
            self.ids = {}

        def __len__(self):
            return 4 + len(self.ids)

        def add_special_tokens(self, payload):
            self.additional_special_tokens = payload[
                "additional_special_tokens"
            ]
            for token in self.additional_special_tokens:
                self.ids.setdefault(token, 4 + len(self.ids))

        def encode(self, token, add_special_tokens=False):
            del add_special_tokens
            return [self.ids[token]]

        def save_pretrained(self, destination):
            Path(destination, "tokenizer_config.json").write_text(
                "{}", encoding="utf-8"
            )

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    metadata = export_router_bundle._save_checkpoint_tokenizer(
        tokenizer_source=tokenizer_source,
        output_dir=output,
        virtual_tokens_path=tokens,
        full_model_config_path=config,
        trust_remote_code=False,
    )

    assert metadata["vocab_size"] == 8
    assert metadata["num_virtual_tokens"] == 4
    assert (output / "tokenizer_config.json").is_file()
