from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from llmgen.router import code_token_id_map, encode_target_only_example


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROUTER = REPOSITORY_ROOT / "scripts" / "train_router.py"
INFER_ROUTER = REPOSITORY_ROOT / "scripts" / "infer_router.py"
EXPORT_ROUTER = REPOSITORY_ROOT / "scripts" / "export_router_bundle.py"
MERGE_ROUTER_ADAPTER = REPOSITORY_ROOT / "scripts" / "merge_router_adapter.py"
VIRTUAL_TOKENS = ("<R0>", "<R1>")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_local_causal_lm(path: Path) -> int:
    """Create a complete HF model/tokenizer pair without touching the network."""

    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    # ByteLevel retains the newline as the ``Ċ`` token, which exercises the
    # router's multi-path separator contract in addition to its code tokens.
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "Ċ": 4,
        "System": 5,
        "User": 6,
        "Assistant": 7,
        ":": 8,
        "route": 9,
        "alpha": 10,
        "skill": 11,
    }
    raw_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    assert tokenizer.encode("\n", add_special_tokens=False)
    path.mkdir()
    tokenizer.save_pretrained(path)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    ).save_pretrained(path)
    return len(tokenizer)


def _run_script(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    # The test passes only local paths. These settings turn an accidental Hub
    # resolution into a deterministic failure instead of a network dependency.
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
        }
    )
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _training_arguments(
    *,
    model: Path,
    output: Path,
    virtual_tokens: Path,
    train_data: Path,
    catalog: Path,
    codes: Path,
    registry: Path,
    epochs: int,
) -> list[str]:
    return [
        "--model-name-or-path",
        str(model),
        "--virtual-tokens",
        str(virtual_tokens),
        "--output-dir",
        str(output),
        "--stage",
        "retrieval",
        "--retrieval-train",
        str(train_data),
        "--num-levels",
        "2",
        "--max-length",
        "48",
        "--per-device-train-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--retrieval-epochs",
        str(epochs),
        "--retrieval-learning-rate",
        "0.001",
        "--logging-steps",
        "1",
        "--save-steps",
        "1",
        "--eval-steps",
        "1",
        "--save-total-limit",
        "2",
        "--skill-catalog",
        str(catalog),
        "--skill-codes",
        str(codes),
        "--skill-registry",
        str(registry),
    ]


def test_local_tiny_causal_lm_training_resume_constrained_inference_and_export(
    tmp_path: Path,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = tmp_path / "base-model"
    base_vocab_size = _build_local_causal_lm(base_model)
    router_data = tmp_path / "router-data"
    index = tmp_path / "index"
    router_data.mkdir()
    index.mkdir()
    catalog = tmp_path / "catalog.jsonl"
    codes = index / "train_codes.jsonl"
    registry = index / "train_registry.json"
    virtual_tokens = index / "virtual_tokens.txt"
    train_data = router_data / "retrieval_train.jsonl"
    queries = tmp_path / "queries.jsonl"

    _write_jsonl(
        catalog,
        [
            {
                "skill_id": "alpha",
                "name": "Alpha",
                "description": "Route alpha requests.",
            }
        ],
    )
    _write_jsonl(
        codes,
        [
            {
                "skill_id": "alpha",
                "indices": [0, 0],
                "tokens": list(VIRTUAL_TOKENS),
            }
        ],
    )
    registry.write_text(
        json.dumps({"num_levels": 2, "buckets": {"0/0": ["alpha"]}}),
        encoding="utf-8",
    )
    virtual_tokens.write_text("\n".join(VIRTUAL_TOKENS) + "\n", encoding="utf-8")
    training_row = {
        "phase": "retrieval",
        "group_id": "q1",
        "query_id": "q1",
        "input_text": "route alpha",
        "target_paths": [list(VIRTUAL_TOKENS)],
        "target_tokens": list(VIRTUAL_TOKENS),
        "target_skill_ids": ["alpha"],
    }
    _write_jsonl(train_data, [training_row])
    _write_jsonl(queries, [{"id": "q1", "query": "route alpha"}])

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(VIRTUAL_TOKENS)})
    target_ids = code_token_id_map(tokenizer, VIRTUAL_TOKENS)
    encoded = encode_target_only_example(
        tokenizer,
        training_row,
        code_token_ids=target_ids,
        num_levels=2,
        max_length=48,
        system_prompt="route skill",
    )
    first_target = next(
        index for index, label in enumerate(encoded["labels"]) if label != -100
    )
    assert encoded["labels"][:first_target] == [-100] * first_target
    assert encoded["labels"][first_target:] == [
        target_ids["<R0>"],
        target_ids["<R1>"],
        tokenizer.eos_token_id,
    ]

    output = tmp_path / "router"
    first_train = _run_script(
        TRAIN_ROUTER,
        *_training_arguments(
            model=base_model,
            output=output,
            virtual_tokens=virtual_tokens,
            train_data=train_data,
            catalog=catalog,
            codes=codes,
            registry=registry,
            epochs=1,
        ),
    )
    assert first_train.returncode == 0, first_train.stderr
    retrieval = output / "retrieval"
    checkpoint_one = retrieval / "checkpoint-1"
    assert (checkpoint_one / "trainer_state.json").is_file()
    assert (retrieval / "router_manifest.json").is_file()
    trained_model = AutoModelForCausalLM.from_pretrained(retrieval, local_files_only=True)
    assert trained_model.get_input_embeddings().num_embeddings == base_vocab_size + 2

    resumed = _run_script(
        TRAIN_ROUTER,
        *_training_arguments(
            model=base_model,
            output=output,
            virtual_tokens=virtual_tokens,
            train_data=train_data,
            catalog=catalog,
            codes=codes,
            registry=registry,
            epochs=2,
        ),
        "--resume-retrieval-from-checkpoint",
        str(checkpoint_one),
    )
    assert resumed.returncode == 0, resumed.stderr
    resumed_state = retrieval / "checkpoint-2" / "trainer_state.json"
    assert resumed_state.is_file()
    assert json.loads(resumed_state.read_text(encoding="utf-8"))["global_step"] == 2

    predictions = tmp_path / "predictions.jsonl"
    inference = _run_script(
        INFER_ROUTER,
        "--model-name-or-path",
        str(retrieval),
        "--virtual-tokens",
        str(virtual_tokens),
        "--codes",
        str(codes),
        "--registry",
        str(registry),
        "--queries",
        str(queries),
        "--output-jsonl",
        str(predictions),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--max-code-paths",
        "1",
        "--top-k",
        "1",
    )
    assert inference.returncode == 0, inference.stderr
    prediction = json.loads(predictions.read_text(encoding="utf-8"))
    assert len(prediction["paths"]) == 1
    assert prediction["paths"][0]["code_tokens"] == list(VIRTUAL_TOKENS)
    assert prediction["paths"][0]["code_text"] == "<R0><R1>"
    assert isinstance(prediction["paths"][0]["score"], float)
    assert prediction["paths"][0]["skill_ids"] == ["alpha"]
    assert prediction["skill_ids"] == ["alpha"]

    exported = _run_script(
        EXPORT_ROUTER,
        "--model-dir",
        str(retrieval),
        "--catalog",
        str(catalog),
        "--codes",
        str(codes),
        "--registry",
        str(registry),
        "--virtual-tokens",
        str(virtual_tokens),
        "--training-data",
        str(train_data),
        "--phase",
        "retrieval",
    )
    assert exported.returncode == 0, exported.stderr
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "router_manifest.json",
        "skill_decode_map.json",
        "virtual_tokens.txt",
    ):
        assert (retrieval / name).is_file()
    assert not (retrieval / "adapter_config.json").exists()

    portable = tmp_path / "portable-router"
    shutil.copytree(retrieval, portable)
    portable_predictions = tmp_path / "portable-predictions.jsonl"
    portable_inference = _run_script(
        INFER_ROUTER,
        "--model-name-or-path",
        str(portable),
        "--candidate-state-dir",
        str(portable),
        "--virtual-tokens",
        str(portable / "virtual_tokens.txt"),
        "--query",
        "route alpha",
        "--output-jsonl",
        str(portable_predictions),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--max-code-paths",
        "1",
        "--top-k",
        "1",
    )
    assert portable_inference.returncode == 0, portable_inference.stderr
    portable_prediction = json.loads(
        portable_predictions.read_text(encoding="utf-8")
    )
    assert portable_prediction["skill_ids"] == ["alpha"]


def test_local_tiny_lora_training_merges_to_portable_constrained_router(
    tmp_path: Path,
) -> None:
    base_model = tmp_path / "base-model"
    _build_local_causal_lm(base_model)
    router_data = tmp_path / "router-data"
    index = tmp_path / "index"
    router_data.mkdir()
    index.mkdir()
    catalog = tmp_path / "catalog.jsonl"
    codes = index / "train_codes.jsonl"
    registry = index / "train_registry.json"
    virtual_tokens = index / "virtual_tokens.txt"
    train_data = router_data / "retrieval_train.jsonl"
    _write_jsonl(
        catalog,
        [
            {
                "skill_id": "alpha",
                "name": "Alpha",
                "description": "Route alpha requests.",
            }
        ],
    )
    _write_jsonl(
        codes,
        [
            {
                "skill_id": "alpha",
                "indices": [0, 0],
                "tokens": list(VIRTUAL_TOKENS),
            }
        ],
    )
    registry.write_text(
        json.dumps({"num_levels": 2, "buckets": {"0/0": ["alpha"]}}),
        encoding="utf-8",
    )
    virtual_tokens.write_text("\n".join(VIRTUAL_TOKENS) + "\n", encoding="utf-8")
    _write_jsonl(
        train_data,
        [
            {
                "phase": "retrieval",
                "group_id": "q1",
                "query_id": "q1",
                "input_text": "route alpha",
                "target_paths": [list(VIRTUAL_TOKENS)],
                "target_tokens": list(VIRTUAL_TOKENS),
                "target_skill_ids": ["alpha"],
            }
        ],
    )

    output = tmp_path / "router"
    trained = _run_script(
        TRAIN_ROUTER,
        *_training_arguments(
            model=base_model,
            output=output,
            virtual_tokens=virtual_tokens,
            train_data=train_data,
            catalog=catalog,
            codes=codes,
            registry=registry,
            epochs=1,
        ),
        "--lora",
        "--lora-target-modules",
        "c_attn",
    )
    assert trained.returncode == 0, trained.stderr
    adapter = output / "retrieval"
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    assert (adapter / "skill_decode_map.json").is_file()

    merged = tmp_path / "merged-router"
    merge = _run_script(
        MERGE_ROUTER_ADAPTER,
        "--base-model",
        str(base_model),
        "--adapter",
        str(adapter),
        "--output-dir",
        str(merged),
    )
    assert merge.returncode == 0, merge.stderr
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "router_manifest.json",
        "skill_decode_map.json",
        "virtual_tokens.txt",
    ):
        assert (merged / name).is_file()
    assert not (merged / "adapter_config.json").exists()

    import service
    import service_910b
    import service_openai

    for serving_module in (service, service_openai, service_910b):
        serving_module._validate_full_model_bundle(merged)
        assert serving_module._load_candidate_bundle(merged).skills["alpha"][
            "name"
        ] == "Alpha"

    predictions = tmp_path / "merged-predictions.jsonl"
    inference = _run_script(
        INFER_ROUTER,
        "--model-name-or-path",
        str(merged),
        "--candidate-state-dir",
        str(merged),
        "--virtual-tokens",
        str(merged / "virtual_tokens.txt"),
        "--query",
        "route alpha",
        "--output-jsonl",
        str(predictions),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--max-code-paths",
        "1",
        "--top-k",
        "1",
    )
    assert inference.returncode == 0, inference.stderr
    assert json.loads(predictions.read_text(encoding="utf-8"))["skill_ids"] == [
        "alpha"
    ]
