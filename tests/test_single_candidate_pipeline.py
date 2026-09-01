"""Single-candidate contracts for the concrete generic pipeline handlers.

The child programs are deliberately replaced here: this keeps the test local
and fast while exercising the real Stage DAG, artifact promotion, and handler
arguments that join those programs together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from llmgen.pipeline.config import load_pipeline_config
from llmgen.pipeline.io import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl
from llmgen.pipeline.ledger import JsonlShardLedger
from llmgen.pipeline.runner import create_pipeline_run
from llmgen.pipeline.stages.base import StageContext


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "router_pipeline.yaml"


def _option(argv: Sequence[str], name: str) -> Path:
    return Path(argv[argv.index(name) + 1])


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(path, rows)


def _empty_ledger(environment: Mapping[str, str], key: str) -> None:
    root = Path(environment[key])
    batch_key = (
        "LLMGEN_EMBEDDING_LEDGER_BATCH_RECORDS"
        if key == "LLMGEN_EMBEDDING_LEDGER_ROOT"
        else "LLMGEN_LLM_LEDGER_BATCH_RECORDS"
    )
    JsonlShardLedger(root, batch_size=int(environment[batch_key])).initialize()


def test_single_candidate_executes_alignment_path_and_retrieval_passthrough(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run concrete handlers through SFT and verify training passthrough aliases."""

    candidates = tmp_path / "candidate.jsonl"
    _jsonl(
        candidates,
        [{"id": "weather", "name": "Weather", "description": "Forecasts"}],
    )
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    atomic_write_json(base_model / "config.json", {"model_type": "test"})
    config = load_pipeline_config(
        DEFAULT_CONFIG,
        candidates=candidates,
        output=tmp_path / "run",
        overrides=(
            f"router.base_model={base_model}",
            "runtime.device=cpu",
            "runtime.num_devices=1",
            "runtime.deepspeed=none",
            "router.alignment.enabled=false",
            "router.alignment.epochs=0",
            "data_generation.alignment_backfill_rounds=0",
            "data_generation.final_alignment_backfill_rounds=0",
        ),
        environment={},
    )
    for key in ("GENERATION_API_KEY", "REVIEW_API_KEY", "EMBEDDING_API_KEY"):
        monkeypatch.setenv(key, "test-key")

    commands: list[tuple[str | None, list[str], dict[str, str]]] = []

    def fake_run_command(
        self: StageContext,
        argv: Sequence[str | Path],
        *,
        environment: Mapping[str, str] | None = None,
        label: str | None = None,
    ) -> None:
        command = [str(value) for value in argv]
        env = dict(environment or {})
        commands.append((label, command, env))
        script = command[1] if len(command) > 1 else ""

        if "LLMGEN_LLM_LEDGER_ROOT" in env:
            _empty_ledger(env, "LLMGEN_LLM_LEDGER_ROOT")
        if "LLMGEN_EMBEDDING_LEDGER_ROOT" in env:
            _empty_ledger(env, "LLMGEN_EMBEDDING_LEDGER_ROOT")

        if script.endswith("00_profile_skills.py"):
            _jsonl(_option(command, "--output"), [{"skill_id": "weather"}])
        elif script.endswith("02a_generate_alignment_queries.py"):
            _jsonl(
                _option(command, "--output"),
                [{"id": "alignment-weather", "skill_ids": ["weather"]}],
            )
        elif script.endswith("03a_review_alignment_queries.py"):
            _jsonl(_option(command, "--output"), [{"query_id": "alignment-weather", "passed": True}])
        elif script.endswith("04_export_dataset.py"):
            dataset = _option(command, "--output-dir")
            _jsonl(dataset / "skills.jsonl", [{"skill_id": "weather"}])
            atomic_write_json(dataset / "manifest.json", {"artifacts": {"skills.jsonl": {}}})
        elif script.endswith("04a_export_alignment.py"):
            dataset = _option(command, "--output-dir")
            _jsonl(
                dataset / "queries_alignment.jsonl",
                [{"id": "alignment-weather", "skill_ids": ["weather"]}],
            )
            _jsonl(
                dataset / "qrels_alignment.jsonl",
                [{"query_id": "alignment-weather", "skill_id": "weather", "relevance": 1}],
            )
        elif label == "generic-router-prepare":
            for name in ("PROCESSED_DIR", "EMBEDDING_DIR"):
                directory = Path(env[name])
                directory.mkdir(parents=True, exist_ok=True)
                atomic_write_json(directory / "manifest.json", {"schema_version": 1})
        elif label == "generic-router-train-tokenizer":
            stage1 = Path(env["STAGE1_DIR"])
            stage1.mkdir(parents=True, exist_ok=True)
            (stage1 / "best.pt").write_bytes(b"test checkpoint")
        elif label == "generic-router-export-codes":
            index = Path(env["INDEX_DIR"])
            index.mkdir(parents=True, exist_ok=True)
            _jsonl(index / "train_codes.jsonl", [{"skill_id": "weather", "code": [0]}])
            atomic_write_json(index / "train_registry.json", {"weather": [0]})
            (index / "virtual_tokens.txt").write_text("<code_0>\n", encoding="utf-8")
            atomic_write_json(index / "manifest.json", {"schema_version": 1})
        elif label == "generic-router-build-router-data":
            router_data = Path(env["ROUTER_DATA_DIR"])
            router_data.mkdir(parents=True, exist_ok=True)
            for name in (
                "memorization_train",
                "memorization_validation",
                "retrieval_alignment_train",
            ):
                _jsonl(router_data / f"{name}.jsonl", [])
            atomic_write_json(router_data / "manifest.json", {"schema_version": 1})
        elif label == "generic-router-train-memorization":
            model = Path(env["ROUTER_OUTPUT_DIR"]) / "memorization"
            model.mkdir(parents=True, exist_ok=True)
            atomic_write_json(model / "config.json", {"model_type": "test"})

    monkeypatch.setattr(StageContext, "run_command", fake_run_command)
    runner = create_pipeline_run(config, repo_root=REPO_ROOT)

    runner.run(to_stage="build-sft")
    generated = runner.registry.resolve("data.queries.generated")
    alignment = runner.registry.resolve("data.queries.alignment.generated")
    assert generated == runner.state.stage_dir("generate-queries") / "output" / "queries.generated.jsonl"
    assert read_jsonl(generated) == []
    assert read_json(generated.with_suffix(".manifest.json"))["execution_mode"] == "alignment_only"
    assert read_jsonl(alignment) == [{"id": "alignment-weather", "skill_ids": ["weather"]}]
    assert runner.registry.resolve("ledger.generate-queries.generation").is_dir()
    assert runner.registry.resolve("sft.directory") == runner.state.stage_dir("build-sft") / "output" / "router_data"
    assert runner.registry.resolve("codes.virtual_tokens").is_file()

    runner.run(from_stage="train-memorization", to_stage="train-retrieval")
    memorization = runner.registry.resolve("model.memorization")
    assert runner.registry.resolve("model.alignment") == memorization
    assert runner.registry.resolve("model.retrieval") == memorization
    assert runner.registry.all()["model.retrieval"].metadata == {
        "passthrough": True,
        "reason": "single-candidate alignment-only run",
    }

    labels = [label for label, _argv, _env in commands]
    assert "plan-workflows" not in labels
    assert "generate-multiskill-queries" not in labels
    assert "review-multiskill-queries" not in labels
    assert "generic-router-prepare" in labels
    assert "generic-router-build-router-data" in labels
    assert "generic-router-train-memorization" in labels
    generated_call = next(call for call in commands if call[0] == "generate-alignment-queries")
    assert _option(generated_call[1], "--profiles") == runner.registry.resolve("data.profiles")
    assert _option(generated_call[1], "--output") == (
        runner.state.stage_dir("generate-queries")
        / "attempts"
        / "0001"
        / "output"
        / "queries.alignment.generated.jsonl"
    )
    assert generated_call[2]["LLMGEN_LLM_LEDGER_BATCH_RECORDS"] == str(
        config.require("checkpointing.llm_batch_records")
    )
