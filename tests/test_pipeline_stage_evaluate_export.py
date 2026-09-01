from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmgen.pipeline.stages import evaluate, export


class _Config:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.hash = "config-hash"

    def get(self, key: str):
        return self.values.get(key)

    def require(self, key: str):
        return self.values[key]


class _State:
    def __init__(self, root: Path) -> None:
        self.root = root

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage


def _decode_map(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "num_levels": 1,
                "skills": {"alpha": {"name": "Alpha"}},
                "skill_to_code": {"alpha": {"tokens": ["<0>"]}},
                "paths": [{"tokens": ["<0>"], "skill_ids": ["alpha"]}],
                "virtual_tokens": ["<0>"],
            }
        ),
        encoding="utf-8",
    )


def _context(tmp_path: Path, *, stage: str) -> SimpleNamespace:
    run_dir = tmp_path / "run"
    model = tmp_path / "model"
    source = run_dir / "source"
    source.mkdir(parents=True)
    model.mkdir()
    _decode_map(model / "skill_decode_map.json")
    (source / "candidate_manifest.json").write_text(
        json.dumps({"candidate_count": 1}), encoding="utf-8"
    )
    values: dict[str, object] = {
        "export.output_dir": "export/model",
        "evaluation.metric_thresholds": {"recall@1": 0.5},
        "evaluation.require_format_valid_rate": 1.0,
        "evaluation.require_candidate_coverage": 1.0,
        "export.require_all_gates": True,
        "export.allow_failed_gates": False,
        "export.smoke_test": False,
        "runtime.device": "cpu",
        "evaluation.dtype": "float32",
        "router.trust_remote_code": False,
    }
    artifacts = {
        "model.retrieval": model,
        "candidates.manifest": source / "candidate_manifest.json",
    }
    return SimpleNamespace(
        run_dir=run_dir,
        output_dir=run_dir / "stages" / stage / "attempts" / "0001" / "output",
        attempt_dir=run_dir / "stages" / stage / "attempts" / "0001",
        spec=SimpleNamespace(name=stage),
        state=_State(run_dir / "stages"),
        config=_Config(values),
        artifact=lambda name: artifacts[name],
    )


def test_evaluate_stage_writes_the_same_audited_artifact_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path, stage="evaluate")
    events: list[str] = []

    def verify(context) -> None:
        events.append("verified")

    def run_router(context, command, *, environment_overrides):
        assert events == ["verified"]
        assert command == "evaluate"
        assert environment_overrides == {
            "ROUTER_EVAL_MODEL_DIR": str(context.artifact("model.retrieval")),
            "EVAL_WORK_DIR": str(context.attempt_dir / "evaluation-work"),
        }
        evaluation_dir = context.output_dir / "evaluation"
        evaluation_dir.mkdir(parents=True)
        (evaluation_dir / "metrics.json").write_text(
            json.dumps({"metrics": {"recall@1": 1.0}}), encoding="utf-8"
        )
        (evaluation_dir / "predictions.jsonl").write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "paths": [
                        {"code_tokens": ["<0>"], "skill_ids": ["alpha"]}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(evaluate, "verify_training_provenance", verify)
    monkeypatch.setattr(evaluate, "router_pipeline", run_router)

    result = evaluate.evaluate(context)

    assert [artifact.logical_name for artifact in result.artifacts] == [
        "evaluation.directory",
        "evaluation.audit",
        "evaluation.quality_gates",
        "evaluation.failed_examples",
        "evaluation.metrics",
        "evaluation.predictions",
    ]
    quality = json.loads(
        (context.output_dir / "evaluation" / "quality_gates.json").read_text(
            encoding="utf-8"
        )
    )
    assert quality["passed"] is True


def test_evaluate_refuses_provenance_drift_before_model_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path, stage="evaluate")
    model_called = False

    def fail_verification(context) -> None:
        raise ValueError("base model provenance drift")

    def run_router(*args, **kwargs) -> None:
        nonlocal model_called
        model_called = True

    monkeypatch.setattr(evaluate, "verify_training_provenance", fail_verification)
    monkeypatch.setattr(evaluate, "router_pipeline", run_router)

    with pytest.raises(ValueError, match="provenance drift"):
        evaluate.evaluate(context)
    assert model_called is False


def test_export_refuses_provenance_drift_before_lora_merge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path, stage="export")
    context.config.values.update(
        {
            "router.finetune_mode": "lora",
            "router.base_model": str(tmp_path / "base-model"),
        }
    )
    command_called = False

    def fail_verification(context) -> None:
        raise ValueError("base model provenance drift")

    def run_command(*args, **kwargs) -> None:
        nonlocal command_called
        command_called = True

    context.run_command = run_command
    monkeypatch.setattr(export, "verify_training_provenance", fail_verification)

    with pytest.raises(ValueError, match="provenance drift"):
        export.export(context)
    assert command_called is False


def test_export_helpers_preserve_copy_and_quality_gate_contracts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "checkpoint-1").mkdir()
    (source / "checkpoint-1" / "ignored").write_text("x", encoding="utf-8")
    copied = tmp_path / "copied-model"
    export.copy_model_tree(source, copied)
    assert (copied / "model.safetensors").read_bytes() == b"weights"
    assert (copied / "config.json").read_text(encoding="utf-8") == "{}"
    assert not (copied / "checkpoint-1").exists()

    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors"}}),
        encoding="utf-8",
    )
    (shards / "model-00001.safetensors").write_bytes(b"shard")
    assert export.full_weights_are_present(shards, export.root_weight_files(shards))
    (shards / "model-00001.safetensors").unlink()
    assert not export.full_weights_are_present(shards, export.root_weight_files(shards))

    context = _context(tmp_path / "quality", stage="export")
    model = context.artifact("model.retrieval")
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "router_manifest.json",
    ):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "virtual_tokens.txt").write_text("<0>\n", encoding="utf-8")
    evaluation_dir = context.state.stage_dir("evaluate") / "output" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "metrics.json").write_text(
        json.dumps({"metrics": {"recall@1": 1.0}}), encoding="utf-8"
    )
    (evaluation_dir / "predictions.jsonl").write_text(
        json.dumps({"query_id": "q1", "paths": [{"skill_ids": ["alpha"]}]})
        + "\n",
        encoding="utf-8",
    )

    quality, failures = export.export_quality(context, model)

    assert quality["passed"] is True
    assert quality["gates"]["full_model_weights"] is True
    assert failures == []
