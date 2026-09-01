"""Contracts for the split generic Pipeline codebook/SFT/training adapters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from llmgen.pipeline.stages import assign_codes, build_sft, train_codebook, train_router
from llmgen.pipeline.stages import common


class _Config:
    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = dict(values)

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def require(self, key: str) -> Any:
        return self.values[key]


class _State:
    def __init__(self, root: Path) -> None:
        self.root = root

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage


class _Context:
    def __init__(
        self,
        tmp_path: Path,
        *,
        stage: str,
        artifacts: Mapping[str, Path] = (),
        alignment_enabled: bool = True,
    ) -> None:
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir(exist_ok=True)
        self.repo_root = tmp_path / "repo"
        self.state = _State(self.run_dir / "stages")
        self.spec = SimpleNamespace(name=stage)
        self.attempt = 1
        self.output_dir = self.state.stage_dir(stage) / "attempts" / "0001" / "output"
        self.output_dir.mkdir(parents=True)
        self.checkpoint_lineage_path = self.output_dir.parent / "checkpoint_lineage.json"
        self.checkpoint_lineage = {"stage": stage}
        self.config = _Config(
            {
                "export.output_dir": "export/model",
                "code": {
                    "mode": "manual",
                    "branching_factors": [2],
                    "num_levels": 1,
                    "spare_capacity_ratio": 1.0,
                    "max_virtual_tokens": 8,
                    "max_branching_factor": 8,
                    "latency_priority": "balanced",
                },
                "router.alignment.enabled": alignment_enabled,
                "router.alignment.epochs": 1 if alignment_enabled else 0,
            }
        )
        self._artifacts = dict(artifacts)
        self.selected: list[tuple[str, Path]] = []
        self.code_plan_path: Path | None = None
        self.commands: list[tuple[list[str], dict[str, str], str | None]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.logger = SimpleNamespace(
            event=lambda name, **fields: self.events.append((name, fields))
        )

    @property
    def stage_dir(self) -> Path:
        return self.state.stage_dir(self.spec.name)

    def artifact(self, name: str) -> Path:
        return self._artifacts[name]

    def set_checkpoint_code_plan(self, path: str | Path) -> None:
        self.code_plan_path = Path(path)
        self.checkpoint_lineage["code_plan_sha256"] = "bound"

    def select_resume_checkpoint(self, *, kind: str, root: Path) -> str | None:
        self.selected.append((kind, root))
        return f"/resume/{self.spec.name}" if kind == "router" else "/resume/codebook"

    def run_command(
        self,
        argv: list[str],
        *,
        environment: Mapping[str, str] | None = None,
        label: str | None = None,
    ) -> None:
        self.commands.append((list(argv), dict(environment or {}), label))


def _artifact_names(result) -> list[str]:
    return [artifact.logical_name for artifact in result.artifacts]


def test_codebook_and_code_assignment_modules_preserve_plan_resume_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "candidates.manifest.json"
    manifest.write_text(json.dumps({"candidate_count": 2}), encoding="utf-8")
    context = _Context(tmp_path, stage="train-codebook", artifacts={"candidates.manifest": manifest})
    calls: list[tuple[str, dict[str, str]]] = []

    monkeypatch.setattr(common, "verify_training_provenance", lambda _: None)

    def fake_router_pipeline(ctx, command, *, environment_overrides=None) -> None:
        calls.append((command, dict(environment_overrides or {})))
        stage_paths = common.paths(ctx)
        stage_paths["stage1"].mkdir(parents=True)
        (stage_paths["stage1"] / "best.pt").write_bytes(b"checkpoint")

    monkeypatch.setattr(common, "router_pipeline", fake_router_pipeline)
    result = train_codebook.train_codebook(context)  # type: ignore[arg-type]

    assert context.code_plan_path == context.output_dir / "code_plan.json"
    assert context.selected == [("codebook", context.stage_dir / "attempts")]
    assert calls == [("train-tokenizer", {"LLMGEN_PIPELINE_CHECKPOINT_LINEAGE": str(context.checkpoint_lineage_path), "TOKENIZER_RESUME": "/resume/codebook"})]
    assert _artifact_names(result) == ["code.plan", "codebook.directory", "codebook.best"]
    assert result.progress["code_plan"]["candidate_count"] == 2

    assignment = _Context(tmp_path, stage="assign-codes")

    def fake_export(ctx, command, *, environment_overrides=None) -> None:
        assert command == "export-codes"
        assert environment_overrides is None
        index = common.paths(ctx)["index"]
        index.mkdir(parents=True)
        for name in ("train_codes.jsonl", "train_registry.json", "virtual_tokens.txt", "manifest.json"):
            (index / name).write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(common, "router_pipeline", fake_export)
    assigned = assign_codes.assign_codes(assignment)  # type: ignore[arg-type]
    assert _artifact_names(assigned) == [
        "codes.directory",
        "codes.train",
        "codes.registry",
        "codes.virtual_tokens",
        "codes.manifest",
    ]


def test_build_sft_module_publishes_only_existing_curriculum_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _Context(tmp_path, stage="build-sft")

    def fake_router_pipeline(ctx, command, *, environment_overrides=None) -> None:
        assert command == "build-router-data"
        assert environment_overrides is None
        router_data = common.paths(ctx)["router_data"]
        router_data.mkdir(parents=True)
        (router_data / "manifest.json").write_text("{}\n", encoding="utf-8")
        for name in ("memorization_train", "retrieval_train"):
            (router_data / f"{name}.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(common, "router_pipeline", fake_router_pipeline)
    result = build_sft.build_sft(context)  # type: ignore[arg-type]
    assert _artifact_names(result) == [
        "sft.directory",
        "sft.manifest",
        "sft.memorization.train",
        "sft.retrieval.train",
    ]


def test_router_training_modules_preserve_checkpoint_and_model_handoffs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memorization = tmp_path / "memorization"
    alignment = tmp_path / "alignment"
    context = _Context(tmp_path, stage="train-memorization")
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(common, "verify_training_provenance", lambda _: None)

    def fake_router_pipeline(ctx, command, *, environment_overrides=None) -> None:
        calls.append((command, dict(environment_overrides or {})))
        model = ctx.output_dir / "memorization"
        model.mkdir(exist_ok=True)

    monkeypatch.setattr(common, "router_pipeline", fake_router_pipeline)
    trained = train_router.train_memorization(context)  # type: ignore[arg-type]
    assert trained.artifacts[0].path == context.output_dir / "memorization"
    assert calls == [
        (
            "train-memorization",
            {
                "LLMGEN_PIPELINE_CHECKPOINT_LINEAGE": str(context.checkpoint_lineage_path),
                "ROUTER_OUTPUT_DIR": str(context.output_dir),
                "ROUTER_RESUME_MEMORIZATION": "/resume/train-memorization",
            },
        )
    ]

    alignment_context = _Context(
        tmp_path,
        stage="train-alignment",
        artifacts={"model.memorization": memorization},
    )
    monkeypatch.setattr(common, "legacy_environment", lambda _: {"BASE": "1"})
    aligned = train_router.train_alignment(alignment_context)  # type: ignore[arg-type]
    assert aligned.artifacts[0].path == alignment_context.output_dir / "retrieval_alignment"
    command, environment, label = alignment_context.commands[0]
    assert command == ["bash", "scripts/skillret/06a_train_alignment.sh"]
    assert label == "legacy-router-train-alignment"
    assert environment["ROUTER_MEMORIZATION_MODEL_DIR"] == str(memorization)
    assert environment["ROUTER_RESUME_ALIGNMENT"] == "/resume/train-alignment"

    retrieval_context = _Context(
        tmp_path,
        stage="train-retrieval",
        artifacts={"model.alignment": alignment},
    )
    monkeypatch.setattr(common, "alignment_only", lambda _: False)
    retrieved = train_router.train_retrieval(retrieval_context)  # type: ignore[arg-type]
    assert retrieved.artifacts[0].path == retrieval_context.output_dir / "retrieval"
    command, environment, label = retrieval_context.commands[0]
    assert command == ["bash", "scripts/skillret/06b_train_retrieval.sh"]
    assert label == "legacy-router-train-retrieval"
    assert environment["ROUTER_RETRIEVAL_INIT_DIR"] == str(alignment)
    assert environment["ROUTER_RESUME_RETRIEVAL"] == "/resume/train-retrieval"


def test_router_training_modules_keep_passthrough_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memorization = tmp_path / "memorization"
    alignment_context = _Context(
        tmp_path,
        stage="train-alignment",
        artifacts={"model.memorization": memorization},
        alignment_enabled=False,
    )
    aligned = train_router.train_alignment(alignment_context)  # type: ignore[arg-type]
    assert aligned.artifacts[0].path == memorization
    assert aligned.artifacts[0].metadata == {
        "passthrough": True,
        "reason": "alignment disabled",
    }
    assert alignment_context.commands == []

    retrieval_context = _Context(
        tmp_path,
        stage="train-retrieval",
        artifacts={"model.alignment": memorization},
    )
    monkeypatch.setattr(common, "alignment_only", lambda _: True)
    retrieved = train_router.train_retrieval(retrieval_context)  # type: ignore[arg-type]
    assert retrieved.artifacts[0].path == memorization
    assert retrieved.artifacts[0].metadata == {
        "passthrough": True,
        "reason": "single-candidate alignment-only run",
    }
    assert retrieval_context.commands == []
