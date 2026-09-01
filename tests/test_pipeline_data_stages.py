"""Direct contracts for the split generic data Stage modules."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmgen.pipeline.stages import (
    enrich,
    finalize_dataset,
    generate_queries,
    ingest,
    plan_queries,
    review_queries,
)
from llmgen.pipeline.stages.base import ArtifactOutput


class _Config:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def get(self, name: str):
        return self.values.get(name)

    def require(self, name: str):
        return self.values[name]


class _State:
    def __init__(self, root: Path) -> None:
        self.root = root

    def stage_dir(self, name: str) -> Path:
        return self.root / name


def _context(tmp_path: Path, *, stage: str, values: dict[str, object]) -> SimpleNamespace:
    run_dir = tmp_path / "run"
    (run_dir / "source").mkdir(parents=True)
    return SimpleNamespace(
        repo_root=tmp_path,
        run_dir=run_dir,
        state=_State(run_dir / "stages"),
        spec=SimpleNamespace(name=stage),
        output_dir=run_dir / "stages" / stage / "attempts" / "0001" / "output",
        stage_dir=run_dir / "stages" / stage,
        config=_Config(values),
        calls=[],
        progress=[],
        update_progress=lambda **kwargs: None,
    )


def test_split_data_stage_modules_do_not_import_legacy_handlers() -> None:
    for module in (ingest, enrich, plan_queries, generate_queries, review_queries, finalize_dataset):
        assert "legacy import" not in inspect.getsource(module)


def test_split_ingest_preserves_candidate_artifact_contract(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text('{"id":"weather","name":"Weather","description":"Forecast"}\n', encoding="utf-8")
    context = _context(
        tmp_path,
        stage="ingest",
        values={
            "input.candidates": str(candidate),
            "input.id_policy": "explicit_or_name",
            "input.preserve_metadata": True,
            "input.single_candidate_policy": "alignment_only",
        },
    )
    frozen = context.run_dir / "source" / "candidates.input.jsonl"
    frozen.write_bytes(candidate.read_bytes())
    manual = context.run_dir / "source" / "manual_alignment.input.jsonl"
    manual.write_bytes(b"")
    import hashlib

    (context.run_dir / "config").mkdir()
    (context.run_dir / "config" / "candidate_input.json").write_text(
        json.dumps({"frozen_path": "source/candidates.input.jsonl", "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    (context.run_dir / "config" / "manual_alignment_input.json").write_text(
        json.dumps(
            {
                "frozen_path": "source/manual_alignment.input.jsonl",
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    context.update_progress = lambda **kwargs: context.progress.append(kwargs)

    result = ingest.ingest(context)

    assert [output.logical_name for output in result.artifacts] == [
        "candidates.input", "candidates.normalized", "candidates.catalog", "candidates.manifest", "inputs.manual_alignment",
    ]
    manifest = json.loads((context.run_dir / "source" / "candidate_manifest.json").read_text())
    assert manifest["execution_mode"] == "alignment_only"
    assert context.progress == [{"completed": 1, "total": 1}]


def test_split_plan_and_generate_alignment_only_keep_stage_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[str, object] = {
        "export.output_dir": "export/model",
        "input.single_candidate_policy": "alignment_only",
        "data_generation.alignment_queries_per_skill": 1,
        "data_generation.alignment_batch_size": 2,
        "checkpointing.llm_batch_records": 5,
    }
    plan_context = _context(tmp_path, stage="plan-queries", values=values)
    (plan_context.run_dir / "source" / "candidate_manifest.json").write_text('{"candidate_count":1}', encoding="utf-8")
    planned = plan_queries.plan_queries(plan_context)
    assert planned.artifacts[0].logical_name == "data.workflows"
    assert planned.artifacts[0].metadata == {"execution_mode": "alignment_only"}

    generation_context = _context(tmp_path / "generation", stage="generate-queries", values={**values, "providers.generation": {"model": "mock"}})
    (generation_context.run_dir / "source" / "candidate_manifest.json").write_text('{"candidate_count":1}', encoding="utf-8")
    commands: list[tuple[list[str], str | None]] = []
    generation_context.run_command = lambda argv, **kwargs: commands.append(([str(value) for value in argv], kwargs.get("label")))
    monkeypatch.setattr(generate_queries, "provider_api_config", lambda *_args: "api.conf")
    monkeypatch.setattr(generate_queries, "provider_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(generate_queries, "ledger_outputs", lambda *_args: ((ArtifactOutput("ledger.generate-queries.generation", generation_context.stage_dir / "ledger", "provider_ledger/v1"),), {"generation": {}}))

    generated = generate_queries.generate_queries(generation_context)

    assert commands[0][1] == "generate-alignment-queries"
    assert all(label != "generate-multiskill-queries" for _command, label in commands)
    assert [output.logical_name for output in generated.artifacts] == [
        "data.queries.generated", "data.queries.alignment.generated", "ledger.generate-queries.generation",
    ]
    assert (generation_context.output_dir / "queries.generated.jsonl").read_text(encoding="utf-8") == ""
