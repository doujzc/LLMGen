from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from llmgen.pipeline.artifacts import ArtifactError, ArtifactRegistry
from llmgen.pipeline.code_plan import CodePlanError, plan_codes
from llmgen.pipeline.config import PipelineConfigError, load_pipeline_config
from llmgen.pipeline.io import atomic_write_json, read_jsonl, sha256_file
from llmgen.pipeline.runner import (
    PipelineRunner,
    PipelineRunnerError,
    StageDependencyError,
    create_pipeline_run,
)
from llmgen.pipeline.schema import (
    ensure_ordered_qrels,
    normalize_candidate_rows,
    validate_ordered_qrels,
)
from llmgen.pipeline.stages import ArtifactOutput, StageResult, StageSpec
from llmgen.pipeline.stages.legacy import (
    _copy_model_tree,
    _export_quality,
    _full_weights_are_present,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "router_pipeline.yaml"
STAGE_NAMES = (
    "ingest",
    "enrich",
    "plan-queries",
    "generate-queries",
    "review-queries",
    "finalize-dataset",
    "train-codebook",
    "assign-codes",
    "build-sft",
    "train-memorization",
    "train-alignment",
    "train-retrieval",
    "evaluate",
    "export",
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(
    tmp_path: Path,
    *,
    overrides: tuple[str, ...] = (),
):
    candidates = tmp_path / "candidates.jsonl"
    if not candidates.exists():
        _write_jsonl(
            candidates,
            [
                {"id": "weather", "name": "Weather", "description": "Forecasts"},
                {"id": "calendar", "name": "Calendar", "description": "Events"},
            ],
        )
    return load_pipeline_config(
        DEFAULT_CONFIG,
        overrides=overrides,
        candidates=candidates,
        output=tmp_path / "run",
        environment={},
    )


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_config_is_strict_and_projects_only_relevant_training_phase(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    changed = _config(
        tmp_path,
        overrides=("router.retrieval.learning_rate=1e-5",),
    )

    assert base.stage_hash("train-memorization") == changed.stage_hash(
        "train-memorization"
    )
    assert base.stage_hash("train-alignment") == changed.stage_hash(
        "train-alignment"
    )
    assert base.stage_hash("train-retrieval") != changed.stage_hash(
        "train-retrieval"
    )

    invalid = base.to_dict()
    invalid["unexpected"] = True
    with pytest.raises(PipelineConfigError, match="unknown configuration key"):
        load_pipeline_config(_write_config(tmp_path, invalid), environment={})


def test_config_rejects_secret_persistence_and_unsupported_legacy_knobs(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path).to_dict()
    base["runtime"]["environment"] = {"MY_API_TOKEN": "must-not-persist"}
    with pytest.raises(PipelineConfigError, match="may not contain secrets"):
        load_pipeline_config(_write_config(tmp_path, base), environment={})

    base = _config(tmp_path).to_dict()
    base["runtime"]["environment"] = {
        "INNOCENT_NAME": "${GENERATION_API_KEY}"
    }
    with pytest.raises(PipelineConfigError, match="contains a provider credential"):
        load_pipeline_config(
            _write_config(tmp_path, base),
            environment={"GENERATION_API_KEY": "super-secret-provider-key"},
        )

    base = _config(tmp_path).to_dict()
    base["providers"]["generation"]["api_config"] = "/tmp/legacy-secret.conf"
    with pytest.raises(PipelineConfigError, match="not accepted"):
        load_pipeline_config(_write_config(tmp_path, base), environment={})

    base = _config(tmp_path).to_dict()
    base["data_generation"]["split"] = {
        "train": 0.8,
        "validation": 0.1,
        "test": 0.1,
    }
    with pytest.raises(PipelineConfigError, match="0.90/0.05/0.05"):
        load_pipeline_config(_write_config(tmp_path, base), environment={})

    base = _config(tmp_path).to_dict()
    base["router"]["finetune_mode"] = "lora"
    with pytest.raises(PipelineConfigError, match="self-contained"):
        load_pipeline_config(_write_config(tmp_path, base), environment={})


@pytest.mark.parametrize("candidate_count", [2, 3, 10, 301, 1000])
def test_auto_code_plan_satisfies_capacity_and_training_constraints(
    candidate_count: int,
) -> None:
    plan = plan_codes(
        candidate_count,
        {
            "mode": "auto",
            "latency_priority": "balanced",
            "spare_capacity_ratio": 1.25,
            "max_virtual_tokens": 512,
            "max_branching_factor": 256,
            "num_levels": None,
        },
    )
    assert plan.capacity >= plan.target_capacity >= candidate_count
    assert max(plan.branching_factors) <= candidate_count
    assert sum(plan.branching_factors) <= 512


def test_code_plan_supports_one_token_and_checks_manual_reserve() -> None:
    plan = plan_codes(
        7,
        {
            "mode": "auto",
            "latency_priority": "latency",
            "spare_capacity_ratio": 1.0,
            "max_virtual_tokens": 32,
            "max_branching_factor": 32,
            "num_levels": 1,
        },
    )
    assert plan.branching_factors == (7,)
    with pytest.raises(CodePlanError, match="target capacity"):
        plan_codes(
            10,
            {
                "mode": "manual",
                "latency_priority": "balanced",
                "spare_capacity_ratio": 1.2,
                "max_virtual_tokens": 32,
                "max_branching_factor": 32,
                "num_levels": 1,
                "branching_factors": [10],
            },
        )


def test_candidate_normalization_and_ordered_qrels(tmp_path: Path) -> None:
    candidates = normalize_candidate_rows(
        [{"name": "Weather Tool", "description": "Forecast", "metadata": {"x": 1}}],
        id_policy="explicit_or_name",
        preserve_metadata=True,
    )
    assert candidates[0]["skill_id"] == "Weather-Tool"
    assert candidates[0]["metadata"] == {"x": 1}

    dataset = tmp_path / "dataset"
    _write_jsonl(
        dataset / "queries_train.jsonl",
        [{"id": "q1", "query": "do both", "skill_ids": ["b", "a"]}],
    )
    _write_jsonl(
        dataset / "qrels_train.jsonl",
        [
            {"query_id": "q1", "skill_id": "a", "relevance": 1},
            {"query_id": "q1", "skill_id": "b", "relevance": 1},
        ],
    )
    atomic_write_json(
        dataset / "manifest.json",
        {
            "artifacts": {
                "queries_train.jsonl": {},
                "qrels_train.jsonl": {},
            }
        },
    )

    details = ensure_ordered_qrels(dataset)
    validate_ordered_qrels(dataset)

    assert details["train"]["ordered"] is True
    qrels = read_jsonl(dataset / "qrels_train.jsonl")
    assert [(row["skill_id"], row["position"]) for row in qrels] == [
        ("b", 0),
        ("a", 1),
    ]


def test_artifact_registry_detects_mutation_and_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "output.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    registry = ArtifactRegistry(run_dir)
    registry.initialize()
    record = registry.build_record(
        logical_name="example.output",
        path=artifact,
        producer="ingest",
        artifact_schema="example/v1",
    )
    registry.register_many({record.logical_name: record})
    assert registry.verify("example.output").sha256 == sha256_file(artifact)

    artifact.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="hash changed"):
        registry.verify("example.output")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="escapes run directory"):
        registry.build_record(
            logical_name="outside",
            path=outside,
            producer="ingest",
            artifact_schema="example/v1",
        )

    replacement = registry.build_record(
        logical_name="example.output",
        path=artifact,
        producer="different-stage",
        artifact_schema="example/v1",
    )
    with pytest.raises(ArtifactError, match="may not overwrite"):
        registry.register_many({replacement.logical_name: replacement})


def test_export_copy_preserves_source_metadata_and_omits_checkpoints(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "config.json").write_text('{"source":true}\n', encoding="utf-8")
    checkpoint = source / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")

    destination = tmp_path / "destination"
    _copy_model_tree(source, destination)
    (destination / "config.json").write_text('{"export":true}\n', encoding="utf-8")

    assert (source / "config.json").read_text(encoding="utf-8") == '{"source":true}\n'
    assert (destination / "model.safetensors").read_bytes() == b"weights"
    assert not (destination / "checkpoint-10").exists()


def test_export_weight_gate_rejects_zero_byte_and_incomplete_shards(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    single = model / "model.safetensors"
    single.write_bytes(b"")
    assert not _full_weights_are_present(model, [single])
    single.unlink()

    index = model / "model.safetensors.index.json"
    atomic_write_json(
        index,
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}},
    )
    first = model / "model-00001-of-00002.safetensors"
    first.write_bytes(b"first")
    assert not _full_weights_are_present(model, [index, first])
    second = model / "model-00002-of-00002.safetensors"
    second.write_bytes(b"second")
    assert _full_weights_are_present(model, [index, first, second])


def test_export_quality_uses_real_decoder_and_prediction_gates(tmp_path: Path) -> None:
    model = tmp_path / "attempt" / "model"
    model.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "router_manifest.json"):
        (model / name).write_text("{}\n", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    tokens = ["<SK_0>", "<SK_1>"]
    (model / "virtual_tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    atomic_write_json(
        model / "skill_decode_map.json",
        {
            "schema_version": 1,
            "num_levels": 1,
            "num_skills": 2,
            "num_paths": 2,
            "virtual_tokens": tokens,
            "skills": {"a": {"name": "A"}, "b": {"name": "B"}},
            "skill_to_code": {
                "a": {"tokens": [tokens[0]], "code_text": tokens[0]},
                "b": {"tokens": [tokens[1]], "code_text": tokens[1]},
            },
            "paths": [
                {"tokens": [tokens[0]], "skill_ids": ["a"]},
                {"tokens": [tokens[1]], "skill_ids": ["b"]},
            ],
        },
    )
    candidate_manifest = tmp_path / "candidate_manifest.json"
    atomic_write_json(candidate_manifest, {"candidate_count": 2})
    evaluation = tmp_path / "stages" / "12_evaluate" / "output" / "evaluation"
    _write_jsonl(
        evaluation / "predictions.jsonl",
        [{"query_id": "q1", "paths": [{"skill_ids": ["a"]}]}],
    )
    atomic_write_json(
        evaluation / "metrics.json",
        {"metrics": {"recall@1": 0.75}},
    )

    class Config:
        values = {
            "export.output_dir": "export/model",
            "export.require_all_gates": True,
            "export.allow_failed_gates": False,
            "evaluation.require_format_valid_rate": 1.0,
            "evaluation.require_candidate_coverage": 1.0,
            "evaluation.metric_thresholds": {"recall@1": 0.5},
        }

        def require(self, name: str):
            return self.values[name]

    class State:
        def stage_dir(self, name: str) -> Path:
            directory = "12_evaluate" if name == "evaluate" else name
            return tmp_path / "stages" / directory

    context = SimpleNamespace(
        run_dir=tmp_path,
        config=Config(),
        state=State(),
        artifact=lambda name: candidate_manifest,
    )
    quality, failures = _export_quality(context, model)

    assert quality["passed"] is True
    assert failures == []
    assert _full_weights_are_present(model, [model / "model.safetensors"])

    _write_jsonl(evaluation / "predictions.jsonl", [{"query_id": "q1", "paths": []}])
    quality, failures = _export_quality(context, model)
    assert quality["passed"] is False
    assert quality["gates"]["format_valid_rate"] is False
    assert failures == [{"query_id": "q1", "paths": []}]


def _mock_specs(
    calls: dict[str, int],
    *,
    fail_once: set[str] | None = None,
    resume_values: dict[str, str | None] | None = None,
) -> tuple[StageSpec, ...]:
    fail_once = fail_once if fail_once is not None else set()
    specs = []
    previous: str | None = None
    for stage in STAGE_NAMES:
        dependency = previous
        logical_name = f"mock.{stage}"

        def handler(context, *, name=stage, source=dependency, output=logical_name):
            calls[name] = calls.get(name, 0) + 1
            if resume_values is not None:
                resume_values[name] = context.resume_checkpoint
            if name in fail_once:
                fail_once.remove(name)
                raise RuntimeError(f"planned failure in {name}")
            upstream = None
            if source is not None:
                upstream = json.loads(
                    context.artifact(f"mock.{source}").read_text(encoding="utf-8")
                )
            path = context.output_dir / "result.json"
            atomic_write_json(
                path,
                {
                    "stage": name,
                    "attempt": context.attempt,
                    "upstream": upstream,
                },
            )
            return StageResult(
                artifacts=(ArtifactOutput(output, path, "mock/v1"),),
                progress={"completed": 1, "total": 1},
            )

        specs.append(
            StageSpec(
                name=stage,
                directory=f"{len(specs):02d}_{stage.replace('-', '_')}",
                dependencies=((dependency,) if dependency else ()),
                required_artifacts=((f"mock.{dependency}",) if dependency else ()),
                handler=handler,
                description=f"mock {stage}",
            )
        )
        previous = stage
    return tuple(specs)


def test_mock_pipeline_failure_resume_force_and_fork(tmp_path: Path) -> None:
    calls: dict[str, int] = {}
    failures = {"review-queries"}
    resumes: dict[str, str | None] = {}
    specs = _mock_specs(calls, fail_once=failures, resume_values=resumes)
    config = _config(tmp_path)
    runner = create_pipeline_run(config, stage_specs=specs, repo_root=REPO_ROOT)

    with pytest.raises(PipelineRunnerError, match="planned failure"):
        runner.run()
    failed = runner.state.read_stage("review-queries")
    traceback_path = runner.run_dir / failed["last_error"]["traceback_path"]
    assert failed["status"] == "failed"
    assert traceback_path.is_file()
    assert oct(traceback_path.stat().st_mode & 0o777) == "0o600"
    assert calls["generate-queries"] == 1

    runner.run(from_stage="review-queries")
    assert runner.status()["run"]["status"] == "completed"
    assert calls["generate-queries"] == 1
    assert calls["review-queries"] == 2
    assert (runner.state.stage_dir("export") / "COMPLETED").is_file()
    assert (runner.state.stage_dir("export") / "input_manifest.json").is_file()
    assert (runner.state.stage_dir("export") / "output" / "manifest.json").is_file()

    before = dict(calls)
    runner.run(force_stage=["assign-codes"])
    for stage in STAGE_NAMES[: STAGE_NAMES.index("assign-codes")]:
        assert calls[stage] == before[stage]
    for stage in STAGE_NAMES[STAGE_NAMES.index("assign-codes") :]:
        assert calls[stage] == before[stage] + 1

    runner.stage("train-retrieval", resume_checkpoint="checkpoint-500")
    assert resumes["train-retrieval"] is None  # completed stages are reused
    runner.run(
        from_stage="train-retrieval",
        to_stage="train-retrieval",
        force_stage=["train-retrieval"],
        resume_checkpoint="checkpoint-500",
    )
    assert resumes["train-retrieval"] == "checkpoint-500"

    child_config = load_pipeline_config(
        runner.run_dir / "config" / "pipeline.resolved.yaml",
        overrides=("router.retrieval.learning_rate=1e-5",),
        output=tmp_path / "fork",
        environment={},
    )
    child_calls: dict[str, int] = {}
    child_specs = _mock_specs(child_calls)
    child = create_pipeline_run(
        child_config,
        stage_specs=child_specs,
        repo_root=REPO_ROOT,
        parent_run_id=runner.state.read_run()["run_id"],
    )
    reused = child.reuse_from(runner)
    assert "mock.train-alignment" in reused
    assert "mock.train-retrieval" not in reused
    assert child.state.read_run()["parent_run_id"] == runner.state.read_run()["run_id"]


def test_runner_redacts_stage_errors_but_keeps_private_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "pipeline-secret-value"
    monkeypatch.setenv("GENERATION_API_KEY", secret)

    def fail(context):
        raise RuntimeError(f"provider rejected {secret}")

    spec = StageSpec("ingest", "00_ingest", (), (), fail, "fail safely")
    runner = create_pipeline_run(
        _config(tmp_path), stage_specs=(spec,), repo_root=REPO_ROOT
    )
    with pytest.raises(PipelineRunnerError) as raised:
        runner.run()

    stage = runner.state.read_stage("ingest")
    run = runner.state.read_run()
    assert secret not in str(raised.value)
    assert secret not in json.dumps(stage)
    assert secret not in json.dumps(run)
    traceback_path = runner.run_dir / stage["last_error"]["traceback_path"]
    assert secret in traceback_path.read_text(encoding="utf-8")
    assert oct(traceback_path.stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize("directory", ["../../outside", "/tmp/outside", "."])
def test_runner_rejects_unsafe_stage_directories(
    tmp_path: Path,
    directory: str,
) -> None:
    spec = StageSpec("ingest", directory, (), (), lambda context: StageResult(), "bad")
    with pytest.raises(PipelineRunnerError, match="unsafe stage directory"):
        PipelineRunner(tmp_path / "run", _config(tmp_path), stage_specs=(spec,))


def test_injected_stage_names_use_conservative_full_config_hash(tmp_path: Path) -> None:
    def complete(context):
        output = context.output_dir / "result.json"
        atomic_write_json(output, {"ok": True})
        return StageResult((ArtifactOutput("custom.output", output, "custom/v1"),))

    spec = StageSpec("alpha", "00_alpha", (), (), complete, "custom")
    runner = create_pipeline_run(
        _config(tmp_path), stage_specs=(spec,), repo_root=REPO_ROOT
    )
    assert runner.run()[0].action == "executed"
    assert runner.run()[0].action == "reused"


def test_create_reserves_run_directory_atomically(tmp_path: Path) -> None:
    config = _config(tmp_path)
    Path(config.require("run.output_dir")).mkdir()
    with pytest.raises(PipelineRunnerError, match="already exists"):
        create_pipeline_run(config, stage_specs=_mock_specs({}), repo_root=REPO_ROOT)


def test_standalone_stage_does_not_run_missing_dependency(tmp_path: Path) -> None:
    specs = _mock_specs({})
    runner = create_pipeline_run(_config(tmp_path), stage_specs=specs, repo_root=REPO_ROOT)
    with pytest.raises(StageDependencyError, match="depends on ingest"):
        runner.stage("enrich")


def test_cli_can_create_resume_and_report_an_ingest_only_run(tmp_path: Path) -> None:
    candidates = tmp_path / "cli-candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            {"id": "a", "name": "A", "description": "first"},
            {"id": "b", "name": "B", "description": "second"},
        ],
    )
    run_dir = tmp_path / "cli-run"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_candidates.py"),
        "run",
        "--candidates",
        str(candidates),
        "--config",
        str(DEFAULT_CONFIG),
        "--output",
        str(run_dir),
        "--from",
        "ingest",
        "--to",
        "ingest",
    ]
    first = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout.splitlines()[-1])[0]["action"] == "executed"

    resumed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_candidates.py"),
            "run",
            "--run-dir",
            str(run_dir),
            "--from",
            "ingest",
            "--to",
            "ingest",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout.splitlines()[-1])[0]["action"] == "reused"

    status = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_candidates.py"),
            "status",
            "--run-dir",
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["run"]["status"] == "partial"

    original_snapshot = (run_dir / "source" / "candidates.input.jsonl").read_bytes()
    _write_jsonl(
        candidates,
        [
            {"id": "changed-a", "name": "Changed A", "description": "new"},
            {"id": "changed-b", "name": "Changed B", "description": "new"},
        ],
    )
    fork_dir = tmp_path / "cli-fork"
    forked = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_candidates.py"),
            "fork",
            "--from-run",
            str(run_dir),
            "--output",
            str(fork_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert forked.returncode == 0, forked.stderr
    assert (fork_dir / "source" / "candidates.input.jsonl").read_bytes() == original_snapshot

    conflicting = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_candidates.py"),
            "run",
            "--run-dir",
            str(run_dir),
            "--candidates",
            str(candidates),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert conflicting.returncode == 2
    assert "cannot be combined" in conflicting.stderr


def test_fork_does_not_reuse_stage_without_final_marker(tmp_path: Path) -> None:
    parent_calls: dict[str, int] = {}
    specs = _mock_specs(parent_calls)
    parent = create_pipeline_run(
        _config(tmp_path), stage_specs=specs, repo_root=REPO_ROOT
    )
    parent.run(from_stage="ingest", to_stage="ingest")
    (parent.state.stage_dir("ingest") / "COMPLETED").unlink()

    child_config = load_pipeline_config(
        parent.run_dir / "config" / "pipeline.resolved.yaml",
        candidates=parent.run_dir / "source" / "candidates.input.jsonl",
        output=tmp_path / "child-with-uncommitted-parent",
        environment={},
    )
    child = create_pipeline_run(
        child_config,
        stage_specs=_mock_specs({}),
        repo_root=REPO_ROOT,
        parent_run_id=parent.state.read_run()["run_id"],
    )
    assert child.reuse_from(parent) == ()
