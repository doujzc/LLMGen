from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmgen.pipeline.stages import common, legacy
from llmgen.pipeline.config import load_pipeline_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Config:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def get(self, key: str):
        return self.values.get(key)

    def require(self, key: str):
        return self.values[key]


class _State:
    def __init__(self, root: Path) -> None:
        self.root = root

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage


def _context(tmp_path: Path, *, stage: str = "evaluate") -> SimpleNamespace:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return SimpleNamespace(
        repo_root=tmp_path / "repo",
        run_dir=run_dir,
        config=_Config(
            {
                "runtime.python": "",
                "runtime.num_devices": 0,
                "runtime.devices": "",
                "runtime.device": "cpu",
                "export.output_dir": "export/model",
                "input.single_candidate_policy": "alignment_only",
                "runtime": {},
                "router.base_model": "local/model",
            }
        ),
        state=_State(run_dir / "stages"),
        spec=SimpleNamespace(name=stage),
        output_dir=run_dir / "stages" / stage / "attempts" / "0001" / "output",
        stage_dir=run_dir / "stages" / stage,
        checkpoint_lineage_path=run_dir / "lineage.json",
        logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
    )


def test_legacy_helpers_are_direct_common_aliases() -> None:
    assert legacy._python is common.python
    assert legacy._checkpoint_environment is common.checkpoint_environment
    assert legacy._verify_training_provenance is common.verify_training_provenance
    assert legacy._configured is common.configured
    assert legacy._paths is common.paths
    assert legacy._device_count is common.device_count
    assert legacy._alignment_only is common.alignment_only
    assert legacy._legacy_environment is common.legacy_environment
    assert legacy._router_pipeline is common.router_pipeline
    assert legacy._copy is common.copy


def test_common_helpers_keep_path_device_alignment_and_copy_contracts(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    paths = common.paths(context)
    assert paths["evaluation"] == context.output_dir / "evaluation"
    assert paths["router_data"] == (
        context.state.stage_dir("build-sft") / "output" / "router_data"
    )
    assert common.device_count(context) == 1
    assert common.configured({"value": 0}, "value", 5) == 0
    assert common.configured({"value": ""}, "value", 5) == 5

    source_dir = context.run_dir / "source"
    source_dir.mkdir()
    (source_dir / "candidate_manifest.json").write_text(
        json.dumps({"candidate_count": 1}), encoding="utf-8"
    )
    assert common.alignment_only(context) is True

    source = tmp_path / "source.jsonl"
    source.write_text("row\n", encoding="utf-8")
    source.with_name("source.manifest.json").write_text("{}", encoding="utf-8")
    source.with_name("source.errors.jsonl").write_text("error\n", encoding="utf-8")
    destination = tmp_path / "nested" / "destination.jsonl"
    common.copy(source, destination)
    assert destination.read_text(encoding="utf-8") == "row\n"
    assert destination.with_name("destination.manifest.json").is_file()
    assert destination.with_name("destination.errors.jsonl").is_file()


def test_common_training_helpers_verify_and_route_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    assert common.checkpoint_environment(context) == {
        "LLMGEN_PIPELINE_CHECKPOINT_LINEAGE": str(context.checkpoint_lineage_path)
    }
    context.checkpoint_lineage_path = None
    with pytest.raises(ValueError, match="checkpoint lineage"):
        common.checkpoint_environment(context)
    context.checkpoint_lineage_path = context.run_dir / "lineage.json"

    provenance = context.run_dir / "config" / "provenance.json"
    provenance.parent.mkdir()
    provenance.write_text("{}", encoding="utf-8")
    events: list[tuple[tuple[object, ...], dict[str, object]]] = []
    context.logger = SimpleNamespace(event=lambda *args, **kwargs: events.append((args, kwargs)))
    verified: dict[str, object] = {}
    monkeypatch.setattr(
        common,
        "verify_run_provenance",
        lambda frozen, **kwargs: verified.update(frozen=frozen, **kwargs),
    )
    common.verify_training_provenance(context)
    assert verified["base_model"] == "local/model"
    assert events[0][0] == ("training.provenance_verified",)

    commands: list[tuple[list[str], dict[str, str], str]] = []
    context.run_command = lambda argv, *, environment, label: commands.append(
        (list(argv), dict(environment), label)
    )
    monkeypatch.setattr(common, "legacy_environment", lambda _: {"BASE": "1"})
    common.router_pipeline(
        context,
        "evaluate",
        environment_overrides={"EXTRA": "2"},
    )
    assert commands == [
        (
            ["bash", "scripts/skillret/07_evaluate.sh"],
            {"BASE": "1", "EXTRA": "2"},
            "generic-router-evaluate",
        )
    ]
    with pytest.raises(ValueError, match="unsupported generic router command"):
        common.router_pipeline(context, "invalid")


def test_legacy_environment_uses_device_specific_visibility_and_blocks_overrides(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        '{"id":"a","name":"A","description":"alpha"}\n',
        encoding="utf-8",
    )
    config = load_pipeline_config(
        REPO_ROOT / "configs" / "router_pipeline.yaml",
        candidates=candidates,
        output=tmp_path / "run",
        environment={},
        overrides=(
            "runtime.device=npu",
            "runtime.devices=[1,3]",
            "runtime.num_devices=2",
        ),
    )
    run_dir = tmp_path / "run"
    context = SimpleNamespace(
        repo_root=REPO_ROOT,
        run_dir=run_dir,
        config=config,
        state=_State(run_dir / "stages"),
        spec=SimpleNamespace(name="evaluate"),
        output_dir=run_dir / "stages" / "evaluate" / "output",
        stage_dir=run_dir / "stages" / "evaluate",
    )

    environment = common.legacy_environment(context)
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "1,3"
    assert "CUDA_VISIBLE_DEVICES" not in environment

    config.data["runtime"]["environment"] = {"PROCESSED_DIR": "/tmp/escape"}
    with pytest.raises(ValueError, match="pipeline-managed variable"):
        common.legacy_environment(context)
