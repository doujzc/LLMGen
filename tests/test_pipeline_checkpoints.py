from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llmgen.pipeline.checkpoints import (
    CheckpointError,
    select_checkpoint,
    write_checkpoint_sidecar,
)
from llmgen.pipeline.config import load_pipeline_config
from llmgen.pipeline.runner import PipelineRunnerError, create_pipeline_run
from llmgen.pipeline.stages import StageResult, StageSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "router_pipeline.yaml"


def _lineage() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "stage": "train-retrieval",
        "stage_config_hash": "stage-hash",
        "input_artifacts": {"sft.directory": "input-hash"},
        "code_plan_sha256": "code-plan-hash",
    }


def _router_checkpoint(root: Path, step: int, lineage: dict[str, object]) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    write_checkpoint_sidecar(checkpoint, kind="router", lineage=lineage, global_step=step)
    return checkpoint


def test_router_checkpoint_selects_latest_complete_and_skips_damaged(tmp_path: Path) -> None:
    lineage = _lineage()
    root = tmp_path / "router"
    first = _router_checkpoint(root, 10, lineage)
    _router_checkpoint(root, 20, lineage)
    damaged = root / "checkpoint-30"
    damaged.mkdir()
    (damaged / "trainer_state.json").write_text(
        json.dumps({"global_step": 29}), encoding="utf-8"
    )

    selected = select_checkpoint(root, kind="router", expected_lineage=lineage)
    assert selected is not None
    assert selected.path.name == "checkpoint-20"

    explicit = select_checkpoint(
        root,
        kind="router",
        expected_lineage=lineage,
        explicit=first,
    )
    assert explicit is not None and explicit.explicit is True
    with pytest.raises(CheckpointError, match="disagree"):
        select_checkpoint(
            root,
            kind="router",
            expected_lineage=lineage,
            explicit=damaged,
        )


def test_checkpoint_rejects_wrong_lineage_and_requires_explicit_legacy_opt_in(
    tmp_path: Path,
) -> None:
    lineage = _lineage()
    checkpoint = _router_checkpoint(tmp_path, 7, lineage)
    wrong = {**lineage, "stage_config_hash": "other"}
    with pytest.raises(CheckpointError, match="stage_config_hash"):
        select_checkpoint(
            tmp_path, kind="router", expected_lineage=wrong, explicit=checkpoint
        )

    (checkpoint / "pipeline_lineage.json").unlink()
    with pytest.raises(CheckpointError, match="no pipeline_lineage"):
        select_checkpoint(
            tmp_path, kind="router", expected_lineage=lineage, explicit=checkpoint
        )
    selected = select_checkpoint(
        tmp_path,
        kind="router",
        expected_lineage=lineage,
        explicit=checkpoint,
        allow_legacy=True,
    )
    assert selected is not None and selected.legacy_without_sidecar is True


def test_codebook_checkpoint_uses_resumable_last_pt_format(tmp_path: Path) -> None:
    lineage = _lineage()
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "global_step": 42,
            "model_state": {},
            "optimizer_state": {},
            "scheduler_state": {},
            "rng_state": {},
        },
        checkpoint,
    )
    write_checkpoint_sidecar(checkpoint, kind="codebook", lineage=lineage, global_step=42)
    selected = select_checkpoint(tmp_path, kind="codebook", expected_lineage=lineage)
    assert selected is not None
    assert selected.global_step == 42

    torch.save({"global_step": 43, "model_state": {}}, checkpoint)
    with pytest.raises(CheckpointError, match="not resumable"):
        select_checkpoint(
            tmp_path, kind="codebook", expected_lineage=lineage, explicit=checkpoint
        )


def _config(tmp_path: Path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        '{"id":"a","name":"A","description":"first"}\n'
        '{"id":"b","name":"B","description":"second"}\n',
        encoding="utf-8",
    )
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "config.json").write_text("{}", encoding="utf-8")
    return load_pipeline_config(
        DEFAULT_CONFIG,
        overrides=(f"router.base_model={base_model}",),
        candidates=candidates,
        output=tmp_path / "run",
        environment={},
    )


def test_runner_auto_resumes_new_attempt_and_records_selection(tmp_path: Path) -> None:
    calls = 0

    def train(context):
        nonlocal calls
        root = context.output_dir / "router"
        selected = context.select_resume_checkpoint(kind="router", root=root)
        calls += 1
        if calls == 1:
            checkpoint = root / "checkpoint-9"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 9}), encoding="utf-8"
            )
            write_checkpoint_sidecar(
                checkpoint,
                kind="router",
                lineage=context.checkpoint_lineage,
                global_step=9,
            )
            raise RuntimeError("interrupted after checkpoint")
        assert selected == str(
            context.stage_dir
            / "attempts"
            / "0001"
            / "output"
            / "router"
            / "checkpoint-9"
        )
        return StageResult()

    spec = StageSpec(
        "train-retrieval", "11_train_retrieval", (), (), train, "test training"
    )
    runner = create_pipeline_run(_config(tmp_path), stage_specs=(spec,), repo_root=REPO_ROOT)
    with pytest.raises(PipelineRunnerError, match="interrupted"):
        runner.stage("train-retrieval")
    runner.stage("train-retrieval")
    selection = runner.state.read_stage("train-retrieval")["progress"]["checkpoint_resume"]
    assert selection["selected"]["global_step"] == 9
    assert selection["selected"]["explicit"] is False
