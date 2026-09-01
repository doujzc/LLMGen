"""Codebook-training Stage adapter for the generic candidate pipeline."""

from __future__ import annotations

from ..code_plan import plan_codes
from ..io import atomic_write_json, read_json
from . import common
from .base import ArtifactOutput, StageContext, StageResult


def train_codebook(context: StageContext) -> StageResult:
    """Freeze a CodePlan, resume a compatible codebook, and train it."""

    common.verify_training_provenance(context)
    stage_paths = common.paths(context)
    candidate_manifest = read_json(context.artifact("candidates.manifest"))
    count = int(candidate_manifest["candidate_count"])
    code_plan = plan_codes(count, context.config.require("code"))
    stage_paths["code_plan"].parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stage_paths["code_plan"], code_plan.to_dict())
    context.set_checkpoint_code_plan(stage_paths["code_plan"])
    checkpoint = context.select_resume_checkpoint(
        kind="codebook", root=context.stage_dir / "attempts"
    )
    environment = common.checkpoint_environment(context)
    if checkpoint:
        environment["TOKENIZER_RESUME"] = checkpoint
    common.router_pipeline(
        context,
        "train-tokenizer",
        environment_overrides=environment,
    )
    return StageResult(
        artifacts=(
            ArtifactOutput("code.plan", stage_paths["code_plan"], "code_plan/v1"),
            ArtifactOutput(
                "codebook.directory",
                stage_paths["stage1"],
                "toolweaver_codebook/v1",
            ),
            ArtifactOutput(
                "codebook.best",
                stage_paths["stage1"] / "best.pt",
                "toolweaver_checkpoint/v1",
            ),
        ),
        progress={"candidate_count": count, "code_plan": code_plan.to_dict()},
    )


handler = train_codebook
