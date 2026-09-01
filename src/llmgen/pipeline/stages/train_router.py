"""Router curriculum training Stage adapters for the generic pipeline."""

from __future__ import annotations

from . import common
from .base import ArtifactOutput, StageContext, StageResult


def train_memorization(context: StageContext) -> StageResult:
    """Train (or resume) the Skill memorization phase."""

    common.verify_training_provenance(context)
    output_root = context.output_dir
    model = output_root / "memorization"
    checkpoint = context.select_resume_checkpoint(kind="router", root=model)
    environment = common.checkpoint_environment(context)
    environment["ROUTER_OUTPUT_DIR"] = str(output_root)
    if checkpoint:
        environment["ROUTER_RESUME_MEMORIZATION"] = checkpoint
    common.router_pipeline(
        context,
        "train-memorization",
        environment_overrides=environment,
    )
    return StageResult(
        artifacts=(ArtifactOutput("model.memorization", model, "router_model/v1"),)
    )


def train_alignment(context: StageContext) -> StageResult:
    """Train the single-Skill alignment phase or preserve memorization output."""

    enabled = bool(context.config.require("router.alignment.enabled"))
    if enabled and int(context.config.require("router.alignment.epochs")) > 0:
        common.verify_training_provenance(context)
        output_root = context.output_dir
        model = output_root / "retrieval_alignment"
        environment = common.legacy_environment(context)
        environment.update(common.checkpoint_environment(context))
        environment.update(
            {
                "ROUTER_OUTPUT_DIR": str(output_root),
                "ROUTER_MEMORIZATION_MODEL_DIR": str(
                    context.artifact("model.memorization")
                ),
            }
        )
        checkpoint = context.select_resume_checkpoint(kind="router", root=model)
        if checkpoint:
            environment["ROUTER_RESUME_ALIGNMENT"] = checkpoint
        context.run_command(
            ["bash", "scripts/skillret/06a_train_alignment.sh"],
            environment=environment,
            label="legacy-router-train-alignment",
        )
        metadata = {"passthrough": False}
    else:
        model = context.artifact("model.memorization")
        metadata = {"passthrough": True, "reason": "alignment disabled"}
        context.logger.event(
            "stage.passthrough", source=str(model), reason="alignment disabled"
        )
    return StageResult(
        artifacts=(
            ArtifactOutput("model.alignment", model, "router_model/v1", metadata=metadata),
        )
    )


def train_retrieval(context: StageContext) -> StageResult:
    """Train the multi-Skill retrieval phase or preserve alignment-only output."""

    if common.alignment_only(context):
        model = context.artifact("model.alignment")
        context.logger.event(
            "stage.passthrough",
            source=str(model),
            reason="single-candidate alignment-only run",
        )
        return StageResult(
            artifacts=(
                ArtifactOutput(
                    "model.retrieval",
                    model,
                    "router_model/v1",
                    metadata={
                        "passthrough": True,
                        "reason": "single-candidate alignment-only run",
                    },
                ),
            )
        )

    common.verify_training_provenance(context)
    output_root = context.output_dir
    model = output_root / "retrieval"
    environment = common.legacy_environment(context)
    environment.update(common.checkpoint_environment(context))
    environment.update(
        {
            "ROUTER_OUTPUT_DIR": str(output_root),
            "ROUTER_RETRIEVAL_INIT_DIR": str(context.artifact("model.alignment")),
        }
    )
    checkpoint = context.select_resume_checkpoint(kind="router", root=model)
    if checkpoint:
        environment["ROUTER_RESUME_RETRIEVAL"] = checkpoint
    context.run_command(
        ["bash", "scripts/skillret/06b_train_retrieval.sh"],
        environment=environment,
        label="legacy-router-train-retrieval",
    )
    return StageResult(
        artifacts=(ArtifactOutput("model.retrieval", model, "router_model/v1"),)
    )


memorization_handler = train_memorization
alignment_handler = train_alignment
retrieval_handler = train_retrieval
