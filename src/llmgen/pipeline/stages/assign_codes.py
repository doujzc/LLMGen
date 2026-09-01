"""Code-assignment Stage adapter for the generic candidate pipeline."""

from __future__ import annotations

from . import common
from .base import ArtifactOutput, StageContext, StageResult


def assign_codes(context: StageContext) -> StageResult:
    """Export the trained codebook's train split and decoder namespace."""

    stage_paths = common.paths(context)
    common.router_pipeline(context, "export-codes")
    return StageResult(
        artifacts=(
            ArtifactOutput("codes.directory", stage_paths["index"], "skill_code_index/v1"),
            ArtifactOutput(
                "codes.train", stage_paths["index"] / "train_codes.jsonl", "skill_code/v1"
            ),
            ArtifactOutput(
                "codes.registry",
                stage_paths["index"] / "train_registry.json",
                "skill_registry/v1",
            ),
            ArtifactOutput(
                "codes.virtual_tokens",
                stage_paths["index"] / "virtual_tokens.txt",
                "virtual_tokens/v1",
            ),
            ArtifactOutput(
                "codes.manifest", stage_paths["index"] / "manifest.json", "code_index_manifest/v1"
            ),
        )
    )


handler = assign_codes
