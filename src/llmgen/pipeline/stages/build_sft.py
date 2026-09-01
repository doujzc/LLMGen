"""Router SFT-data Stage adapter for the generic candidate pipeline."""

from __future__ import annotations

from . import common
from .base import ArtifactOutput, StageContext, StageResult


_SFT_SPLITS = (
    "memorization_train",
    "memorization_validation",
    "retrieval_alignment_train",
    "retrieval_train",
    "retrieval_validation",
)


def build_sft(context: StageContext) -> StageResult:
    """Build target-only curriculum data from final data and assigned codes."""

    stage_paths = common.paths(context)
    common.router_pipeline(context, "build-router-data")
    artifacts = [
        ArtifactOutput("sft.directory", stage_paths["router_data"], "router_sft_bundle/v1"),
        ArtifactOutput(
            "sft.manifest",
            stage_paths["router_data"] / "manifest.json",
            "router_sft_manifest/v1",
        ),
    ]
    for name in _SFT_SPLITS:
        path = stage_paths["router_data"] / f"{name}.jsonl"
        if path.is_file():
            artifacts.append(
                ArtifactOutput(f"sft.{name.replace('_', '.')}", path, "router_sft/v1")
            )
    return StageResult(artifacts=tuple(artifacts))


handler = build_sft
