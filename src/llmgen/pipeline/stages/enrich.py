"""Candidate capability profiling Stage adapter."""

from __future__ import annotations

from ..providers import ledger_outputs, provider_api_config, provider_config, provider_environment, workers
from .base import ArtifactOutput, StageContext, StageResult
from .common import paths, python


def enrich(context: StageContext) -> StageResult:
    """Invoke the existing profile builder with the generic Provider contract."""

    stage_paths = paths(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    provider = provider_config(context, "generation")
    context.run_command(
        [python(context), "scripts/clawhub_data/00_profile_skills.py", "--catalog", str(stage_paths["catalog"]), "--output", str(stage_paths["profiles"]), "--api-config", provider_api_config(context, "generation"), "--model", str(provider["model"]), "--workers", workers(provider), "--batch-size", str(int(context.config.get("data_generation.profile_batch_size") or 10))],
        environment=provider_environment(context, "generation", operation="profile-candidates"),
        label="profile-candidates",
    )
    ledger_artifacts, ledger_progress = ledger_outputs(context, "generation")
    return StageResult(artifacts=(ArtifactOutput("data.profiles", stage_paths["profiles"], "skill_profile/v1"), *ledger_artifacts), progress={"provider_ledgers": ledger_progress})

