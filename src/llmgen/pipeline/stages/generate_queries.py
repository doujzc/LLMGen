"""Alignment and multi-candidate query-generation Stage adapter."""

from __future__ import annotations

from ..io import atomic_write_json, atomic_write_jsonl, utc_now
from ..providers import ledger_outputs, provider_api_config, provider_config, provider_environment, workers
from .base import ArtifactOutput, StageContext, StageResult
from .common import alignment_only, paths, python


def generate_queries(context: StageContext) -> StageResult:
    """Generate alignment data first, then optional multi-candidate queries."""

    stage_paths = paths(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    provider = provider_config(context, "generation")
    common = ["--api-config", provider_api_config(context, "generation"), "--model", str(provider["model"]), "--workers", workers(provider)]
    context.run_command(
        [python(context), "scripts/clawhub_data/02a_generate_alignment_queries.py", "--profiles", str(stage_paths["profiles"]), "--output", str(stage_paths["alignment_queries"]), "--variants", str(int(context.config.require("data_generation.alignment_queries_per_skill"))), "--batch-size", str(int(context.config.require("data_generation.alignment_batch_size"))), *common],
        environment=provider_environment(context, "generation", operation="generate-alignment"),
        label="generate-alignment-queries",
    )
    if alignment_only(context):
        ledger_artifacts, ledger_progress = ledger_outputs(context, "generation")
        atomic_write_jsonl(stage_paths["generated_queries"], [])
        atomic_write_json(stage_paths["generated_queries"].with_suffix(".manifest.json"), {"schema_version": 1, "stage": "query_generation", "created_at": utc_now(), "execution_mode": "alignment_only", "query_count": 0})
        return StageResult(
            artifacts=(ArtifactOutput("data.queries.generated", stage_paths["generated_queries"], "query_draft/v1", metadata={"execution_mode": "alignment_only"}), ArtifactOutput("data.queries.alignment.generated", stage_paths["alignment_queries"], "alignment_query_draft/v1"), *ledger_artifacts),
            progress={"execution_mode": "alignment_only", "provider_ledgers": ledger_progress},
        )
    context.run_command(
        [python(context), "scripts/clawhub_data/02_generate_queries.py", "--workflows", str(stage_paths["workflows"]), "--output", str(stage_paths["generated_queries"]), "--variants", str(int(context.config.require("data_generation.explicit_variants"))), "--implicit-variants", str(int(context.config.require("data_generation.implicit_variants"))), "--batch-size", str(int(context.config.require("data_generation.query_batch_size"))), "--validation-retry-rounds", str(int(context.config.require("data_generation.validation_retry_rounds"))), "--min-completion-rate", str(float(context.config.require("data_generation.min_completion_rate"))), *common],
        environment=provider_environment(context, "generation", operation="generate-multiskill"),
        label="generate-multiskill-queries",
    )
    ledger_artifacts, ledger_progress = ledger_outputs(context, "generation")
    return StageResult(artifacts=(ArtifactOutput("data.queries.generated", stage_paths["generated_queries"], "query_draft/v1"), ArtifactOutput("data.queries.alignment.generated", stage_paths["alignment_queries"], "alignment_query_draft/v1"), *ledger_artifacts), progress={"provider_ledgers": ledger_progress})
