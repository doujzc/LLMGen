"""Query-review and coverage-backfill Stage adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..io import atomic_write_json, atomic_write_jsonl, utc_now
from ..providers import ledger_outputs, provider_api_config, provider_config, provider_environment, workers
from .base import ArtifactOutput, StageContext, StageResult
from .common import alignment_only, copy, paths, python


def _review_alignment(context: StageContext, stage_paths: Mapping[str, Path]) -> None:
    provider = provider_config(context, "review")
    common = ["--api-config", provider_api_config(context, "review"), "--model", str(provider["model"]), "--workers", workers(provider), "--batch-size", str(int(context.config.require("data_generation.review_batch_size")))]
    context.run_command(
        [python(context), "scripts/clawhub_data/03a_review_alignment_queries.py", "--queries", str(stage_paths["review_alignment_queries"]), "--profiles", str(stage_paths["profiles"]), "--output", str(stage_paths["alignment_reviews"]), *common],
        environment=provider_environment(context, "review", operation="review-alignment"),
        label="review-alignment-queries",
    )


def _review_multiskill(context: StageContext, stage_paths: Mapping[str, Path]) -> None:
    provider = provider_config(context, "review")
    common = ["--api-config", provider_api_config(context, "review"), "--model", str(provider["model"]), "--workers", workers(provider), "--batch-size", str(int(context.config.require("data_generation.review_batch_size")))]
    context.run_command(
        [python(context), "scripts/clawhub_data/03_review_queries.py", "--queries", str(stage_paths["review_queries"]), "--workflows", str(stage_paths["review_workflows"]), "--output", str(stage_paths["reviews"]), *common],
        environment=provider_environment(context, "review", operation="review-multiskill"),
        label="review-multiskill-queries",
    )


def review_queries(context: StageContext) -> StageResult:
    """Review generated data and run the established alignment/coverage backfills."""

    stage_paths = paths(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    copy(stage_paths["workflows"], stage_paths["review_workflows"])
    copy(stage_paths["generated_queries"], stage_paths["review_queries"])
    copy(stage_paths["alignment_queries"], stage_paths["review_alignment_queries"])
    _review_alignment(context, stage_paths)
    generation = provider_config(context, "generation")
    generation_common = ["--api-config", provider_api_config(context, "generation"), "--model", str(generation["model"]), "--workers", workers(generation)]
    alignment_rounds = int(context.config.require("data_generation.alignment_backfill_rounds"))
    min_alignment = int(context.config.require("data_generation.alignment_queries_per_skill"))
    for round_index in range(1, alignment_rounds + 1):
        context.run_command(
            [python(context), "scripts/clawhub_data/03a2_backfill_alignment.py", "--profiles", str(stage_paths["profiles"]), "--queries", str(stage_paths["review_alignment_queries"]), "--reviews", str(stage_paths["alignment_reviews"]), "--round", str(round_index), "--variants", str(int(context.config.require("data_generation.alignment_queries_per_skill"))), "--batch-size", str(int(context.config.require("data_generation.alignment_batch_size"))), "--min-passed-per-skill", str(min_alignment), *generation_common],
            environment=provider_environment(context, "generation", operation=f"alignment-backfill-{round_index}"),
            label=f"alignment-backfill-{round_index}",
        )
        _review_alignment(context, stage_paths)
    manual = str(context.config.get("data_generation.manual_alignment_path") or "").strip()
    if manual:
        context.run_command(
            [python(context), "scripts/light_data/02b_apply_manual_alignment.py", "--profiles", str(stage_paths["profiles"]), "--queries", str(stage_paths["review_alignment_queries"]), "--reviews", str(stage_paths["alignment_reviews"]), "--curated", str(context.artifact("inputs.manual_alignment"))],
            label="apply-manual-alignment",
        )
    is_alignment_only = alignment_only(context)
    if is_alignment_only:
        atomic_write_jsonl(stage_paths["reviews"], [])
        atomic_write_json(stage_paths["reviews"].with_suffix(".manifest.json"), {"schema_version": 1, "stage": "query_review", "created_at": utc_now(), "execution_mode": "alignment_only", "reviewed_count": 0})
    else:
        _review_multiskill(context, stage_paths)
    coverage_rounds = 0 if is_alignment_only else int(context.config.require("data_generation.max_backfill_rounds"))
    skills_per_query = context.config.require("data_generation.skills_per_query")
    split = context.config.require("data_generation.split")
    for round_index in range(1, coverage_rounds + 1):
        context.run_command(
            [python(context), "scripts/clawhub_data/03b_build_coverage_workflows.py", "--profiles", str(stage_paths["profiles"]), "--workflows", str(stage_paths["review_workflows"]), "--queries", str(stage_paths["review_queries"]), "--reviews", str(stage_paths["reviews"]), "--alignment-queries", str(stage_paths["review_alignment_queries"]), "--alignment-reviews", str(stage_paths["alignment_reviews"]), "--round", str(round_index), "--min-train-positives-per-skill", str(int(context.config.require("data_generation.retrieval_positives_per_skill"))), "--variants-per-workflow", str(int(context.config.require("data_generation.explicit_variants"))), "--oversample-factor", str(float(context.config.require("data_generation.coverage_oversample_factor"))), "--min-skills-per-query", str(int(skills_per_query["min"])), "--max-skills-per-query", str(int(skills_per_query["max"])), "--train-fraction", str(float(split["train"])), "--validation-fraction", str(float(split["validation"])), "--test-fraction", str(float(split["test"])), "--seed", str(int(context.config.require("run.seed")))],
            label=f"plan-coverage-backfill-{round_index}",
        )
        context.run_command(
            [python(context), "scripts/clawhub_data/02_generate_queries.py", "--workflows", str(stage_paths["review_workflows"]), "--output", str(stage_paths["review_queries"]), "--variants", str(int(context.config.require("data_generation.explicit_variants"))), "--implicit-variants", str(int(context.config.require("data_generation.implicit_variants"))), "--batch-size", str(int(context.config.require("data_generation.query_batch_size"))), "--validation-retry-rounds", str(int(context.config.require("data_generation.validation_retry_rounds"))), "--min-completion-rate", str(float(context.config.require("data_generation.min_completion_rate"))), *generation_common],
            environment=provider_environment(context, "generation", operation=f"coverage-backfill-{round_index}"),
            label=f"generate-coverage-backfill-{round_index}",
        )
        _review_multiskill(context, stage_paths)
    final_rounds = int(context.config.require("data_generation.final_alignment_backfill_rounds"))
    for offset in range(1, final_rounds + 1):
        round_index = alignment_rounds + offset
        context.run_command(
            [python(context), "scripts/clawhub_data/03a2_backfill_alignment.py", "--profiles", str(stage_paths["profiles"]), "--queries", str(stage_paths["review_alignment_queries"]), "--reviews", str(stage_paths["alignment_reviews"]), "--round", str(round_index), "--variants", str(int(context.config.require("data_generation.alignment_queries_per_skill"))), "--batch-size", str(int(context.config.require("data_generation.alignment_batch_size"))), "--min-passed-per-skill", str(min_alignment), "--multiskill-queries", str(stage_paths["review_queries"]), "--multiskill-reviews", str(stage_paths["reviews"]), "--workflows", str(stage_paths["review_workflows"]), "--min-combined-per-skill", str(int(context.config.require("data_generation.retrieval_positives_per_skill"))), *generation_common],
            environment=provider_environment(context, "generation", operation=f"final-alignment-backfill-{round_index}"),
            label=f"final-alignment-backfill-{offset}",
        )
        _review_alignment(context, stage_paths)
    ledger_artifacts, ledger_progress = ledger_outputs(context, "generation", "review")
    return StageResult(
        artifacts=(ArtifactOutput("data.workflows.reviewed", stage_paths["review_workflows"], "workflow_plan/v1"), ArtifactOutput("data.queries.reviewed", stage_paths["review_queries"], "query_draft/v1"), ArtifactOutput("data.reviews", stage_paths["reviews"], "query_review/v1"), ArtifactOutput("data.queries.alignment.reviewed", stage_paths["review_alignment_queries"], "alignment_query_draft/v1"), ArtifactOutput("data.reviews.alignment", stage_paths["alignment_reviews"], "alignment_query_review/v1"), *ledger_artifacts),
        progress={"provider_ledgers": ledger_progress},
    )
