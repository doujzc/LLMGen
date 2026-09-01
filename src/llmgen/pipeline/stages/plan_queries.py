"""Workflow-planning Stage adapter."""

from __future__ import annotations

from ..io import atomic_write_json, atomic_write_jsonl, utc_now
from .base import ArtifactOutput, StageContext, StageResult
from .common import alignment_only, paths, python


def plan_queries(context: StageContext) -> StageResult:
    """Build multi-candidate workflows, or a durable empty alignment-only plan."""

    stage_paths = paths(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    if alignment_only(context):
        atomic_write_jsonl(stage_paths["workflows"], [])
        atomic_write_json(stage_paths["workflows"].with_suffix(".manifest.json"), {"schema_version": 1, "stage": "workflow_specs", "created_at": utc_now(), "execution_mode": "alignment_only", "workflow_count": 0, "candidate_skill_count": 1})
        return StageResult(artifacts=(ArtifactOutput("data.workflows", stage_paths["workflows"], "workflow_plan/v1", metadata={"execution_mode": "alignment_only"}),), progress={"workflow_count": 0, "execution_mode": "alignment_only"})
    skills_per_query = context.config.require("data_generation.skills_per_query")
    context.run_command(
        [python(context), "scripts/clawhub_data/01_build_workflows.py", "--profiles", str(stage_paths["profiles"]), "--output", str(stage_paths["workflows"]), "--workflows-per-skill", str(int(context.config.require("data_generation.workflows_per_skill"))), "--min-skills-per-query", str(int(skills_per_query["min"])), "--max-skills-per-query", str(int(skills_per_query["max"])), "--seed", str(int(context.config.require("run.seed")))],
        label="plan-workflows",
    )
    return StageResult(artifacts=(ArtifactOutput("data.workflows", stage_paths["workflows"], "workflow_plan/v1"),))

