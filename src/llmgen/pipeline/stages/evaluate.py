"""Evaluation stage adapter for the generic candidate pipeline."""

from __future__ import annotations

from ..io import atomic_write_json, atomic_write_jsonl
from ..quality import audit_evaluation
from .base import ArtifactOutput, StageContext, StageResult
from .common import paths, router_pipeline, verify_training_provenance


def evaluate(context: StageContext) -> StageResult:
    """Run constrained evaluation and persist its audited evidence."""

    verify_training_provenance(context)
    stage_paths = paths(context)
    router_pipeline(
        context,
        "evaluate",
        environment_overrides={
            "ROUTER_EVAL_MODEL_DIR": str(context.artifact("model.retrieval")),
            "EVAL_WORK_DIR": str(context.attempt_dir / "evaluation-work"),
        },
    )
    metrics = stage_paths["evaluation"] / "metrics.json"
    predictions = stage_paths["evaluation"] / "predictions.jsonl"
    audit = audit_evaluation(
        metrics_path=metrics,
        predictions_path=predictions,
        decode_map_path=context.artifact("model.retrieval")
        / "skill_decode_map.json",
        metric_thresholds=context.config.require("evaluation.metric_thresholds"),
        required_format_valid_rate=float(
            context.config.require("evaluation.require_format_valid_rate")
        ),
    )
    evaluation_report = stage_paths["evaluation"] / "evaluation.json"
    quality_report = stage_paths["evaluation"] / "quality_gates.json"
    failed_examples = stage_paths["evaluation"] / "failed_examples.jsonl"
    atomic_write_json(evaluation_report, dict(audit.evaluation))
    atomic_write_json(quality_report, dict(audit.quality_gates))
    atomic_write_jsonl(failed_examples, audit.failed_examples)
    artifacts = [
        ArtifactOutput(
            "evaluation.directory",
            stage_paths["evaluation"],
            "router_evaluation/v1",
        ),
        ArtifactOutput(
            "evaluation.audit",
            evaluation_report,
            "router_evaluation_audit/v1",
        ),
        ArtifactOutput(
            "evaluation.quality_gates",
            quality_report,
            "quality_gates/v1",
        ),
        ArtifactOutput(
            "evaluation.failed_examples",
            failed_examples,
            "evaluation_failures/v1",
        ),
        ArtifactOutput("evaluation.metrics", metrics, "router_metrics/v1"),
        ArtifactOutput(
            "evaluation.predictions",
            predictions,
            "router_predictions/v1",
        ),
    ]
    return StageResult(artifacts=tuple(artifacts))


_evaluate = evaluate
