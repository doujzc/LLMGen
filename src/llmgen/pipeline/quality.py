"""Portable evaluation and export-quality report helpers.

The functions in this module deliberately do not know about a Runner or a
StageContext.  Stage adapters can therefore build the same audited report for
an in-process evaluator, a legacy subprocess, or a future Provider-backed
implementation without duplicating schema and integrity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..router_bundle import validate_skill_decode_map
from .io import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file, utc_now


class QualityValidationError(ValueError):
    """Raised when an evaluation or export report input has no valid schema."""


@dataclass(frozen=True)
class EvaluationAudit:
    """Validated evaluation data plus its derived quality-gate evidence."""

    evaluation: Mapping[str, Any]
    quality_gates: Mapping[str, Any]
    failed_examples: tuple[Mapping[str, Any], ...]


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityValidationError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QualityValidationError(f"{field} must be a finite number")
    return normalized


def _json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, ValueError) as error:
        raise QualityValidationError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise QualityValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _decode_paths(payload: Mapping[str, Any]) -> tuple[dict[tuple[str, ...], tuple[str, ...]], set[tuple[str, ...]]]:
    try:
        validate_skill_decode_map(payload)
    except Exception as error:
        raise QualityValidationError(f"invalid skill decode map: {error}") from error
    by_tokens: dict[tuple[str, ...], tuple[str, ...]] = {}
    by_skills: set[tuple[str, ...]] = set()
    for raw_path in payload["paths"]:
        assert isinstance(raw_path, Mapping)  # guaranteed by validate_skill_decode_map
        tokens = tuple(str(value) for value in raw_path["tokens"])
        skill_ids = tuple(str(value) for value in raw_path["skill_ids"])
        by_tokens[tokens] = skill_ids
        by_skills.add(skill_ids)
    return by_tokens, by_skills


def _prediction_failure(
    row: Mapping[str, Any],
    *,
    row_number: int,
    paths_by_tokens: Mapping[tuple[str, ...], tuple[str, ...]],
    paths_by_skills: set[tuple[str, ...]],
) -> dict[str, Any] | None:
    """Return normalized failure evidence for one prediction, if any."""

    reasons: list[str] = []
    query_id = row.get("query_id")
    if not isinstance(query_id, str) or not query_id.strip():
        reasons.append("query_id must be a non-empty string")
    raw_paths = row.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        reasons.append("paths must be a non-empty list")
        raw_paths = []
    for path_number, raw_path in enumerate(raw_paths, start=1):
        prefix = f"paths[{path_number}]"
        if not isinstance(raw_path, Mapping):
            reasons.append(f"{prefix} must be an object")
            continue
        raw_skills = raw_path.get("skill_ids")
        if not isinstance(raw_skills, list) or not raw_skills or any(
            not isinstance(value, str) or not value for value in raw_skills
        ):
            reasons.append(f"{prefix}.skill_ids must be a non-empty string list")
            continue
        skill_ids = tuple(raw_skills)
        raw_tokens = raw_path.get("code_tokens")
        if raw_tokens is None:
            if skill_ids not in paths_by_skills:
                reasons.append(f"{prefix} is not a decode-map path")
            continue
        if not isinstance(raw_tokens, list) or not raw_tokens or any(
            not isinstance(value, str) or not value for value in raw_tokens
        ):
            reasons.append(f"{prefix}.code_tokens must be a non-empty string list")
            continue
        expected_skills = paths_by_tokens.get(tuple(raw_tokens))
        if expected_skills is None:
            reasons.append(f"{prefix}.code_tokens are not a decode-map path")
        elif skill_ids != expected_skills:
            reasons.append(f"{prefix}.skill_ids disagree with decode-map path")
    if not reasons:
        return None
    return {
        "row_number": row_number,
        "query_id": query_id if isinstance(query_id, str) else None,
        "failure_reasons": reasons,
        "prediction": dict(row),
    }


def audit_evaluation(
    *,
    metrics_path: str | Path,
    predictions_path: str | Path,
    decode_map_path: str | Path,
    metric_thresholds: Mapping[str, float] | None = None,
    required_format_valid_rate: float = 1.0,
) -> EvaluationAudit:
    """Validate evaluator outputs and derive a non-silent quality report.

    A malformed top-level metrics/decode-map document is rejected immediately.
    Prediction-row defects are retained as ``failed_examples`` so callers can
    publish useful diagnostics while the resulting quality gate remains false.
    Each generated path must be an exact path from ``skill_decode_map.json``;
    a row that supplies ``code_tokens`` must also agree with that path's exact
    ordered ``skill_ids`` membership.
    """

    required_rate = _finite_number(
        required_format_valid_rate, field="required_format_valid_rate"
    )
    if not 0.0 <= required_rate <= 1.0:
        raise QualityValidationError("required_format_valid_rate must be in [0, 1]")
    thresholds = dict(metric_thresholds or {})
    for name, threshold in thresholds.items():
        if not isinstance(name, str) or not name:
            raise QualityValidationError("metric threshold names must be non-empty strings")
        _finite_number(threshold, field=f"metric_thresholds.{name}")

    metrics_document = _json_object(metrics_path, label="metrics")
    raw_metrics = metrics_document.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise QualityValidationError("metrics.json must contain a metrics object")
    metrics: dict[str, float] = {}
    for name, value in raw_metrics.items():
        if not isinstance(name, str) or not name:
            raise QualityValidationError("metrics keys must be non-empty strings")
        metrics[name] = _finite_number(value, field=f"metrics.{name}")

    decode_map = _json_object(decode_map_path, label="skill decode map")
    paths_by_tokens, paths_by_skills = _decode_paths(decode_map)
    try:
        predictions = read_jsonl(predictions_path)
    except (OSError, ValueError) as error:
        raise QualityValidationError(f"cannot read predictions: {predictions_path}") from error
    failures = tuple(
        failure
        for row_number, row in enumerate(predictions, start=1)
        if (failure := _prediction_failure(
            row,
            row_number=row_number,
            paths_by_tokens=paths_by_tokens,
            paths_by_skills=paths_by_skills,
        )) is not None
    )
    prediction_count = len(predictions)
    format_valid_rate = (prediction_count - len(failures)) / prediction_count if prediction_count else 0.0
    metric_gates = {
        f"metric:{name}": metrics.get(name, float("-inf")) >= float(threshold)
        for name, threshold in thresholds.items()
    }
    gates = {
        "metrics_schema": True,
        "predictions_schema": bool(predictions),
        "decode_map_paths": not failures,
        "format_valid_rate": format_valid_rate >= required_rate,
        **metric_gates,
    }
    evaluation = {
        "schema_version": 1,
        "created_at": utc_now(),
        "metrics_path": str(Path(metrics_path)),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_path": str(Path(predictions_path)),
        "predictions_sha256": sha256_file(predictions_path),
        "decode_map_path": str(Path(decode_map_path)),
        "decode_map_sha256": sha256_file(decode_map_path),
        "metrics": metrics,
        "prediction_count": prediction_count,
        "invalid_prediction_count": len(failures),
        "format_valid_rate": format_valid_rate,
        "required_format_valid_rate": required_rate,
    }
    quality = {
        "schema_version": 1,
        "created_at": utc_now(),
        "gates": gates,
        "metric_thresholds": thresholds,
        "passed": all(gates.values()),
        "evaluation": evaluation,
    }
    return EvaluationAudit(evaluation=evaluation, quality_gates=quality, failed_examples=failures)


def model_file_manifest(model_dir: str | Path) -> dict[str, Any]:
    """Return a deterministic, file-level SHA-256 manifest for an export tree.

    Symlinks are refused so a report cannot attest to bytes outside the portable
    model directory.  Empty directories are intentionally omitted because they
    carry no model content.
    """

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise QualityValidationError(f"export model directory does not exist: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise QualityValidationError(f"export model must not contain symlinks: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files:
        raise QualityValidationError(f"export model directory has no files: {root}")
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "model_dir": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def build_run_lineage_payload(
    *,
    run_id: str,
    candidate_input_sha256: str,
    config_sha256: str,
    model_manifest: Mapping[str, Any],
    artifact_lineage: Mapping[str, Any],
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build the export-side lineage record without relying on Runner internals."""

    for name, value in {
        "run_id": run_id,
        "candidate_input_sha256": candidate_input_sha256,
        "config_sha256": config_sha256,
    }.items():
        if not isinstance(value, str) or not value:
            raise QualityValidationError(f"{name} must be a non-empty string")
    files = model_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise QualityValidationError("model_manifest must contain a non-empty files list")
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": run_id,
        "candidate_input_sha256": candidate_input_sha256,
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "artifacts": dict(artifact_lineage),
        "model_files": dict(model_manifest),
    }


def write_export_reports(
    report_dir: str | Path,
    *,
    audit: EvaluationAudit,
    model_manifest: Mapping[str, Any],
    run_lineage: Mapping[str, Any],
) -> dict[str, Path]:
    """Atomically persist the standard evaluation/export evidence artifacts."""

    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": root / "evaluation.json",
        "quality_gates": root / "quality_gates.json",
        "failed_examples": root / "failed_examples.jsonl",
        "model_files": root / "model_files.json",
        "run_lineage": root / "run_lineage.json",
    }
    atomic_write_json(paths["evaluation"], dict(audit.evaluation))
    atomic_write_json(paths["quality_gates"], dict(audit.quality_gates))
    atomic_write_jsonl(paths["failed_examples"], audit.failed_examples)
    atomic_write_json(paths["model_files"], dict(model_manifest))
    atomic_write_json(paths["run_lineage"], dict(run_lineage))
    return paths
