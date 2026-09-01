"""Export stage adapter and deployment-quality gates for generic pipelines."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

from ...router_bundle import validate_skill_decode_map
from ..io import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)
from ..quality import (
    audit_evaluation,
    build_run_lineage_payload,
    model_file_manifest,
    write_export_reports,
)
from .base import ArtifactOutput, StageContext, StageResult
from .common import legacy_environment, paths, python, verify_training_provenance


def copy_model_tree(source: Path, destination: Path) -> None:
    """Copy metadata and safely share immutable model weight blobs."""

    def link_weights_or_copy(src: str, dst: str) -> str:
        source_path = Path(src)
        immutable_weight = (
            source_path.suffix == ".safetensors"
            or source_path.name.startswith("pytorch_model-")
            or source_path.name in {"pytorch_model.bin", "adapter_model.bin"}
        )
        if immutable_weight:
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        shutil.copy2(src, dst)
        return dst

    shutil.copytree(
        source,
        destination,
        copy_function=link_weights_or_copy,
        ignore=shutil.ignore_patterns("checkpoint-*"),
    )


def root_weight_files(model_dir: Path) -> list[Path]:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    files = [model_dir / name for name in names if (model_dir / name).is_file()]
    files.extend(sorted(model_dir.glob("model-*.safetensors")))
    files.extend(sorted(model_dir.glob("pytorch_model-*.bin")))
    return list(dict.fromkeys(files))


def full_weights_are_present(model_dir: Path, weight_files: Sequence[Path]) -> bool:
    """Check that a root weight file or every shard named by its index exists."""

    if any(
        path.is_file() and path.stat().st_size > 0
        for path in (
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
        )
    ):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / name
        if not index_path.is_file():
            continue
        try:
            payload = read_json(index_path)
            weight_map = payload.get("weight_map")
            shard_names = (
                {str(value) for value in weight_map.values()}
                if isinstance(weight_map, dict)
                else set()
            )
        except (OSError, TypeError, ValueError):
            shard_names = set()
        safe_shards = [
            model_dir / shard
            for shard in shard_names
            if Path(shard).name == shard
        ]
        if (
            len(safe_shards) == len(shard_names)
            and safe_shards
            and all(
                path.is_file() and path.stat().st_size > 0
                for path in safe_shards
            )
        ):
            return True
    return False


def export_quality(
    context: StageContext,
    model_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required_names = (
        "config.json",
        "tokenizer_config.json",
        "router_manifest.json",
        "skill_decode_map.json",
        "virtual_tokens.txt",
    )
    required_files = {name: (model_dir / name).is_file() for name in required_names}
    weight_files = root_weight_files(model_dir)
    tokenizer_files = [
        name
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
        if (model_dir / name).is_file()
    ]

    decode_map: dict[str, Any] = {}
    decode_error: str | None = None
    try:
        raw_decode = read_json(model_dir / "skill_decode_map.json")
        if not isinstance(raw_decode, dict):
            raise ValueError("decode map must be an object")
        validate_skill_decode_map(raw_decode)
        decode_map = raw_decode
    except Exception as error:
        decode_error = f"{type(error).__name__}: {error}"
    candidate_count = int(
        read_json(context.artifact("candidates.manifest"))["candidate_count"]
    )
    decoded_skills = decode_map.get("skills")
    decoded_count = len(decoded_skills) if isinstance(decoded_skills, dict) else 0
    candidate_coverage = decoded_count / candidate_count if candidate_count else 0.0

    token_consistency = False
    if decode_map:
        try:
            file_tokens = [
                line.strip()
                for line in (model_dir / "virtual_tokens.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            map_tokens = [str(value) for value in decode_map["virtual_tokens"]]
            token_consistency = file_tokens == map_tokens and len(file_tokens) == len(
                set(file_tokens)
            )
        except (KeyError, OSError, TypeError, ValueError):
            token_consistency = False

    stage_paths = paths(context)
    predictions_path = stage_paths["evaluation"] / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.is_file() else []
    invalid_predictions = [
        row
        for row in predictions
        if not isinstance(row.get("paths"), list) or not row.get("paths")
    ]
    format_valid_rate = (
        (len(predictions) - len(invalid_predictions)) / len(predictions)
        if predictions
        else 0.0
    )
    required_format_rate = float(
        context.config.require("evaluation.require_format_valid_rate")
    )
    required_candidate_coverage = float(
        context.config.require("evaluation.require_candidate_coverage")
    )
    metrics_path = stage_paths["evaluation"] / "metrics.json"
    metrics_payload: dict[str, Any] = {}
    metrics_error: str | None = None
    try:
        raw_metrics = read_json(metrics_path)
        if not isinstance(raw_metrics, dict) or not isinstance(
            raw_metrics.get("metrics"), dict
        ):
            raise ValueError("metrics.json must contain a metrics object")
        metrics_payload = raw_metrics
    except Exception as error:
        metrics_error = f"{type(error).__name__}: {error}"
    metric_values = metrics_payload.get("metrics", {})
    metric_thresholds = context.config.require("evaluation.metric_thresholds")
    gates = {
        "required_files": all(required_files.values()),
        "full_model_weights": full_weights_are_present(model_dir, weight_files),
        "tokenizer_assets": bool(tokenizer_files),
        "decoder_token_consistency": token_consistency,
        "evaluation_metrics_present": metrics_error is None,
        "format_valid_rate": format_valid_rate >= required_format_rate,
        "candidate_coverage": candidate_coverage >= required_candidate_coverage,
    }
    for name, threshold in metric_thresholds.items():
        value = metric_values.get(name)
        gates[f"metric:{name}"] = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= float(threshold)
        )
    return (
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "required_files": required_files,
            "weight_files": [path.name for path in weight_files],
            "tokenizer_files": tokenizer_files,
            "decode_error": decode_error,
            "candidate_count": candidate_count,
            "decoded_candidate_count": decoded_count,
            "candidate_coverage": candidate_coverage,
            "required_candidate_coverage": required_candidate_coverage,
            "prediction_count": len(predictions),
            "invalid_prediction_count": len(invalid_predictions),
            "format_valid_rate": format_valid_rate,
            "required_format_valid_rate": required_format_rate,
            "metrics_error": metrics_error,
            "metrics": metric_values,
            "metric_thresholds": metric_thresholds,
            "gates": gates,
            "passed": all(gates.values()),
            "require_all_gates": bool(
                context.config.require("export.require_all_gates")
            ),
            "allow_failed_gates": bool(
                context.config.require("export.allow_failed_gates")
            ),
            "model_load_smoke_test": "not_requested",
        },
        invalid_predictions,
    )


def run_export_model_smoke(context: StageContext, model_dir: Path) -> None:
    if not context.config.require("export.smoke_test"):
        return
    smoke_dir = context.attempt_dir / "export-smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    predictions = smoke_dir / "predictions.jsonl"
    command = [
        python(context),
        "scripts/infer_router.py",
        "--model-name-or-path",
        str(model_dir),
        "--candidate-state-dir",
        str(model_dir),
        "--query",
        "Select the single best available skill for this request.",
        "--query-id",
        "export-smoke",
        "--output-jsonl",
        str(predictions),
        "--batch-size",
        "1",
        "--max-code-paths",
        "1",
        "--top-k",
        "1",
        "--cutoffs",
        "1",
        "--device",
        str(context.config.require("runtime.device")),
        "--dtype",
        str(context.config.require("evaluation.dtype")),
    ]
    if context.config.require("router.trust_remote_code"):
        command.append("--trust-remote-code")
    context.run_command(command, label="export-constrained-decode-smoke-test")
    smoke_metrics = smoke_dir / "metrics.json"
    atomic_write_json(smoke_metrics, {"metrics": {}})
    audit = audit_evaluation(
        metrics_path=smoke_metrics,
        predictions_path=predictions,
        decode_map_path=model_dir / "skill_decode_map.json",
        required_format_valid_rate=1.0,
    )
    if not audit.quality_gates["passed"]:
        raise ValueError(
            "export constrained-decode smoke produced an invalid decoder path"
        )
    atomic_write_json(smoke_dir / "evaluation.json", dict(audit.evaluation))


def export(context: StageContext) -> StageResult:
    verify_training_provenance(context)
    stage_paths = paths(context)
    source = context.artifact("model.retrieval")
    temporary = context.attempt_dir / "model"
    if temporary.exists():
        shutil.rmtree(temporary)
    if context.config.require("router.finetune_mode") == "lora":
        command = [
            python(context),
            "scripts/merge_router_adapter.py",
            "--base-model",
            str(context.config.require("router.base_model")),
            "--adapter",
            str(source),
            "--output-dir",
            str(temporary),
        ]
        if context.config.require("router.trust_remote_code"):
            command.append("--trust-remote-code")
        context.run_command(command, label="merge-router-lora-adapter")
    else:
        copy_model_tree(source, temporary)
    environment = legacy_environment(context)
    environment["ROUTER_EXPORT_MODEL_DIR"] = str(temporary)
    context.run_command(
        ["bash", "scripts/skillret/10_export_web_bundle.sh"],
        environment=environment,
        label="materialize-router-bundle",
    )
    run_manifest = context.state.read_run()
    candidate_input_sha256 = str(
        read_json(context.run_dir / "config" / "candidate_input.json")["sha256"]
    )
    registry_snapshot = {
        name: record.to_dict() for name, record in context.registry.all().items()
    }
    router_manifest_path = temporary / "router_manifest.json"
    if router_manifest_path.is_file():
        router_manifest = read_json(router_manifest_path)
        if not isinstance(router_manifest, dict):
            raise ValueError("exported router_manifest.json must be an object")
        training_stage_lineage = router_manifest.get("pipeline_lineage")
        provenance_path = context.run_dir / "config" / "provenance.json"
        router_manifest["pipeline_lineage"] = {
            "schema_version": 1,
            "run_id": run_manifest["run_id"],
            "config_sha256": context.config.hash,
            "git_commit": run_manifest.get("git_commit"),
            "candidate_input_sha256": candidate_input_sha256,
            "run_provenance_sha256": (
                sha256_file(provenance_path) if provenance_path.is_file() else None
            ),
            "source_artifact": "model.retrieval",
            "training_stage": (
                training_stage_lineage
                if isinstance(training_stage_lineage, dict)
                else None
            ),
            "artifacts": {
                name: {
                    "sha256": record["sha256"],
                    "artifact_schema": record["artifact_schema"],
                    "format": record["format"],
                    "producer": record["producer"],
                    "inputs": record["inputs"],
                    "config_hash": record["config_hash"],
                }
                for name, record in sorted(registry_snapshot.items())
            },
        }
        atomic_write_json(router_manifest_path, router_manifest)

    smoke_status = "not_requested"
    smoke_error: str | None = None
    if context.config.require("export.smoke_test"):
        try:
            run_export_model_smoke(context, temporary)
            smoke_status = "passed"
        except Exception as error:
            smoke_status = "failed"
            smoke_error = f"{type(error).__name__}: {error}"
            context.logger.event(
                "export.smoke_test_failed",
                level="ERROR",
                error_type=type(error).__name__,
                error=str(error),
            )

    quality, _legacy_failed_examples = export_quality(context, temporary)
    evaluation_audit = audit_evaluation(
        metrics_path=stage_paths["evaluation"] / "metrics.json",
        predictions_path=stage_paths["evaluation"] / "predictions.jsonl",
        decode_map_path=temporary / "skill_decode_map.json",
        metric_thresholds=context.config.require("evaluation.metric_thresholds"),
        required_format_valid_rate=float(
            context.config.require("evaluation.require_format_valid_rate")
        ),
    )
    quality["evaluation"] = dict(evaluation_audit.evaluation)
    quality["gates"].update(
        {
            f"evaluation:{name}": bool(passed)
            for name, passed in evaluation_audit.quality_gates["gates"].items()
        }
    )
    quality["model_load_smoke_test"] = smoke_status
    if smoke_error is not None:
        quality["model_load_smoke_error"] = smoke_error
    if smoke_status != "not_requested":
        quality["gates"]["model_load_smoke_test"] = smoke_status == "passed"
    quality["passed"] = all(quality["gates"].values())
    quality["deployment_qualified"] = quality["passed"] and smoke_status == "passed"

    if router_manifest_path.is_file():
        router_manifest = read_json(router_manifest_path)
        router_manifest["pipeline_quality"] = {
            "schema_version": 1,
            "passed": quality["passed"],
            "deployment_qualified": quality["deployment_qualified"],
            "failed_gates": [
                name for name, passed in quality["gates"].items() if not passed
            ],
            "allow_failed_gates": bool(
                context.config.require("export.allow_failed_gates")
            ),
            "model_load_smoke_test": smoke_status,
        }
        preliminary_files = model_file_manifest(temporary)
        pipeline_lineage = router_manifest.setdefault("pipeline_lineage", {})
        pipeline_lineage["model_files"] = [
            item
            for item in preliminary_files["files"]
            if item["path"] != "router_manifest.json"
        ]
        pipeline_lineage["model_file_hash_scope"] = (
            "all exported model files except router_manifest.json; the complete "
            "post-write inventory is export/report/model_files.json"
        )
        atomic_write_json(router_manifest_path, router_manifest)

    attempt_report = context.attempt_dir / "report"
    if attempt_report.exists():
        shutil.rmtree(attempt_report)
    attempt_report.mkdir(parents=True)
    exported_files = model_file_manifest(temporary)
    run_lineage = build_run_lineage_payload(
        run_id=str(run_manifest["run_id"]),
        candidate_input_sha256=candidate_input_sha256,
        config_sha256=context.config.hash,
        model_manifest=exported_files,
        artifact_lineage=registry_snapshot,
        git_commit=run_manifest.get("git_commit"),
    )
    write_export_reports(
        attempt_report,
        audit=evaluation_audit,
        model_manifest=exported_files,
        run_lineage=run_lineage,
    )
    atomic_write_json(attempt_report / "quality_gates.json", quality)
    atomic_write_json(attempt_report / "artifact_lineage.json", run_lineage)

    require_all = bool(context.config.require("export.require_all_gates"))
    allow_failed = bool(context.config.require("export.allow_failed_gates"))
    if not quality["passed"] and require_all and not allow_failed:
        failed = ", ".join(
            name for name, passed in quality["gates"].items() if not passed
        )
        raise ValueError(
            "export quality gates failed: "
            + failed
            + f"; diagnostics remain in {attempt_report}"
        )

    destination = stage_paths["export_model"]
    report_dir = stage_paths["export_report"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": run_manifest["run_id"],
        "model": str(destination),
        "candidate_count": read_json(context.artifact("candidates.manifest"))["candidate_count"],
        "config_hash": context.config.hash,
        "model_file_count": exported_files["file_count"],
        "model_total_bytes": exported_files["total_bytes"],
        "quality_gates": quality,
    }
    atomic_write_json(attempt_report / "run_summary.json", summary)
    summary_text = (
        f"# Pipeline Run {summary['run_id']}\n\n"
        f"- Model: `{destination}`\n"
        f"- Candidates: {summary['candidate_count']}\n"
        f"- Config SHA-256: `{context.config.hash}`\n"
        f"- Quality gates: {'passed' if quality['passed'] else 'failed'}\n"
        f"- Deployment qualified: {str(quality['deployment_qualified']).lower()}\n"
    )
    atomic_write_text(attempt_report / "run_summary.md", summary_text)

    previous_model = context.attempt_dir / "previous-model"
    previous_report = context.attempt_dir / "previous-report"
    for previous in (previous_model, previous_report):
        if previous.exists():
            shutil.rmtree(previous)
    if destination.exists():
        shutil.move(destination, previous_model)
    if report_dir.exists():
        shutil.move(report_dir, previous_report)
    try:
        os.replace(temporary, destination)
        os.replace(attempt_report, report_dir)
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        if report_dir.exists():
            shutil.rmtree(report_dir)
        if previous_model.exists():
            shutil.move(previous_model, destination)
        if previous_report.exists():
            shutil.move(previous_report, report_dir)
        raise
    return StageResult(
        artifacts=(
            ArtifactOutput("export.model", destination, "deployable_router/v1"),
            ArtifactOutput("export.report", report_dir, "pipeline_run_report/v1"),
            ArtifactOutput(
                "export.summary",
                report_dir / "run_summary.json",
                "pipeline_run_summary/v1",
            ),
            ArtifactOutput(
                "export.evaluation",
                report_dir / "evaluation.json",
                "router_evaluation_audit/v1",
            ),
            ArtifactOutput(
                "export.quality_gates",
                report_dir / "quality_gates.json",
                "quality_gates/v1",
            ),
            ArtifactOutput(
                "export.lineage",
                report_dir / "artifact_lineage.json",
                "artifact_lineage/v1",
            ),
            ArtifactOutput(
                "export.model_files",
                report_dir / "model_files.json",
                "model_file_manifest/v1",
            ),
            ArtifactOutput(
                "export.run_lineage",
                report_dir / "run_lineage.json",
                "artifact_lineage/v1",
            ),
        )
    )


_copy_model_tree = copy_model_tree
_root_weight_files = root_weight_files
_full_weights_are_present = full_weights_are_present
_export_quality = export_quality
_run_export_model_smoke = run_export_model_smoke
_export = export
