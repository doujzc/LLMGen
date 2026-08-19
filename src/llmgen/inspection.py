"""Read-only projections of Top1 training and evaluation artifacts."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .experiment import read_json_object
from .top1 import Top1DataError, messages_from_row, read_jsonl, sha256_file


def discover_evaluation_runs(root: str | Path) -> list[dict[str, Any]]:
    """Return newest-first summaries for evaluation runs below ``root``."""

    base = Path(root).expanduser().resolve()
    if not base.exists():
        return []
    records = []
    for manifest_path in base.rglob("eval_manifest.json"):
        run_dir = manifest_path.parent
        try:
            manifest = read_json_object(manifest_path)
            summary = _optional_json(run_dir / "summary.json")
            status = _optional_json(run_dir / "status.json")
            dataset = _mapping(manifest.get("dataset"))
            model = _mapping(manifest.get("model"))
            semantic = _mapping(manifest.get("semantic_inference"))
            available_oos = _mapping(summary.get("available_oos"))
            evaluation_id = manifest.get("evaluation_id")
            model_id_short = _short_identifier(model.get("model_id"))
            records.append(
                {
                    "run_ref": _run_reference(model_id_short, evaluation_id),
                    "run_dir": str(run_dir),
                    "created_at": manifest.get("created_at"),
                    "state": status.get("state", "UNKNOWN"),
                    "evaluation_id": evaluation_id,
                    "suite_id": manifest.get("suite_id"),
                    "model_id": model.get("model_id"),
                    "model_id_short": model_id_short,
                    "dataset": dataset.get("path"),
                    "dataset_name": _path_name(dataset.get("path")),
                    "dataset_sha256": dataset.get("sha256"),
                    "decoding": semantic.get("decoding_mode"),
                    "route_threshold": semantic.get("route_threshold"),
                    "rows": summary.get("rows"),
                    "backend_accuracy": summary.get("backend_accuracy"),
                    "raw_candidate_accuracy": summary.get(
                        "raw_candidate_accuracy"
                    ),
                    "unsafe_oos_accept_rate": available_oos.get(
                        "unsafe_oos_accept_rate"
                    ),
                }
            )
        except (OSError, Top1DataError, ValueError) as exc:
            records.append(
                {
                    "run_dir": str(run_dir),
                    "created_at": None,
                    "state": "INVALID",
                    "evaluation_id": run_dir.name,
                    "error": str(exc),
                }
            )
    return sorted(
        records,
        key=lambda row: (str(row.get("created_at") or ""), row["run_dir"]),
        reverse=True,
    )


def discover_training_runs(root: str | Path) -> list[dict[str, Any]]:
    """Return newest-first summaries for training runs below ``root``."""

    base = Path(root).expanduser().resolve()
    if not base.exists():
        return []
    records = []
    for manifest_path in base.rglob("run_manifest.json"):
        run_dir = manifest_path.parent
        try:
            manifest = read_json_object(manifest_path)
            summary = _optional_json(run_dir / "final" / "summary.json")
            status = _optional_json(run_dir / "status.json")
            records.append(
                {
                    "run_dir": str(run_dir),
                    "created_at": manifest.get("created_at"),
                    "state": status.get("state", "UNKNOWN"),
                    "run_id": manifest.get("run_id"),
                    "experiment_name": manifest.get("experiment_name"),
                    "model_id": summary.get("model_id"),
                    "best_eval_loss": summary.get("best_eval_loss"),
                    "best_checkpoint": summary.get("best_checkpoint"),
                    "completed_at": summary.get("completed_at"),
                }
            )
        except (OSError, Top1DataError, ValueError) as exc:
            records.append(
                {
                    "run_dir": str(run_dir),
                    "created_at": None,
                    "state": "INVALID",
                    "run_id": run_dir.name,
                    "error": str(exc),
                }
            )
    return sorted(
        records,
        key=lambda row: (str(row.get("created_at") or ""), row["run_dir"]),
        reverse=True,
    )


def load_evaluation_run(
    run_dir: str | Path,
    *,
    target_candidate: str | None = None,
    predicted_candidate: str | None = None,
    errors_only: bool = False,
    backend_correct: bool | None = None,
) -> dict[str, Any]:
    """Load one evaluation run and project its cases without changing artifacts."""

    root = _required_directory(run_dir)
    manifest = read_json_object(root / "eval_manifest.json")
    summary = _optional_json(root / "summary.json")
    metrics = _optional_json(root / "metrics.json")
    predictions_path = root / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.is_file() else []
    dataset_rows, dataset_status = _verified_evaluation_dataset(manifest)

    projected = []
    matching_rows = 0
    for prediction in predictions:
        target = prediction.get("target_candidate_name")
        predicted = prediction.get("predicted_candidate_name")
        if target_candidate and target != target_candidate:
            continue
        if predicted_candidate and predicted != predicted_candidate:
            continue
        if errors_only and prediction.get("correct") is not False:
            continue
        backend = _mapping(prediction.get("backend_decision"))
        if (
            backend_correct is not None
            and backend.get("correct") is not backend_correct
        ):
            continue
        matching_rows += 1
        row_index = _integer(prediction.get("row_index"), default=-1)
        source_row = (
            dataset_rows[row_index]
            if 0 <= row_index < len(dataset_rows)
            else None
        )
        diagnostics = _mapping(prediction.get("diagnostics"))
        history = _mapping(prediction.get("history_ablation"))
        messages = _safe_messages(source_row)
        projected.append(
            {
                "row_index": row_index,
                "target": target,
                "predicted": predicted,
                "candidate_correct": prediction.get("correct"),
                "target_backend": backend.get("target_backend_label"),
                "predicted_backend": backend.get("predicted_backend_label"),
                "backend_correct": backend.get("correct"),
                "route_status": backend.get("status"),
                "confidence": prediction.get("candidate_confidence"),
                "message_count": diagnostics.get("original_message_count"),
                "history_dropped": diagnostics.get("history_messages_dropped"),
                "current_user_truncated": diagnostics.get(
                    "current_user_truncated"
                ),
                "history_changed_prediction": history.get("changed_prediction"),
                "last_user": _last_user(messages),
                "dialogue": _format_dialogue(messages),
                "prediction_record": dict(prediction),
            }
        )
    return {
        "run_dir": str(root),
        "manifest": manifest,
        "summary": summary,
        "metrics": metrics,
        "dataset_status": dataset_status,
        "matching_rows": matching_rows,
        "cases": projected,
    }


def evaluation_statistics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Project one metrics payload into dashboard-ready read-only statistics."""

    backend = _mapping(metrics.get("backend"))
    backend_labels = _backend_label_accuracy_rows(backend)
    positive_rows = [
        row for row in backend_labels if row["sample_type"] == "positive"
    ]
    negative_rows = [
        row for row in backend_labels if row["sample_type"] == "negative"
    ]
    available_oos = _mapping(backend.get("available_oos"))
    routing = _mapping(metrics.get("routing_policy"))
    kpis = [
        {
            "name": "后端总准确率",
            "value": backend.get("accuracy"),
            "format": "percent",
            "tone": "primary",
        },
        {
            "name": "正样本准确率",
            "value": _combined_accuracy(positive_rows),
            "format": "percent",
            "tone": "neutral",
        },
        {
            "name": "负样本准确率",
            "value": _combined_accuracy(negative_rows),
            "format": "percent",
            "tone": "neutral",
        },
        {
            "name": "样本数",
            "value": metrics.get("rows"),
            "format": "integer",
            "tone": "neutral",
        },
    ]
    return {
        "kpis": kpis,
        "backend_labels": backend_labels,
        "routing": _named_numeric_rows(
            {
                "原始路由覆盖率": routing.get("raw_route_coverage"),
                "最终路由覆盖率": routing.get("output_route_coverage"),
                "阈值拒绝率": routing.get("threshold_abstention_rate"),
                "拒绝后候选准确率": routing.get("selective_candidate_accuracy"),
                "Available Precision": available_oos.get("available_precision"),
                "Available Recall": available_oos.get("available_recall"),
                "OOS Precision": available_oos.get("oos_precision"),
                "OOS Recall": available_oos.get("oos_recall"),
            }
        ),
        "history_ablation": _named_numeric_rows(
            _mapping(metrics.get("history_ablation"))
        ),
    }


def compare_evaluation_runs(
    first_dir: str | Path,
    second_dir: str | Path,
) -> dict[str, Any]:
    """Compare aggregate results and same-dataset case changes for two runs."""

    first_root = _required_directory(first_dir)
    second_root = _required_directory(second_dir)
    first_manifest = read_json_object(first_root / "eval_manifest.json")
    second_manifest = read_json_object(second_root / "eval_manifest.json")
    first_summary = _optional_json(first_root / "summary.json")
    second_summary = _optional_json(second_root / "summary.json")
    aggregate = []
    for name, path in (
        ("backend_accuracy", ("backend_accuracy",)),
        ("raw_candidate_accuracy", ("raw_candidate_accuracy",)),
        ("expected_calibration_error", ("expected_calibration_error",)),
        (
            "unsafe_oos_accept_rate",
            ("available_oos", "unsafe_oos_accept_rate"),
        ),
        (
            "output_route_coverage",
            ("routing_policy", "output_route_coverage"),
        ),
    ):
        first_value = _nested(first_summary, path)
        second_value = _nested(second_summary, path)
        aggregate.append(
            {
                "metric": name,
                "first": first_value,
                "second": second_value,
                "delta": _numeric_delta(first_value, second_value),
            }
        )

    first_dataset = _mapping(first_manifest.get("dataset"))
    second_dataset = _mapping(second_manifest.get("dataset"))
    same_dataset = bool(
        first_dataset.get("sha256")
        and first_dataset.get("sha256") == second_dataset.get("sha256")
    )
    changes = []
    if same_dataset:
        first_predictions = _predictions_by_index(first_root / "predictions.jsonl")
        second_predictions = _predictions_by_index(second_root / "predictions.jsonl")
        dataset_rows, _ = _verified_evaluation_dataset(first_manifest)
        for row_index in sorted(set(first_predictions) & set(second_predictions)):
            first = first_predictions[row_index]
            second = second_predictions[row_index]
            first_candidate = first.get("predicted_candidate_name")
            second_candidate = second.get("predicted_candidate_name")
            first_backend = _mapping(first.get("backend_decision")).get(
                "predicted_backend_label"
            )
            second_backend = _mapping(second.get("backend_decision")).get(
                "predicted_backend_label"
            )
            if first_candidate == second_candidate and first_backend == second_backend:
                continue
            source_row = (
                dataset_rows[row_index]
                if 0 <= row_index < len(dataset_rows)
                else None
            )
            messages = _safe_messages(source_row)
            changes.append(
                {
                    "row_index": row_index,
                    "target": first.get("target_candidate_name"),
                    "first_candidate": first_candidate,
                    "second_candidate": second_candidate,
                    "first_backend": first_backend,
                    "second_backend": second_backend,
                    "first_correct": first.get("correct"),
                    "second_correct": second.get("correct"),
                    "change": _correctness_change(first, second),
                    "last_user": _last_user(messages),
                }
            )
    return {
        "same_dataset": same_dataset,
        "aggregate": aggregate,
        "case_changes": changes,
        "case_change_count": len(changes),
    }


def load_training_run(run_dir: str | Path) -> dict[str, Any]:
    """Load the existing summaries, curves, and event tail for one training run."""

    root = _required_directory(run_dir)
    curves = _optional_json(root / "final" / "curves.json")
    train_curve = curves.get("train") if isinstance(curves.get("train"), list) else []
    validation_curve = (
        curves.get("validation")
        if isinstance(curves.get("validation"), list)
        else []
    )
    loss_rows = [
        {
            "step": row.get("step"),
            "epoch": row.get("main_epoch", row.get("epoch")),
            "stage": row.get("stage"),
            "metric": "train_loss",
            "value": row.get("loss"),
        }
        for row in train_curve
        if isinstance(row, Mapping) and row.get("loss") is not None
    ]
    loss_rows.extend(
        {
            "step": row.get("step"),
            "epoch": row.get("main_epoch", row.get("epoch")),
            "stage": row.get("stage"),
            "metric": "eval_loss",
            "value": row.get("eval_loss"),
        }
        for row in validation_curve
        if isinstance(row, Mapping) and row.get("eval_loss") is not None
    )
    optimization_rows = []
    for row in train_curve:
        if not isinstance(row, Mapping):
            continue
        for metric in ("learning_rate", "grad_norm"):
            if row.get(metric) is not None:
                optimization_rows.append(
                    {
                        "step": row.get("step"),
                        "stage": row.get("stage"),
                        "metric": metric,
                        "value": row.get(metric),
                    }
                )
    return {
        "run_dir": str(root),
        "manifest": read_json_object(root / "run_manifest.json"),
        "status": _optional_json(root / "status.json"),
        "summary": _optional_json(root / "final" / "summary.json"),
        "curves": curves,
        "loss_rows": loss_rows,
        "optimization_rows": optimization_rows,
        "event_tail": _jsonl_tail(root / "logs" / "events.jsonl", limit=100),
    }


def _required_directory(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise Top1DataError(f"run directory does not exist: {path}")
    return path


def _optional_json(path: Path) -> dict[str, Any]:
    return read_json_object(path) if path.is_file() else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _short_identifier(value: Any, length: int = 12) -> str | None:
    if not isinstance(value, str):
        return None
    return value if len(value) <= length else value[:length]


def _path_name(value: Any) -> str | None:
    return Path(value).name if isinstance(value, str) and value else None


def _run_reference(model_id: Any, evaluation_id: Any) -> str:
    model = str(model_id or "unknown-model")
    evaluation = str(evaluation_id or "unknown-evaluation")
    return f"{model}/{evaluation}"


def _backend_label_accuracy_rows(
    backend: Mapping[str, Any],
) -> list[dict[str, Any]]:
    policy = _mapping(backend.get("decision_policy"))
    fallback = policy.get("fallback_backend_label")
    per_label = _mapping(backend.get("per_label"))
    configured_labels = policy.get("backend_labels")
    labels = (
        [str(label) for label in configured_labels]
        if isinstance(configured_labels, list)
        else [str(label) for label in per_label]
    )
    rows = []
    for label in labels:
        item = _mapping(per_label.get(label))
        support = item.get("support")
        correct = item.get("correct")
        normalized_support = (
            int(support)
            if isinstance(support, int) and not isinstance(support, bool)
            else 0
        )
        normalized_correct = (
            int(correct)
            if isinstance(correct, int) and not isinstance(correct, bool)
            else 0
        )
        rows.append(
            {
                "label": label,
                "sample_type": "negative" if label == fallback else "positive",
                "accuracy": (
                    normalized_correct / normalized_support
                    if normalized_support
                    else None
                ),
                "correct": normalized_correct,
                "support": normalized_support,
            }
        )
    return rows


def _combined_accuracy(rows: Sequence[Mapping[str, Any]]) -> float | None:
    support = sum(int(row.get("support") or 0) for row in rows)
    correct = sum(int(row.get("correct") or 0) for row in rows)
    return correct / support if support else None


def _named_numeric_rows(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        rows.append({"metric": str(name), "value": value})
    return rows


def _integer(value: Any, *, default: int) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def _verified_evaluation_dataset(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = _mapping(manifest.get("dataset"))
    raw_path = metadata.get("path")
    expected_sha256 = metadata.get("sha256")
    status = {
        "path": raw_path,
        "expected_sha256": expected_sha256,
        "state": "missing_metadata",
    }
    if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
        return [], status
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return [], {**status, "state": "missing"}
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        return [], {**status, "state": "hash_mismatch", "actual_sha256": actual_sha256}
    return read_jsonl(path), {**status, "state": "verified", "actual_sha256": actual_sha256}


def _safe_messages(row: Mapping[str, Any] | None) -> list[dict[str, str]]:
    if row is None:
        return []
    try:
        return messages_from_row(row)
    except Top1DataError:
        return []


def _last_user(messages: Sequence[Mapping[str, str]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return None


def _format_dialogue(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(
        f'{message.get("role", "unknown")}: {message.get("content", "")}'
        for message in messages
    )


def _nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for component in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(component)
    return value


def _numeric_delta(first: Any, second: Any) -> float | None:
    if isinstance(first, bool) or isinstance(second, bool):
        return None
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return None
    return float(second) - float(first)


def _predictions_by_index(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    result = {}
    for row in read_jsonl(path):
        row_index = _integer(row.get("row_index"), default=-1)
        if row_index >= 0:
            result[row_index] = row
    return result


def _correctness_change(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> str:
    before = first.get("correct")
    after = second.get("correct")
    if before is False and after is True:
        return "improved"
    if before is True and after is False:
        return "regressed"
    return "changed"


def _jsonl_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise Top1DataError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise Top1DataError(f"invalid JSONL file: {path}") from exc
    return list(rows)
