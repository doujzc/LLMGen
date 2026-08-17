"""Constrained-generation records and aggregate Top1 evaluation metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .diagnostics import numeric_summary
from .top1 import INFERENCE_DECISION_RULE, ROUTING_MODE, Top1DataError


@dataclass(frozen=True)
class BackendDecisionPolicy:
    """Explicit mapping from trained candidates to deployed backend labels."""

    candidate_to_backend: Mapping[str, str]
    backend_labels: tuple[str, ...]
    fallback_backend_label: str

    @property
    def available_backend_labels(self) -> tuple[str, ...]:
        """Return backend labels that represent executable routes."""

        return tuple(
            label
            for label in self.backend_labels
            if label != self.fallback_backend_label
        )

    def payload(self) -> dict[str, Any]:
        """Return the normalized, portable routing-policy payload."""

        return {
            "schema_version": 2,
            "routing_mode": ROUTING_MODE,
            "decision_rule": INFERENCE_DECISION_RULE,
            "backend_labels": list(self.backend_labels),
            "fallback_backend_label": self.fallback_backend_label,
            "candidate_to_backend": dict(self.candidate_to_backend),
        }


def load_backend_decision_policy(
    path: str | Path,
    candidate_names: Sequence[str],
) -> BackendDecisionPolicy:
    """Load and validate one routing policy against model candidates."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Top1DataError(f"invalid backend decision policy: {source}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError("backend decision policy must be a JSON object")
    if payload.get("schema_version") != 2:
        raise Top1DataError("backend decision policy schema_version must be 2")
    if payload.get("routing_mode") != ROUTING_MODE:
        raise Top1DataError("backend decision policy has an incompatible routing mode")
    if payload.get("decision_rule") != INFERENCE_DECISION_RULE:
        raise Top1DataError("backend decision policy has an incompatible decision rule")

    backend_values = payload.get("backend_labels")
    if not isinstance(backend_values, list) or not backend_values:
        raise Top1DataError("backend_labels must be a non-empty list")
    backend_labels: list[str] = []
    for index, value in enumerate(backend_values):
        if not isinstance(value, str) or not value.strip():
            raise Top1DataError(f"backend_labels[{index}] must be a non-empty string")
        label = value.strip()
        if label in backend_labels:
            raise Top1DataError(f"duplicate backend label: {label!r}")
        backend_labels.append(label)

    fallback = payload.get("fallback_backend_label")
    if not isinstance(fallback, str) or fallback not in backend_labels:
        raise Top1DataError("fallback_backend_label must exist in backend_labels")
    if len(backend_labels) < 2:
        raise Top1DataError("backend decision policy requires an available backend label")

    raw_mapping = payload.get("candidate_to_backend")
    if not isinstance(raw_mapping, dict):
        raise Top1DataError("candidate_to_backend must be an object")
    expected_candidates = tuple(candidate_names)
    if set(raw_mapping) != set(expected_candidates):
        raise Top1DataError(
            "candidate_to_backend must cover exactly the candidate registry"
        )
    candidate_to_backend: dict[str, str] = {}
    for candidate in expected_candidates:
        backend = raw_mapping[candidate]
        if not isinstance(backend, str) or backend not in backend_labels:
            raise Top1DataError(
                f"candidate {candidate!r} maps to an unknown backend label"
            )
        candidate_to_backend[candidate] = backend
    if set(candidate_to_backend.values()) != set(backend_labels):
        raise Top1DataError("every backend label must receive at least one candidate")

    return BackendDecisionPolicy(
        candidate_to_backend=candidate_to_backend,
        backend_labels=tuple(backend_labels),
        fallback_backend_label=fallback,
    )


def candidate_confidence(path_logprob: float | None) -> float | None:
    """Convert one normalized constrained-path log probability to probability."""

    if path_logprob is None:
        return None
    value = float(path_logprob)
    if math.isnan(value):
        raise Top1DataError("candidate generation produced a NaN path score")
    if value == -math.inf:
        return 0.0
    if value == math.inf:
        raise Top1DataError("candidate generation produced an infinite path score")
    return max(0.0, min(1.0, math.exp(min(0.0, value))))


def prediction_from_generation(
    *,
    row_index: int,
    candidate_names: Sequence[str],
    generated_candidate_name: str,
    path_logprob: float | None,
    path_tokens: int,
    target_candidate_name: str | None,
    diagnostics: Mapping[str, Any],
    decision_policy: BackendDecisionPolicy,
    route_threshold: float | None,
    decoding: Mapping[str, Any],
    history_ablation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one privacy-safe record from constrained generation output."""

    candidates = tuple(candidate_names)
    if tuple(decision_policy.candidate_to_backend) != candidates:
        raise Top1DataError("decision policy candidate order differs from the registry")
    if generated_candidate_name not in decision_policy.candidate_to_backend:
        raise Top1DataError("generated candidate does not exist in the registry")
    if (
        target_candidate_name is not None
        and target_candidate_name not in decision_policy.candidate_to_backend
    ):
        raise Top1DataError("target candidate does not exist in the registry")
    confidence = candidate_confidence(path_logprob)
    backend_decision = _route_decision(
        predicted_candidate_name=generated_candidate_name,
        target_candidate_name=target_candidate_name,
        confidence=confidence,
        route_threshold=route_threshold,
        policy=decision_policy,
    )
    record: dict[str, Any] = {
        "schema_version": 2,
        "row_index": row_index,
        "target_candidate_name": target_candidate_name,
        "predicted_candidate_name": generated_candidate_name,
        "correct": (
            generated_candidate_name == target_candidate_name
            if target_candidate_name is not None
            else None
        ),
        "score_mode": "constrained_generate_path_logprob",
        "path_logprob": path_logprob,
        "path_tokens": path_tokens,
        "candidate_confidence": confidence,
        "diagnostics": dict(diagnostics),
        "decoding": dict(decoding),
        "backend_decision": backend_decision,
    }
    if history_ablation is not None:
        ablated_candidate = str(history_ablation["candidate_name"])
        ablated_confidence = candidate_confidence(history_ablation.get("path_logprob"))
        ablated_backend = _route_decision(
            predicted_candidate_name=ablated_candidate,
            target_candidate_name=target_candidate_name,
            confidence=ablated_confidence,
            route_threshold=route_threshold,
            policy=decision_policy,
        )
        record["history_ablation"] = {
            "predicted_candidate_name": ablated_candidate,
            "correct": (
                ablated_candidate == target_candidate_name
                if target_candidate_name is not None
                else None
            ),
            "changed_prediction": ablated_candidate != generated_candidate_name,
            "path_logprob": history_ablation.get("path_logprob"),
            "path_tokens": int(history_ablation["path_tokens"]),
            "candidate_confidence": ablated_confidence,
            "backend_decision": {
                **ablated_backend,
                "changed_prediction": (
                    ablated_backend["predicted_backend_label"]
                    != backend_decision["predicted_backend_label"]
                ),
            },
        }
    return record


def _route_decision(
    *,
    predicted_candidate_name: str,
    target_candidate_name: str | None,
    confidence: float | None,
    route_threshold: float | None,
    policy: BackendDecisionPolicy,
) -> dict[str, Any]:
    if route_threshold is not None and (
        isinstance(route_threshold, bool)
        or not isinstance(route_threshold, (int, float))
        or not math.isfinite(float(route_threshold))
        or not 0.0 <= float(route_threshold) <= 1.0
    ):
        raise Top1DataError("route_threshold must be between 0 and 1")
    raw_backend = policy.candidate_to_backend[predicted_candidate_name]
    raw_should_route = raw_backend != policy.fallback_backend_label
    if raw_should_route and route_threshold is not None and confidence is None:
        raise Top1DataError(
            "route threshold requires normalized generation transition scores"
        )
    threshold_triggered = bool(
        raw_should_route
        and route_threshold is not None
        and confidence is not None
        and confidence < route_threshold
    )
    predicted_backend = (
        policy.fallback_backend_label if threshold_triggered else raw_backend
    )
    target_backend = (
        policy.candidate_to_backend[target_candidate_name]
        if target_candidate_name is not None
        else None
    )
    return {
        "target_backend_label": target_backend,
        "raw_predicted_backend_label": raw_backend,
        "predicted_backend_label": predicted_backend,
        "raw_correct": raw_backend == target_backend if target_backend else None,
        "correct": predicted_backend == target_backend if target_backend else None,
        "raw_should_route": raw_should_route,
        "should_route": predicted_backend != policy.fallback_backend_label,
        "candidate_confidence": confidence,
        "route_threshold": route_threshold,
        "threshold_triggered": threshold_triggered,
        "threshold_margin": (
            confidence - route_threshold
            if raw_should_route
            and confidence is not None
            and route_threshold is not None
            else None
        ),
        "status": (
            "abstained"
            if threshold_triggered
            else "routed"
            if raw_should_route
            else "no_route"
        ),
    }


def aggregate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
    decision_policy: BackendDecisionPolicy,
) -> dict[str, Any]:
    """Aggregate candidate, route-threshold, and backend diagnostics."""

    labeled = [row for row in predictions if row.get("target_candidate_name") is not None]
    confusion = {
        target: {predicted: 0 for predicted in candidate_names}
        for target in candidate_names
    }
    target_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    for row in labeled:
        target = str(row["target_candidate_name"])
        predicted = str(row["predicted_candidate_name"])
        if target not in confusion or predicted not in confusion[target]:
            raise Top1DataError("prediction record contains an unknown candidate")
        confusion[target][predicted] += 1
        target_counts[target] += 1
        predicted_counts[predicted] += 1
        correct_counts[target] += target == predicted

    per_candidate = {}
    recalls = []
    for name in candidate_names:
        support = target_counts[name]
        predicted = predicted_counts[name]
        correct = correct_counts[name]
        recall = correct / support if support else None
        precision = correct / predicted if predicted else None
        if recall is not None:
            recalls.append(recall)
        per_candidate[name] = {
            "support": support,
            "predicted": predicted,
            "correct": correct,
            "recall": recall,
            "precision": precision,
        }

    history_rows = [row for row in labeled if row.get("history_ablation") is not None]
    single_turn = [
        row
        for row in labeled
        if int(row.get("diagnostics", {}).get("original_message_count", 1)) == 1
    ]
    multi_turn = [
        row
        for row in labeled
        if int(row.get("diagnostics", {}).get("original_message_count", 1)) > 1
    ]
    history_truncated = [
        row
        for row in labeled
        if int(row.get("diagnostics", {}).get("history_messages_dropped", 0)) > 0
    ]
    current_truncated = [
        row
        for row in labeled
        if bool(row.get("diagnostics", {}).get("current_user_truncated", False))
    ]
    untouched = [
        row
        for row in labeled
        if int(row.get("diagnostics", {}).get("history_messages_dropped", 0)) == 0
        and not bool(row.get("diagnostics", {}).get("current_user_truncated", False))
    ]
    confidence_rows = [
        row for row in labeled if row.get("candidate_confidence") is not None
    ]
    return {
        "schema_version": 2,
        "rows": len(predictions),
        "labeled_rows": len(labeled),
        "top1_accuracy": _accuracy([bool(row["correct"]) for row in labeled]),
        "macro_recall_observed_candidates": mean(recalls) if recalls else None,
        "per_candidate": per_candidate,
        "confusion_matrix": confusion,
        "candidate_confidence": numeric_summary(
            float(row["candidate_confidence"])
            for row in predictions
            if row.get("candidate_confidence") is not None
        ),
        "calibration": _calibration(confidence_rows),
        "conversation_strata": {
            "single_turn": _stratum(single_turn),
            "multi_turn": _stratum(multi_turn),
        },
        "prompt_fitting_strata": {
            "history_truncated": _stratum(history_truncated),
            "current_user_truncated": _stratum(current_truncated),
            "untouched": _stratum(untouched),
        },
        "history_ablation": _history_ablation(history_rows),
        "routing_policy": _routing_policy_metrics(predictions),
        "backend": _aggregate_backend_predictions(predictions, decision_policy),
        "hard_examples": _hard_examples(predictions),
    }


def _routing_policy_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in predictions if row.get("target_candidate_name") is not None]
    threshold_values = {
        row["backend_decision"].get("route_threshold") for row in predictions
    }
    triggered = sum(
        bool(row["backend_decision"]["threshold_triggered"]) for row in predictions
    )
    raw_routed = sum(
        bool(row["backend_decision"]["raw_should_route"]) for row in predictions
    )
    routed = sum(bool(row["backend_decision"]["should_route"]) for row in predictions)
    accepted_labeled = [
        row
        for row in labeled
        if not bool(row["backend_decision"]["threshold_triggered"])
    ]
    return {
        "route_threshold": (
            next(iter(threshold_values)) if len(threshold_values) == 1 else None
        ),
        "raw_route_coverage": raw_routed / len(predictions) if predictions else None,
        "output_route_coverage": routed / len(predictions) if predictions else None,
        "threshold_triggered_examples": triggered,
        "threshold_abstention_rate": (
            triggered / len(predictions) if predictions else None
        ),
        "selective_candidate_accuracy": _accuracy(
            [bool(row["correct"]) for row in accepted_labeled]
        ),
    }


def _aggregate_backend_predictions(
    predictions: Sequence[Mapping[str, Any]],
    policy: BackendDecisionPolicy,
) -> dict[str, Any]:
    labeled = [
        row
        for row in predictions
        if row.get("backend_decision", {}).get("target_backend_label") is not None
    ]
    confusion = {
        target: {predicted: 0 for predicted in policy.backend_labels}
        for target in policy.backend_labels
    }
    target_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    binary = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in labeled:
        decision = row["backend_decision"]
        target = str(decision["target_backend_label"])
        predicted = str(decision["predicted_backend_label"])
        if target not in confusion or predicted not in confusion[target]:
            raise Top1DataError("backend decision contains an unknown backend label")
        confusion[target][predicted] += 1
        target_counts[target] += 1
        predicted_counts[predicted] += 1
        correct_counts[target] += target == predicted
        target_available = target != policy.fallback_backend_label
        predicted_available = predicted != policy.fallback_backend_label
        key = (
            "tp"
            if target_available and predicted_available
            else "fp"
            if not target_available and predicted_available
            else "fn"
            if target_available
            else "tn"
        )
        binary[key] += 1

    per_label = {}
    recalls = []
    for label in policy.backend_labels:
        support = target_counts[label]
        predicted = predicted_counts[label]
        correct = correct_counts[label]
        recall = correct / support if support else None
        precision = correct / predicted if predicted else None
        if recall is not None:
            recalls.append(recall)
        per_label[label] = {
            "support": support,
            "predicted": predicted,
            "correct": correct,
            "recall": recall,
            "precision": precision,
        }

    tp, fp, fn, tn = (binary[key] for key in ("tp", "fp", "fn", "tn"))
    return {
        "decision_policy": policy.payload(),
        "labels": list(policy.backend_labels),
        "labeled_rows": len(labeled),
        "accuracy": _accuracy(
            [bool(row["backend_decision"]["correct"]) for row in labeled]
        ),
        "macro_recall_observed_labels": mean(recalls) if recalls else None,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "available_oos": {
            **binary,
            "accuracy": (tp + tn) / len(labeled) if labeled else None,
            "available_precision": tp / (tp + fp) if tp + fp else None,
            "available_recall": tp / (tp + fn) if tp + fn else None,
            "oos_precision": tn / (tn + fn) if tn + fn else None,
            "oos_recall": tn / (tn + fp) if tn + fp else None,
            "unsafe_oos_accept_rate": fp / (fp + tn) if fp + tn else None,
            "available_false_reject_rate": fn / (tp + fn) if tp + fn else None,
        },
    }


def _accuracy(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "accuracy": _accuracy([bool(row["correct"]) for row in rows]),
        "backend_accuracy": _accuracy(
            [bool(row["backend_decision"]["correct"]) for row in rows]
        ),
    }


def _history_ablation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "full_history_accuracy": _accuracy([bool(row["correct"]) for row in rows]),
        "latest_user_only_accuracy": _accuracy(
            [bool(row["history_ablation"]["correct"]) for row in rows]
        ),
        "prediction_changes": sum(
            bool(row["history_ablation"]["changed_prediction"]) for row in rows
        ),
        "history_helped": sum(
            bool(row["correct"]) and not bool(row["history_ablation"]["correct"])
            for row in rows
        ),
        "history_hurt": sum(
            not bool(row["correct"]) and bool(row["history_ablation"]["correct"])
            for row in rows
        ),
        "backend_full_history_accuracy": _accuracy(
            [bool(row["backend_decision"]["correct"]) for row in rows]
        ),
        "backend_latest_user_only_accuracy": _accuracy(
            [
                bool(row["history_ablation"]["backend_decision"]["correct"])
                for row in rows
            ]
        ),
        "backend_prediction_changes": sum(
            bool(row["history_ablation"]["backend_decision"]["changed_prediction"])
            for row in rows
        ),
    }


def _calibration(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"bins": [], "expected_calibration_error": None}
    buckets = []
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            row
            for row in rows
            if lower <= float(row["candidate_confidence"]) <= upper
            and (index == bins - 1 or float(row["candidate_confidence"]) < upper)
        ]
        if not members:
            buckets.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "rows": 0,
                    "accuracy": None,
                    "confidence": None,
                }
            )
            continue
        accuracy = mean(bool(row["correct"]) for row in members)
        confidence = mean(float(row["candidate_confidence"]) for row in members)
        error += (len(members) / len(rows)) * abs(accuracy - confidence)
        buckets.append(
            {
                "lower": lower,
                "upper": upper,
                "rows": len(members),
                "accuracy": accuracy,
                "confidence": confidence,
            }
        )
    return {"bins": buckets, "expected_calibration_error": error}


def _hard_examples(
    rows: Sequence[Mapping[str, Any]],
    limit: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    def compact(row: Mapping[str, Any]) -> dict[str, Any]:
        backend = row["backend_decision"]
        return {
            "row_index": row["row_index"],
            "target_candidate_name": row.get("target_candidate_name"),
            "predicted_candidate_name": row["predicted_candidate_name"],
            "correct": row.get("correct"),
            "candidate_confidence": row.get("candidate_confidence"),
            "target_backend_label": backend["target_backend_label"],
            "predicted_backend_label": backend["predicted_backend_label"],
            "backend_correct": backend["correct"],
            "route_threshold": backend["route_threshold"],
            "threshold_triggered": backend["threshold_triggered"],
        }

    confidence_rows = [row for row in rows if row.get("candidate_confidence") is not None]
    low_confidence_routes = sorted(
        (
            row
            for row in confidence_rows
            if bool(row["backend_decision"]["raw_should_route"])
        ),
        key=lambda row: float(row["candidate_confidence"]),
    )[:limit]
    high_confidence_errors = sorted(
        (row for row in confidence_rows if row.get("correct") is False),
        key=lambda row: float(row["candidate_confidence"]),
        reverse=True,
    )[:limit]
    closest_to_threshold = sorted(
        (
            row
            for row in confidence_rows
            if row["backend_decision"].get("threshold_margin") is not None
        ),
        key=lambda row: abs(float(row["backend_decision"]["threshold_margin"])),
    )[:limit]
    return {
        "lowest_confidence_routes": [compact(row) for row in low_confidence_routes],
        "highest_confidence_errors": [compact(row) for row in high_confidence_errors],
        "closest_to_route_threshold": [compact(row) for row in closest_to_threshold],
    }
