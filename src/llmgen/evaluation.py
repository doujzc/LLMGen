"""Candidate-path scoring records and aggregate Top1 evaluation metrics."""

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
    available_threshold: float
    temperature: float

    @property
    def available_backend_labels(self) -> tuple[str, ...]:
        """Return backend labels that represent supported operations."""

        return tuple(
            label
            for label in self.backend_labels
            if label != self.fallback_backend_label
        )

    def payload(self) -> dict[str, Any]:
        """Return the normalized, portable decision-policy payload."""

        return {
            "schema_version": 1,
            "routing_mode": ROUTING_MODE,
            "decision_rule": INFERENCE_DECISION_RULE,
            "backend_labels": list(self.backend_labels),
            "fallback_backend_label": self.fallback_backend_label,
            "candidate_to_backend": dict(self.candidate_to_backend),
            "available_threshold": self.available_threshold,
            "temperature": self.temperature,
        }


def load_backend_decision_policy(
    path: str | Path,
    candidate_names: Sequence[str],
    *,
    available_threshold: float | None = None,
) -> BackendDecisionPolicy:
    """Load and validate one backend decision policy against model candidates."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Top1DataError(f"invalid backend decision policy: {source}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError("backend decision policy must be a JSON object")
    if payload.get("schema_version") != 1:
        raise Top1DataError("backend decision policy schema_version must be 1")
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
    used_backend_labels = set(candidate_to_backend.values())
    if used_backend_labels != set(backend_labels):
        raise Top1DataError("every backend label must receive at least one candidate")

    threshold = (
        available_threshold
        if available_threshold is not None
        else payload.get("available_threshold")
    )
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise Top1DataError("available_threshold must be between 0 and 1")
    temperature = payload.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise Top1DataError("temperature must be a finite positive number")

    return BackendDecisionPolicy(
        candidate_to_backend=candidate_to_backend,
        backend_labels=tuple(backend_labels),
        fallback_backend_label=fallback,
        available_threshold=float(threshold),
        temperature=float(temperature),
    )


def prediction_from_scores(
    *,
    row_index: int,
    candidate_names: Sequence[str],
    scores: Mapping[str, Mapping[str, float | int]],
    target_candidate_name: str | None,
    diagnostics: Mapping[str, Any],
    decision_policy: BackendDecisionPolicy,
    history_ablation_scores: Mapping[str, Mapping[str, float | int]] | None = None,
) -> dict[str, Any]:
    """Create one privacy-safe prediction record from candidate path scores."""

    if set(scores) != set(candidate_names):
        raise Top1DataError("candidate scores do not match the candidate registry")
    if tuple(decision_policy.candidate_to_backend) != tuple(candidate_names):
        raise Top1DataError("decision policy candidate order differs from the registry")
    if target_candidate_name is not None and target_candidate_name not in scores:
        raise Top1DataError("target candidate does not exist in candidate scores")

    sum_prediction = max(candidate_names, key=lambda name: float(scores[name]["sum_logprob"]))
    mean_prediction = max(
        candidate_names,
        key=lambda name: float(scores[name]["mean_logprob"]),
    )
    prediction = sum_prediction
    selected = [
        float(scores[name]["sum_logprob"]) / decision_policy.temperature
        for name in candidate_names
    ]
    probabilities = _softmax(selected)
    candidate_probabilities = dict(zip(candidate_names, probabilities, strict=True))
    backend_decision = _backend_decision(
        candidate_probabilities,
        decision_policy,
        target_candidate_name=target_candidate_name,
    )
    entropy = -sum(
        probability * math.log(max(probability, 1e-45))
        for probability in probabilities
    )
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    selected_scores = sorted(selected, reverse=True)
    margin = selected_scores[0] - selected_scores[1] if len(selected_scores) > 1 else None

    record: dict[str, Any] = {
        "schema_version": 1,
        "row_index": row_index,
        "target_candidate_name": target_candidate_name,
        "predicted_candidate_name": prediction,
        "correct": prediction == target_candidate_name if target_candidate_name else None,
        "sum_logprob_prediction": sum_prediction,
        "mean_logprob_prediction": mean_prediction,
        "score_mode": "sum_logprob",
        "confidence": max(probabilities),
        "normalized_entropy": normalized_entropy,
        "margin": margin,
        "diagnostics": dict(diagnostics),
        "candidate_scores": {
            name: {
                "sum_logprob": float(scores[name]["sum_logprob"]),
                "mean_logprob": float(scores[name]["mean_logprob"]),
                "path_tokens": int(scores[name]["path_tokens"]),
                "probability": candidate_probabilities[name],
            }
            for name in candidate_names
        },
        "backend_decision": backend_decision,
    }
    if history_ablation_scores is not None:
        if set(history_ablation_scores) != set(candidate_names):
            raise Top1DataError(
                "history-ablation scores do not match the candidate registry"
            )
        ablated_prediction = max(
            candidate_names,
            key=lambda name: float(history_ablation_scores[name]["sum_logprob"]),
        )
        ablated_probabilities = _softmax(
            [
                float(history_ablation_scores[name]["sum_logprob"])
                / decision_policy.temperature
                for name in candidate_names
            ]
        )
        ablated_backend = _backend_decision(
            dict(zip(candidate_names, ablated_probabilities, strict=True)),
            decision_policy,
            target_candidate_name=target_candidate_name,
        )
        record["history_ablation"] = {
            "predicted_candidate_name": ablated_prediction,
            "correct": (
                ablated_prediction == target_candidate_name
                if target_candidate_name
                else None
            ),
            "changed_prediction": ablated_prediction != prediction,
            "backend_decision": {
                **ablated_backend,
                "changed_prediction": (
                    ablated_backend["predicted_backend_label"]
                    != backend_decision["predicted_backend_label"]
                ),
            },
        }
    return record


def _backend_decision(
    candidate_probabilities: Mapping[str, float],
    policy: BackendDecisionPolicy,
    *,
    target_candidate_name: str | None,
) -> dict[str, Any]:
    backend_probabilities = {label: 0.0 for label in policy.backend_labels}
    for candidate, probability in candidate_probabilities.items():
        backend = policy.candidate_to_backend[candidate]
        backend_probabilities[backend] += float(probability)
    fallback_probability = backend_probabilities[policy.fallback_backend_label]
    available_probability = max(0.0, min(1.0, 1.0 - fallback_probability))
    if available_probability >= policy.available_threshold:
        predicted_backend = max(
            policy.available_backend_labels,
            key=lambda label: backend_probabilities[label],
        )
    else:
        predicted_backend = policy.fallback_backend_label
    target_backend = (
        policy.candidate_to_backend[target_candidate_name]
        if target_candidate_name is not None
        else None
    )
    return {
        "target_backend_label": target_backend,
        "predicted_backend_label": predicted_backend,
        "correct": (
            predicted_backend == target_backend if target_backend is not None else None
        ),
        "confidence": backend_probabilities[predicted_backend],
        "available_probability": available_probability,
        "oos_probability": fallback_probability,
        "available_threshold": policy.available_threshold,
        "threshold_margin": available_probability - policy.available_threshold,
        "backend_probabilities": backend_probabilities,
    }


def aggregate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
    decision_policy: BackendDecisionPolicy,
) -> dict[str, Any]:
    """Aggregate accuracy, calibration, confusion, and limitation diagnostics."""

    labeled = [row for row in predictions if row.get("target_candidate_name") is not None]
    confusion = {
        target: {predicted: 0 for predicted in candidate_names}
        for target in candidate_names
    }
    predicted_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    target_nll: dict[str, list[float]] = {name: [] for name in candidate_names}
    correct_values = []
    for row in labeled:
        target = str(row["target_candidate_name"])
        predicted = str(row["predicted_candidate_name"])
        if target not in confusion or predicted not in confusion[target]:
            raise Top1DataError("prediction record contains an unknown candidate")
        confusion[target][predicted] += 1
        target_counts[target] += 1
        predicted_counts[predicted] += 1
        is_correct = target == predicted
        correct_values.append(is_correct)
        correct_counts[target] += is_correct
        target_score = row["candidate_scores"][target]
        target_nll[target].append(-float(target_score["mean_logprob"]))

    per_candidate = {}
    observed_recalls = []
    for name in candidate_names:
        support = target_counts[name]
        predicted = predicted_counts[name]
        correct = correct_counts[name]
        recall = correct / support if support else None
        precision = correct / predicted if predicted else None
        if recall is not None:
            observed_recalls.append(recall)
        per_candidate[name] = {
            "support": support,
            "predicted": predicted,
            "correct": correct,
            "recall": recall,
            "precision": precision,
            "path_tokens": (
                int(predictions[0]["candidate_scores"][name]["path_tokens"])
                if predictions
                else None
            ),
            "mean_target_token_nll": (
                mean(target_nll[name]) if target_nll[name] else None
            ),
        }

    sum_correct = [
        row["sum_logprob_prediction"] == row["target_candidate_name"] for row in labeled
    ]
    mean_correct = [
        row["mean_logprob_prediction"] == row["target_candidate_name"] for row in labeled
    ]
    score_disagreements = sum(
        row["sum_logprob_prediction"] != row["mean_logprob_prediction"]
        for row in predictions
    )
    history_rows = [row for row in labeled if row.get("history_ablation") is not None]
    history_help = sum(
        bool(row["correct"]) and not bool(row["history_ablation"]["correct"])
        for row in history_rows
    )
    history_hurt = sum(
        not bool(row["correct"]) and bool(row["history_ablation"]["correct"])
        for row in history_rows
    )
    backend_history_help = sum(
        bool(row["backend_decision"]["correct"])
        and not bool(row["history_ablation"]["backend_decision"]["correct"])
        for row in history_rows
    )
    backend_history_hurt = sum(
        not bool(row["backend_decision"]["correct"])
        and bool(row["history_ablation"]["backend_decision"]["correct"])
        for row in history_rows
    )
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
        and not bool(
            row.get("diagnostics", {}).get("current_user_truncated", False)
        )
    ]

    return {
        "schema_version": 1,
        "rows": len(predictions),
        "labeled_rows": len(labeled),
        "top1_accuracy": _accuracy(correct_values),
        "macro_recall_observed_candidates": (
            mean(observed_recalls) if observed_recalls else None
        ),
        "per_candidate": per_candidate,
        "confusion_matrix": confusion,
        "confidence": numeric_summary(float(row["confidence"]) for row in predictions),
        "normalized_entropy": numeric_summary(
            float(row["normalized_entropy"]) for row in predictions
        ),
        "margin": numeric_summary(
            float(row["margin"])
            for row in predictions
            if row.get("margin") is not None
        ),
        "calibration": _calibration(labeled),
        "score_mode_comparison": {
            "sum_logprob_accuracy": _accuracy(sum_correct),
            "mean_logprob_accuracy": _accuracy(mean_correct),
            "sum_logprob_prediction_counts": dict(
                Counter(str(row["sum_logprob_prediction"]) for row in predictions)
            ),
            "mean_logprob_prediction_counts": dict(
                Counter(str(row["mean_logprob_prediction"]) for row in predictions)
            ),
            "prediction_disagreements": score_disagreements,
            "prediction_disagreement_rate": (
                score_disagreements / len(predictions) if predictions else None
            ),
        },
        "conversation_strata": {
            "single_turn": _stratum(single_turn),
            "multi_turn": _stratum(multi_turn),
        },
        "prompt_fitting_strata": {
            "history_truncated": _stratum(history_truncated),
            "current_user_truncated": _stratum(current_truncated),
            "untouched": _stratum(untouched),
        },
        "history_ablation": {
            "rows": len(history_rows),
            "full_history_accuracy": _accuracy(
                [bool(row["correct"]) for row in history_rows]
            ),
            "latest_user_only_accuracy": _accuracy(
                [bool(row["history_ablation"]["correct"]) for row in history_rows]
            ),
            "prediction_changes": sum(
                bool(row["history_ablation"]["changed_prediction"])
                for row in history_rows
            ),
            "history_helped": history_help,
            "history_hurt": history_hurt,
            "net_help": history_help - history_hurt,
            "backend_full_history_accuracy": _accuracy(
                [
                    bool(row["backend_decision"]["correct"])
                    for row in history_rows
                ]
            ),
            "backend_latest_user_only_accuracy": _accuracy(
                [
                    bool(row["history_ablation"]["backend_decision"]["correct"])
                    for row in history_rows
                ]
            ),
            "backend_prediction_changes": sum(
                bool(
                    row["history_ablation"]["backend_decision"][
                        "changed_prediction"
                    ]
                )
                for row in history_rows
            ),
            "backend_history_helped": backend_history_help,
            "backend_history_hurt": backend_history_hurt,
            "backend_net_help": backend_history_help - backend_history_hurt,
        },
        "backend": _aggregate_backend_predictions(predictions, decision_policy),
        "hard_examples": _hard_examples(predictions),
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
    calibration_rows = []
    brier_values = []
    for row in labeled:
        decision = row["backend_decision"]
        target = str(decision["target_backend_label"])
        predicted = str(decision["predicted_backend_label"])
        if target not in confusion or predicted not in confusion[target]:
            raise Top1DataError("backend decision contains an unknown backend label")
        confusion[target][predicted] += 1
        target_counts[target] += 1
        predicted_counts[predicted] += 1
        is_correct = target == predicted
        correct_counts[target] += is_correct
        calibration_rows.append(
            {
                "confidence": float(decision["confidence"]),
                "correct": is_correct,
            }
        )
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
        brier_values.append(
            (
                float(decision["available_probability"])
                - float(target_available)
            )
            ** 2
        )

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
        "confidence": numeric_summary(
            float(row["backend_decision"]["confidence"])
            for row in predictions
        ),
        "calibration": _calibration(calibration_rows),
        "available_oos": {
            "available_threshold": policy.available_threshold,
            "available_probability": numeric_summary(
                float(row["backend_decision"]["available_probability"])
                for row in predictions
            ),
            "brier_score": mean(brier_values) if brier_values else None,
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


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _accuracy(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "accuracy": _accuracy([bool(row["correct"]) for row in rows]),
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
            if lower <= float(row["confidence"]) <= upper
            and (index == bins - 1 or float(row["confidence"]) < upper)
        ]
        if not members:
            buckets.append(
                {"lower": lower, "upper": upper, "rows": 0, "accuracy": None, "confidence": None}
            )
            continue
        accuracy = mean(bool(row["correct"]) for row in members)
        confidence = mean(float(row["confidence"]) for row in members)
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
            "margin": row.get("margin"),
            "confidence": row["confidence"],
            "normalized_entropy": row["normalized_entropy"],
            "target_backend_label": backend["target_backend_label"],
            "predicted_backend_label": backend["predicted_backend_label"],
            "backend_correct": backend["correct"],
            "available_probability": backend["available_probability"],
            "available_threshold": backend["available_threshold"],
        }

    low_margin = sorted(
        (row for row in rows if row.get("margin") is not None),
        key=lambda row: float(row["margin"]),
    )[:limit]
    high_confidence_errors = sorted(
        (row for row in rows if row.get("correct") is False),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )[:limit]
    closest_to_threshold = sorted(
        rows,
        key=lambda row: abs(
            float(row["backend_decision"]["threshold_margin"])
        ),
    )[:limit]
    return {
        "lowest_margin": [compact(row) for row in low_margin],
        "highest_confidence_errors": [
            compact(row) for row in high_confidence_errors
        ],
        "closest_to_available_threshold": [
            compact(row) for row in closest_to_threshold
        ],
    }
