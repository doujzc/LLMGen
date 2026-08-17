"""Candidate-path scoring records and aggregate Top1 evaluation metrics."""

from __future__ import annotations

from collections import Counter
import math
from statistics import mean
from typing import Any, Mapping, Sequence

from .diagnostics import numeric_summary
from .top1 import Top1DataError


SCORE_MODES = ("sum_logprob", "mean_logprob")


def prediction_from_scores(
    *,
    row_index: int,
    candidate_names: Sequence[str],
    scores: Mapping[str, Mapping[str, float | int]],
    score_mode: str,
    target_candidate_name: str | None,
    diagnostics: Mapping[str, Any],
    history_ablation_scores: Mapping[str, Mapping[str, float | int]] | None = None,
) -> dict[str, Any]:
    """Create one privacy-safe prediction record from candidate path scores."""

    if score_mode not in SCORE_MODES:
        raise Top1DataError(f"unsupported score mode: {score_mode}")
    if set(scores) != set(candidate_names):
        raise Top1DataError("candidate scores do not match the candidate registry")

    sum_prediction = max(candidate_names, key=lambda name: float(scores[name]["sum_logprob"]))
    mean_prediction = max(
        candidate_names,
        key=lambda name: float(scores[name]["mean_logprob"]),
    )
    prediction = sum_prediction if score_mode == "sum_logprob" else mean_prediction
    selected = [float(scores[name][score_mode]) for name in candidate_names]
    probabilities = _softmax(selected)
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
        "score_mode": score_mode,
        "confidence": max(probabilities),
        "normalized_entropy": normalized_entropy,
        "margin": margin,
        "diagnostics": dict(diagnostics),
        "candidate_scores": {
            name: {
                "sum_logprob": float(scores[name]["sum_logprob"]),
                "mean_logprob": float(scores[name]["mean_logprob"]),
                "path_tokens": int(scores[name]["path_tokens"]),
            }
            for name in candidate_names
        },
    }
    if history_ablation_scores is not None:
        ablated_prediction = max(
            candidate_names,
            key=lambda name: float(history_ablation_scores[name][score_mode]),
        )
        record["history_ablation"] = {
            "predicted_candidate_name": ablated_prediction,
            "correct": (
                ablated_prediction == target_candidate_name
                if target_candidate_name
                else None
            ),
            "changed_prediction": ablated_prediction != prediction,
        }
    return record


def aggregate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
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
        },
        "hard_examples": _hard_examples(predictions),
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
        return {
            "row_index": row["row_index"],
            "target_candidate_name": row.get("target_candidate_name"),
            "predicted_candidate_name": row["predicted_candidate_name"],
            "correct": row.get("correct"),
            "margin": row.get("margin"),
            "confidence": row["confidence"],
            "normalized_entropy": row["normalized_entropy"],
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
    return {
        "lowest_margin": [compact(row) for row in low_margin],
        "highest_confidence_errors": [
            compact(row) for row in high_confidence_errors
        ],
    }
