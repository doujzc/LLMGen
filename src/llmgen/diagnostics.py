"""Pure diagnostic summaries for Top1 training and evaluation."""

from __future__ import annotations

from collections import Counter
import math
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


def numeric_summary(values: Iterable[int | float]) -> dict[str, float | int | None]:
    """Summarize a numeric sequence with stable nearest-rank percentiles."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": mean(ordered),
    }


def build_data_profile(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    candidate_names: Sequence[str],
    candidate_tokens: Mapping[str, Sequence[int]],
    max_length: int,
) -> dict[str, Any]:
    """Build a content-free profile of prepared examples and token pressure."""

    target_counts = Counter(str(row["target_candidate_name"]) for row in diagnostics)
    input_lengths = [int(row["input_tokens"]) for row in diagnostics]
    prompt_lengths = [int(row["prompt_tokens"]) for row in diagnostics]
    target_lengths = [int(row["target_tokens"]) for row in diagnostics]
    message_counts = [int(row["original_message_count"]) for row in diagnostics]
    utilization = [length / max_length for length in input_lengths]
    token_lengths = {name: len(candidate_tokens[name]) + 1 for name in candidate_names}

    first_token_groups: dict[int, list[str]] = {}
    for name in candidate_names:
        first_token_groups.setdefault(int(candidate_tokens[name][0]), []).append(name)
    shared_first = [
        {"token_id": token_id, "candidates": names}
        for token_id, names in sorted(first_token_groups.items())
        if len(names) > 1
    ]

    observed_counts = [target_counts[name] for name in candidate_names if target_counts[name]]
    imbalance_ratio = None
    if observed_counts:
        imbalance_ratio = max(observed_counts) / min(observed_counts)
    return {
        "schema_version": 1,
        "rows": len(diagnostics),
        "candidate_counts": {name: target_counts[name] for name in candidate_names},
        "missing_candidates": [name for name in candidate_names if not target_counts[name]],
        "observed_class_imbalance_ratio": imbalance_ratio,
        "multi_turn_rows": sum(count > 1 for count in message_counts),
        "message_count": numeric_summary(message_counts),
        "input_tokens": numeric_summary(input_lengths),
        "prompt_tokens": numeric_summary(prompt_lengths),
        "target_tokens": numeric_summary(target_lengths),
        "max_length_utilization": numeric_summary(utilization),
        "rows_at_max_length": sum(length == max_length for length in input_lengths),
        "history_truncated_rows": sum(
            int(row["history_messages_dropped"]) > 0 for row in diagnostics
        ),
        "history_messages_dropped": sum(
            int(row["history_messages_dropped"]) for row in diagnostics
        ),
        "current_user_truncated_rows": sum(
            bool(row["current_user_truncated"]) for row in diagnostics
        ),
        "candidate_tokenization": {
            "path_tokens_including_eos": token_lengths,
            "min_path_tokens": min(token_lengths.values()),
            "max_path_tokens": max(token_lengths.values()),
            "shared_first_token_groups": shared_first,
            "length_bias_risk": len(set(token_lengths.values())) > 1,
        },
    }


def build_curve_summary(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract compact loss and learning-rate curves from Trainer history."""

    train_curve = []
    eval_curve = []
    for row in history:
        common = {
            "step": row.get("step"),
            "epoch": row.get("epoch"),
            "main_epoch": row.get("main_epoch", row.get("epoch")),
            "main_epochs": row.get("main_epochs"),
            "trainer_epoch": row.get("trainer_epoch"),
            "stage": row.get("stage"),
            "stage_progress": row.get("stage_progress"),
        }
        if "loss" in row:
            train_curve.append(
                {
                    **common,
                    "loss": row.get("loss"),
                    "learning_rate": row.get("learning_rate"),
                    "grad_norm": row.get("grad_norm"),
                }
            )
        validation_key = next(
            (
                key
                for key in ("eval_loss", "final_loss")
                if key in row
            ),
            None,
        )
        if validation_key is not None:
            eval_curve.append(
                {
                    **common,
                    "eval_loss": row.get(validation_key),
                    "source": validation_key,
                }
            )
    finite_eval = [
        float(row["eval_loss"])
        for row in eval_curve
        if isinstance(row.get("eval_loss"), (int, float))
        and math.isfinite(float(row["eval_loss"]))
    ]
    return {
        "schema_version": 1,
        "train": train_curve,
        "validation": eval_curve,
        "best_eval_loss": min(finite_eval) if finite_eval else None,
    }
