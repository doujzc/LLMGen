from __future__ import annotations

import math
import unittest

from llmgen.evaluation import (
    BackendDecisionPolicy,
    aggregate_predictions,
    load_backend_decision_policy,
    prediction_from_scores,
)


def _scores(a_sum: float, b_sum: float, *, a_tokens: int = 2, b_tokens: int = 4):
    return {
        "A": {
            "sum_logprob": a_sum,
            "mean_logprob": a_sum / a_tokens,
            "path_tokens": a_tokens,
        },
        "B": {
            "sum_logprob": b_sum,
            "mean_logprob": b_sum / b_tokens,
            "path_tokens": b_tokens,
        },
    }


def _policy(threshold: float = 0.5) -> BackendDecisionPolicy:
    return BackendDecisionPolicy(
        candidate_to_backend={"A": "A", "B": "Fallback"},
        backend_labels=("A", "Fallback"),
        fallback_backend_label="Fallback",
        available_threshold=threshold,
        temperature=1.0,
    )


class EvaluationTests(unittest.TestCase):
    def test_prediction_retains_length_bias_comparison(self) -> None:
        record = prediction_from_scores(
            row_index=0,
            candidate_names=("A", "B"),
            scores=_scores(-2.0, -3.0),
            target_candidate_name="B",
            diagnostics={"original_message_count": 1},
            decision_policy=_policy(),
        )

        self.assertEqual(record["sum_logprob_prediction"], "A")
        self.assertEqual(record["mean_logprob_prediction"], "B")
        self.assertEqual(record["predicted_candidate_name"], "A")
        self.assertFalse(record["correct"])

    def test_metrics_include_confusion_calibration_and_history_ablation(self) -> None:
        first = prediction_from_scores(
            row_index=0,
            candidate_names=("A", "B"),
            scores=_scores(-1.0, -4.0),
            target_candidate_name="A",
            diagnostics={"original_message_count": 1},
            decision_policy=_policy(),
        )
        second = prediction_from_scores(
            row_index=1,
            candidate_names=("A", "B"),
            scores=_scores(-2.0, -1.0),
            target_candidate_name="B",
            diagnostics={"original_message_count": 3},
            decision_policy=_policy(),
            history_ablation_scores=_scores(-1.0, -2.0),
        )

        metrics = aggregate_predictions(
            (first, second),
            ("A", "B"),
            _policy(),
        )

        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["A"]["A"], 1)
        self.assertEqual(metrics["history_ablation"]["history_helped"], 1)
        self.assertEqual(metrics["conversation_strata"]["multi_turn"]["rows"], 1)
        self.assertIsNotNone(
            metrics["calibration"]["expected_calibration_error"]
        )
        self.assertEqual(metrics["hard_examples"]["lowest_margin"][0]["row_index"], 1)
        self.assertEqual(metrics["backend"]["accuracy"], 1.0)
        self.assertEqual(metrics["backend"]["available_oos"]["oos_recall"], 1.0)

    def test_backend_decision_aggregates_oos_before_thresholding(self) -> None:
        candidates = ("Available", "KnownOos", "NoAvailable")
        policy = BackendDecisionPolicy(
            candidate_to_backend={
                "Available": "Available",
                "KnownOos": "NoAvailable",
                "NoAvailable": "NoAvailable",
            },
            backend_labels=("Available", "NoAvailable"),
            fallback_backend_label="NoAvailable",
            available_threshold=0.5,
            temperature=1.0,
        )
        scores = {
            name: {
                "sum_logprob": math.log(probability),
                "mean_logprob": math.log(probability),
                "path_tokens": 1,
            }
            for name, probability in {
                "Available": 0.40,
                "KnownOos": 0.35,
                "NoAvailable": 0.25,
            }.items()
        }

        record = prediction_from_scores(
            row_index=0,
            candidate_names=candidates,
            scores=scores,
            target_candidate_name="KnownOos",
            diagnostics={"original_message_count": 1},
            decision_policy=policy,
        )

        self.assertEqual(record["predicted_candidate_name"], "Available")
        self.assertFalse(record["correct"])
        self.assertEqual(
            record["backend_decision"]["predicted_backend_label"],
            "NoAvailable",
        )
        self.assertTrue(record["backend_decision"]["correct"])
        self.assertAlmostEqual(
            record["backend_decision"]["available_probability"],
            0.40,
        )
        self.assertAlmostEqual(record["backend_decision"]["oos_probability"], 0.60)

    def test_repository_policy_encodes_operational_oos_mapping(self) -> None:
        candidates = (
            "StockAdvice",
            "StockOther",
            "StockQuery",
            "ProductOther",
            "Ecommerce",
            "ChitChat",
            "NoAvailable",
        )
        policy = load_backend_decision_policy(
            "configs/top1_decision_policy.json",
            candidates,
            available_threshold=0.7,
        )

        self.assertEqual(
            policy.available_backend_labels,
            ("StockQuery", "Ecommerce"),
        )
        self.assertEqual(policy.candidate_to_backend["StockAdvice"], "NoAvailable")
        self.assertEqual(policy.candidate_to_backend["ChitChat"], "NoAvailable")
        self.assertEqual(policy.available_threshold, 0.7)


if __name__ == "__main__":
    unittest.main()
