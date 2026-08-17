from __future__ import annotations

import unittest

from llmgen.evaluation import aggregate_predictions, prediction_from_scores


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


class EvaluationTests(unittest.TestCase):
    def test_prediction_retains_length_bias_comparison(self) -> None:
        record = prediction_from_scores(
            row_index=0,
            candidate_names=("A", "B"),
            scores=_scores(-2.0, -3.0),
            score_mode="sum_logprob",
            target_candidate_name="B",
            diagnostics={"original_message_count": 1},
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
            score_mode="sum_logprob",
            target_candidate_name="A",
            diagnostics={"original_message_count": 1},
        )
        second = prediction_from_scores(
            row_index=1,
            candidate_names=("A", "B"),
            scores=_scores(-2.0, -1.0),
            score_mode="sum_logprob",
            target_candidate_name="B",
            diagnostics={"original_message_count": 3},
            history_ablation_scores=_scores(-1.0, -2.0),
        )

        metrics = aggregate_predictions((first, second), ("A", "B"))

        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["A"]["A"], 1)
        self.assertEqual(metrics["history_ablation"]["history_helped"], 1)
        self.assertEqual(metrics["conversation_strata"]["multi_turn"]["rows"], 1)
        self.assertIsNotNone(
            metrics["calibration"]["expected_calibration_error"]
        )
        self.assertEqual(metrics["hard_examples"]["lowest_margin"][0]["row_index"], 1)


if __name__ == "__main__":
    unittest.main()
