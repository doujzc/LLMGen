from __future__ import annotations

import unittest

from llmgen.diagnostics import build_curve_summary, build_data_profile


class DiagnosticsTests(unittest.TestCase):
    def test_data_profile_surfaces_truncation_and_candidate_length_risk(self) -> None:
        profile = build_data_profile(
            [
                {
                    "target_candidate_name": "A",
                    "input_tokens": 16,
                    "prompt_tokens": 13,
                    "target_tokens": 3,
                    "original_message_count": 3,
                    "history_messages_dropped": 1,
                    "current_user_truncated": False,
                },
                {
                    "target_candidate_name": "A",
                    "input_tokens": 20,
                    "prompt_tokens": 17,
                    "target_tokens": 3,
                    "original_message_count": 1,
                    "history_messages_dropped": 0,
                    "current_user_truncated": True,
                },
            ],
            candidate_names=("A", "B"),
            candidate_tokens={"A": (1, 2), "B": (3, 4, 5)},
            max_length=20,
        )

        self.assertEqual(profile["missing_candidates"], ["B"])
        self.assertEqual(profile["history_truncated_rows"], 1)
        self.assertEqual(profile["current_user_truncated_rows"], 1)
        self.assertTrue(profile["candidate_tokenization"]["length_bias_risk"])

    def test_curve_summary_keeps_periodic_and_final_validation(self) -> None:
        curves = build_curve_summary(
            [
                {
                    "step": 5,
                    "epoch": 1.0,
                    "main_epoch": 1.0,
                    "trainer_epoch": 0.5,
                    "stage_progress": 0.25,
                    "loss": 1.2,
                    "margin_loss": 0.4,
                    "margin_violation_rate": 0.25,
                    "grad_norm": 2.0,
                },
                {
                    "step": 5,
                    "epoch": 1.0,
                    "eval_loss": 1.0,
                    "eval_margin_loss": 0.3,
                    "eval_margin_violation_rate": 0.2,
                },
                {"step": 10, "epoch": 3.0, "final_loss": 0.8},
            ]
        )

        self.assertEqual(len(curves["validation"]), 2)
        self.assertEqual(curves["best_eval_loss"], 0.8)
        self.assertEqual(curves["validation"][-1]["source"], "final_loss")
        self.assertEqual(curves["train"][0]["main_epoch"], 1.0)
        self.assertEqual(curves["train"][0]["trainer_epoch"], 0.5)
        self.assertEqual(curves["train"][0]["margin_loss"], 0.4)
        self.assertEqual(curves["validation"][0]["margin_violation_rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
