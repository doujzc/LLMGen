from __future__ import annotations

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from llmgen.evaluation import BackendDecisionPolicy
from llmgen.inspection import (
    compare_evaluation_runs,
    discover_evaluation_runs,
    discover_training_runs,
    evaluation_statistics,
    load_evaluation_run,
    load_training_run,
)
from llmgen.top1 import sha256_file, write_json, write_jsonl
from scripts import debug_top1


class InspectionTests(unittest.TestCase):
    def test_evaluation_artifacts_are_joined_read_only_by_verified_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "cases.jsonl"
            write_jsonl(
                dataset,
                (
                    {
                        "messages": [{"role": "user", "content": "case zero"}],
                        "target_candidate_name": "A",
                    },
                    {
                        "messages": [
                            {"role": "user", "content": "old request"},
                            {"role": "assistant", "content": "answer"},
                            {"role": "user", "content": "case one"},
                        ],
                        "target_candidate_name": "B",
                    },
                ),
            )
            run = root / "evaluations" / "model" / "eval-1"
            run.mkdir(parents=True)
            self._write_evaluation_run(run, dataset, row_index=1)
            before = self._snapshot(root)

            discovered = discover_evaluation_runs(root / "evaluations")
            detail = load_evaluation_run(run, errors_only=True)
            backend_errors = load_evaluation_run(run, backend_correct=False)

            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0]["evaluation_id"], "eval-1")
            self.assertEqual(detail["dataset_status"]["state"], "verified")
            self.assertEqual(detail["matching_rows"], 1)
            self.assertEqual(backend_errors["matching_rows"], 1)
            self.assertEqual(detail["cases"][0]["last_user"], "case one")
            self.assertIn("assistant: answer", detail["cases"][0]["dialogue"])
            self.assertEqual(before, self._snapshot(root))

    def test_evaluation_case_projection_has_no_row_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "cases.jsonl"
            write_jsonl(
                dataset,
                (
                    {
                        "messages": [{"role": "user", "content": f"case {index}"}],
                        "target_candidate_name": "A",
                    }
                    for index in range(501)
                ),
            )
            run = root / "evaluation"
            run.mkdir()
            write_json(
                run / "eval_manifest.json",
                {
                    "evaluation_id": "all-cases",
                    "dataset": {
                        "path": str(dataset),
                        "sha256": sha256_file(dataset),
                    },
                },
            )
            write_jsonl(
                run / "predictions.jsonl",
                (
                    {
                        "row_index": index,
                        "target_candidate_name": "A",
                        "predicted_candidate_name": "A",
                        "correct": True,
                        "backend_decision": {
                            "target_backend_label": "A",
                            "predicted_backend_label": "A",
                            "correct": True,
                        },
                    }
                    for index in range(501)
                ),
            )

            detail = load_evaluation_run(run)

            self.assertEqual(detail["matching_rows"], 501)
            self.assertEqual(len(detail["cases"]), 501)

    def test_evaluation_comparison_only_diffs_cases_for_same_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "cases.jsonl"
            write_jsonl(
                dataset,
                (
                    {
                        "messages": [{"role": "user", "content": "example"}],
                        "target_candidate_name": "A",
                    },
                ),
            )
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            self._write_evaluation_run(first, dataset, predicted="B")
            self._write_evaluation_run(second, dataset, predicted="A")

            comparison = compare_evaluation_runs(first, second)

            self.assertTrue(comparison["same_dataset"])
            self.assertEqual(comparison["case_change_count"], 1)
            self.assertEqual(comparison["case_changes"][0]["change"], "improved")

    def test_training_curves_and_event_tail_are_projected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs" / "experiment" / "run-1"
            (root / "final").mkdir(parents=True)
            (root / "logs").mkdir()
            write_json(
                root / "run_manifest.json",
                {
                    "created_at": "2026-08-19T00:00:00Z",
                    "run_id": "run-1",
                    "experiment_name": "experiment",
                },
            )
            write_json(root / "status.json", {"state": "COMPLETED"})
            write_json(
                root / "final" / "summary.json",
                {"model_id": "model-1", "best_eval_loss": 0.2},
            )
            write_json(
                root / "final" / "curves.json",
                {
                    "train": [
                        {
                            "step": 1,
                            "main_epoch": 0.1,
                            "stage": "main",
                            "loss": 0.5,
                            "learning_rate": 1e-5,
                            "grad_norm": 1.2,
                        }
                    ],
                    "validation": [
                        {
                            "step": 1,
                            "main_epoch": 0.1,
                            "stage": "main",
                            "eval_loss": 0.4,
                        }
                    ],
                },
            )
            write_jsonl(root / "logs" / "events.jsonl", ({"event": "done"},))

            discovered = discover_training_runs(root.parents[1])
            detail = load_training_run(root)

            self.assertEqual(len(discovered), 1)
            self.assertEqual(len(detail["loss_rows"]), 2)
            self.assertEqual(len(detail["optimization_rows"]), 2)
            self.assertEqual(detail["event_tail"][0]["event"], "done")

    def test_debug_case_inputs_and_margins_are_validated_without_ui(self) -> None:
        messages = debug_top1._normalize_messages(
            [["user", "first"], ["assistant", "reply"], ["user", "current"]]
        )
        margin = debug_top1._score_margin(
            (
                {"candidate": "A", "path_logprob": -0.1, "confidence": 0.9},
                {"candidate": "B", "path_logprob": -0.4, "confidence": 0.6},
            )
        )

        self.assertEqual(messages[-1], {"role": "user", "content": "current"})
        self.assertAlmostEqual(margin["logprob_margin"], 0.3)
        self.assertAlmostEqual(margin["confidence_margin"], 0.3)
        with self.assertRaisesRegex(Exception, "between 0 and 1"):
            debug_top1._normalize_threshold(1.1)

    def test_detailed_candidate_scores_follow_the_constrained_trie(self) -> None:
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Vector:
            def __init__(self, values):
                self.values = values

            def float(self):
                return self

            def __getitem__(self, index):
                return Scalar(self.values[index])

        class Logits:
            def __getitem__(self, key):
                _row, position, allowed = key
                values = {10: 2.0, 11: 1.0, 2: 0.0}
                self.position = position
                return Vector([values[token_id] for token_id in allowed])

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        class Torch:
            long = "long"

            @staticmethod
            def tensor(values, **_kwargs):
                return values

            @staticmethod
            def inference_mode():
                return InferenceMode()

            @staticmethod
            def log_softmax(values, dim):
                if dim != -1:
                    raise AssertionError("unexpected log_softmax dimension")
                maximum = max(values.values)
                denominator = sum(
                    math.exp(value - maximum) for value in values.values
                )
                return Vector(
                    [
                        value - maximum - math.log(denominator)
                        for value in values.values
                    ]
                )

        class Model:
            def __call__(self, **_kwargs):
                return SimpleNamespace(logits=Logits())

        runtime = object.__new__(debug_top1.Top1DebugRuntime)
        runtime.candidate_names = ("A", "B")
        runtime.candidate_tokens = {"A": (10,), "B": (11,)}
        runtime.trie = debug_top1.CandidateNameTokenTrie(
            runtime.candidate_tokens,
            eos_token_id=2,
        )
        runtime.tokenizer = SimpleNamespace(pad_token_id=0)
        runtime.decision_policy = BackendDecisionPolicy(
            candidate_to_backend={"A": "Available", "B": "Fallback"},
            backend_labels=("Available", "Fallback"),
            fallback_backend_label="Fallback",
        )
        runtime.torch = Torch()
        runtime.model = Model()
        runtime.device = "cpu"

        scores = runtime._candidate_path_scores(
            (90,),
            generated_candidate_name="A",
        )

        self.assertEqual([row["candidate"] for row in scores], ["A", "B"])
        self.assertTrue(scores[0]["generated"])
        self.assertAlmostEqual(
            sum(float(row["confidence"]) for row in scores),
            1.0,
        )

    def test_evaluation_statistics_prioritize_backend_accuracy(self) -> None:
        statistics = evaluation_statistics(
            {
                "rows": 10,
                "top1_accuracy": 0.8,
                "macro_recall_observed_candidates": 0.75,
                "per_candidate": {
                    "A": {
                        "support": 6,
                        "predicted": 5,
                        "precision": 0.8,
                        "recall": 2 / 3,
                    }
                },
                "conversation_strata": {
                    "single_turn": {
                        "rows": 4,
                        "accuracy": 0.75,
                        "backend_accuracy": 1.0,
                    }
                },
                "prompt_fitting_strata": {},
                "calibration": {
                    "expected_calibration_error": 0.12,
                    "bins": [
                        {
                            "lower": 0.8,
                            "upper": 0.9,
                            "rows": 4,
                            "accuracy": 0.75,
                            "confidence": 0.84,
                        }
                    ],
                },
                "routing_policy": {"output_route_coverage": 0.6},
                "history_ablation": {"rows": 3, "history_helped": 1},
                "backend": {
                    "accuracy": 0.9,
                    "decision_policy": {
                        "backend_labels": ["Available", "Fallback"],
                        "fallback_backend_label": "Fallback",
                    },
                    "per_label": {
                        "Available": {
                            "support": 5,
                            "predicted": 5,
                            "correct": 4,
                            "precision": 0.8,
                            "recall": 0.8,
                        },
                        "Fallback": {
                            "support": 5,
                            "predicted": 5,
                            "correct": 5,
                            "precision": 1.0,
                            "recall": 1.0,
                        }
                    },
                    "available_oos": {
                        "unsafe_oos_accept_rate": 0.1,
                        "available_false_reject_rate": 0.2,
                    },
                },
            }
        )

        self.assertEqual(statistics["kpis"][0]["value"], 0.9)
        self.assertEqual(statistics["kpis"][1]["value"], 0.8)
        self.assertEqual(statistics["kpis"][2]["value"], 1.0)
        self.assertEqual(statistics["backend_labels"][0]["accuracy"], 0.8)
        self.assertEqual(statistics["backend_labels"][1]["sample_type"], "negative")
        self.assertNotIn("candidate_metrics", statistics)

    def test_evaluation_visual_helpers_escape_labels_and_highlight_errors(self) -> None:
        heatmap = debug_top1._confusion_heatmap_html(
            {"A<script>": {"A<script>": 3, "B": 1}},
            title="Matrix",
        )
        kpis = debug_top1._kpi_html(
            (
                {
                    "name": "Unsafe <rate>",
                    "value": 0.125,
                    "format": "percent",
                    "tone": "danger",
                },
            )
        )
        labels = debug_top1._backend_label_accuracy_html(
            (
                {
                    "label": "Available<script>",
                    "sample_type": "positive",
                    "accuracy": 0.8,
                    "correct": 4,
                    "support": 5,
                },
            )
        )

        self.assertNotIn("<script>", heatmap)
        self.assertNotIn("<script>", labels)
        self.assertIn("220,38,38", heatmap)
        self.assertIn("80.00%", labels)
        self.assertIn("12.50%", kpis)
        self.assertIn("Unsafe &lt;rate&gt;", kpis)

    def test_case_filter_and_styling_follow_backend_correctness(self) -> None:
        import pandas as pd

        styled = debug_top1._evaluation_cases_frame(
            pd,
            (
                {
                    "row_index": 3,
                    "backend_correct": False,
                    "target_backend": "NoAvailable",
                    "predicted_backend": "Ecommerce",
                    "target": "GeneralProduct",
                    "predicted": "EcommerceProduct",
                },
            ),
        )

        self.assertEqual(debug_top1._backend_correct_filter(["正确"]), True)
        self.assertEqual(debug_top1._backend_correct_filter(["错误"]), False)
        self.assertIsNone(debug_top1._backend_correct_filter(["正确", "错误"]))
        self.assertEqual(styled.data.iloc[0]["后端结果"], "✗ 错误")
        self.assertIn("rgba(239, 68, 68, 0.12)", styled.to_html())

    @staticmethod
    def _snapshot(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }

    @staticmethod
    def _write_evaluation_run(
        run: Path,
        dataset: Path,
        *,
        predicted: str = "B",
        row_index: int = 0,
    ) -> None:
        write_json(
            run / "eval_manifest.json",
            {
                "created_at": "2026-08-19T00:00:00Z",
                "evaluation_id": run.name,
                "suite_id": "suite",
                "model": {"model_id": "model-1"},
                "dataset": {"path": str(dataset), "sha256": sha256_file(dataset)},
                "semantic_inference": {
                    "decoding_mode": "greedy",
                    "route_threshold": None,
                },
            },
        )
        write_json(run / "status.json", {"state": "COMPLETED"})
        correct = predicted == "A"
        write_json(
            run / "summary.json",
            {
                "rows": 1,
                "backend_accuracy": float(correct),
                "raw_candidate_accuracy": float(correct),
                "expected_calibration_error": 0.1,
                "routing_policy": {"output_route_coverage": 0.5},
                "available_oos": {"unsafe_oos_accept_rate": 0.0},
            },
        )
        write_json(
            run / "metrics.json",
            {
                "confusion_matrix": {"A": {"A": int(correct), "B": int(not correct)}},
                "backend": {
                    "confusion_matrix": {
                        "A": {"A": int(correct), "Fallback": int(not correct)}
                    }
                },
            },
        )
        write_jsonl(
            run / "predictions.jsonl",
            (
                {
                    "row_index": row_index,
                    "target_candidate_name": "A",
                    "predicted_candidate_name": predicted,
                    "correct": correct,
                    "candidate_confidence": 0.8,
                    "diagnostics": {
                        "original_message_count": 1,
                        "history_messages_dropped": 0,
                        "current_user_truncated": False,
                    },
                    "backend_decision": {
                        "target_backend_label": "A",
                        "predicted_backend_label": "A" if correct else "Fallback",
                        "correct": correct,
                        "status": "routed" if correct else "no_route",
                    },
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
