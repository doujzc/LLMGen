from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import unittest

from llmgen.top1 import read_jsonl
from scripts.build_top1_stock_prediction_v1 import (
    MAX_QUERY_CHARACTERS,
    TRAIN_CONTRASTS,
    VALIDATION_CONTRASTS,
    build_stock_prediction_rows,
    validate_stock_prediction_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPOSITORY_ROOT / "data_top1/top1_stock_prediction_v1.jsonl"
VALIDATION_PATH = (
    REPOSITORY_ROOT / "data_top1/top1_stock_prediction_v1_validation.jsonl"
)
COMBINED_PATH = REPOSITORY_ROOT / "data_top1/top1_train_combined_v1.jsonl"


class StockPredictionDataTests(unittest.TestCase):
    def test_plan_is_balanced_concise_and_family_disjoint(self) -> None:
        train_rows = build_stock_prediction_rows("train")
        validation_rows = build_stock_prediction_rows("validation")
        validate_stock_prediction_rows(train_rows, validation_rows)

        self.assertEqual(len(train_rows), 48)
        self.assertEqual(len(validation_rows), 16)
        self.assertEqual(len(TRAIN_CONTRASTS), 24)
        self.assertEqual(len(VALIDATION_CONTRASTS), 8)
        self.assertEqual(
            Counter(row["target_candidate_name"] for row in train_rows),
            Counter({"StockAdvice": 24, "StockQuery": 24}),
        )
        self.assertEqual(
            Counter(row["target_candidate_name"] for row in validation_rows),
            Counter({"StockAdvice": 8, "StockQuery": 8}),
        )
        self.assertTrue(
            all(
                len(row["messages"][0]["content"]) <= MAX_QUERY_CHARACTERS
                for row in [*train_rows, *validation_rows]
            )
        )
        train_families = {row["contrast_family"] for row in train_rows}
        validation_families = {
            row["contrast_family"] for row in validation_rows
        }
        self.assertFalse(train_families & validation_families)

    def test_every_pair_has_future_and_observed_sides(self) -> None:
        for split in ("train", "validation"):
            grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in build_stock_prediction_rows(split):
                grouped[str(row["contrast_pair_id"])].append(row)
            for pair_rows in grouped.values():
                self.assertEqual(len(pair_rows), 2)
                self.assertEqual(
                    {row["contrast_side"] for row in pair_rows},
                    {"future_prediction", "observed_fact"},
                )
                self.assertEqual(
                    {row["target_candidate_name"] for row in pair_rows},
                    {"StockAdvice", "StockQuery"},
                )
                temporal_values = {
                    row["temporal_decision"]["requires_future_market_judgment"]
                    for row in pair_rows
                }
                self.assertEqual(temporal_values, {True, False})

    def test_versioned_files_match_the_deterministic_builder(self) -> None:
        self.assertEqual(
            read_jsonl(TRAIN_PATH),
            build_stock_prediction_rows("train"),
        )
        self.assertEqual(
            read_jsonl(VALIDATION_PATH),
            build_stock_prediction_rows("validation"),
        )

    def test_requested_prediction_and_factual_contrast_labels(self) -> None:
        labels_by_text = {
            row["messages"][0]["content"]: row["target_candidate_name"]
            for row in read_jsonl(TRAIN_PATH)
        }
        self.assertEqual(
            labels_by_text["贵州茅台下周会不会涨？"],
            "StockAdvice",
        )
        self.assertEqual(
            labels_by_text["贵州茅台今天涨了多少？"],
            "StockQuery",
        )
        self.assertEqual(
            labels_by_text["深证成指下周会不会涨？"],
            "StockAdvice",
        )

    def test_validation_is_not_merged_into_training(self) -> None:
        combined_ids = {str(row["id"]) for row in read_jsonl(COMBINED_PATH)}
        train_ids = {str(row["id"]) for row in read_jsonl(TRAIN_PATH)}
        validation_ids = {
            str(row["id"]) for row in read_jsonl(VALIDATION_PATH)
        }
        self.assertTrue(train_ids.issubset(combined_ids))
        self.assertFalse(validation_ids & combined_ids)


if __name__ == "__main__":
    unittest.main()
