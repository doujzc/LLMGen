from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import unittest

from llmgen.top1 import read_jsonl
from scripts.build_top1_short_queries_v1 import (
    MAX_QUERY_CHARACTERS,
    TRAIN_CONTRASTS,
    VALIDATION_CONTRASTS,
    build_short_query_rows,
    validate_short_query_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPOSITORY_ROOT / "data_top1/top1_short_queries_v1.jsonl"
VALIDATION_PATH = (
    REPOSITORY_ROOT / "data_top1/top1_short_queries_v1_validation.jsonl"
)
COMBINED_PATH = REPOSITORY_ROOT / "data_top1/top1_train_combined_v1.jsonl"


class ShortQueryDataTests(unittest.TestCase):
    def test_plan_is_balanced_concise_and_family_disjoint(self) -> None:
        train_rows = build_short_query_rows("train")
        validation_rows = build_short_query_rows("validation")
        validate_short_query_rows(train_rows, validation_rows)

        self.assertEqual(len(train_rows), 48)
        self.assertEqual(len(validation_rows), 16)
        self.assertEqual(len(TRAIN_CONTRASTS), 24)
        self.assertEqual(len(VALIDATION_CONTRASTS), 8)
        self.assertEqual(
            Counter(row["target_candidate_name"] for row in train_rows),
            Counter({"ProductEcommerce": 24, "ProductGeneral": 24}),
        )
        self.assertEqual(
            Counter(row["target_candidate_name"] for row in validation_rows),
            Counter({"ProductEcommerce": 8, "ProductGeneral": 8}),
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

    def test_every_pair_contains_both_retail_decisions(self) -> None:
        for split in ("train", "validation"):
            grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in build_short_query_rows(split):
                grouped[str(row["contrast_pair_id"])].append(row)
            for pair_rows in grouped.values():
                self.assertEqual(len(pair_rows), 2)
                self.assertEqual(
                    {row["contrast_side"] for row in pair_rows},
                    {"retail", "general"},
                )
                self.assertEqual(
                    {row["target_candidate_name"] for row in pair_rows},
                    {"ProductEcommerce", "ProductGeneral"},
                )
                self.assertEqual(
                    {row["short_query_intent"] for row in pair_rows},
                    {pair_rows[0]["short_query_intent"]},
                )

    def test_versioned_files_match_the_deterministic_builder(self) -> None:
        self.assertEqual(read_jsonl(TRAIN_PATH), build_short_query_rows("train"))
        self.assertEqual(
            read_jsonl(VALIDATION_PATH),
            build_short_query_rows("validation"),
        )

    def test_requested_short_form_and_boundary_have_reviewed_labels(self) -> None:
        labels_by_text = {
            row["messages"][0]["content"]: row["target_candidate_name"]
            for row in read_jsonl(TRAIN_PATH)
        }
        self.assertEqual(
            labels_by_text["运动鞋哪个牌子好？"],
            "ProductEcommerce",
        )
        self.assertEqual(
            labels_by_text["七座SUV哪个牌子好？"],
            "ProductGeneral",
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
