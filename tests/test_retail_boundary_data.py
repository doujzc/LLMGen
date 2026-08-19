from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import unittest

from llmgen.top1 import read_jsonl
from scripts.build_top1_retail_boundary_v1 import (
    BOUNDARY_AXES,
    build_boundary_rows,
    validate_boundary_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPOSITORY_ROOT / "data_top1/top1_retail_boundary_v1.jsonl"
VALIDATION_PATH = (
    REPOSITORY_ROOT / "data_top1/top1_retail_boundary_v1_validation.jsonl"
)
COMBINED_PATH = REPOSITORY_ROOT / "data_top1/top1_train_combined_v1.jsonl"


class RetailBoundaryDataTests(unittest.TestCase):
    def test_structured_plan_is_balanced_and_family_disjoint(self) -> None:
        train_rows = build_boundary_rows("train")
        validation_rows = build_boundary_rows("validation")
        validate_boundary_rows(train_rows, validation_rows)

        self.assertEqual(len(train_rows), 192)
        self.assertEqual(len(validation_rows), 64)
        self.assertEqual(
            Counter(row["target_candidate_name"] for row in train_rows),
            Counter({"GeneralProduct": 96, "EcommerceProduct": 96}),
        )
        self.assertEqual(
            Counter(row["target_candidate_name"] for row in validation_rows),
            Counter({"GeneralProduct": 32, "EcommerceProduct": 32}),
        )
        self.assertEqual(len(BOUNDARY_AXES), 4)
        train_families = {row["boundary_family"] for row in train_rows}
        validation_families = {
            row["boundary_family"] for row in validation_rows
        }
        self.assertEqual(len(train_families), 24)
        self.assertEqual(len(validation_families), 8)
        self.assertFalse(train_families & validation_families)

    def test_every_minimal_pair_changes_object_and_label_only(self) -> None:
        for split in ("train", "validation"):
            grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in build_boundary_rows(split):
                grouped[str(row["boundary_pair_id"])].append(row)
            for pair_rows in grouped.values():
                self.assertEqual(len(pair_rows), 2)
                self.assertEqual(
                    {row["boundary_side"] for row in pair_rows},
                    {"unsupported", "retail"},
                )
                self.assertEqual(
                    {row["target_candidate_name"] for row in pair_rows},
                    {"GeneralProduct", "EcommerceProduct"},
                )
                self.assertEqual(
                    {row["boundary_axis"] for row in pair_rows},
                    {pair_rows[0]["boundary_axis"]},
                )
                self.assertNotEqual(
                    pair_rows[0]["messages"],
                    pair_rows[1]["messages"],
                )

    def test_versioned_files_match_the_deterministic_builder(self) -> None:
        train_rows = read_jsonl(TRAIN_PATH)
        validation_rows = read_jsonl(VALIDATION_PATH)
        self.assertEqual(train_rows, build_boundary_rows("train"))
        self.assertEqual(validation_rows, build_boundary_rows("validation"))

    def test_reported_bad_cases_have_direct_retail_contrasts(self) -> None:
        rows = read_jsonl(TRAIN_PATH)
        labels_by_text = {
            row["messages"][0]["content"]: row["target_candidate_name"]
            for row in rows
        }
        self.assertEqual(
            labels_by_text["送他一架小型直升机大概要多少钱？"],
            "GeneralProduct",
        )
        self.assertEqual(
            labels_by_text["送他一架遥控玩具直升机大概要多少钱？"],
            "EcommerceProduct",
        )
        self.assertEqual(
            labels_by_text["想买这本绘本的电子版权，能直接下单吗？"],
            "GeneralProduct",
        )
        self.assertEqual(
            labels_by_text["想买这本绘本的纸质版，能直接下单吗？"],
            "EcommerceProduct",
        )

    def test_boundary_validation_is_not_in_combined_training_data(self) -> None:
        combined_ids = {str(row["id"]) for row in read_jsonl(COMBINED_PATH)}
        train_ids = {str(row["id"]) for row in read_jsonl(TRAIN_PATH)}
        validation_ids = {
            str(row["id"]) for row in read_jsonl(VALIDATION_PATH)
        }
        self.assertTrue(train_ids.issubset(combined_ids))
        self.assertFalse(validation_ids & combined_ids)


if __name__ == "__main__":
    unittest.main()
