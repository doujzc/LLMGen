from __future__ import annotations

import copy
from pathlib import Path
import unittest

from llmgen.synthesis import load_taxonomy_descriptions
from llmgen.top1 import Top1DataError, load_candidate_names, read_jsonl
from scripts.repair_top1_ecommerce_labels import (
    apply_label_repairs,
    load_repair_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPAIR_PATH = REPOSITORY_ROOT / "configs/top1_ecommerce_boundary_repairs_v1.json"
AUGMENTATION_PATH = (
    REPOSITORY_ROOT
    / "data_top1/generated/top1_controlled_multiturn_v1/train.jsonl"
)
COMBINED_PATH = REPOSITORY_ROOT / "data_top1/top1_train_combined_v1.jsonl"
TAXONOMY_PATH = REPOSITORY_ROOT / "data_top1/top1_labeldesc_paper_v1.jsonl"
CANDIDATE_PATH = REPOSITORY_ROOT / "configs/top1_candidates.json"


class EcommerceBoundaryDataTests(unittest.TestCase):
    def test_repair_preserves_model_judgment_and_records_human_override(self) -> None:
        row = {
            "id": "case-1",
            "messages": [{"role": "user", "content": "哪款更适合办公"}],
            "target_candidate_name": "ProductGeneral",
            "synthesis": {
                "labeler_predicted_candidate_name": "ProductGeneral",
                "reviewer_predicted_candidate_name": "ProductGeneral",
            },
        }
        spec = {
            "repair_version": "review-v1",
            "repairs": [
                {
                    "id": "case-1",
                    "from_candidate_name": "ProductGeneral",
                    "to_candidate_name": "ProductEcommerce",
                    "reason_code": "ordinary_goods_recommendation",
                }
            ],
        }

        repaired = apply_label_repairs([row], spec)

        self.assertEqual(repaired[0]["target_candidate_name"], "ProductEcommerce")
        self.assertEqual(repaired[0]["synthesis"], row["synthesis"])
        self.assertEqual(
            repaired[0]["label_review_correction"],
            {
                "repair_version": "review-v1",
                "previous_target_candidate_name": "ProductGeneral",
                "corrected_target_candidate_name": "ProductEcommerce",
                "reason_code": "ordinary_goods_recommendation",
            },
        )

    def test_repair_is_idempotent_and_rejects_missing_ids(self) -> None:
        spec = {
            "repair_version": "review-v1",
            "repairs": [
                {
                    "id": "case-1",
                    "from_candidate_name": "ProductGeneral",
                    "to_candidate_name": "ProductEcommerce",
                    "reason_code": "ordinary_goods_recommendation",
                }
            ],
        }
        row = {
            "id": "case-1",
            "messages": [{"role": "user", "content": "推荐一个"}],
            "target_candidate_name": "ProductGeneral",
        }
        once = apply_label_repairs([row], spec)
        self.assertEqual(apply_label_repairs(once, spec), once)
        with self.assertRaisesRegex(Top1DataError, "missing from dataset"):
            apply_label_repairs([], spec)

    def test_reviewed_repair_manifest_matches_both_training_datasets(self) -> None:
        spec = load_repair_spec(REPAIR_PATH)
        expected = {str(repair["id"]): repair for repair in spec["repairs"]}
        self.assertEqual(len(expected), 24)

        for path in (AUGMENTATION_PATH, COMBINED_PATH):
            rows = {str(row["id"]): row for row in read_jsonl(path)}
            self.assertTrue(set(expected).issubset(rows))
            for row_id, repair in expected.items():
                row = rows[row_id]
                self.assertEqual(
                    row["target_candidate_name"],
                    repair["to_candidate_name"],
                )
                correction = row.get("label_review_correction")
                self.assertIsInstance(correction, dict)
                self.assertEqual(
                    correction["repair_version"],
                    spec["repair_version"],
                )
                self.assertEqual(
                    correction["previous_target_candidate_name"],
                    repair["from_candidate_name"],
                )

    def test_taxonomy_assigns_pre_purchase_retail_consultation_to_ecommerce(self) -> None:
        candidates = load_candidate_names(CANDIDATE_PATH)
        descriptions = load_taxonomy_descriptions(TAXONOMY_PATH, candidates)

        ecommerce = descriptions["ProductEcommerce"]["extended_definition"]
        general = descriptions["ProductGeneral"]["extended_definition"]
        self.assertIn("购买前", ecommerce)
        self.assertIn("多少钱", ecommerce)
        self.assertIn("有什么优惠", ecommerce)
        self.assertIn("是否适合某种用途", ecommerce)
        self.assertIn("应归入 ProductEcommerce", general)

    def test_conflicting_correction_metadata_is_rejected(self) -> None:
        spec = load_repair_spec(REPAIR_PATH)
        repair = copy.deepcopy(spec["repairs"][0])
        row = next(
            row
            for row in read_jsonl(AUGMENTATION_PATH)
            if row["id"] == repair["id"]
        )
        row["label_review_correction"] = {"repair_version": "different"}
        with self.assertRaisesRegex(Top1DataError, "conflicting correction metadata"):
            apply_label_repairs([row], {**spec, "repairs": [repair]})


if __name__ == "__main__":
    unittest.main()
