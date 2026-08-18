from __future__ import annotations

from collections import Counter
import unittest

from scripts.build_top1_training_v1 import (
    EXPECTED_CANDIDATES,
    directed_pair_plan,
    reviewed_source_candidate,
)


class Top1TrainingDataV1Tests(unittest.TestCase):
    def test_reviewed_taxonomy_migration_excludes_policy_drift(self) -> None:
        self.assertEqual(
            reviewed_source_candidate(
                {
                    "expected_candidate_id": "no_route_stock_research",
                    "scenario_family": "research",
                }
            ),
            "StockOther",
        )
        self.assertEqual(
            reviewed_source_candidate(
                {
                    "expected_candidate_id": "no_route_product_other",
                    "scenario_family": "product",
                }
            ),
            "GeneralProduct",
        )
        self.assertIsNone(
            reviewed_source_candidate(
                {
                    "expected_candidate_id": "no_route_stock_other",
                    "scenario_family": "oos_stock_other_software_discovery",
                }
            )
        )
        self.assertIsNone(
            reviewed_source_candidate(
                {
                    "expected_candidate_id": "no_route_multi_product",
                    "scenario_family": "oos_multi_product_unrelated_pair",
                }
            )
        )
        self.assertEqual(
            reviewed_source_candidate(
                {
                    "expected_candidate_id": "no_route_no_available",
                    "scenario_family": "oos_no_available_insufficient_reference",
                }
            ),
            "NoAvailable",
        )

    def test_directed_pair_plan_is_balanced_and_totals_300(self) -> None:
        plan = directed_pair_plan(EXPECTED_CANDIDATES)
        self.assertEqual(len(plan), 42)
        self.assertEqual(sum(plan.values()), 300)
        self.assertTrue(all(source != target for source, target in plan))
        self.assertEqual(set(plan.values()), {7, 8})
        sources = Counter()
        targets = Counter()
        for (source, target), count in plan.items():
            sources[source] += count
            targets[target] += count
        self.assertLessEqual(max(sources.values()) - min(sources.values()), 1)
        self.assertLessEqual(max(targets.values()) - min(targets.values()), 1)


if __name__ == "__main__":
    unittest.main()
