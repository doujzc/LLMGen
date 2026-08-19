from __future__ import annotations

from collections import Counter
import unittest

from scripts.build_top1_intent_change import CHITCHAT_SEEDS, build_intent_change_rows


CANDIDATES = (
    "StockAdvice",
    "StockOther",
    "StockQuery",
    "ProductGeneral",
    "ProductEcommerce",
    "ChitChat",
    "NoAvailable",
)


def _source_row(row_id: str, legacy_label: str, content: str) -> dict:
    return {
        "id": row_id,
        "split": "train",
        "bucket": "test",
        "scenario_family": f"family_{row_id}",
        "expected_candidate_id": legacy_label,
        "messages": [{"role": "user", "content": content}],
    }


class IntentChangeDataTests(unittest.TestCase):
    def test_builder_covers_every_directed_pair_without_reserved_seed(self) -> None:
        source_rows = [
            _source_row("stock_advice", "no_route_stock_advice", "该怎么安排仓位"),
            _source_row("stock_other", "no_route_stock_other", "介绍这家公司的业务"),
            _source_row("reserved_stock_query", "stock_market_information", "查询旧行情"),
            _source_row("stock_query", "stock_market_information", "查询当前行情"),
            _source_row("product_other", "no_route_product_other", "这款耳机怎么连接"),
            _source_row(
                "ecommerce",
                "ecommerce_product_recommendation",
                "推荐一款通勤耳机",
            ),
            _source_row("no_available", "no_route_no_available", "设置明早的闹钟"),
        ]

        rows = build_intent_change_rows(
            source_rows,
            reserved_ids={"reserved_stock_query"},
            candidate_names=CANDIDATES,
            per_pair=1,
            seed=7,
        )

        self.assertEqual(len(rows), len(CANDIDATES) * (len(CANDIDATES) - 1))
        self.assertEqual(
            Counter(row["source_candidate_name"] for row in rows),
            Counter({candidate: 6 for candidate in CANDIDATES}),
        )
        self.assertEqual(
            Counter(row["target_candidate_name"] for row in rows),
            Counter({candidate: 6 for candidate in CANDIDATES}),
        )
        self.assertTrue(
            all(
                row["source_candidate_name"] != row["target_candidate_name"]
                for row in rows
            )
        )
        self.assertTrue(
            all(row["target_candidate_name"] in CANDIDATES for row in rows)
        )
        self.assertTrue(
            all(row["transition_behavior"] == "IntentChange" for row in rows)
        )
        self.assertTrue(all(row["transition_style"] == "direct" for row in rows))
        target_seed_contents = {
            row["id"]: row["messages"][-1]["content"] for row in source_rows
        }
        target_seed_contents.update(
            {
                f"independent_chitchat_seed_{index:03d}": content
                for index, content in enumerate(CHITCHAT_SEEDS, start=1)
            }
        )
        self.assertTrue(
            all(
                row["messages"][-1]["content"]
                == target_seed_contents[row["target_seed_id"]]
                for row in rows
            )
        )
        self.assertTrue(
            all("reserved_stock_query" not in row["source_seed_id"] for row in rows)
        )
        self.assertTrue(
            all("reserved_stock_query" not in row["target_seed_id"] for row in rows)
        )
        self.assertTrue(all(row["messages"][-1]["role"] == "user" for row in rows))
        self.assertEqual(
            rows,
            build_intent_change_rows(
                source_rows,
                reserved_ids={"reserved_stock_query"},
                candidate_names=CANDIDATES,
                per_pair=1,
                seed=7,
            ),
        )


if __name__ == "__main__":
    unittest.main()
