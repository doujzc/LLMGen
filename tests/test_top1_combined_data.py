from __future__ import annotations

import unittest

from llmgen.top1 import Top1DataError
from scripts.build_top1_combined_v1 import combine_training_rows


def _row(row_id: str, content: str, target: str) -> dict[str, object]:
    return {
        "id": row_id,
        "messages": [{"role": "user", "content": content}],
        "target_candidate_name": target,
    }


class Top1CombinedDataTests(unittest.TestCase):
    def test_combiner_preserves_source_order(self) -> None:
        base = [_row("base-1", "查询当前股价", "StockQuery")]
        augmentation = [_row("aug-1", "预测下周走势", "StockAdvice")]

        combined = combine_training_rows(
            (("base", base), ("augmentation", augmentation))
        )

        self.assertEqual([row["id"] for row in combined], ["base-1", "aug-1"])

    def test_combiner_rejects_duplicate_id_or_conversation(self) -> None:
        first = _row("row-1", "查询当前股价", "StockQuery")
        duplicate_id = _row("row-1", "预测下周走势", "StockAdvice")
        duplicate_conversation = _row("row-2", "查询当前股价", "StockQuery")

        with self.assertRaisesRegex(Top1DataError, "duplicate id"):
            combine_training_rows((("one", [first]), ("two", [duplicate_id])))
        with self.assertRaisesRegex(Top1DataError, "duplicate conversation"):
            combine_training_rows(
                (("one", [first]), ("two", [duplicate_conversation]))
            )


if __name__ == "__main__":
    unittest.main()
