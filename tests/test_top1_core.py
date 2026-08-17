from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from llmgen.top1 import (
    CONVERSATION_TEMPLATE,
    Top1DataError,
    build_user_prompt,
    candidate_token_sequences,
    fit_prompt,
    load_candidate_names,
    normalize_messages,
    prepare_example,
    validate_training_rows,
)


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return [100 + ord(character) for character in text]


class Top1CoreTests(unittest.TestCase):
    def test_registry_contains_only_ordered_candidate_names(self) -> None:
        self.assertEqual(
            load_candidate_names("configs/top1_candidates.json"),
            (
                "StockAdvice",
                "StockOther",
                "StockQuery",
                "ProductOther",
                "Ecommerce",
                "ChitChat",
                "NoAvailable",
            ),
        )

    def test_registry_rejects_non_string_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "routing_mode": "candidate_name_top1",
                        "candidates": [{"name": "legacy-shape"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Top1DataError, r"candidates\[0\]"):
                load_candidate_names(registry)

    def test_normalization_drops_source_system_and_requires_final_user(self) -> None:
        normalized = normalize_messages(
            [
                {"role": "system", "content": "untrusted"},
                {"role": "user", "content": "推荐耳机"},
                {"role": "assistant", "content": "预算是多少？"},
                {"role": "user", "content": "500 元以内"},
            ]
        )

        self.assertEqual(
            normalized[-1],
            {"role": "user", "content": "500 元以内"},
        )
        self.assertTrue(all(message["role"] != "system" for message in normalized))
        with self.assertRaisesRegex(Top1DataError, "final non-system"):
            normalize_messages(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            )

    def test_prompt_uses_standalone_request_contract(self) -> None:
        prompt = build_user_prompt(
            [
                {"role": "user", "content": "推荐耳机"},
                {"role": "assistant", "content": "预算是多少？"},
                {"role": "user", "content": "500 元以内"},
            ]
        )

        self.assertEqual(CONVERSATION_TEMPLATE, "standalone_request_v2")
        self.assertIn('"history"', prompt)
        self.assertIn('"current_user_request":"500 元以内"', prompt)
        self.assertIn("还原为可独立理解的请求", prompt)
        self.assertTrue(prompt.endswith("仅输出候选名称："))

    def test_prompt_fitting_removes_old_history_first(self) -> None:
        tokenizer = CharacterTokenizer()
        prompt, kept = fit_prompt(
            tokenizer,
            [
                {"role": "user", "content": "very old " * 30},
                {"role": "assistant", "content": "old answer " * 30},
                {"role": "user", "content": "最新请求"},
            ],
            "route",
            max_prompt_tokens=420,
        )

        self.assertEqual(kept, ({"role": "user", "content": "最新请求"},))
        self.assertIn("最新请求", prompt)
        self.assertNotIn("very old", prompt)

    def test_prepared_example_supervises_only_candidate_and_eos(self) -> None:
        tokenizer = CharacterTokenizer()
        candidate_tokens = candidate_token_sequences(
            tokenizer,
            ("StockQuery", "Ecommerce"),
        )
        prepared = prepare_example(
            tokenizer,
            {
                "messages": [{"role": "user", "content": "查询贵州茅台"}],
                "target_candidate_name": "StockQuery",
            },
            candidate_tokens=candidate_tokens,
            max_length=512,
            system_prompt="route",
        )

        target = tokenizer.encode("StockQuery") + [tokenizer.eos_token_id]
        self.assertEqual(prepared.encoded["input_ids"][-len(target) :], target)
        self.assertEqual(prepared.encoded["labels"][-len(target) :], target)
        self.assertTrue(
            all(value == -100 for value in prepared.encoded["labels"][: -len(target)])
        )
        self.assertEqual(
            prepared.sft_row["messages"][-1]["content"],
            "StockQuery",
        )

    def test_validation_requires_canonical_fields_and_allows_partial_coverage(
        self,
    ) -> None:
        report = validate_training_rows(
            [
                {
                    "messages": [{"role": "user", "content": "查询股价"}],
                    "target_candidate_name": "StockQuery",
                    "ignored_metadata": {"anything": True},
                }
            ],
            ("StockQuery", "Ecommerce"),
            source="train.jsonl",
        )

        self.assertEqual(report["candidate_counts"], {"StockQuery": 1})
        with self.assertRaisesRegex(Top1DataError, "must contain messages"):
            validate_training_rows(
                [{"query": "legacy", "target_candidate_name": "StockQuery"}],
                ("StockQuery",),
                source="train.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
