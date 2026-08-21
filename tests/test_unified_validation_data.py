from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from llmgen.top1 import (
    candidate_token_sequences,
    load_candidate_names,
    prepare_example,
    read_jsonl,
    sha256_file,
)
from scripts.build_top1_unified_validation_v1 import (
    CONTEXTUAL_MULTI_TURN_PER_CANDIDATE,
    DATASET_VERSION,
    DEFAULT_RETAIL_VALIDATION,
    DEFAULT_SHORT_VALIDATION,
    DEFAULT_SOURCE_DATA,
    DEFAULT_STOCK_VALIDATION,
    EXPECTED_CANDIDATES,
    EXPECTED_CONTEXT_REQUIREMENT_COUNTS,
    EXPECTED_DIFFICULTY_COUNTS,
    EXPECTED_PHENOMENON_COUNTS,
    EXPECTED_ROWS,
    INTENT_CHANGE_PER_DIRECTED_PAIR,
    ROWS_PER_CANDIDATE,
    SINGLE_TURN_PER_CANDIDATE,
    build_unified_validation_rows,
    validate_unified_validation_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPOSITORY_ROOT / "data_top1/top1_validation_unified_v1.jsonl"
SUMMARY_PATH = (
    REPOSITORY_ROOT / "data_top1/top1_validation_unified_v1_summary.json"
)
TRAIN_PATH = REPOSITORY_ROOT / "data_top1/top1_train_combined_v1.jsonl"


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return [100 + ord(character) for character in text]


class UnifiedValidationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidate_names = load_candidate_names(
            REPOSITORY_ROOT / "configs/top1_candidates.json"
        )
        cls.source_path = (REPOSITORY_ROOT / DEFAULT_SOURCE_DATA).resolve()
        cls.retail_rows = read_jsonl(REPOSITORY_ROOT / DEFAULT_RETAIL_VALIDATION)
        cls.short_rows = read_jsonl(REPOSITORY_ROOT / DEFAULT_SHORT_VALIDATION)
        cls.stock_rows = read_jsonl(REPOSITORY_ROOT / DEFAULT_STOCK_VALIDATION)
        cls.train_rows = read_jsonl(TRAIN_PATH)
        cls.built_rows = read_jsonl(OUTPUT_PATH)

    def test_balanced_turn_and_intent_change_contract(self) -> None:
        rows = self.built_rows
        self.assertEqual(len(rows), EXPECTED_ROWS)
        self.assertEqual(
            Counter(row["target_candidate_name"] for row in rows),
            Counter(
                {
                    candidate: ROWS_PER_CANDIDATE
                    for candidate in EXPECTED_CANDIDATES
                }
            ),
        )
        for candidate in EXPECTED_CANDIDATES:
            candidate_rows = [
                row
                for row in rows
                if row["target_candidate_name"] == candidate
            ]
            single = sum(len(row["messages"]) == 1 for row in candidate_rows)
            contextual = sum(
                row["conversation_phenomenon"] == "contextual_multiturn"
                for row in candidate_rows
            )
            self.assertEqual(single, SINGLE_TURN_PER_CANDIDATE)
            self.assertEqual(
                contextual,
                CONTEXTUAL_MULTI_TURN_PER_CANDIDATE,
            )

        transitions = Counter(
            (
                row.get("source_candidate_name"),
                row["target_candidate_name"],
            )
            for row in rows
            if row["conversation_phenomenon"] == "intent_change"
        )
        self.assertEqual(len(transitions), 42)
        self.assertEqual(
            set(transitions.values()),
            {INTENT_CHANGE_PER_DIRECTED_PAIR},
        )
        self.assertEqual(
            Counter(row["conversation_phenomenon"] for row in rows),
            Counter(EXPECTED_PHENOMENON_COUNTS),
        )
        self.assertEqual(
            Counter(row["difficulty"] for row in rows),
            Counter(EXPECTED_DIFFICULTY_COUNTS),
        )
        self.assertEqual(
            Counter(row["context_requirement"] for row in rows),
            Counter(EXPECTED_CONTEXT_REQUIREMENT_COUNTS),
        )

    def test_every_provenance_record_is_used_once(self) -> None:
        references: list[str] = []
        for row in self.built_rows:
            references.append(str(row["source_record_id"]))
            target_record_id = row.get("target_record_id")
            if isinstance(target_record_id, str):
                references.append(target_record_id)
        self.assertEqual(len(references), len(set(references)))

    def test_versioned_file_is_train_disjoint(self) -> None:
        rows = self.built_rows
        report = validate_unified_validation_rows(
            rows,
            self.train_rows,
            self.candidate_names,
        )
        self.assertEqual(report["train_id_overlap"], 0)
        self.assertEqual(report["train_conversation_overlap"], 0)
        self.assertEqual(report["train_current_utterance_overlap"], 0)
        self.assertEqual(report["train_source_id_overlap"], 0)
        self.assertEqual(report["train_source_family_overlap"], 0)

    def test_versioned_file_matches_builder_when_source_repo_is_available(
        self,
    ) -> None:
        if not self.source_path.is_file():
            self.skipTest("PromptGen sibling source repository is not available")
        rebuilt = build_unified_validation_rows(
            read_jsonl(self.source_path),
            self.retail_rows,
            self.short_rows,
            self.stock_rows,
            candidate_names=self.candidate_names,
        )
        self.assertEqual(self.built_rows, rebuilt)

    def test_summary_hash_and_contract_match_artifact(self) -> None:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(summary["dataset_version"], DATASET_VERSION)
        self.assertFalse(summary["blind_test"])
        self.assertEqual(summary["rows"], EXPECTED_ROWS)
        self.assertEqual(summary["output"]["sha256"], sha256_file(OUTPUT_PATH))
        self.assertEqual(
            summary["leakage_checks"]["unique_current_utterances"],
            EXPECTED_ROWS,
        )
        self.assertEqual(
            summary["near_duplicate_audit"]["threshold_counts"]["0.90"],
            0,
        )

    def test_every_row_fits_the_shared_prompt_without_truncation(self) -> None:
        tokenizer = CharacterTokenizer()
        candidate_tokens = candidate_token_sequences(
            tokenizer,
            self.candidate_names,
        )
        system_prompt = (
            REPOSITORY_ROOT / "configs/top1_system_prompt.md"
        ).read_text(encoding="utf-8")
        for row in self.built_rows:
            prepared = prepare_example(
                tokenizer,
                row,
                candidate_tokens=candidate_tokens,
                max_length=8192,
                system_prompt=system_prompt,
            )
            self.assertEqual(prepared.diagnostics["history_messages_dropped"], 0)
            self.assertFalse(prepared.diagnostics["current_user_truncated"])
            target_length = prepared.diagnostics["target_tokens"]
            self.assertTrue(
                all(
                    value == -100
                    for value in prepared.encoded["labels"][:-target_length]
                )
            )


if __name__ == "__main__":
    unittest.main()
