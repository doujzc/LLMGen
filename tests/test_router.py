from __future__ import annotations

import math

import pytest

from llmgen.router import (
    GeneratedPath,
    RouterDataError,
    TokenTrie,
    active_skill_ids_from_registry,
    aggregate_retrieval_metrics,
    buckets_from_codes,
    build_closed_set_evaluation_rows,
    build_memorization_examples,
    build_retrieval_examples,
    canonical_query_group,
    code_token_id_map,
    encode_target_only_example,
    grouped_train_validation_split,
    normalize_code_rows,
    qrels_by_query,
    query_code_path_metrics,
    query_retrieval_metrics,
    rank_bucket_candidates,
    render_router_prompt,
    validate_registry_assignments,
)


CODE_ROWS = [
    {"skill_id": "s1", "indices": [0, 0], "tokens": ["<L1_0>", "<L2_0>"]},
    {"skill_id": "s2", "indices": [0, 0], "tokens": ["<L1_0>", "<L2_0>"]},
    {"skill_id": "s3", "indices": [1, 1], "tokens": ["<L1_1>", "<L2_1>"]},
]


def test_code_rows_preserve_collision_buckets_and_active_deletions() -> None:
    codes, levels = normalize_code_rows(CODE_ROWS)
    assert levels == 2
    assert buckets_from_codes(codes) == {
        ("<L1_0>", "<L2_0>"): ("s1", "s2"),
        ("<L1_1>", "<L2_1>"): ("s3",),
    }
    assert buckets_from_codes(codes, ["s2", "s3"]) == {
        ("<L1_0>", "<L2_0>"): ("s2",),
        ("<L1_1>", "<L2_1>"): ("s3",),
    }


def test_code_rows_reject_variable_length_suffixes() -> None:
    rows = CODE_ROWS[:1] + [
        {"skill_id": "suffix", "tokens": ["<L1_0>", "<L2_0>", "<Z_0>"]}
    ]
    with pytest.raises(RouterDataError, match="same fixed length"):
        normalize_code_rows(rows)


def test_active_registry_detects_duplicate_membership() -> None:
    with pytest.raises(RouterDataError, match="multiple buckets"):
        active_skill_ids_from_registry(
            {"buckets": {"0/0": ["s1"], "0/1": ["s1"]}}
        )


def test_registry_bucket_indices_are_cross_checked() -> None:
    validate_registry_assignments(
        {"buckets": {"0/0": ["s1", "s2"], "1/1": ["s3"]}},
        CODE_ROWS,
    )
    with pytest.raises(RouterDataError, match="codes assign"):
        validate_registry_assignments(
            {"buckets": {"1/1": ["s1"]}},
            CODE_ROWS,
        )


def test_multi_positive_queries_expand_by_distinct_code_path() -> None:
    codes, _ = normalize_code_rows(CODE_ROWS)
    queries = [{"id": "q1", "query": "do both", "skill_ids": ["s1", "s2", "s3"]}]
    examples = build_retrieval_examples(queries, codes)
    assert len(examples) == 2
    assert {tuple(row["target_tokens"]) for row in examples} == {
        ("<L1_0>", "<L2_0>"),
        ("<L1_1>", "<L2_1>"),
    }
    collided = next(row for row in examples if row["target_skill_ids"] == ["s1", "s2"])
    assert collided["positive_skill_ids"] == ["s1", "s2", "s3"]
    assert all(row["group_id"] == canonical_query_group("do both") for row in examples)


def test_closed_set_export_collapses_multi_target_sft_rows() -> None:
    rows = [
        {
            "query_id": "q1",
            "input_text": "do both",
            "positive_skill_ids": ["s1", "s2", "s3"],
            "target_skill_ids": ["s1", "s2"],
        },
        {
            "query_id": "q1",
            "input_text": "do both",
            "positive_skill_ids": ["s1", "s2", "s3"],
            "target_skill_ids": ["s3"],
        },
    ]
    queries, qrels = build_closed_set_evaluation_rows(
        rows, allowed_skill_ids={"s1", "s2", "s3"}
    )
    assert queries == [
        {
            "id": "q1",
            "query": "do both",
            "skill_ids": ["s1", "s2", "s3"],
        }
    ]
    assert [row["skill_id"] for row in qrels] == ["s1", "s2", "s3"]


def test_closed_set_export_rejects_unknown_candidate_skill() -> None:
    with pytest.raises(RouterDataError, match="outside the candidate corpus"):
        build_closed_set_evaluation_rows(
            [
                {
                    "query_id": "q1",
                    "input_text": "unknown",
                    "positive_skill_ids": ["missing"],
                }
            ],
            allowed_skill_ids={"s1"},
        )


def test_duplicate_query_texts_share_a_split_group() -> None:
    codes, _ = normalize_code_rows(CODE_ROWS)
    rows = build_retrieval_examples(
        [
            {"id": "q1", "query": "  Do BOTH\n", "skill_ids": ["s1"]},
            {"id": "q2", "query": "do both", "skill_ids": ["s3"]},
        ],
        codes,
    )
    assert rows[0]["query_id"] != rows[1]["query_id"]
    assert rows[0]["group_id"] == rows[1]["group_id"]


def test_qrels_are_authoritative_when_supplied() -> None:
    codes, _ = normalize_code_rows(CODE_ROWS)
    queries = [{"id": "q1", "query": "third", "skill_ids": ["s1"]}]
    qrels = qrels_by_query(
        [
            {"query_id": "q1", "skill_id": "s3", "relevance": 1},
            {"query_id": "q1", "skill_id": "s2", "relevance": 0},
        ]
    )
    examples = build_retrieval_examples(queries, codes, qrels)
    assert len(examples) == 1
    assert examples[0]["target_skill_ids"] == ["s3"]


def test_memorization_uses_official_document_order() -> None:
    codes, _ = normalize_code_rows(CODE_ROWS)
    rows = build_memorization_examples(
        [
            {
                "skill_id": "s1",
                "name": "Name",
                "description": "Description",
                "skill_md": "Body",
            }
        ],
        codes,
    )
    assert rows[0]["input_text"] == "Name | Description | Body"
    assert rows[0]["phase"] == "memorization"


def test_grouped_split_never_leaks_multi_target_query() -> None:
    rows = [
        {"group_id": "q1", "target": "a"},
        {"group_id": "q1", "target": "b"},
        {"group_id": "q2", "target": "c"},
        {"group_id": "q3", "target": "d"},
        {"group_id": "q4", "target": "e"},
    ]
    first = grouped_train_validation_split(
        rows, validation_fraction=0.25, seed=9
    )
    second = grouped_train_validation_split(
        rows, validation_fraction=0.25, seed=9
    )
    assert first == second
    train, validation = first
    train_groups = {row["group_id"] for row in train}
    validation_groups = {row["group_id"] for row in validation}
    assert train_groups.isdisjoint(validation_groups)
    assert sum(row["group_id"] == "q1" for row in train + validation) == 2


def test_trie_requires_exactly_l_tokens_then_eos() -> None:
    trie = TokenTrie([(10, 20), (10, 21), (11, 22)], eos_token_id=2)
    assert trie.num_levels == 2
    assert trie.allowed_next(()) == (10, 11)
    assert trie.allowed_next((10,)) == (20, 21)
    assert trie.allowed_next((10, 20)) == (2,)
    assert trie.allowed_next((12,)) == ()
    assert trie.allowed_next((10, 20, 2)) == ()
    assert trie.is_active_path((11, 22))


def test_bucket_expansion_keeps_all_colliding_skills() -> None:
    path_a = ("<L1_0>", "<L2_0>")
    path_b = ("<L1_1>", "<L2_1>")
    candidates = rank_bucket_candidates(
        [GeneratedPath(path_b, -0.2), GeneratedPath(path_a, -0.1)],
        {path_a: ("s1", "s2"), path_b: ("s3",)},
    )
    assert [row["skill_id"] for row in candidates] == ["s1", "s2", "s3"]
    assert candidates[0]["score"] == candidates[1]["score"] == -0.1


def test_skillret_metrics_for_multi_positive_query() -> None:
    metrics = query_retrieval_metrics(
        ["a", "x", "b"], {"a", "b"}, cutoffs=(1, 3)
    )
    assert metrics["recall@1"] == 0.5
    assert metrics["ndcg@1"] == 1.0
    assert metrics["map@1"] == 1.0
    assert metrics["mrr@1"] == 1.0
    assert metrics["completeness@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    expected_ndcg = (1.0 + 1.0 / math.log2(4)) / (
        1.0 + 1.0 / math.log2(3)
    )
    assert metrics["ndcg@3"] == pytest.approx(expected_ndcg)
    assert metrics["map@3"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert metrics["completeness@3"] == 1.0
    aggregate = aggregate_retrieval_metrics([metrics, metrics])
    assert aggregate == metrics


def test_code_path_metrics_ignore_collision_bucket_tie_breaks() -> None:
    codes, _ = normalize_code_rows(CODE_ROWS)
    buckets = buckets_from_codes(codes)
    path_a = ("<L1_0>", "<L2_0>")
    path_b = ("<L1_1>", "<L2_1>")
    metrics = query_code_path_metrics(
        [path_a, path_b],
        {"s2", "s3"},
        codes,
        buckets,
        cutoffs=(1, 2),
    )
    assert metrics["code_recall@1"] == 0.5
    assert metrics["bucket_recall@1"] == 0.5
    assert metrics["bucket_completeness@1"] == 0.0
    assert metrics["code_recall@2"] == 1.0
    assert metrics["bucket_recall@2"] == 1.0
    assert metrics["bucket_completeness@2"] == 1.0


class FakeTokenizer:
    eos_token_id = 2
    chat_template = None

    def __init__(self) -> None:
        self.special = {"<L1_0>": 10, "<L2_0>": 20}

    def encode(self, text, add_special_tokens=False, **kwargs):
        if text in self.special:
            return [self.special[text]]
        return [100 + index for index, _ in enumerate(text)]


def test_target_only_encoding_masks_the_entire_prompt() -> None:
    tokenizer = FakeTokenizer()
    token_ids = code_token_id_map(tokenizer, tokenizer.special)
    encoded = encode_target_only_example(
        tokenizer,
        {
            "input_text": "hello",
            "target_tokens": ["<L1_0>", "<L2_0>"],
        },
        code_token_ids=token_ids,
        num_levels=2,
        max_length=64,
    )
    assert encoded["input_ids"][-3:] == [10, 20, 2]
    assert encoded["labels"][-3:] == [10, 20, 2]
    assert set(encoded["labels"][:-3]) == {-100}
    assert len(encoded["attention_mask"]) == len(encoded["input_ids"])


def test_qwen3_style_chat_template_disables_thinking() -> None:
    class ChatTokenizer:
        chat_template = "{{ messages }}"

        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "rendered"

    tokenizer = ChatTokenizer()
    assert render_router_prompt(tokenizer, "find a skill", "route") == "rendered"
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
