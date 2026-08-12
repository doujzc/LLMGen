from __future__ import annotations

from argparse import Namespace
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from llmgen.direct_router import CandidateNameTokenTrie, CandidateRoute
from scripts.infer_candidate_router import (
    _generate_batch,
    _load_queries,
    _logits_processor_class,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return [100 + ord(character) for character in text]

    def __call__(self, prompts, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[90, 91]] * len(prompts)),
            "attention_mask": torch.tensor([[1, 1]] * len(prompts)),
        }


class FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs = None
        self.output = SimpleNamespace(
            sequences=torch.tensor([[90, 91, 11, 12, 2]]),
            scores=(torch.zeros((1, 32)),) * 3,
            beam_indices=None,
        )

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.output

    def compute_transition_scores(
        self,
        sequences,
        scores,
        beam_indices=None,
        normalize_logits=False,
    ):
        del sequences, scores
        assert beam_indices is None
        assert normalize_logits is True
        return torch.tensor([[-0.1, -0.2, -0.3]])


def test_messages_json_preserves_structured_multiturn_input(tmp_path) -> None:
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "推荐耳机"},
                    {"role": "assistant", "content": "预算多少？"},
                    {"role": "user", "content": "500以内"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows = _load_queries(
        Namespace(
            query=None,
            query_txt=None,
            queries=None,
            messages_json=str(path),
            query_id="dialog-1",
        )
    )

    assert rows[0]["id"] == "dialog-1"
    assert [message["role"] for message in rows[0]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]


def test_query_jsonl_does_not_validate_ids(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    source_rows = [
        {"query": "missing id"},
        {"id": 42, "query": "numeric id"},
        {"id": "duplicate", "query": "first duplicate"},
        {"query_id": "duplicate", "query": "second duplicate"},
        {"id": {"opaque": True}, "query": "structured id"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    rows = _load_queries(
        Namespace(
            query=None,
            query_txt=None,
            queries=str(path),
            messages_json=None,
            query_id="unused",
        )
    )

    assert [row["id"] for row in rows] == [
        "row-000001",
        42,
        "duplicate",
        "duplicate",
        {"opaque": True},
    ]


def test_top1_generation_returns_name_and_downstream_route() -> None:
    trie = CandidateNameTokenTrie(
        {"StockQuery": (10,), "Ecommerce": (11, 12)},
        eos_token_id=2,
    )
    model = FakeModel()
    routes = {
        "StockQuery": CandidateRoute(
            "StockQuery", "stock_market_information", "SearchStockQuotes", False
        ),
        "Ecommerce": CandidateRoute(
            "Ecommerce",
            "ecommerce_product_recommendation",
            "RecommendProduct",
            False,
        ),
    }

    result = _generate_batch(
        batch=[
            {
                "id": "q1",
                "messages": [{"role": "user", "content": "推荐耳机"}],
            }
        ],
        tokenizer=FakeTokenizer(),
        model=model,
        torch=torch,
        trie=trie,
        routes_by_name=routes,
        system_prompt="route",
        args=Namespace(
            max_input_length=256,
            device="cpu",
            decoding_mode="greedy",
            num_beams=4,
        ),
    )[0]

    assert model.generate_kwargs["num_return_sequences"] == 1
    assert result["candidate_name"] == "Ecommerce"
    assert result["generated_text"] == "Ecommerce"
    assert result["selected_candidate_id"] == "ecommerce_product_recommendation"
    assert result["intent_label"] == "RecommendProduct"
    assert result["score"] == pytest.approx(-0.6)


def test_finished_rows_can_only_repeat_eos_while_batch_continues() -> None:
    trie = CandidateNameTokenTrie(
        {"A": (10,), "BC": (11, 12)},
        eos_token_id=2,
    )
    Processor = _logits_processor_class(torch)
    processor = Processor(trie, prompt_width=2)
    scores = torch.zeros((2, 32))
    input_ids = torch.tensor(
        [
            [90, 91, 10, 2],
            [90, 91, 11, 12],
        ]
    )

    constrained = processor(input_ids, scores)

    assert torch.isfinite(constrained[0, 2])
    assert not torch.isfinite(constrained[0, 10])
    assert torch.isfinite(constrained[1, 2])
