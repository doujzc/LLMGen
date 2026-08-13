from __future__ import annotations

from argparse import ArgumentTypeError, Namespace
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from llmgen.direct_router import CandidateNameTokenTrie, CandidateRoute
from llmgen.router import RouterDataError
from scripts.infer_candidate_router import (
    _generate_batch,
    _load_queries,
    _logits_processor_class,
    _metrics,
    _parse_route_threshold,
    _validate_model_tokenizer_vocabulary,
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


class VocabularyModel:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_embeddings = SimpleNamespace(num_embeddings=input_size)
        self.output_embeddings = SimpleNamespace(out_features=output_size)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


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


def test_vocabulary_validation_allows_qwen_reserved_embedding_rows() -> None:
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"first": 0, "last": 151668}
    )
    model = VocabularyModel(input_size=151936, output_size=151936)

    _validate_model_tokenizer_vocabulary(model, tokenizer)


@pytest.mark.parametrize(
    ("input_size", "output_size", "expected"),
    ((10, 20, "input"), (20, 10, "output")),
)
def test_vocabulary_validation_rejects_unrepresentable_token_ids(
    input_size, output_size, expected
) -> None:
    tokenizer = SimpleNamespace(get_vocab=lambda: {"last": 10})
    model = VocabularyModel(input_size=input_size, output_size=output_size)

    with pytest.raises(RouterDataError, match=f"{expected} vocabulary"):
        _validate_model_tokenizer_vocabulary(model, tokenizer)


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
            route_threshold=None,
        ),
    )[0]

    assert model.generate_kwargs["num_return_sequences"] == 1
    assert result["candidate_name"] == "Ecommerce"
    assert result["generated_text"] == "Ecommerce"
    assert result["selected_candidate_id"] == "ecommerce_product_recommendation"
    assert result["intent_label"] == "RecommendProduct"
    assert result["score"] == pytest.approx(-0.6)
    assert result["candidate_confidence"] == pytest.approx(0.5488116361)
    assert result["threshold_triggered"] is False


def test_route_threshold_abstains_below_confidence_and_preserves_raw_route() -> None:
    trie = CandidateNameTokenTrie({"Ecommerce": (11, 12)}, eos_token_id=2)
    model = FakeModel()
    route = CandidateRoute(
        "Ecommerce",
        "ecommerce_product_recommendation",
        "RecommendProduct",
        False,
    )

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
        routes_by_name={"Ecommerce": route},
        system_prompt="route",
        args=Namespace(
            max_input_length=256,
            device="cpu",
            decoding_mode="greedy",
            num_beams=4,
            route_threshold=0.6,
        ),
    )[0]

    assert result["candidate_name"] == "Ecommerce"
    assert result["raw_selected_candidate_id"] == route.candidate_id
    assert result["raw_intent_label"] == "RecommendProduct"
    assert result["raw_should_route"] is True
    assert result["candidate_confidence"] == pytest.approx(0.5488116361)
    assert result["route_threshold"] == 0.6
    assert result["threshold_triggered"] is True
    assert result["selected_candidate_id"] is None
    assert result["intent_label"] is None
    assert result["should_route"] is False
    assert result["status"] == "abstained"


def test_route_threshold_does_not_reject_virtual_candidate() -> None:
    trie = CandidateNameTokenTrie({"Ecommerce": (11, 12)}, eos_token_id=2)
    model = FakeModel()
    route = CandidateRoute("Ecommerce", "no_route_product_other", None, True)

    result = _generate_batch(
        batch=[
            {
                "id": "q1",
                "messages": [{"role": "user", "content": "随便聊聊"}],
            }
        ],
        tokenizer=FakeTokenizer(),
        model=model,
        torch=torch,
        trie=trie,
        routes_by_name={"Ecommerce": route},
        system_prompt="route",
        args=Namespace(
            max_input_length=256,
            device="cpu",
            decoding_mode="greedy",
            num_beams=4,
            route_threshold=0.99,
        ),
    )[0]

    assert result["threshold_triggered"] is False
    assert result["selected_candidate_id"] == "no_route_product_other"
    assert result["should_route"] is False
    assert result["status"] == "no_route"


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf", "not-a-number"])
def test_route_threshold_rejects_invalid_values(value) -> None:
    with pytest.raises(ArgumentTypeError, match="route threshold"):
        _parse_route_threshold(value)


def test_route_threshold_requires_transition_scores() -> None:
    trie = CandidateNameTokenTrie({"Ecommerce": (11, 12)}, eos_token_id=2)
    model = FakeModel()
    model.compute_transition_scores = None
    route = CandidateRoute(
        "Ecommerce",
        "ecommerce_product_recommendation",
        "RecommendProduct",
        False,
    )

    with pytest.raises(RouterDataError, match="transition scores"):
        _generate_batch(
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
            routes_by_name={"Ecommerce": route},
            system_prompt="route",
            args=Namespace(
                max_input_length=256,
                device="cpu",
                decoding_mode="greedy",
                num_beams=4,
                route_threshold=0.6,
            ),
        )


def test_metrics_keep_raw_accuracy_separate_from_threshold_policy() -> None:
    routes = {
        "StockQuery": CandidateRoute(
            "StockQuery", "stock", "SearchStockQuotes", False
        ),
        "NoAvailable": CandidateRoute(
            "NoAvailable", "no_route", None, True
        ),
    }
    queries = [
        {
            "target_candidate_name": "StockQuery",
            "expected_system_output": "SearchStockQuotes",
        },
        {"target_candidate_name": "NoAvailable", "expected_system_output": None},
    ]
    results = [
        {
            "candidate_name": "StockQuery",
            "intent_label": None,
            "should_route": False,
            "threshold_triggered": True,
            "route_threshold": 0.6,
        },
        {
            "candidate_name": "NoAvailable",
            "intent_label": None,
            "should_route": False,
            "threshold_triggered": False,
            "route_threshold": 0.6,
        },
    ]

    metrics = _metrics(queries, results, routes)

    assert metrics is not None
    assert metrics["candidate_accuracy"] == 1.0
    assert metrics["system_output_accuracy"] == 0.5
    assert metrics["routing_policy"]["route_threshold"] == 0.6
    assert metrics["routing_policy"]["output_route_coverage"] == 0.0
    assert metrics["routing_policy"]["threshold_abstention_rate"] == 0.5
    assert metrics["routing_policy"]["selective_candidate_accuracy"] == 1.0
    assert metrics["routing_policy"]["false_no_route_rate"] == 1.0


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
