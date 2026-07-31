from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.infer_router import (
    _generate_batch,
    _load_queries,
    _logits_processor_class,
    _resolve_decoding,
)
from llmgen.router import MultiPathTokenTrie, RouterDataError


def test_query_txt_uses_each_nonempty_line_in_original_order(tmp_path) -> None:
    query_txt = tmp_path / "queries.txt"
    query_txt.write_text(
        "\ufeff第一个 Query\n\n重复 Query\r\n  重复 Query  \n",
        encoding="utf-8",
    )

    rows = _load_queries(
        Namespace(
            query=None,
            query_id="interactive",
            query_txt=str(query_txt),
            queries=None,
        )
    )

    assert [row["query"] for row in rows] == [
        "第一个 Query",
        "重复 Query",
        "重复 Query",
    ]
    assert [row["source_line"] for row in rows] == [1, 3, 4]
    assert [row["id"] for row in rows] == [
        "line-000001",
        "line-000003",
        "line-000004",
    ]


def test_logits_processor_keeps_finished_batch_rows_padded_with_eos() -> None:
    trie = MultiPathTokenTrie(
        [(10, 20), (11, 21)],
        eos_token_id=2,
        separator_token_ids=(13,),
        max_paths=2,
    )
    Processor = _logits_processor_class(torch)
    processor = Processor(trie, prompt_width=2)
    input_ids = torch.tensor(
        [
            [90, 91, 10, 20, 2],
            [90, 91, 10, 20, 13],
        ]
    )
    scores = torch.zeros((2, 32))

    constrained = processor(input_ids, scores)

    assert torch.isfinite(constrained[0, 2])
    assert not torch.isfinite(constrained[0, 10])
    assert torch.isfinite(constrained[1, 11])
    assert not torch.isfinite(constrained[1, 10])


def test_decoding_mode_normalizes_greedy_and_validates_beam_width() -> None:
    assert _resolve_decoding(Namespace()) == ("greedy", 1)
    assert _resolve_decoding(
        Namespace(decoding_mode="greedy", num_beams=8)
    ) == ("greedy", 1)
    assert _resolve_decoding(
        Namespace(decoding_mode="beam_search", num_beams=4)
    ) == ("beam_search", 4)
    assert _resolve_decoding(
        Namespace(decoding_mode="greedy_beam_fill", num_beams=6)
    ) == ("greedy_beam_fill", 6)

    with pytest.raises(RouterDataError, match="num_beams >= 2"):
        _resolve_decoding(
            Namespace(decoding_mode="beam_search", num_beams=1)
        )
    with pytest.raises(RouterDataError, match="num_beams >= 2"):
        _resolve_decoding(
            Namespace(decoding_mode="greedy_beam_fill", num_beams=1)
        )
    with pytest.raises(RouterDataError, match="decoding_mode"):
        _resolve_decoding(Namespace(decoding_mode="sampling", num_beams=4))


class _FakeTokenizer:
    pad_token_id = 0
    chat_template = None

    def __call__(self, prompts, **kwargs):
        return {
            "input_ids": torch.tensor([[90, 91]] * len(prompts)),
            "attention_mask": torch.tensor([[1, 1]] * len(prompts)),
        }


class _FakeBeamModel:
    def __init__(self) -> None:
        self.generate_kwargs = None
        self.received_beam_indices = None
        self.output = SimpleNamespace(
            sequences=torch.tensor(
                [
                    [90, 91, 10, 20],
                    [90, 91, 11, 21],
                ]
            ),
            scores=(torch.zeros((2, 32)),) * 2,
            beam_indices=torch.tensor(
                [
                    [-1, -1, 0, 0],
                    [-1, -1, 1, 1],
                ]
            ),
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
        self.received_beam_indices = beam_indices
        assert normalize_logits is True
        return torch.tensor(
            [
                [-0.1, -0.2],
                [-0.4, -0.5],
            ]
        )


def test_beam_search_returns_top_k_single_codes_and_tracks_beam_scores() -> None:
    trie = MultiPathTokenTrie(
        [(10, 20), (11, 21)],
        eos_token_id=2,
        separator_token_ids=(13,),
        max_paths=2,
    )
    model = _FakeBeamModel()

    results = _generate_batch(
        batch=[{"id": "q1", "query": "查天气"}],
        tokenizer=_FakeTokenizer(),
        model=model,
        torch=torch,
        trie=trie,
        id_to_token={
            10: "<L1_0>",
            11: "<L1_1>",
            20: "<L2_0>",
            21: "<L2_1>",
        },
        buckets={
            ("<L1_0>", "<L2_0>"): ("weather",),
            ("<L1_1>", "<L2_1>"): ("calendar",),
        },
        args=Namespace(
            system_prompt="route",
            max_input_length=32,
            device="cpu",
            top_k=10,
            decoding_mode="beam_search",
            num_beams=2,
        ),
    )

    assert model.generate_kwargs["num_beams"] == 2
    assert model.generate_kwargs["num_return_sequences"] == 2
    assert model.generate_kwargs["max_new_tokens"] == 2
    assert model.generate_kwargs["logits_processor"][0].trie.max_paths == 1
    assert model.generate_kwargs["early_stopping"] is True
    assert model.generate_kwargs["use_cache"] is True
    assert model.generate_kwargs["renormalize_logits"] is True
    assert model.received_beam_indices is model.output.beam_indices
    assert results[0]["decoding"] == {
        "mode": "beam_search",
        "num_beams": 2,
        "scope": "single_code_top_k",
        "num_return_sequences": 2,
    }
    assert results[0]["generated_text"] == (
        "<L1_0><L2_0>\n<L1_1><L2_1>"
    )
    assert [path["code_text"] for path in results[0]["paths"]] == [
        "<L1_0><L2_0>",
        "<L1_1><L2_1>",
    ]
    assert results[0]["paths"][0]["score"] == pytest.approx(-0.3)
    assert results[0]["paths"][1]["score"] == pytest.approx(-0.9)
    assert [row["skill_id"] for row in results[0]["candidates"]] == [
        "weather",
        "calendar",
    ]
    assert results[0]["skill_ids"] == ["weather", "calendar"]


class _FakeGreedyModel:
    def __init__(self) -> None:
        self.generate_kwargs = None
        self.output = SimpleNamespace(
            sequences=torch.tensor([[90, 91, 10, 20, 13, 11, 21, 2]]),
            scores=(torch.zeros((1, 32)),) * 6,
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
        assert beam_indices is None
        assert normalize_logits is True
        return torch.tensor([[-0.1, -0.2, -0.01, -0.3, -0.4, -0.01]])


def test_greedy_keeps_autoregressive_multi_path_output() -> None:
    trie = MultiPathTokenTrie(
        [(10, 20), (11, 21)],
        eos_token_id=2,
        separator_token_ids=(13,),
        max_paths=2,
    )
    model = _FakeGreedyModel()

    results = _generate_batch(
        batch=[{"id": "q1", "query": "查天气并写入日历"}],
        tokenizer=_FakeTokenizer(),
        model=model,
        torch=torch,
        trie=trie,
        id_to_token={
            10: "<L1_0>",
            11: "<L1_1>",
            20: "<L2_0>",
            21: "<L2_1>",
        },
        buckets={
            ("<L1_0>", "<L2_0>"): ("weather",),
            ("<L1_1>", "<L2_1>"): ("calendar",),
        },
        args=Namespace(
            system_prompt="route",
            max_input_length=32,
            device="cpu",
            top_k=10,
            decoding_mode="greedy",
            num_beams=8,
        ),
    )

    assert model.generate_kwargs["num_beams"] == 1
    assert model.generate_kwargs["num_return_sequences"] == 1
    assert model.generate_kwargs["max_new_tokens"] == 6
    assert model.generate_kwargs["logits_processor"][0].trie.max_paths == 2
    assert results[0]["decoding"] == {
        "mode": "greedy",
        "num_beams": 1,
        "scope": "autoregressive_multi_path",
        "num_return_sequences": 1,
    }
    assert results[0]["generated_text"] == (
        "<L1_0><L2_0>\n<L1_1><L2_1>"
    )
    assert len(results[0]["paths"]) == 2


def test_beam_search_groups_top_codes_by_input_query() -> None:
    trie = MultiPathTokenTrie(
        [(10, 20), (11, 21)],
        eos_token_id=2,
        separator_token_ids=(13,),
        max_paths=2,
    )
    model = _FakeBeamModel()
    model.output = SimpleNamespace(
        sequences=torch.tensor(
            [
                [90, 91, 10, 20],
                [90, 91, 11, 21],
                [90, 91, 11, 21],
                [90, 91, 10, 20],
            ]
        ),
        scores=(torch.zeros((4, 32)),) * 2,
        beam_indices=torch.tensor(
            [
                [-1, -1, 0, 0],
                [-1, -1, 1, 1],
                [-1, -1, 2, 2],
                [-1, -1, 3, 3],
            ]
        ),
    )

    def transition_scores(*args, **kwargs):
        model.received_beam_indices = kwargs["beam_indices"]
        return torch.tensor(
            [
                [-0.1, -0.2],
                [-0.4, -0.5],
                [-0.2, -0.3],
                [-0.5, -0.6],
            ]
        )

    model.compute_transition_scores = transition_scores
    results = _generate_batch(
        batch=[
            {"id": "q1", "query": "查天气"},
            {"id": "q2", "query": "写入日历"},
        ],
        tokenizer=_FakeTokenizer(),
        model=model,
        torch=torch,
        trie=trie,
        id_to_token={
            10: "<L1_0>",
            11: "<L1_1>",
            20: "<L2_0>",
            21: "<L2_1>",
        },
        buckets={
            ("<L1_0>", "<L2_0>"): ("weather",),
            ("<L1_1>", "<L2_1>"): ("calendar",),
        },
        args=Namespace(
            system_prompt="route",
            max_input_length=32,
            device="cpu",
            top_k=10,
            decoding_mode="beam_search",
            num_beams=2,
        ),
    )

    assert [
        [path["code_text"] for path in result["paths"]]
        for result in results
    ] == [
        ["<L1_0><L2_0>", "<L1_1><L2_1>"],
        ["<L1_1><L2_1>", "<L1_0><L2_0>"],
    ]


class _FakeGreedyBeamModel:
    def __init__(self) -> None:
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if len(self.generate_calls) == 1:
            return SimpleNamespace(
                sequences=torch.tensor([[90, 91, 10, 20, 2]]),
                scores=(torch.zeros((1, 32)),) * 3,
                beam_indices=None,
            )
        return SimpleNamespace(
            sequences=torch.tensor(
                [
                    [90, 91, 10, 20],
                    [90, 91, 11, 21],
                ]
            ),
            scores=(torch.zeros((2, 32)),) * 2,
            beam_indices=torch.tensor(
                [
                    [-1, -1, 0, 0],
                    [-1, -1, 1, 1],
                ]
            ),
        )

    def compute_transition_scores(
        self,
        sequences,
        scores,
        beam_indices=None,
        normalize_logits=False,
    ):
        assert normalize_logits is True
        if int(sequences.shape[0]) == 1:
            return torch.tensor([[-0.1, -0.2, -0.01]])
        assert beam_indices is not None
        return torch.tensor(
            [
                [-0.1, -0.2],
                [-0.4, -0.5],
            ]
        )


def test_greedy_beam_fill_keeps_greedy_then_appends_unique_beam_candidates() -> None:
    trie = MultiPathTokenTrie(
        [(10, 20), (11, 21)],
        eos_token_id=2,
        separator_token_ids=(13,),
        max_paths=2,
    )
    model = _FakeGreedyBeamModel()

    results = _generate_batch(
        batch=[{"id": "q1", "query": "查天气并安排日历"}],
        tokenizer=_FakeTokenizer(),
        model=model,
        torch=torch,
        trie=trie,
        id_to_token={
            10: "<L1_0>",
            11: "<L1_1>",
            20: "<L2_0>",
            21: "<L2_1>",
        },
        buckets={
            ("<L1_0>", "<L2_0>"): ("weather",),
            ("<L1_1>", "<L2_1>"): ("calendar",),
        },
        args=Namespace(
            system_prompt="route",
            max_input_length=32,
            device="cpu",
            top_k=2,
            decoding_mode="greedy_beam_fill",
            num_beams=2,
        ),
    )

    assert len(model.generate_calls) == 2
    assert model.generate_calls[0]["num_beams"] == 1
    assert model.generate_calls[1]["num_beams"] == 2
    result = results[0]
    assert result["generated_text"] == "<L1_0><L2_0>"
    assert [path["code_text"] for path in result["paths"]] == [
        "<L1_0><L2_0>"
    ]
    assert [path["code_text"] for path in result["beam_fill_paths"]] == [
        "<L1_0><L2_0>",
        "<L1_1><L2_1>",
    ]
    assert [
        (candidate["skill_id"], candidate["selection_source"])
        for candidate in result["candidates"]
    ] == [
        ("weather", "greedy"),
        ("calendar", "beam_fill"),
    ]
    assert result["decoding"]["beam_executed"] is True
    assert result["decoding"]["beam_candidates_added"] == 1
    assert result["decoding"]["target_reached"] is True
    assert result["skill_ids"] == ["weather", "calendar"]


def test_greedy_beam_fill_skips_beam_when_greedy_already_reaches_top_k() -> None:
    trie = MultiPathTokenTrie(
        [(10, 20), (11, 21)],
        eos_token_id=2,
        separator_token_ids=(13,),
        max_paths=2,
    )
    model = _FakeGreedyModel()

    result = _generate_batch(
        batch=[{"id": "q1", "query": "查天气并写入日历"}],
        tokenizer=_FakeTokenizer(),
        model=model,
        torch=torch,
        trie=trie,
        id_to_token={
            10: "<L1_0>",
            11: "<L1_1>",
            20: "<L2_0>",
            21: "<L2_1>",
        },
        buckets={
            ("<L1_0>", "<L2_0>"): ("weather",),
            ("<L1_1>", "<L2_1>"): ("calendar",),
        },
        args=Namespace(
            system_prompt="route",
            max_input_length=32,
            device="cpu",
            top_k=2,
            decoding_mode="greedy_beam_fill",
            num_beams=2,
        ),
    )[0]

    assert model.generate_kwargs["num_beams"] == 1
    assert result["beam_fill_paths"] == []
    assert result["decoding"]["beam_executed"] is False
    assert [candidate["selection_source"] for candidate in result["candidates"]] == [
        "greedy",
        "greedy",
    ]


def test_greedy_beam_fill_preserves_batch_order_and_only_beams_short_rows(
    monkeypatch,
) -> None:
    batch = [
        {"id": "q1", "query": "Greedy 已满"},
        {"id": "q2", "query": "缺少一个"},
        {"id": "q3", "query": "全部补充"},
    ]
    calls = []

    def make_result(row, mode, skill_ids):
        return {
            "query_id": row["id"],
            "query": row["query"],
            "generated_text": "",
            "decoding": {"mode": mode, "num_beams": 1 if mode == "greedy" else 3},
            "paths": [],
            "candidates": [
                {
                    "skill_id": skill_id,
                    "score": -float(index),
                    "path_rank": index,
                    "code_tokens": [skill_id],
                }
                for index, skill_id in enumerate(skill_ids)
            ],
        }

    def fake_generate_decoding_batch(*, batch, args, **kwargs):
        calls.append((args.decoding_mode, [row["id"] for row in batch]))
        skill_ids = (
            {
                "q1": ("a", "b"),
                "q2": ("c",),
                "q3": (),
            }
            if args.decoding_mode == "greedy"
            else {
                "q2": ("c", "d", "unused"),
                "q3": ("e", "f", "unused"),
            }
        )
        return [
            make_result(row, args.decoding_mode, skill_ids[row["id"]])
            for row in batch
        ]

    monkeypatch.setattr(
        "scripts.infer_router._generate_decoding_batch",
        fake_generate_decoding_batch,
    )

    results = _generate_batch(
        batch=batch,
        tokenizer=None,
        model=None,
        torch=None,
        trie=None,
        id_to_token={},
        buckets={},
        args=Namespace(
            decoding_mode="greedy_beam_fill",
            num_beams=3,
            top_k=2,
        ),
    )

    assert calls == [
        ("greedy", ["q1", "q2", "q3"]),
        ("beam_search", ["q2", "q3"]),
    ]
    assert [row["query_id"] for row in results] == ["q1", "q2", "q3"]
    assert [
        [candidate["skill_id"] for candidate in row["candidates"]]
        for row in results
    ] == [["a", "b"], ["c", "d"], ["e", "f"]]
    assert [
        [candidate["selection_source"] for candidate in row["candidates"]]
        for row in results
    ] == [
        ["greedy", "greedy"],
        ["greedy", "beam_fill"],
        ["beam_fill", "beam_fill"],
    ]
    assert results[1]["decoding"]["result_fields"] == {
        "greedy_paths": "paths",
        "beam_supplement_paths": "beam_fill_paths",
        "final_skill_ranking": "candidates",
        "final_skill_ids": "skill_ids",
    }
    assert [row["skill_ids"] for row in results] == [
        ["a", "b"],
        ["c", "d"],
        ["e", "f"],
    ]
