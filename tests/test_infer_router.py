from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.infer_router import (
    _generate_batch,
    _logits_processor_class,
    _resolve_decoding,
)
from llmgen.router import MultiPathTokenTrie, RouterDataError


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

    with pytest.raises(RouterDataError, match="num_beams >= 2"):
        _resolve_decoding(
            Namespace(decoding_mode="beam_search", num_beams=1)
        )
    with pytest.raises(RouterDataError, match="decoding_mode"):
        _resolve_decoding(Namespace(decoding_mode="sampling", num_beams=4))


class _FakeTokenizer:
    pad_token_id = 0
    chat_template = None

    def __call__(self, prompts, **kwargs):
        assert len(prompts) == 1
        return {
            "input_ids": torch.tensor([[90, 91]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class _FakeBeamModel:
    def __init__(self) -> None:
        self.generate_kwargs = None
        self.received_beam_indices = None
        self.output = SimpleNamespace(
            sequences=torch.tensor([[90, 91, 10, 20, 2]]),
            scores=(torch.zeros((4, 32)),) * 3,
            beam_indices=torch.tensor([[-1, -1, 0, 0, 0]]),
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
        return torch.tensor([[-0.1, -0.2, -0.3]])


def test_beam_search_selects_one_complete_sequence_and_tracks_beam_scores() -> None:
    trie = MultiPathTokenTrie(
        [(10, 20)],
        eos_token_id=2,
        separator_token_ids=(13,),
        max_paths=1,
    )
    model = _FakeBeamModel()

    results = _generate_batch(
        batch=[{"id": "q1", "query": "查天气"}],
        tokenizer=_FakeTokenizer(),
        model=model,
        torch=torch,
        trie=trie,
        id_to_token={10: "<L1_0>", 20: "<L2_0>"},
        buckets={("<L1_0>", "<L2_0>"): ("weather",)},
        args=Namespace(
            system_prompt="route",
            max_input_length=32,
            device="cpu",
            top_k=10,
            decoding_mode="beam_search",
            num_beams=4,
        ),
    )

    assert model.generate_kwargs["num_beams"] == 4
    assert model.generate_kwargs["num_return_sequences"] == 1
    assert model.generate_kwargs["early_stopping"] is True
    assert model.generate_kwargs["use_cache"] is True
    assert model.generate_kwargs["renormalize_logits"] is True
    assert model.received_beam_indices is model.output.beam_indices
    assert results[0]["decoding"] == {
        "mode": "beam_search",
        "num_beams": 4,
    }
    assert results[0]["generated_text"] == "<L1_0><L2_0>"
    assert results[0]["paths"][0]["score"] == pytest.approx(-0.3)
