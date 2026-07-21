from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.infer_router import _logits_processor_class
from llmgen.router import MultiPathTokenTrie


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

