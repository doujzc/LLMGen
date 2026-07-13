from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from llmgen.embeddings import (
    EmbeddingServiceError,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
)


class FakeEmbeddings:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=self.rows)


class FakeClient:
    def __init__(self, rows):
        self.embeddings = FakeEmbeddings(rows)
        self.closed = False

    def close(self):
        self.closed = True


def test_openai_embedding_adapter_orders_normalizes_and_forwards_dimensions():
    client = FakeClient(
        [
            SimpleNamespace(index=1, embedding=[0.0, 3.0]),
            SimpleNamespace(index=0, embedding=[4.0, 0.0]),
        ]
    )
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingConfig(
            model="Qwen/Qwen3-Embedding-8B",
            base_url="http://embed.test/v1",
            api_key="secret",
            dimensions=2,
        ),
        client=client,
    )

    result = model.encode(
        ["first", "second"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 1.0]])
    assert result.dtype == np.float32
    assert client.embeddings.calls == [
        {
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": ["first", "second"],
            "encoding_format": "float",
            "dimensions": 2,
        }
    ]
    model.close()
    assert client.closed


@pytest.mark.parametrize(
    "rows,match",
    [
        ([{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}], "duplicate"),
        ([{"index": 0, "embedding": [0.0, 0.0]}], "zero-norm"),
        ([{"index": 0, "embedding": [float("nan"), 1.0]}], "non-finite"),
    ],
)
def test_openai_embedding_adapter_rejects_invalid_responses(rows, match):
    model = OpenAIEmbeddingModel(
        OpenAIEmbeddingConfig(base_url="http://embed.test/v1"),
        client=FakeClient(rows),
    )
    inputs = ["a", "b"] if len(rows) == 2 else ["a"]

    with pytest.raises(EmbeddingServiceError, match=match):
        model.encode(inputs)


def test_openai_embedding_config_hides_api_key_and_validates_dimensions():
    config = OpenAIEmbeddingConfig(api_key="do-not-print")
    assert "do-not-print" not in repr(config)
    with pytest.raises(ValueError, match="dimensions"):
        OpenAIEmbeddingConfig(dimensions=0)
