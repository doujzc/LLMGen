"""Embedding backends used by the offline SkillRet preparation stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_OPENAI_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"


class EmbeddingServiceError(RuntimeError):
    """Raised when an embedding service returns an invalid or failed response."""


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingConfig:
    """Configuration for an OpenAI-compatible embeddings endpoint."""

    model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    base_url: str = DEFAULT_OPENAI_BASE_URL
    api_key: str = field(default="EMPTY", repr=False)
    dimensions: int | None = None
    timeout: float = 600.0
    max_retries: int = 5

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("embedding model must be non-empty")
        if not self.base_url.strip():
            raise ValueError("embedding base_url must be non-empty")
        if not self.api_key:
            raise ValueError("embedding api_key must be non-empty")
        if self.dimensions is not None and self.dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if self.timeout <= 0:
            raise ValueError("embedding timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("embedding max_retries must be non-negative")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise EmbeddingServiceError(
            "embedding service returned a zero-norm or non-finite vector"
        )
    return np.asarray(values / norms, dtype=np.float32)


class OpenAIEmbeddingModel:
    """Small ``SentenceTransformer.encode``-compatible API adapter.

    ``client`` is injectable so response validation can be tested without a live
    service. Production calls use the official ``openai`` Python SDK against any
    endpoint that implements ``POST /v1/embeddings``.
    """

    tokenizer = None
    max_seq_length = None

    def __init__(self, config: OpenAIEmbeddingConfig, *, client: Any = None) -> None:
        self.config = config
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - real environment only
                raise EmbeddingServiceError(
                    "OpenAI embedding mode requires the 'openai' package; install "
                    "the project's training dependencies"
                ) from exc
            client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        self.client = client

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int | None = None,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Embed one already-batched sequence and validate the indexed response."""

        del batch_size, show_progress_bar
        if not convert_to_numpy:
            raise ValueError("OpenAIEmbeddingModel only supports NumPy output")
        texts = list(sentences)
        if not texts:
            raise ValueError("embedding input must be non-empty")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("every embedding input must be a non-empty string")

        request: dict[str, Any] = {
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.config.dimensions is not None:
            request["dimensions"] = self.config.dimensions
        try:
            response = self.client.embeddings.create(**request)
        except Exception as exc:
            raise EmbeddingServiceError(
                "OpenAI-compatible embedding request failed for "
                f"{self.config.base_url!r} using model {self.config.model!r}"
            ) from exc

        rows = list(_field(response, "data", ()) or ())
        if len(rows) != len(texts):
            raise EmbeddingServiceError(
                "embedding response count does not match the request count"
            )
        indexed: dict[int, Sequence[float]] = {}
        for fallback_index, row in enumerate(rows):
            index = _field(row, "index", fallback_index)
            vector = _field(row, "embedding")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(texts)
            ):
                raise EmbeddingServiceError("embedding response contains an invalid index")
            if index in indexed:
                raise EmbeddingServiceError("embedding response contains a duplicate index")
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
                raise EmbeddingServiceError("embedding response contains no float vector")
            indexed[index] = vector
        if set(indexed) != set(range(len(texts))):
            raise EmbeddingServiceError("embedding response indices are incomplete")

        try:
            values = np.asarray(
                [indexed[index] for index in range(len(texts))], dtype=np.float32
            )
        except (TypeError, ValueError) as exc:
            raise EmbeddingServiceError(
                "embedding response vectors have inconsistent dimensions"
            ) from exc
        if values.ndim != 2 or values.shape[1] < 1:
            raise EmbeddingServiceError(
                "embedding response must be a non-empty two-dimensional matrix"
            )
        if (
            self.config.dimensions is not None
            and values.shape[1] != self.config.dimensions
        ):
            raise EmbeddingServiceError(
                "embedding response dimension differs from the requested dimensions"
            )
        if not np.all(np.isfinite(values)):
            raise EmbeddingServiceError("embedding response contains non-finite values")
        return _normalize_rows(values) if normalize_embeddings else values

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
