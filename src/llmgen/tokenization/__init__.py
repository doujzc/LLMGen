"""Hierarchical skill-tokenization strategies and shared API."""

from collections.abc import Mapping
from typing import Any

from ..config import TokenizerConfig
from .base import (
    BaseSkillTokenizer,
    EncodingError,
    SerializationError,
    SkillTokenizer,
    StrategyMismatchError,
    TokenizerError,
    TokenizerNotFittedError,
)
from .balanced import BalancedSkillTokenizer, BalancedTokenizer
from .interpretable import InterpretableSkillTokenizer, TaxonomyEncodingError


def create_tokenizer(config: TokenizerConfig) -> BaseSkillTokenizer:
    """Create the configured strategy behind the shared tokenizer API."""

    if not isinstance(config, TokenizerConfig):
        raise TypeError("config must be a TokenizerConfig")
    if config.strategy == "interpretable":
        return InterpretableSkillTokenizer(config)
    if config.strategy == "balanced":
        return BalancedSkillTokenizer(config)
    # TokenizerConfig already validates this; retain a defensive boundary for
    # future strategy extensions and deserialized subclasses.
    raise StrategyMismatchError(f"unsupported tokenizer strategy: {config.strategy!r}")


def tokenizer_from_snapshot(payload: Mapping[str, Any]) -> BaseSkillTokenizer:
    """Restore the strategy implementation declared by a saved snapshot."""

    config_payload = payload.get("config") if isinstance(payload, Mapping) else None
    strategy = config_payload.get("strategy") if isinstance(config_payload, Mapping) else None
    if strategy == "interpretable":
        from .interpretable import InterpretableSkillTokenizer

        return InterpretableSkillTokenizer.from_snapshot(payload)
    if strategy == "balanced":
        from .balanced import BalancedSkillTokenizer

        return BalancedSkillTokenizer.from_snapshot(payload)
    raise SerializationError(f"unknown or missing tokenizer strategy: {strategy!r}")

__all__ = [
    "BaseSkillTokenizer",
    "BalancedSkillTokenizer",
    "BalancedTokenizer",
    "EncodingError",
    "InterpretableSkillTokenizer",
    "SerializationError",
    "SkillTokenizer",
    "StrategyMismatchError",
    "TaxonomyEncodingError",
    "TokenizerError",
    "TokenizerNotFittedError",
    "create_tokenizer",
    "tokenizer_from_snapshot",
]
