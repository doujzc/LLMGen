"""Serializable data models used by all skill-tokenization strategies."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # Avoid a runtime cycle while retaining precise annotations.
    from .config import TokenizerConfig


class ModelValidationError(ValueError):
    """Raised when a SkillRecord or HierarchicalCode is malformed."""


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _float_tuple(name: str, values: Sequence[float]) -> tuple[float, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ModelValidationError(f"{name} must be a sequence of finite numbers") from exc
    if any(isinstance(value, bool) for value in raw):
        raise ModelValidationError(f"{name} must contain only finite numbers")
    try:
        result = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must contain only finite numbers") from exc
    if any(not math.isfinite(value) for value in result):
        raise ModelValidationError(f"{name} must contain only finite numbers")
    return result


@dataclass(frozen=True, slots=True)
class SkillRecord:
    """Offline representation of one dynamically registered skill."""

    skill_id: str
    name: str = ""
    description: str = ""
    hierarchy: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()
    collaborative_embedding: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ModelValidationError("skill_id must be a non-empty string")
        if not isinstance(self.name, str):
            raise ModelValidationError("name must be a string")
        if not isinstance(self.description, str):
            raise ModelValidationError("description must be a string")

        try:
            hierarchy = tuple(self.hierarchy)
        except TypeError as exc:
            raise ModelValidationError("hierarchy must be a sequence of labels") from exc
        if any(not isinstance(label, str) or not label.strip() for label in hierarchy):
            raise ModelValidationError(
                "every hierarchy label must be a non-empty string"
            )
        object.__setattr__(self, "hierarchy", hierarchy)
        object.__setattr__(self, "embedding", _float_tuple("embedding", self.embedding))
        object.__setattr__(
            self,
            "collaborative_embedding",
            _float_tuple("collaborative_embedding", self.collaborative_embedding),
        )

        if not isinstance(self.metadata, Mapping):
            raise ModelValidationError("metadata must be a mapping")
        metadata = dict(self.metadata)
        try:
            # JSON round-tripping both validates and severs aliases to nested
            # mutable objects owned by the caller.
            metadata = json.loads(json.dumps(metadata))
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("metadata must be JSON serializable") from exc
        object.__setattr__(self, "metadata", _freeze_json(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "hierarchy": list(self.hierarchy),
            "embedding": list(self.embedding),
            "collaborative_embedding": list(self.collaborative_embedding),
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillRecord":
        if not isinstance(payload, Mapping):
            raise ModelValidationError("skill record snapshot must be a mapping")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise ModelValidationError(f"invalid skill record fields: {exc}") from exc


@dataclass(frozen=True, slots=True)
class HierarchicalCode:
    """A full, fixed-depth code and its rendered special tokens."""

    indices: tuple[int, ...]
    tokens: tuple[str, ...]
    codebook_version: str

    def __post_init__(self) -> None:
        try:
            indices = tuple(self.indices)
            tokens = tuple(self.tokens)
        except TypeError as exc:
            raise ModelValidationError("indices and tokens must be sequences") from exc
        if not indices:
            raise ModelValidationError("a hierarchical code must contain at least one level")
        if len(indices) != len(tokens):
            raise ModelValidationError("indices and tokens must have identical lengths")
        for index in indices:
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ModelValidationError("code indices must be non-negative integers")
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise ModelValidationError("code tokens must be non-empty strings")
        if not isinstance(self.codebook_version, str) or not self.codebook_version.strip():
            raise ModelValidationError("codebook_version must be a non-empty string")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "tokens", tokens)

    def validate_against(self, config: "TokenizerConfig") -> None:
        if len(self.indices) != config.num_levels:
            raise ModelValidationError(
                f"code has {len(self.indices)} levels; expected {config.num_levels}"
            )
        if self.codebook_version != config.codebook_version:
            raise ModelValidationError(
                "codebook version mismatch: "
                f"{self.codebook_version!r} != {config.codebook_version!r}"
            )
        expected: list[str] = []
        for level, index in enumerate(self.indices):
            if index >= config.branching_factors[level]:
                raise ModelValidationError(
                    f"index {index} exceeds branching factor at level {level + 1}"
                )
            expected.append(config.token_for(level, index))
        if tuple(expected) != self.tokens:
            raise ModelValidationError(
                "stored tokens do not match indices and configured token_format"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": list(self.indices),
            "tokens": list(self.tokens),
            "codebook_version": self.codebook_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HierarchicalCode":
        if not isinstance(payload, Mapping):
            raise ModelValidationError("hierarchical code snapshot must be a mapping")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise ModelValidationError(f"invalid hierarchical code fields: {exc}") from exc
