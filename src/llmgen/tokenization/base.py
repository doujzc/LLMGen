"""Strategy-independent API for hierarchical skill tokenizers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar

from ..config import TokenizerConfig
from ..models import HierarchicalCode, SkillRecord
from ..registry import (
    DuplicateSkillError,
    InvalidCodeError,
    RegistryError,
    SkillRegistry,
    UnknownSkillError,
)


class TokenizerError(RuntimeError):
    """Base exception for tokenizer lifecycle and strategy failures."""


class TokenizerNotFittedError(TokenizerError):
    """Raised when an operation needs a fitted/restored codebook."""


class StrategyMismatchError(TokenizerError):
    """Raised when a strategy class receives another strategy's config."""


class SerializationError(TokenizerError):
    """Raised when a tokenizer snapshot cannot be saved or restored."""


class EncodingError(TokenizerError):
    """Raised when a strategy produces an invalid code."""


class BaseSkillTokenizer(ABC):
    """Common lifecycle, registry, token conversion, and serialization.

    Subclasses implement only code construction and their private codebook
    state. Dynamic membership and prefix decoding are intentionally shared.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    STRATEGY: ClassVar[str | None] = None

    def __init__(self, config: TokenizerConfig) -> None:
        if not isinstance(config, TokenizerConfig):
            raise TypeError("config must be a TokenizerConfig")
        if self.STRATEGY is not None and config.strategy != self.STRATEGY:
            raise StrategyMismatchError(
                f"{type(self).__name__} requires strategy={self.STRATEGY!r}; "
                f"got {config.strategy!r}"
            )
        self.config = config
        self.registry = SkillRegistry(config)
        self._is_fitted = False
        # Registry operations are individually thread-safe, but tokenizer
        # transactions also touch strategy state. This outer lock keeps those
        # two pieces atomic with respect to each other.
        self._lock = RLock()

    @property
    def is_fitted(self) -> bool:
        with self._lock:
            return self._is_fitted

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return self.config.special_tokens

    def fit(self, skills: Sequence[SkillRecord]) -> None:
        """Create one offline codebook version and replace registry contents."""

        with self._lock:
            records = self._validate_skill_batch(skills)
            # Strategy implementations necessarily build temporary codebook
            # state before the common registry validates every assignment.
            # Preserve it so a failed re-fit cannot pair new strategy state
            # with the old active registry.
            previous_strategy_state = deepcopy(dict(self._strategy_snapshot()))
            try:
                raw_codes = self._fit_strategy(records)
                if isinstance(raw_codes, Mapping):
                    try:
                        codes = tuple(
                            raw_codes[record.skill_id] for record in records
                        )
                    except KeyError as exc:
                        raise EncodingError(
                            f"strategy did not return a code for skill {exc.args[0]!r}"
                        ) from exc
                else:
                    codes = tuple(raw_codes)
                if len(codes) != len(records):
                    raise EncodingError(
                        "strategy returned a different number of codes and skills: "
                        f"{len(codes)} != {len(records)}"
                    )

                replacement = SkillRegistry(self.config)
                try:
                    for record, code in zip(records, codes, strict=True):
                        replacement.add(record, code)
                except RegistryError as exc:
                    raise EncodingError(
                        f"strategy produced an invalid assignment: {exc}"
                    ) from exc
            except Exception:
                self._restore_strategy(previous_strategy_state)
                raise
            self.registry = replacement
            self._is_fitted = True

    def encode(self, skill_or_id: SkillRecord | str) -> HierarchicalCode:
        """Return an existing code or preview an unregistered skill's code."""

        with self._lock:
            self._require_fitted()
            if isinstance(skill_or_id, str):
                return self.registry.code_for(skill_or_id)
            if not isinstance(skill_or_id, SkillRecord):
                raise TypeError("encode expects a SkillRecord or skill_id string")
            if skill_or_id.skill_id in self.registry:
                return self.registry.code_for(skill_or_id.skill_id)
            return self._validated_strategy_code(self._encode_new(skill_or_id))

    def add(self, skill: SkillRecord) -> HierarchicalCode:
        """Encode and register a new skill without moving existing skills."""

        with self._lock:
            self._require_fitted()
            if not isinstance(skill, SkillRecord):
                raise TypeError("add expects a SkillRecord")
            if skill.skill_id in self.registry:
                raise DuplicateSkillError(
                    f"skill_id {skill.skill_id!r} is already registered"
                )
            code = self._validated_strategy_code(self._encode_new(skill))
            self.registry.add(skill, code)
            try:
                self._on_add(skill, code)
            except Exception:
                self.registry.remove(skill.skill_id)
                raise
            return code

    def remove(self, skill_id: str) -> bool:
        """Remove active membership while leaving token/codebook IDs frozen."""

        with self._lock:
            self._require_fitted()
            if skill_id not in self.registry:
                return False
            skill = self.registry.get(skill_id)
            code = self.registry.code_for(skill_id)
            self.registry.remove(skill_id)
            try:
                self._on_remove(skill_id, code)
            except Exception:
                # Preserve the registry contract if strategy bookkeeping fails.
                self.registry.add(skill, code)
                raise
            return True

    def decode(
        self, prefix: HierarchicalCode | Sequence[int] | Sequence[str] | None = None
    ) -> tuple[str, ...]:
        with self._lock:
            self._require_fitted()
            return self.registry.decode(prefix)

    def valid_next_tokens(
        self, prefix: HierarchicalCode | Sequence[int] | Sequence[str] | None = None
    ) -> tuple[str, ...]:
        with self._lock:
            self._require_fitted()
            return self.registry.valid_next_tokens(prefix)

    def indices_to_tokens(self, indices: Sequence[int]) -> tuple[str, ...]:
        """Render a valid full code or prefix with positional namespaces."""

        with self._lock:
            normalized = self.registry.normalize_prefix(tuple(indices))
            return tuple(
                self.config.token_for(level, index)
                for level, index in enumerate(normalized)
            )

    def tokens_to_indices(self, tokens: Sequence[str]) -> tuple[int, ...]:
        """Parse a valid full code or prefix into integer indices."""

        with self._lock:
            return self.registry.normalize_prefix(tuple(tokens))

    def make_code(self, indices: Sequence[int]) -> HierarchicalCode:
        """Build and cross-validate a full HierarchicalCode."""

        with self._lock:
            normalized = self.registry.normalize_prefix(tuple(indices))
            if len(normalized) != self.config.num_levels:
                raise EncodingError(
                    f"full code requires {self.config.num_levels} indices; "
                    f"got {len(normalized)}"
                )
            return HierarchicalCode(
                indices=normalized,
                tokens=self.indices_to_tokens(normalized),
                codebook_version=self.config.codebook_version,
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a complete, JSON-serializable tokenizer snapshot."""

        with self._lock:
            strategy_state = self._strategy_snapshot()
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "tokenizer_class": type(self).__name__,
                "is_fitted": self._is_fitted,
                "config": self.config.to_dict(),
                "strategy_state": strategy_state,
                "registry": self.registry.snapshot(),
            }
            try:
                json.dumps(payload)
            except (TypeError, ValueError) as exc:
                raise SerializationError(
                    "strategy snapshot must contain only JSON-serializable values"
                ) from exc
            return payload

    def save(self, path: str | Path) -> None:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise SerializationError(f"failed to save snapshot to {target}: {exc}") from exc

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> "BaseSkillTokenizer":
        if not isinstance(payload, Mapping):
            raise SerializationError("tokenizer snapshot must be a mapping")
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise SerializationError(
                f"unsupported tokenizer schema_version: {payload.get('schema_version')!r}"
            )
        try:
            config = TokenizerConfig.from_dict(payload["config"])
            instance = cls(config)
            with instance._lock:
                state = payload.get("strategy_state", {})
                if not isinstance(state, Mapping):
                    raise SerializationError("strategy_state must be a mapping")
                instance._restore_strategy(dict(state))
                instance.registry = SkillRegistry.from_snapshot(
                    config, payload["registry"]
                )
                fitted = payload.get("is_fitted")
                if not isinstance(fitted, bool):
                    raise SerializationError("is_fitted must be a boolean")
                instance._is_fitted = fitted
                if not fitted and len(instance.registry):
                    raise SerializationError(
                        "an unfitted snapshot cannot contain active skills"
                    )
                instance._validate_restored_state()
            return instance
        except SerializationError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise SerializationError(f"invalid tokenizer snapshot: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "BaseSkillTokenizer":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SerializationError(f"failed to load snapshot from {source}: {exc}") from exc
        return cls.from_snapshot(payload)

    @abstractmethod
    def _fit_strategy(
        self, skills: tuple[SkillRecord, ...]
    ) -> Sequence[HierarchicalCode] | Mapping[str, HierarchicalCode]:
        """Fit strategy state and return one full code per input skill."""

    @abstractmethod
    def _encode_new(self, skill: SkillRecord) -> HierarchicalCode:
        """Purely encode an unregistered skill against frozen strategy state."""

    def _on_add(self, skill: SkillRecord, code: HierarchicalCode) -> None:
        """Update optional strategy bookkeeping after a successful add."""

    def _on_remove(self, skill_id: str, code: HierarchicalCode) -> None:
        """Update optional strategy bookkeeping after a successful remove."""

    def _strategy_snapshot(self) -> Mapping[str, Any]:
        return {}

    def _restore_strategy(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise SerializationError(
                f"{type(self).__name__} does not support non-empty strategy state"
            )

    def _validate_restored_state(self) -> None:
        """Cross-check strategy state against the restored common registry."""

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise TokenizerNotFittedError(
                "tokenizer is not fitted; call fit() or restore a snapshot first"
            )

    def _validated_strategy_code(self, code: HierarchicalCode) -> HierarchicalCode:
        try:
            code.validate_against(self.config)
        except (AttributeError, ValueError) as exc:
            raise EncodingError(f"strategy returned an invalid code: {exc}") from exc
        return code

    @staticmethod
    def _validate_skill_batch(
        skills: Sequence[SkillRecord],
    ) -> tuple[SkillRecord, ...]:
        if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
            raise TypeError("fit expects a sequence of SkillRecord objects")
        records = tuple(skills)
        if not records:
            raise TokenizerError("fit requires at least one skill")
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, SkillRecord):
                raise TypeError("fit expects only SkillRecord objects")
            if record.skill_id in seen:
                raise DuplicateSkillError(
                    f"duplicate skill_id in fit input: {record.skill_id!r}"
                )
            seen.add(record.skill_id)
        return records


# Concise public alias; BaseSkillTokenizer remains the explicit implementation
# name used by strategy authors.
SkillTokenizer = BaseSkillTokenizer
