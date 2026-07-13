"""Dynamic mapping between hierarchical codes and active skills."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Any

from .config import TokenizerConfig
from .models import HierarchicalCode, ModelValidationError, SkillRecord


class RegistryError(RuntimeError):
    """Base class for registry failures."""


class DuplicateSkillError(RegistryError):
    """Raised when an add would replace an existing skill implicitly."""


class UnknownSkillError(RegistryError, KeyError):
    """Raised when a requested skill is not active."""


class InvalidCodeError(RegistryError, ValueError):
    """Raised when a code/prefix does not match the configured namespace."""


class BucketCapacityError(RegistryError):
    """Raised when a hard-capacity leaf bucket cannot accept another skill."""


CodePrefix = HierarchicalCode | Sequence[int] | Sequence[str] | None


class SkillRegistry:
    """Thread-safe active registry and constrained-decoding trie.

    Codes are many-to-one by design: a leaf bucket may contain multiple skills.
    Python dict insertion order makes bucket and prefix decode results stable.
    """

    SCHEMA_VERSION = 1

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self._records: dict[str, SkillRecord] = {}
        self._codes: dict[str, HierarchicalCode] = {}
        self._buckets: dict[tuple[int, ...], list[str]] = {}
        self._next_indices: dict[tuple[int, ...], set[int]] = {}
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __contains__(self, skill_id: object) -> bool:
        with self._lock:
            return skill_id in self._records

    @property
    def skill_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def active_codes(self) -> tuple[tuple[int, ...], ...]:
        with self._lock:
            return tuple(self._buckets)

    def get(self, skill_id: str) -> SkillRecord:
        with self._lock:
            try:
                return self._records[skill_id]
            except KeyError as exc:
                raise UnknownSkillError(f"unknown skill_id: {skill_id!r}") from exc

    def code_for(self, skill_id: str) -> HierarchicalCode:
        with self._lock:
            try:
                return self._codes[skill_id]
            except KeyError as exc:
                raise UnknownSkillError(f"unknown skill_id: {skill_id!r}") from exc

    def add(self, skill: SkillRecord, code: HierarchicalCode) -> None:
        """Atomically add an active skill without overwriting existing state."""

        if not isinstance(skill, SkillRecord):
            raise RegistryError("skill must be a SkillRecord")
        self._validate_code(code)
        with self._lock:
            if skill.skill_id in self._records:
                raise DuplicateSkillError(
                    f"skill_id {skill.skill_id!r} is already registered"
                )
            members = self._buckets.get(code.indices, ())
            if (
                self.config.bucket_capacity is not None
                and len(members) >= self.config.bucket_capacity
                and self.config.overflow_policy == "error"
            ):
                raise BucketCapacityError(
                    f"bucket {code.indices!r} reached configured capacity "
                    f"{self.config.bucket_capacity}"
                )
            self._records[skill.skill_id] = skill
            self._codes[skill.skill_id] = code
            self._buckets.setdefault(code.indices, []).append(skill.skill_id)
            for depth, index in enumerate(code.indices):
                prefix = code.indices[:depth]
                self._next_indices.setdefault(prefix, set()).add(index)

    def remove(self, skill_id: str) -> bool:
        """Idempotently remove a skill and hide newly empty paths."""

        with self._lock:
            if skill_id not in self._records:
                return False
            code = self._codes.pop(skill_id)
            del self._records[skill_id]
            members = self._buckets[code.indices]
            members.remove(skill_id)
            if not members:
                del self._buckets[code.indices]
            self._rebuild_trie()
            return True

    def decode(self, prefix: CodePrefix = None) -> tuple[str, ...]:
        """Return all active skills below a full code or a code prefix."""

        indices = self.normalize_prefix(prefix)
        with self._lock:
            if len(indices) == self.config.num_levels:
                return tuple(self._buckets.get(indices, ()))
            return tuple(
                skill_id
                for skill_id, code in self._codes.items()
                if code.indices[: len(indices)] == indices
            )

    def valid_next_indices(self, prefix: CodePrefix = None) -> tuple[int, ...]:
        indices = self.normalize_prefix(prefix)
        if len(indices) >= self.config.num_levels:
            return ()
        with self._lock:
            return tuple(sorted(self._next_indices.get(indices, ())))

    def valid_next_tokens(self, prefix: CodePrefix = None) -> tuple[str, ...]:
        indices = self.normalize_prefix(prefix)
        level = len(indices)
        if level >= self.config.num_levels:
            return ()
        return tuple(
            self.config.token_for(level, index)
            for index in self.valid_next_indices(indices)
        )

    def normalize_prefix(self, prefix: CodePrefix) -> tuple[int, ...]:
        """Validate and convert an integer/token prefix to integer indices."""

        if prefix is None:
            return ()
        if isinstance(prefix, HierarchicalCode):
            self._validate_code(prefix)
            return prefix.indices
        if isinstance(prefix, (str, bytes)) or not isinstance(prefix, Sequence):
            raise InvalidCodeError(
                "prefix must be a sequence of indices/tokens or a HierarchicalCode"
            )
        raw = tuple(prefix)
        if len(raw) > self.config.num_levels:
            raise InvalidCodeError(
                f"prefix has {len(raw)} levels; maximum is {self.config.num_levels}"
            )
        if not raw:
            return ()

        if all(isinstance(value, int) and not isinstance(value, bool) for value in raw):
            indices = tuple(raw)
            for offset, index in enumerate(indices):
                limit = self.config.branching_factors[offset]
                if index < 0 or index >= limit:
                    raise InvalidCodeError(
                        f"index {index} outside branching factor {limit} "
                        f"at level {offset + 1}"
                    )
            return indices

        if all(isinstance(value, str) for value in raw):
            indices_list: list[int] = []
            for offset, token in enumerate(raw):
                namespace = self.config.tokens_for_level(offset)
                try:
                    indices_list.append(namespace.index(token))
                except ValueError as exc:
                    raise InvalidCodeError(
                        f"token {token!r} is not valid at level {offset + 1}"
                    ) from exc
            return tuple(indices_list)

        raise InvalidCodeError("prefix cannot mix integer indices and string tokens")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            entries = [
                {
                    "skill": record.to_dict(),
                    "code": self._codes[skill_id].to_dict(),
                }
                for skill_id, record in self._records.items()
            ]
        return {
            "schema_version": self.SCHEMA_VERSION,
            "codebook_version": self.config.codebook_version,
            "structural_signature": [
                self.config.strategy,
                self.config.num_levels,
                list(self.config.branching_factors),
                self.config.codebook_version,
                self.config.token_format,
            ],
            "entries": entries,
        }

    @classmethod
    def from_snapshot(
        cls, config: TokenizerConfig, payload: Mapping[str, Any]
    ) -> "SkillRegistry":
        if not isinstance(payload, Mapping):
            raise RegistryError("registry snapshot must be a mapping")
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise RegistryError(
                "unsupported registry schema_version: "
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("codebook_version") != config.codebook_version:
            raise RegistryError("registry and config codebook versions do not match")
        expected_signature = [
            config.strategy,
            config.num_levels,
            list(config.branching_factors),
            config.codebook_version,
            config.token_format,
        ]
        if payload.get("structural_signature") != expected_signature:
            raise RegistryError("registry structural signature does not match config")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise RegistryError("registry snapshot entries must be a list")

        registry = cls(config)
        try:
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise RegistryError("every registry entry must be a mapping")
                registry.add(
                    SkillRecord.from_dict(entry["skill"]),
                    HierarchicalCode.from_dict(entry["code"]),
                )
        except (KeyError, ModelValidationError) as exc:
            raise RegistryError(f"invalid registry entry: {exc}") from exc
        return registry

    def _validate_code(self, code: HierarchicalCode) -> None:
        if not isinstance(code, HierarchicalCode):
            raise InvalidCodeError("code must be a HierarchicalCode")
        try:
            code.validate_against(self.config)
        except ModelValidationError as exc:
            raise InvalidCodeError(str(exc)) from exc

    def _rebuild_trie(self) -> None:
        self._next_indices.clear()
        for code in self._buckets:
            for depth, index in enumerate(code):
                self._next_indices.setdefault(code[:depth], set()).add(index)
