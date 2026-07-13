"""Append-only taxonomy tokenization with human-readable path metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ..models import HierarchicalCode, SkillRecord
from .base import BaseSkillTokenizer, EncodingError, SerializationError


class TaxonomyEncodingError(EncodingError, ValueError):
    """Raised when a Skill Card cannot be represented by the taxonomy."""


class InterpretableSkillTokenizer(BaseSkillTokenizer):
    """Map taxonomy labels to stable, prefix-local child indices.

    Child slots are append-only. Removing the final skill on a path hides that
    path from the active trie but deliberately does not recycle its meanings.
    """

    STRATEGY = "interpretable"

    def __init__(self, config) -> None:  # type annotation inherited at runtime
        super().__init__(config)
        self._label_to_index: dict[tuple[int, ...], dict[str, int]] = {}

    def _fit_strategy(
        self, skills: tuple[SkillRecord, ...]
    ) -> Mapping[str, HierarchicalCode]:
        # Build into temporary state so a bad Skill Card does not corrupt a
        # previously fitted tokenizer.
        mappings: dict[tuple[int, ...], dict[str, int]] = {}
        assignments: dict[str, HierarchicalCode] = {}
        ordered = sorted(
            skills,
            key=lambda record: (
                tuple(record.hierarchy[: self.config.num_levels]),
                record.skill_id,
            ),
        )
        for skill in ordered:
            labels = self._labels_for(skill)
            indices = self._assign_labels(labels, mappings)
            assignments[skill.skill_id] = self.make_code(indices)
        self._label_to_index = mappings
        return assignments

    def _encode_new(self, skill: SkillRecord) -> HierarchicalCode:
        labels = self._labels_for(skill)
        # Preview against a copy. BaseSkillTokenizer.encode(SkillRecord) must
        # not reserve taxonomy slots; a successful add commits in _on_add.
        preview = {
            prefix: dict(children)
            for prefix, children in self._label_to_index.items()
        }
        indices = self._assign_labels(labels, preview)
        return self.make_code(indices)

    def _on_add(self, skill: SkillRecord, code: HierarchicalCode) -> None:
        committed = self._assign_labels(
            self._labels_for(skill), self._label_to_index
        )
        if committed != code.indices:
            raise TaxonomyEncodingError(
                "taxonomy changed between encoding and registration"
            )

    def explain(
        self,
        code_or_prefix: HierarchicalCode | tuple[int, ...] | tuple[str, ...],
    ) -> tuple[str, ...]:
        """Translate code indices back to taxonomy labels for audit/debugging.

        A shared overflow slot can represent several labels. In that explicit
        mode they are rendered with ``" | "`` rather than pretending the code
        has a unique meaning.
        """

        with self._lock:
            self._require_fitted()
            indices = self.registry.normalize_prefix(code_or_prefix)
            labels: list[str] = []
            for depth, index in enumerate(indices):
                prefix = indices[:depth]
                matching = sorted(
                    label
                    for label, child_index in self._label_to_index.get(
                        prefix, {}
                    ).items()
                    if child_index == index
                )
                if not matching:
                    raise TaxonomyEncodingError(
                        f"no taxonomy label for prefix={prefix!r}, index={index}"
                    )
                labels.append(" | ".join(matching))
            return tuple(labels)

    @property
    def taxonomy(self) -> dict[tuple[int, ...], dict[str, int]]:
        """Return a defensive copy of the append-only taxonomy mapping."""

        with self._lock:
            return {
                prefix: dict(labels)
                for prefix, labels in self._label_to_index.items()
            }

    def _labels_for(self, skill: SkillRecord) -> tuple[str, ...]:
        if len(skill.hierarchy) < self.config.num_levels:
            raise TaxonomyEncodingError(
                f"skill {skill.skill_id!r} hierarchy has {len(skill.hierarchy)} "
                f"levels; expected at least {self.config.num_levels}"
            )
        return tuple(skill.hierarchy[: self.config.num_levels])

    def _assign_labels(
        self,
        labels: tuple[str, ...],
        mappings: dict[tuple[int, ...], dict[str, int]],
    ) -> tuple[int, ...]:
        indices: list[int] = []
        for depth, label in enumerate(labels):
            prefix = tuple(indices)
            children = mappings.setdefault(prefix, {})
            if label in children:
                index = children[label]
            else:
                index = self._allocate_index(prefix, label, children, depth)
                children[label] = index
            indices.append(index)
        return tuple(indices)

    def _allocate_index(
        self,
        prefix: tuple[int, ...],
        label: str,
        children: Mapping[str, int],
        depth: int,
    ) -> int:
        limit = self.config.branching_factors[depth]
        used = set(children.values())
        for index in range(limit):
            if index not in used:
                return index

        if self.config.overflow_policy == "error":
            raise TaxonomyEncodingError(
                f"taxonomy prefix {prefix!r} exceeds branching_factors[{depth}]="
                f"{limit}; create a new codebook version or explicitly allow sharing"
            )

        # Explicit overflow sharing is stable across processes and independent
        # of current bucket occupancy. It never moves an existing label.
        digest = hashlib.blake2b(
            (
                f"{self.config.codebook_version}\0{prefix!r}\0{label}"
            ).encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big") % limit

    def _strategy_snapshot(self) -> Mapping[str, Any]:
        nodes = []
        for prefix in sorted(self._label_to_index, key=lambda p: (len(p), p)):
            labels = self._label_to_index[prefix]
            nodes.append(
                {
                    "prefix": list(prefix),
                    "labels": [
                        {"label": label, "index": index}
                        for label, index in sorted(
                            labels.items(), key=lambda item: (item[1], item[0])
                        )
                    ],
                }
            )
        return {"nodes": nodes}

    def _restore_strategy(self, payload: Mapping[str, Any]) -> None:
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise SerializationError("interpretable strategy_state.nodes must be a list")

        restored: dict[tuple[int, ...], dict[str, int]] = {}
        for node in nodes:
            if not isinstance(node, Mapping):
                raise SerializationError("every taxonomy node must be a mapping")
            try:
                prefix = tuple(node["prefix"])
                labels = node["labels"]
            except (KeyError, TypeError) as exc:
                raise SerializationError(f"invalid taxonomy node: {exc}") from exc
            if len(prefix) >= self.config.num_levels:
                raise SerializationError("taxonomy prefix must be shorter than num_levels")
            # Reuse the registry's namespace validation without requiring the
            # prefix to be active.
            try:
                normalized = self.registry.normalize_prefix(prefix)
            except ValueError as exc:
                raise SerializationError(f"invalid taxonomy prefix: {exc}") from exc
            if not isinstance(labels, list):
                raise SerializationError("taxonomy node labels must be a list")
            if normalized in restored:
                raise SerializationError(f"duplicate taxonomy prefix: {normalized!r}")

            children: dict[str, int] = {}
            limit = self.config.branching_factors[len(normalized)]
            for entry in labels:
                if not isinstance(entry, Mapping):
                    raise SerializationError("taxonomy label entry must be a mapping")
                label = entry.get("label")
                index = entry.get("index")
                if not isinstance(label, str) or not label:
                    raise SerializationError("taxonomy label must be a non-empty string")
                if label in children:
                    raise SerializationError(f"duplicate taxonomy label: {label!r}")
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or index >= limit
                ):
                    raise SerializationError(
                        f"taxonomy child index {index!r} outside [0, {limit})"
                    )
                children[label] = index
            restored[normalized] = children
        self._label_to_index = restored

    def _validate_restored_state(self) -> None:
        if not self._is_fitted:
            if self._label_to_index:
                raise SerializationError(
                    "unfitted interpretable state cannot contain taxonomy nodes"
                )
            return
        if not self._label_to_index:
            raise SerializationError(
                "fitted interpretable state must retain its taxonomy"
            )

        for skill_id in self.registry.skill_ids:
            skill = self.registry.get(skill_id)
            labels = self._labels_for(skill)
            expected: list[int] = []
            for label in labels:
                prefix = tuple(expected)
                children = self._label_to_index.get(prefix)
                if children is None or label not in children:
                    raise SerializationError(
                        f"taxonomy does not encode restored skill {skill_id!r}"
                    )
                expected.append(children[label])
            actual = self.registry.code_for(skill_id).indices
            if tuple(expected) != actual:
                raise SerializationError(
                    f"taxonomy code for restored skill {skill_id!r} does not "
                    "match registry"
                )
