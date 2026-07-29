"""Pure-Python utilities for generative skill-router training and inference.

The Hugging Face entry points live under :mod:`scripts`.  This module keeps
artifact parsing, supervision construction, trie constraints, bucket expansion,
and retrieval metrics independent from ``torch`` and ``transformers`` so they
can be validated in a lightweight test environment.
"""

from __future__ import annotations

import json
import hashlib
import math
import random
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODE_PATH_SEPARATOR = "\n"


class RouterDataError(ValueError):
    """Raised when router artifacts violate their shared schema."""


def canonical_query_group(query: str) -> str:
    """Group duplicate query texts so they cannot cross train/validation."""

    normalized = " ".join(unicodedata.normalize("NFKC", query).casefold().split())
    if not normalized:
        raise RouterDataError("query text is empty after normalization")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"query-text:{digest}"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, rejecting non-object rows with a useful location."""

    rows: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RouterDataError(
                    f"invalid JSON at {source}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RouterDataError(
                    f"expected a JSON object at {source}:{line_number}"
                )
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSON objects as UTF-8 JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_virtual_tokens(path: str | Path) -> tuple[str, ...]:
    """Load the complete hierarchical-token namespace, preserving file order."""

    seen: set[str] = set()
    tokens: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            token = line.strip()
            if not token:
                continue
            if any(character.isspace() for character in token):
                raise RouterDataError(
                    f"virtual token at line {line_number} contains whitespace"
                )
            if token in seen:
                raise RouterDataError(f"duplicate virtual token: {token!r}")
            seen.add(token)
            tokens.append(token)
    if not tokens:
        raise RouterDataError("virtual token file is empty")
    return tuple(tokens)


def _nonempty_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_code_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], int]:
    """Normalize Stage-1 ``*_codes.jsonl`` into ``skill_id -> token path``.

    The frozen Stage-1 contract uses ``skill_id`` and ``tokens``.  A small set
    of aliases is accepted to make older experimental artifacts diagnosable,
    but integer-only paths are deliberately rejected: Stage 2 must consume the
    exact stable token strings from the index manifest.
    """

    skill_to_code: dict[str, tuple[str, ...]] = {}
    num_levels: int | None = None
    for row_number, row in enumerate(rows, start=1):
        skill_id = _nonempty_text(row, "skill_id", "id", "item_id")
        if not skill_id:
            raise RouterDataError(f"code row {row_number} has no skill_id")
        raw_tokens = row.get("tokens", row.get("code_tokens"))
        if not isinstance(raw_tokens, (list, tuple)) or not raw_tokens:
            raise RouterDataError(
                f"code row {row_number} for {skill_id!r} has no token path"
            )
        tokens = tuple(raw_tokens)
        if any(not isinstance(token, str) or not token for token in tokens):
            raise RouterDataError(
                f"code row {row_number} for {skill_id!r} contains an invalid token"
            )
        if len(set(tokens)) != len(tokens):
            # Reusing a token across levels makes suffix parsing ambiguous and
            # violates the level-specific namespaces used by this project.
            raise RouterDataError(
                f"code row {row_number} for {skill_id!r} reuses a level token"
            )
        if num_levels is None:
            num_levels = len(tokens)
        elif len(tokens) != num_levels:
            raise RouterDataError(
                "all code paths must have the same fixed length: "
                f"expected {num_levels}, got {len(tokens)} for {skill_id!r}"
            )
        if skill_id in skill_to_code:
            raise RouterDataError(f"duplicate code assignment for {skill_id!r}")
        skill_to_code[skill_id] = tokens
    if not skill_to_code or num_levels is None:
        raise RouterDataError("code artifact is empty")
    return skill_to_code, num_levels


def buckets_from_codes(
    skill_to_code: Mapping[str, Sequence[str]],
    active_skill_ids: Iterable[str] | None = None,
) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Build collision-preserving code buckets.

    ``active_skill_ids`` is the runtime deletion filter.  Inactive skills never
    enter the trie, even if their historical assignment remains in codes.jsonl.
    """

    active = set(skill_to_code) if active_skill_ids is None else set(active_skill_ids)
    unknown = active.difference(skill_to_code)
    if unknown:
        preview = ", ".join(sorted(unknown)[:5])
        raise RouterDataError(f"active registry references unknown skills: {preview}")

    grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for skill_id, raw_code in skill_to_code.items():
        if skill_id in active:
            grouped[tuple(raw_code)].append(skill_id)
    return {
        code: tuple(sorted(skill_ids))
        for code, skill_ids in sorted(grouped.items())
        if skill_ids
    }


def active_skill_ids_from_registry(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return active skills from the Stage-1 registry's bucket mapping."""

    buckets = payload.get("buckets")
    if not isinstance(buckets, Mapping):
        raise RouterDataError("registry must contain a buckets mapping")
    active: list[str] = []
    seen: set[str] = set()
    for key, members in buckets.items():
        if not isinstance(key, str) or not isinstance(members, list):
            raise RouterDataError("registry bucket keys must be strings and values lists")
        for member in members:
            if not isinstance(member, str) or not member:
                raise RouterDataError(f"invalid skill id in registry bucket {key!r}")
            if member in seen:
                raise RouterDataError(f"skill {member!r} appears in multiple buckets")
            seen.add(member)
            active.append(member)
    return tuple(active)


def validate_registry_assignments(
    payload: Mapping[str, Any],
    code_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Cross-check ``i/j/...`` registry buckets against code-row indices."""

    assignments: dict[str, tuple[int, ...]] = {}
    for row_number, row in enumerate(code_rows, start=1):
        skill_id = _nonempty_text(row, "skill_id", "id", "item_id")
        indices = row.get("indices")
        if not skill_id or not isinstance(indices, (list, tuple)) or not indices:
            raise RouterDataError(
                f"code row {row_number} lacks skill_id or integer indices"
            )
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices):
            raise RouterDataError(f"invalid code indices for skill {skill_id!r}")
        assignments[skill_id] = tuple(indices)

    buckets = payload.get("buckets")
    if not isinstance(buckets, Mapping):
        raise RouterDataError("registry must contain a buckets mapping")
    for raw_key, members in buckets.items():
        if not isinstance(raw_key, str) or not isinstance(members, list):
            raise RouterDataError("invalid registry bucket")
        try:
            indices = tuple(int(part) for part in raw_key.split("/"))
        except ValueError as exc:
            raise RouterDataError(
                f"registry bucket key is not an i/j/... path: {raw_key!r}"
            ) from exc
        if not indices or any(index < 0 for index in indices):
            raise RouterDataError(f"invalid registry bucket key: {raw_key!r}")
        for skill_id in members:
            assigned = assignments.get(skill_id)
            if assigned is None:
                raise RouterDataError(
                    f"registry references skill {skill_id!r} absent from code rows"
                )
            if assigned != indices:
                raise RouterDataError(
                    f"registry puts {skill_id!r} in {indices}, but codes assign {assigned}"
                )


def qrels_by_query(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Group positive qrels in source order, ignoring non-relevant rows."""

    grouped: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row_number, row in enumerate(rows, start=1):
        query_id = _nonempty_text(row, "query_id", "qid")
        skill_id = _nonempty_text(row, "skill_id", "doc_id")
        if not query_id or not skill_id:
            raise RouterDataError(f"qrel row {row_number} lacks query_id or skill_id")
        relevance = row.get("relevance", 1)
        try:
            is_positive = float(relevance) > 0
        except (TypeError, ValueError) as exc:
            raise RouterDataError(
                f"qrel row {row_number} has invalid relevance {relevance!r}"
            ) from exc
        if is_positive and skill_id not in seen[query_id]:
            grouped[query_id].append(skill_id)
            seen[query_id].add(skill_id)
    return {
        query_id: tuple(skill_ids)
        for query_id, skill_ids in grouped.items()
    }


def build_retrieval_examples(
    queries: Iterable[Mapping[str, Any]],
    skill_to_code: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build one ordered, newline-delimited target per multi-positive query.

    Each distinct positive code path occupies one output line. Skills colliding
    in one bucket share that line. Positive order is preserved because the
    sequence represents a skill chain rather than an unordered label set.
    """

    examples: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for row_number, query_row in enumerate(queries, start=1):
        query_id = _nonempty_text(query_row, "query_id", "id")
        query = _nonempty_text(query_row, "query", "input_text", "instruction")
        if not query_id or not query:
            raise RouterDataError(f"query row {row_number} lacks id or query text")
        if query_id in seen_query_ids:
            raise RouterDataError(f"duplicate query id: {query_id!r}")
        seen_query_ids.add(query_id)

        if qrels is not None:
            positive_ids = tuple(qrels.get(query_id, ()))
        else:
            raw_ids = query_row.get("skill_ids", ())
            if not isinstance(raw_ids, (list, tuple)):
                raise RouterDataError(
                    f"query {query_id!r} has a non-list skill_ids field"
                )
            positive_ids = tuple(str(value) for value in raw_ids)
        if not positive_ids:
            continue

        code_to_skills: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for skill_id in positive_ids:
            if skill_id not in skill_to_code:
                raise RouterDataError(
                    f"query {query_id!r} references skill {skill_id!r} without a code"
                )
            code_to_skills[tuple(skill_to_code[skill_id])].append(skill_id)

        all_positive_ids = list(dict.fromkeys(positive_ids))
        target_paths = [list(code) for code in code_to_skills]
        examples.append(
            {
                "phase": "retrieval",
                "group_id": canonical_query_group(query),
                "query_id": query_id,
                "input_text": query,
                "target_paths": target_paths,
                "target_text": CODE_PATH_SEPARATOR.join(
                    "".join(path) for path in target_paths
                ),
                "target_skill_ids": all_positive_ids,
                "path_skill_ids": [
                    list(dict.fromkeys(code_to_skills[code]))
                    for code in code_to_skills
                ],
                "positive_skill_ids": all_positive_ids,
            }
        )
    return examples


def build_closed_set_evaluation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_skill_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize unique queries/qrels from held-out retrieval SFT rows.

    New router data already has one SFT row per query. Older expanded artifacts
    are also accepted and collapsed, preserving the exact query-group split.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        query_id = _nonempty_text(row, "query_id", "id")
        query = _nonempty_text(row, "input_text", "query")
        if not query_id or not query:
            raise RouterDataError(
                f"closed-set validation row {row_number} lacks query_id or input_text"
            )
        raw_positives = row.get("positive_skill_ids")
        if not isinstance(raw_positives, (list, tuple)) or not raw_positives:
            raise RouterDataError(
                f"closed-set validation query {query_id!r} has no positive_skill_ids"
            )
        positives = {str(value).strip() for value in raw_positives if str(value).strip()}
        if not positives:
            raise RouterDataError(
                f"closed-set validation query {query_id!r} has no valid positive skills"
            )
        if allowed_skill_ids is not None:
            unknown = positives.difference(allowed_skill_ids)
            if unknown:
                raise RouterDataError(
                    f"closed-set validation query {query_id!r} references a skill "
                    f"outside the candidate corpus: {next(iter(sorted(unknown)))}"
                )

        previous = grouped.get(query_id)
        if previous is None:
            grouped[query_id] = {"query": query, "positive_skill_ids": positives}
            continue
        if previous["query"] != query:
            raise RouterDataError(
                f"closed-set validation query {query_id!r} has inconsistent text"
            )
        if previous["positive_skill_ids"] != positives:
            raise RouterDataError(
                f"closed-set validation query {query_id!r} has inconsistent positives"
            )

    if not grouped:
        raise RouterDataError("closed-set validation artifact is empty")

    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    for query_id in sorted(grouped):
        details = grouped[query_id]
        positives = sorted(details["positive_skill_ids"])
        queries.append(
            {
                "id": query_id,
                "query": details["query"],
                "skill_ids": positives,
            }
        )
        qrels.extend(
            {
                "query_id": query_id,
                "skill_id": skill_id,
                "relevance": 1,
            }
            for skill_id in positives
        )
    return queries, qrels


def skill_document_text(skill: Mapping[str, Any]) -> str:
    """Render the SkillRet document text used for memorization alignment."""

    explicit = _nonempty_text(skill, "text", "document_text")
    if explicit:
        return explicit
    name = _nonempty_text(skill, "name")
    description = _nonempty_text(skill, "description")
    body = _nonempty_text(skill, "skill_md", "body")
    return " | ".join(value for value in (name, description, body) if value)


def build_memorization_examples(
    skills: Iterable[Mapping[str, Any]],
    skill_to_code: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Build document-to-code examples for ToolWeaver-style memorization."""

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, skill in enumerate(skills, start=1):
        skill_id = _nonempty_text(skill, "skill_id", "id", "item_id")
        if not skill_id:
            raise RouterDataError(f"catalog row {row_number} has no skill_id")
        if skill_id in seen:
            raise RouterDataError(f"duplicate catalog skill: {skill_id!r}")
        seen.add(skill_id)
        if skill_id not in skill_to_code:
            continue
        text = skill_document_text(skill)
        if not text:
            raise RouterDataError(f"catalog skill {skill_id!r} has no document text")
        code = tuple(skill_to_code[skill_id])
        examples.append(
            {
                "phase": "memorization",
                "group_id": skill_id,
                "skill_id": skill_id,
                "input_text": text,
                "target_paths": [list(code)],
                "target_tokens": list(code),
                "target_text": "".join(code),
                "target_skill_ids": [skill_id],
            }
        )
    return examples


def grouped_train_validation_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
    group_key: str = "group_id",
    preserve_target_key: str | None = None,
    min_train_target_groups: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by group while optionally preserving target coverage in train."""

    if not 0.0 <= validation_fraction < 1.0:
        raise RouterDataError("validation_fraction must be in [0, 1)")
    if min_train_target_groups < 1:
        raise RouterDataError("min_train_target_groups must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        group = row.get(group_key)
        if not isinstance(group, str) or not group:
            raise RouterDataError(f"row {row_number} has no non-empty {group_key}")
        groups[group].append(dict(row))
    group_ids = sorted(groups)
    if not group_ids:
        return [], []

    shuffled = group_ids.copy()
    random.Random(seed).shuffle(shuffled)
    if validation_fraction == 0.0 or len(shuffled) == 1:
        validation_count = 0
    else:
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation_count = min(validation_count, len(shuffled) - 1)
    if preserve_target_key is None:
        validation_groups = set(shuffled[:validation_count])
    else:
        group_targets: dict[str, set[str]] = {}
        target_group_counts: Counter[str] = Counter()
        for group_id, group_rows in groups.items():
            targets: set[str] = set()
            for row in group_rows:
                raw_targets = row.get(preserve_target_key)
                if not isinstance(raw_targets, (list, tuple, set)) or not raw_targets:
                    raise RouterDataError(
                        f"group {group_id!r} has no non-empty {preserve_target_key}"
                    )
                for target in raw_targets:
                    if not isinstance(target, str) or not target:
                        raise RouterDataError(
                            f"group {group_id!r} has an invalid target in {preserve_target_key}"
                        )
                    targets.add(target)
            group_targets[group_id] = targets
            target_group_counts.update(targets)

        remaining = target_group_counts.copy()
        validation_groups: set[str] = set()
        for group_id in shuffled:
            if len(validation_groups) >= validation_count:
                break
            targets = group_targets[group_id]
            if any(
                remaining[target] - 1 < min_train_target_groups
                for target in targets
            ):
                continue
            validation_groups.add(group_id)
            remaining.subtract(targets)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group_id in group_ids:
        destination = validation if group_id in validation_groups else train
        destination.extend(groups[group_id])
    return train, validation


def mix_replay_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    replay_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Mix one deterministic, repeat-sampled replay source into primary rows."""

    mixed, counts = mix_replay_sources(
        primary_rows,
        (("replay", replay_rows, replay_fraction),),
        seed=seed,
    )
    return mixed, counts["replay"]


def mix_replay_sources(
    primary_rows: Sequence[Mapping[str, Any]],
    replay_sources: Sequence[
        tuple[str, Sequence[Mapping[str, Any]], float]
    ],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Mix named replay sources to requested final-dataset fractions.

    Each source is shuffled and cycled as many times as necessary, so a small
    replay dataset can still reach its configured fraction. The allocation and
    final row order are deterministic for a fixed seed.
    """

    primary = [dict(row) for row in primary_rows]
    normalized_sources: list[
        tuple[str, Sequence[Mapping[str, Any]], float]
    ] = []
    counts: dict[str, int] = {}
    seen_names: set[str] = set()
    for name, rows, fraction in replay_sources:
        if not name or name in seen_names:
            raise RouterDataError(f"replay source name must be unique: {name!r}")
        seen_names.add(name)
        if not 0.0 <= fraction < 1.0:
            raise RouterDataError(
                f"{name} replay fraction must be in [0, 1)"
            )
        if fraction > 0.0 and not rows:
            raise RouterDataError(
                f"{name} replay fraction is positive but replay data is empty"
            )
        normalized_sources.append((name, rows, fraction))
        counts[name] = 0

    total_fraction = sum(source[2] for source in normalized_sources)
    if total_fraction >= 1.0:
        raise RouterDataError("total replay fraction must be less than 1")
    if total_fraction == 0.0:
        return primary, counts
    if not primary:
        raise RouterDataError("primary replay mixture is empty")

    requested_total = max(
        1,
        round(len(primary) * total_fraction / (1.0 - total_fraction)),
    )
    exact_counts = [
        requested_total * fraction / total_fraction
        for _, _, fraction in normalized_sources
    ]
    allocated = [math.floor(value) for value in exact_counts]
    remaining = requested_total - sum(allocated)
    allocation_order = sorted(
        range(len(normalized_sources)),
        key=lambda index: (
            -(exact_counts[index] - allocated[index]),
            index,
        ),
    )
    for index in allocation_order[:remaining]:
        allocated[index] += 1

    selected: list[dict[str, Any]] = []
    for source_index, ((name, rows, _), requested) in enumerate(
        zip(normalized_sources, allocated, strict=True)
    ):
        counts[name] = requested
        if requested == 0:
            continue
        source = [dict(row) for row in rows]
        source_random = random.Random(seed + source_index)
        while requested > 0:
            cycle = [dict(row) for row in source]
            source_random.shuffle(cycle)
            take = min(requested, len(cycle))
            selected.extend(cycle[:take])
            requested -= take

    mixed = [*primary, *selected]
    random.Random(seed + 1).shuffle(mixed)
    return mixed, counts


class TokenTrie:
    """Trie over active, fixed-length code-token paths."""

    _LEAF = object()

    def __init__(self, paths: Iterable[Sequence[int]], *, eos_token_id: int) -> None:
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise RouterDataError("eos_token_id must be a non-negative integer")
        normalized = sorted(set(tuple(path) for path in paths))
        if not normalized:
            raise RouterDataError("cannot build a trie without active paths")
        self.num_levels = len(normalized[0])
        if self.num_levels < 1:
            raise RouterDataError("code paths cannot be empty")
        self.eos_token_id = eos_token_id
        self._paths = frozenset(normalized)
        self._root: dict[Any, Any] = {}
        for path in normalized:
            if len(path) != self.num_levels:
                raise RouterDataError("all trie paths must have the same fixed length")
            if any(not isinstance(token_id, int) or token_id < 0 for token_id in path):
                raise RouterDataError("trie token ids must be non-negative integers")
            node = self._root
            for token_id in path:
                node = node.setdefault(token_id, {})
            node[self._LEAF] = True

    @property
    def paths(self) -> frozenset[tuple[int, ...]]:
        return self._paths

    def allowed_next(self, generated: Sequence[int]) -> tuple[int, ...]:
        """Return valid next ids; only EOS is legal after exactly ``L`` ids."""

        prefix = tuple(int(value) for value in generated)
        if len(prefix) > self.num_levels:
            return ()
        node = self._root
        for token_id in prefix:
            child = node.get(token_id)
            if not isinstance(child, dict):
                return ()
            node = child
        if len(prefix) == self.num_levels:
            return (self.eos_token_id,) if self._LEAF in node else ()
        return tuple(sorted(key for key in node if key is not self._LEAF))

    def is_active_path(self, path: Sequence[int]) -> bool:
        return tuple(path) in self._paths


class MultiPathTokenTrie:
    """Grammar for ``path (separator path)* EOS`` constrained decoding.

    Completed paths cannot be repeated. At a path boundary the model decides
    autoregressively between EOS and the configured textual separator.
    """

    def __init__(
        self,
        paths: Iterable[Sequence[int]],
        *,
        eos_token_id: int,
        separator_token_ids: Sequence[int],
        max_paths: int,
    ) -> None:
        if (
            not isinstance(max_paths, int)
            or isinstance(max_paths, bool)
            or max_paths < 1
        ):
            raise RouterDataError("max_paths must be a positive integer")
        separator = tuple(int(value) for value in separator_token_ids)
        if not separator or any(value < 0 for value in separator):
            raise RouterDataError(
                "separator_token_ids must be non-empty and non-negative"
            )
        if eos_token_id in separator:
            raise RouterDataError("path separator cannot contain EOS")
        self.path_trie = TokenTrie(paths, eos_token_id=eos_token_id)
        self.eos_token_id = eos_token_id
        self.separator_token_ids = separator
        self.max_paths = min(max_paths, len(self.path_trie.paths))

    @property
    def num_levels(self) -> int:
        return self.path_trie.num_levels

    @property
    def paths(self) -> frozenset[tuple[int, ...]]:
        return self.path_trie.paths

    def _state(
        self, generated: Sequence[int]
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], str, int] | None:
        completed: list[tuple[int, ...]] = []
        prefix: list[int] = []
        mode = "path"
        separator_offset = 0
        for raw_token_id in generated:
            token_id = int(raw_token_id)
            if token_id == self.eos_token_id:
                return None
            if mode == "path":
                prefix.append(token_id)
                prefix_tuple = tuple(prefix)
                if not any(
                    path not in completed and path[: len(prefix_tuple)] == prefix_tuple
                    for path in self.paths
                ):
                    return None
                if len(prefix) == self.num_levels:
                    completed.append(prefix_tuple)
                    prefix = []
                    mode = "boundary"
            elif mode == "boundary":
                if token_id != self.separator_token_ids[0]:
                    return None
                if len(self.separator_token_ids) == 1:
                    mode = "path"
                else:
                    mode = "separator"
                    separator_offset = 1
            else:
                if token_id != self.separator_token_ids[separator_offset]:
                    return None
                separator_offset += 1
                if separator_offset == len(self.separator_token_ids):
                    mode = "path"
        return tuple(completed), tuple(prefix), mode, separator_offset

    def allowed_next(self, generated: Sequence[int]) -> tuple[int, ...]:
        state = self._state(generated)
        if state is None:
            return ()
        completed, prefix, mode, separator_offset = state
        if mode == "separator":
            return (self.separator_token_ids[separator_offset],)
        if mode == "boundary":
            if len(completed) >= self.max_paths or len(completed) >= len(self.paths):
                return (self.eos_token_id,)
            return tuple(sorted({self.eos_token_id, self.separator_token_ids[0]}))

        candidates = [
            path
            for path in self.paths
            if path not in completed and path[: len(prefix)] == prefix
        ]
        if not candidates or len(prefix) >= self.num_levels:
            return ()
        return tuple(sorted({path[len(prefix)] for path in candidates}))

    def parse_complete(self, generated: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        """Parse tokens before EOS and require a complete final path."""

        state = self._state(generated)
        if state is None:
            raise RouterDataError("generated multi-path sequence is invalid")
        completed, prefix, mode, _ = state
        if (
            not completed
            or len(completed) > self.max_paths
            or prefix
            or mode != "boundary"
        ):
            raise RouterDataError("generated sequence does not end at a path boundary")
        return completed


@dataclass(frozen=True, slots=True)
class GeneratedPath:
    """One path from an ordered autoregressive result before bucket expansion."""

    tokens: tuple[str, ...]
    score: float


def rank_bucket_candidates(
    paths: Sequence[GeneratedPath],
    buckets: Mapping[tuple[str, ...], Sequence[str]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Expand ordered paths without losing skills in a collision bucket."""

    best: dict[str, tuple[float, int, tuple[str, ...]]] = {}
    for path_rank, generated in enumerate(paths):
        members = buckets.get(generated.tokens, ())
        for skill_id in members:
            previous = best.get(skill_id)
            candidate = (float(generated.score), path_rank, generated.tokens)
            if previous is None or candidate[0] > previous[0]:
                best[skill_id] = candidate
    ordered = sorted(best.items(), key=lambda item: (item[1][1], item[0]))
    if limit is not None:
        if limit < 1:
            raise RouterDataError("candidate limit must be positive")
        ordered = ordered[:limit]
    return [
        {
            "skill_id": skill_id,
            "score": values[0],
            "path_rank": values[1],
            "code_tokens": list(values[2]),
        }
        for skill_id, values in ordered
    ]


def _discounted_gain(relevant_ranks: Iterable[int]) -> float:
    return sum(1.0 / math.log2(rank + 2.0) for rank in relevant_ranks)


def query_retrieval_metrics(
    ranked_skill_ids: Sequence[str],
    relevant_skill_ids: Iterable[str],
    *,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute binary SkillRet metrics for one query.

    ``completeness@k`` is one iff every relevant skill appears in the top ``k``.
    AP@k uses ``min(number_of_relevant, k)`` as denominator, matching the
    bounded retrieval interpretation used by common IR toolkits.
    """

    relevant = set(relevant_skill_ids)
    if not relevant:
        raise RouterDataError("retrieval metrics require at least one relevant skill")
    if len(set(ranked_skill_ids)) != len(ranked_skill_ids):
        raise RouterDataError("ranked_skill_ids must not contain duplicates")
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not normalized_cutoffs or any(k < 1 for k in normalized_cutoffs):
        raise RouterDataError("metric cutoffs must be positive integers")

    metrics: dict[str, float] = {}
    for cutoff in normalized_cutoffs:
        top = list(ranked_skill_ids[:cutoff])
        hits = [index for index, skill_id in enumerate(top) if skill_id in relevant]
        hit_count = len(hits)
        recall = hit_count / len(relevant)
        ideal_count = min(len(relevant), cutoff)
        dcg = _discounted_gain(hits)
        idcg = _discounted_gain(range(ideal_count))
        precision_sum = sum(
            (hit_number + 1) / (rank + 1)
            for hit_number, rank in enumerate(hits)
        )
        reciprocal_rank = 1.0 / (hits[0] + 1) if hits else 0.0
        suffix = f"@{cutoff}"
        metrics[f"recall{suffix}"] = recall
        metrics[f"ndcg{suffix}"] = dcg / idcg if idcg else 0.0
        metrics[f"map{suffix}"] = precision_sum / ideal_count
        metrics[f"mrr{suffix}"] = reciprocal_rank
        metrics[f"completeness{suffix}"] = float(hit_count == len(relevant))
    return metrics


def query_code_path_metrics(
    ranked_code_paths: Sequence[Sequence[str]],
    relevant_skill_ids: Iterable[str],
    skill_to_code: Mapping[str, Sequence[str]],
    buckets: Mapping[tuple[str, ...], Sequence[str]],
    *,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Evaluate code paths without arbitrary ordering inside collision buckets.

    ``code_recall@k`` measures distinct relevant paths among the top-k generated
    paths. ``bucket_recall@k`` expands those paths fully before comparing skill
    IDs, so it remains invariant to the tie-break used for skills sharing a code.
    """

    raw_relevant = list(relevant_skill_ids)
    if isinstance(relevant_skill_ids, (set, frozenset)):
        raw_relevant = sorted(raw_relevant)
    ordered_relevant = list(dict.fromkeys(raw_relevant))
    relevant = set(ordered_relevant)
    if not relevant:
        raise RouterDataError("code-path metrics require at least one relevant skill")
    unknown = relevant.difference(skill_to_code)
    if unknown:
        raise RouterDataError(
            "relevant skills have no code assignment: " + ", ".join(sorted(unknown)[:5])
        )
    ranked = [tuple(path) for path in ranked_code_paths]
    if len(set(ranked)) != len(ranked):
        raise RouterDataError("ranked_code_paths must not contain duplicates")
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not normalized_cutoffs or any(k < 1 for k in normalized_cutoffs):
        raise RouterDataError("metric cutoffs must be positive integers")

    relevant_paths = {tuple(skill_to_code[skill_id]) for skill_id in relevant}
    ordered_relevant_paths = list(
        dict.fromkeys(tuple(skill_to_code[skill_id]) for skill_id in ordered_relevant)
    )
    metrics: dict[str, float] = {
        "ordered_code_exact_match": float(ranked == ordered_relevant_paths),
        "code_count_exact_match": float(len(ranked) == len(ordered_relevant_paths)),
        "code_count_absolute_error": float(
            abs(len(ranked) - len(ordered_relevant_paths))
        ),
    }
    for cutoff in normalized_cutoffs:
        top_paths = ranked[:cutoff]
        retrieved_relevant_paths = relevant_paths.intersection(top_paths)
        expanded_skills = {
            skill_id
            for path in top_paths
            for skill_id in buckets.get(path, ())
        }
        bucket_hits = len(relevant.intersection(expanded_skills))
        suffix = f"@{cutoff}"
        metrics[f"code_recall{suffix}"] = (
            len(retrieved_relevant_paths) / len(relevant_paths)
        )
        metrics[f"bucket_recall{suffix}"] = bucket_hits / len(relevant)
        metrics[f"bucket_completeness{suffix}"] = float(bucket_hits == len(relevant))
    return metrics


def aggregate_retrieval_metrics(
    per_query: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Macro-average per-query metrics."""

    if not per_query:
        raise RouterDataError("cannot aggregate an empty metric collection")
    keys = set(per_query[0])
    if any(set(row) != keys for row in per_query):
        raise RouterDataError("all per-query metric rows must have identical keys")
    return {
        key: sum(float(row[key]) for row in per_query) / len(per_query)
        for key in sorted(keys)
    }


def render_router_prompt(tokenizer: Any, input_text: str, system_prompt: str = "") -> str:
    """Render a generation prompt using the model's chat template when present."""

    if not isinstance(input_text, str) or not input_text.strip():
        raise RouterDataError("input_text must be non-empty")
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_text.strip()})
    chat_template = getattr(tokenizer, "chat_template", None)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if chat_template and callable(apply_template):
        # Qwen3 enables a thinking prefix by default. The router is supervised
        # and constrained to emit a code token immediately, so reasoning text is
        # intentionally disabled. Other Hugging Face templates ignore this
        # template variable when it is not used.
        return apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    system = f"System: {system_prompt.strip()}\n" if system_prompt else ""
    return f"{system}User: {input_text.strip()}\nAssistant:"


def code_token_id_map(tokenizer: Any, tokens: Iterable[str]) -> dict[str, int]:
    """Validate that every hierarchical token is atomic for this tokenizer."""

    mapping: dict[str, int] = {}
    used_ids: dict[int, str] = {}
    for token in tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RouterDataError(
                f"hierarchical token {token!r} does not encode to exactly one id"
            )
        token_id = int(ids[0])
        if token_id in used_ids and used_ids[token_id] != token:
            raise RouterDataError(
                f"tokens {used_ids[token_id]!r} and {token!r} share id {token_id}"
            )
        mapping[token] = token_id
        used_ids[token_id] = token
    return mapping


def encode_target_only_example(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    code_token_ids: Mapping[str, int],
    num_levels: int,
    max_length: int,
    system_prompt: str = "",
) -> dict[str, list[int]]:
    """Encode a newline-delimited code sequence with target-only causal loss."""

    input_text = row.get("input_text")
    if not isinstance(input_text, str) or not input_text.strip():
        raise RouterDataError("training row has no input_text")
    raw_paths = row.get("target_paths")
    if raw_paths is None:
        # Backward compatibility for already-built single-path router data.
        raw_paths = [row.get("target_tokens")]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise RouterDataError("training target must contain at least one code path")
    paths: list[list[str]] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, list) or len(raw_path) != num_levels:
            raise RouterDataError(
                f"each training target path must contain exactly {num_levels} "
                "hierarchical tokens"
            )
        if any(not isinstance(token, str) for token in raw_path):
            raise RouterDataError("training target contains a non-string code token")
        paths.append(raw_path)
    if len({tuple(path) for path in paths}) != len(paths):
        raise RouterDataError("training target contains a duplicate code path")
    try:
        encoded_paths = [
            [code_token_ids[token] for token in path]
            for path in paths
        ]
    except (KeyError, TypeError) as exc:
        raise RouterDataError("training target contains an unknown code token") from exc
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise RouterDataError("causal tokenizer must define eos_token_id")
    separator_ids = list(
        tokenizer.encode(
            CODE_PATH_SEPARATOR,
            add_special_tokens=False,
            verbose=False,
        )
    )
    if not separator_ids:
        raise RouterDataError("tokenizer encodes the code-path separator as empty")
    if int(eos_token_id) in separator_ids:
        raise RouterDataError("code-path separator contains the tokenizer EOS token")
    target_ids: list[int] = []
    for path_index, path_ids in enumerate(encoded_paths):
        if path_index:
            target_ids.extend(int(value) for value in separator_ids)
        target_ids.extend(int(value) for value in path_ids)
    target_ids.append(int(eos_token_id))
    if max_length < len(target_ids) + 1:
        raise RouterDataError("max_length is too small for prompt plus supervised target")

    prompt = render_router_prompt(tokenizer, input_text, system_prompt)
    prompt_ids = list(
        tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
    )
    # Preserve the entire supervised suffix.  Right-truncate the prompt so a
    # long SKILL.md cannot discard the high-signal name and description prefix.
    prompt_budget = max_length - len(target_ids)
    prompt_ids = prompt_ids[:prompt_budget]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids.copy()
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
