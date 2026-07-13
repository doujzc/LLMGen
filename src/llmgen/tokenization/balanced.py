"""Balanced residual-codebook tokenizer implemented with NumPy.

The expensive balancing path is deliberately confined to :meth:`fit`.
Once fitted, codebooks are frozen and a new skill is encoded by deterministic
nearest-residual lookup, optionally biased by current per-level usage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..models import HierarchicalCode, SkillRecord
from .base import BaseSkillTokenizer, SerializationError


_EPSILON = np.finfo(np.float64).eps


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return row-wise L2 normalized values, preserving all-zero rows."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embedding values must form a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embedding values must contain only finite numbers")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)


def _squared_distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute a numerically stable matrix of squared Euclidean distances."""

    value_norms = np.sum(values * values, axis=1, keepdims=True)
    centroid_norms = np.sum(centroids * centroids, axis=1)[None, :]
    distances = value_norms + centroid_norms - 2.0 * values @ centroids.T
    # Roundoff can produce tiny negative values for identical vectors.
    return np.maximum(distances, 0.0)


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))


def _sinkhorn_transport(
    distances: np.ndarray,
    *,
    temperature: float,
    iterations: int,
) -> np.ndarray:
    """Balance a distance matrix in log space.

    The returned transport has approximately unit row sums and ``N / K``
    column sums.  Log-space normalization remains stable at the low
    temperatures commonly used for codebook assignment.
    """

    costs = np.asarray(distances, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError("Sinkhorn requires a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(costs)):
        raise ValueError("Sinkhorn distances must contain only finite values")
    if temperature <= 0.0:
        raise ValueError("Sinkhorn temperature must be positive")
    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be positive")

    sample_count, centroid_count = costs.shape
    log_plan = -costs / float(temperature)
    # Each sample has mass 1/N and each code has mass 1/K.
    log_row_target = -np.log(float(sample_count))
    log_column_target = -np.log(float(centroid_count))
    for _ in range(iterations):
        log_plan += log_row_target - _logsumexp(log_plan, axis=1)
        log_plan += log_column_target - _logsumexp(log_plan, axis=0)

    return np.exp(log_plan + np.log(float(sample_count)))


def _capacity_constrained_rounding(transport: np.ndarray) -> np.ndarray:
    """Deterministically round a soft plan while enforcing balanced capacity.

    Every sample is assigned exactly once.  Code capacities are either
    ``floor(N/K)`` or ``ceil(N/K)``; when a remainder exists, the codes with
    strongest soft support receive the extra slots.  High-confidence samples
    choose first, and all ties are resolved by stable integer index order.
    """

    plan = np.asarray(transport, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[0] == 0 or plan.shape[1] == 0:
        raise ValueError("rounding requires a non-empty two-dimensional plan")
    if not np.all(np.isfinite(plan)) or np.any(plan < 0.0):
        raise ValueError("transport must contain finite non-negative values")

    sample_count, centroid_count = plan.shape
    base_capacity, remainder = divmod(sample_count, centroid_count)
    capacities = np.full(centroid_count, base_capacity, dtype=np.int64)
    if remainder:
        code_indices = np.arange(centroid_count)
        support = np.max(plan, axis=0)
        extra_order = np.lexsort((code_indices, -support))
        capacities[extra_order[:remainder]] += 1

    if centroid_count == 1:
        return np.zeros(sample_count, dtype=np.int64)

    # The margin prioritizes assignments that are expensive to redirect.
    best_two = np.partition(plan, kth=centroid_count - 2, axis=1)[:, -2:]
    confidence = best_two[:, 1] - best_two[:, 0]
    maximum = np.max(plan, axis=1)
    sample_indices = np.arange(sample_count)
    sample_order = np.lexsort((sample_indices, -maximum, -confidence))

    assignments = np.full(sample_count, -1, dtype=np.int64)
    remaining = capacities.copy()
    for sample_index in sample_order:
        available = np.flatnonzero(remaining > 0)
        # np.argmax returns the first maximum, making index ties deterministic.
        chosen = int(available[np.argmax(plan[sample_index, available])])
        assignments[sample_index] = chosen
        remaining[chosen] -= 1

    if np.any(assignments < 0) or np.any(remaining != 0):  # defensive invariant
        raise RuntimeError("capacity-constrained rounding failed to fill all slots")
    return assignments


def _initialize_centroids(
    values: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Deterministic-for-a-seed k-means++ initialization."""

    sample_count, dimension = values.shape
    centroids = np.empty((count, dimension), dtype=np.float64)
    selected: list[int] = []
    first = int(rng.integers(sample_count))
    centroids[0] = values[first]
    selected.append(first)
    closest = _squared_distances(values, centroids[:1])[:, 0]

    for offset in range(1, count):
        total = float(np.sum(closest))
        if total > _EPSILON:
            index = int(rng.choice(sample_count, p=closest / total))
        else:
            unused = [index for index in range(sample_count) if index not in selected]
            index = unused[0] if unused else offset % sample_count
        centroids[offset] = values[index]
        selected.append(index)
        closest = np.minimum(
            closest, _squared_distances(values, centroids[offset : offset + 1])[:, 0]
        )
    return centroids


def _updated_centroids(
    values: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Update means and deterministically reseed unused vanilla-kmeans codes."""

    updated = centroids.copy()
    reconstruction_error = distances[np.arange(len(values)), assignments]
    farthest = np.lexsort((np.arange(len(values)), -reconstruction_error))
    empty_offset = 0
    for index in range(len(centroids)):
        members = values[assignments == index]
        if len(members):
            updated[index] = np.mean(members, axis=0)
        else:
            updated[index] = values[farthest[empty_offset % len(values)]]
            empty_offset += 1
    return updated


def _fit_residual_codebook(
    residuals: np.ndarray,
    *,
    centroid_count: int,
    balanced: bool,
    temperature: float,
    sinkhorn_iterations: int,
    clustering_iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one residual centroid layer and return centroids plus hard codes."""

    centroids = _initialize_centroids(residuals, centroid_count, rng)
    previous: np.ndarray | None = None
    assignments = np.zeros(len(residuals), dtype=np.int64)

    for _ in range(clustering_iterations):
        distances = _squared_distances(residuals, centroids)
        if balanced:
            transport = _sinkhorn_transport(
                distances,
                temperature=temperature,
                iterations=sinkhorn_iterations,
            )
            assignments = _capacity_constrained_rounding(transport)
        else:
            assignments = np.argmin(distances, axis=1).astype(np.int64)

        updated = _updated_centroids(residuals, centroids, assignments, distances)
        converged = previous is not None and np.array_equal(assignments, previous)
        centroids = updated
        previous = assignments.copy()
        if converged:
            break

    return centroids, assignments


class BalancedSkillTokenizer(BaseSkillTokenizer):
    """ToolWeaver-inspired balanced multi-level residual tokenizer."""

    STRATEGY = "balanced"
    STATE_VERSION = 1

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._codebooks: tuple[np.ndarray, ...] = ()
        self._usage: tuple[np.ndarray, ...] = ()
        self._semantic_dim: int | None = None
        self._collaborative_dim: int | None = None

    @property
    def codebooks(self) -> tuple[np.ndarray, ...]:
        """Return defensive copies of the frozen codebooks."""

        with self._lock:
            return tuple(codebook.copy() for codebook in self._codebooks)

    @property
    def usage_counts(self) -> tuple[tuple[int, ...], ...]:
        with self._lock:
            return tuple(
                tuple(int(value) for value in level) for level in self._usage
            )

    def _fit_strategy(
        self, skills: tuple[SkillRecord, ...]
    ) -> Sequence[HierarchicalCode]:
        features, semantic_dim, collaborative_dim = self._features_for_fit(skills)
        rng = np.random.default_rng(self.config.random_seed)
        residuals = features.copy()
        code_columns: list[np.ndarray] = []
        codebooks: list[np.ndarray] = []
        usage: list[np.ndarray] = []

        for level, centroid_count in enumerate(self.config.branching_factors):
            balanced = (
                self.config.balance_scope == "all"
                or level == self.config.num_levels - 1
            )
            codebook, assignments = _fit_residual_codebook(
                residuals,
                centroid_count=centroid_count,
                balanced=balanced,
                temperature=float(self.config.sinkhorn_temperature),
                sinkhorn_iterations=self.config.sinkhorn_iterations,
                clustering_iterations=self.config.clustering_iterations,
                rng=rng,
            )
            codebook.setflags(write=False)
            codebooks.append(codebook)
            code_columns.append(assignments)
            usage.append(np.bincount(assignments, minlength=centroid_count).astype(np.int64))
            residuals = residuals - codebook[assignments]

        self._codebooks = tuple(codebooks)
        self._usage = tuple(usage)
        self._semantic_dim = semantic_dim
        self._collaborative_dim = collaborative_dim

        matrix = np.column_stack(code_columns)
        return tuple(self.make_code(tuple(int(value) for value in row)) for row in matrix)

    def _encode_new(self, skill: SkillRecord) -> HierarchicalCode:
        if not self._codebooks or self._semantic_dim is None:
            raise ValueError("balanced tokenizer has no fitted codebooks")
        residual = self._feature_for_skill(skill)
        indices: list[int] = []
        penalty_weight = float(self.config.dynamic_balance_weight)
        for codebook, usage in zip(self._codebooks, self._usage, strict=True):
            distances = _squared_distances(residual[None, :], codebook)[0]
            usage_total = int(np.sum(usage))
            usage_ratio = usage.astype(np.float64) / max(usage_total, 1)
            scores = distances + penalty_weight * usage_ratio
            index = int(np.argmin(scores))
            indices.append(index)
            residual = residual - codebook[index]
        return self.make_code(indices)

    def _on_add(self, skill: SkillRecord, code: HierarchicalCode) -> None:
        for level, index in enumerate(code.indices):
            self._usage[level][index] += 1

    def _on_remove(self, skill_id: str, code: HierarchicalCode) -> None:
        if any(self._usage[level][index] <= 0 for level, index in enumerate(code.indices)):
            raise RuntimeError(f"usage underflow while removing {skill_id!r}")
        for level, index in enumerate(code.indices):
            self._usage[level][index] -= 1

    def _features_for_fit(
        self, skills: tuple[SkillRecord, ...]
    ) -> tuple[np.ndarray, int, int]:
        semantic_dim = len(skills[0].embedding)
        if semantic_dim == 0:
            raise ValueError("balanced tokenizer requires a non-empty embedding")
        for skill in skills:
            if len(skill.embedding) != semantic_dim:
                raise ValueError("all embedding vectors must have the same dimension")

        collaborative_dims = {
            len(skill.collaborative_embedding)
            for skill in skills
            if skill.collaborative_embedding
        }
        if len(collaborative_dims) > 1:
            raise ValueError(
                "all non-empty collaborative_embedding vectors must have the same dimension"
            )
        collaborative_dim = next(iter(collaborative_dims), 0)
        semantic = _normalize_rows(
            np.asarray([skill.embedding for skill in skills], dtype=np.float64)
        )
        alpha = float(self.config.collaborative_weight)
        if alpha == 1.0 and collaborative_dim == 0:
            raise ValueError(
                "collaborative_weight=1 requires collaborative_embedding data"
            )
        parts = [np.sqrt(1.0 - alpha) * semantic]
        if collaborative_dim:
            collaborative = np.zeros((len(skills), collaborative_dim), dtype=np.float64)
            for row, skill in enumerate(skills):
                if skill.collaborative_embedding:
                    if len(skill.collaborative_embedding) != collaborative_dim:
                        raise ValueError(
                            "all non-empty collaborative_embedding vectors must have "
                            "the same dimension"
                        )
                    collaborative[row] = skill.collaborative_embedding
            parts.append(np.sqrt(alpha) * _normalize_rows(collaborative))
        return np.concatenate(parts, axis=1), semantic_dim, collaborative_dim

    def _feature_for_skill(self, skill: SkillRecord) -> np.ndarray:
        assert self._semantic_dim is not None
        collaborative_dim = self._collaborative_dim or 0
        if len(skill.embedding) != self._semantic_dim:
            raise ValueError(
                f"embedding dimension {len(skill.embedding)} does not match fitted "
                f"dimension {self._semantic_dim}"
            )
        semantic = _normalize_rows(np.asarray([skill.embedding], dtype=np.float64))[0]
        alpha = float(self.config.collaborative_weight)
        parts = [np.sqrt(1.0 - alpha) * semantic]
        if collaborative_dim:
            if (
                skill.collaborative_embedding
                and len(skill.collaborative_embedding) != collaborative_dim
            ):
                raise ValueError(
                    "collaborative_embedding dimension does not match fitted dimension "
                    f"{collaborative_dim}"
                )
            collaborative = np.zeros(collaborative_dim, dtype=np.float64)
            if skill.collaborative_embedding:
                collaborative[:] = skill.collaborative_embedding
            collaborative = _normalize_rows(collaborative[None, :])[0]
            parts.append(np.sqrt(alpha) * collaborative)
        elif skill.collaborative_embedding and alpha > 0.0:
            raise ValueError(
                "collaborative_embedding is unavailable in the fitted feature space"
            )
        return np.concatenate(parts)

    def _strategy_snapshot(self) -> Mapping[str, Any]:
        initialized = bool(self._codebooks)
        return {
            "state_version": self.STATE_VERSION,
            "initialized": initialized,
            "semantic_dim": self._semantic_dim,
            "collaborative_dim": self._collaborative_dim,
            "codebooks": [codebook.tolist() for codebook in self._codebooks],
            "usage": [level.tolist() for level in self._usage],
        }

    def _restore_strategy(self, payload: Mapping[str, Any]) -> None:
        if payload.get("state_version") != self.STATE_VERSION:
            raise SerializationError(
                f"unsupported balanced state_version: {payload.get('state_version')!r}"
            )
        initialized = payload.get("initialized")
        if not isinstance(initialized, bool):
            raise SerializationError("balanced initialized flag must be a boolean")
        if not initialized:
            if payload.get("codebooks") or payload.get("usage"):
                raise SerializationError("uninitialized balanced state cannot contain codebooks")
            self._codebooks = ()
            self._usage = ()
            self._semantic_dim = None
            self._collaborative_dim = None
            return

        semantic_dim = payload.get("semantic_dim")
        collaborative_dim = payload.get("collaborative_dim")
        if not isinstance(semantic_dim, int) or isinstance(semantic_dim, bool) or semantic_dim < 1:
            raise SerializationError("balanced semantic_dim must be a positive integer")
        if (
            not isinstance(collaborative_dim, int)
            or isinstance(collaborative_dim, bool)
            or collaborative_dim < 0
        ):
            raise SerializationError("balanced collaborative_dim must be non-negative")
        raw_codebooks = payload.get("codebooks")
        raw_usage = payload.get("usage")
        if not isinstance(raw_codebooks, list) or not isinstance(raw_usage, list):
            raise SerializationError("balanced codebooks and usage must be lists")
        if len(raw_codebooks) != self.config.num_levels or len(raw_usage) != self.config.num_levels:
            raise SerializationError("balanced state level count does not match config")

        feature_dim = semantic_dim + collaborative_dim
        codebooks: list[np.ndarray] = []
        usages: list[np.ndarray] = []
        for level, centroid_count in enumerate(self.config.branching_factors):
            try:
                codebook = np.asarray(raw_codebooks[level], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise SerializationError("balanced state contains invalid numeric arrays") from exc
            raw_level_usage = raw_usage[level]
            if (
                not isinstance(raw_level_usage, list)
                or len(raw_level_usage) != centroid_count
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in raw_level_usage
                )
            ):
                raise SerializationError(
                    "balanced usage must contain non-negative integer counts"
                )
            usage = np.asarray(raw_level_usage, dtype=np.int64)
            if codebook.shape != (centroid_count, feature_dim):
                raise SerializationError(
                    f"codebook shape at level {level + 1} is {codebook.shape}; "
                    f"expected {(centroid_count, feature_dim)}"
                )
            if not np.all(np.isfinite(codebook)):
                raise SerializationError("balanced codebook contains non-finite values")
            if usage.shape != (centroid_count,) or np.any(usage < 0):
                raise SerializationError("balanced usage has invalid shape or values")
            codebook.setflags(write=False)
            codebooks.append(codebook)
            usages.append(usage)

        self._semantic_dim = semantic_dim
        self._collaborative_dim = collaborative_dim
        self._codebooks = tuple(codebooks)
        self._usage = tuple(usages)

    def _validate_restored_state(self) -> None:
        initialized = bool(self._codebooks)
        if self._is_fitted != initialized:
            raise SerializationError(
                "balanced fitted flag and codebook initialization disagree"
            )
        if not initialized:
            return

        expected = [
            np.zeros(size, dtype=np.int64)
            for size in self.config.branching_factors
        ]
        for skill_id in self.registry.skill_ids:
            skill = self.registry.get(skill_id)
            # Validate that a restored record remains encodable by the frozen
            # feature layout as well as that its stored code is counted.
            self._feature_for_skill(skill)
            code = self.registry.code_for(skill_id)
            for level, index in enumerate(code.indices):
                expected[level][index] += 1
        for level, (actual, recomputed) in enumerate(
            zip(self._usage, expected, strict=True)
        ):
            if not np.array_equal(actual, recomputed):
                raise SerializationError(
                    f"balanced usage at level {level + 1} does not match registry"
                )


BalancedTokenizer = BalancedSkillTokenizer


__all__ = ["BalancedSkillTokenizer", "BalancedTokenizer"]
