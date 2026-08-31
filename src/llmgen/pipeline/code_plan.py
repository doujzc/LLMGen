"""Deterministic code-space planning for arbitrary candidate counts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


class CodePlanError(ValueError):
    """Raised when no valid code plan satisfies the requested constraints."""


@dataclass(frozen=True)
class CodePlan:
    schema_version: int
    candidate_count: int
    mode: str
    latency_priority: str
    num_levels: int
    branching_factors: tuple[int, ...]
    capacity: int
    virtual_token_count: int
    spare_capacity: int
    target_capacity: int
    max_branching_factor: int
    max_virtual_tokens: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["branching_factors"] = list(self.branching_factors)
        return value


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _balanced_factors(
    *,
    target: int,
    levels: int,
    limit: int,
) -> tuple[int, ...] | None:
    minimum = 1 if target == 1 else 2
    if limit < minimum or limit**levels < target:
        return None
    base = max(minimum, min(limit, int(target ** (1.0 / levels))))
    factors = [base] * levels
    while _product(factors) < target:
        choices = [index for index, value in enumerate(factors) if value < limit]
        if not choices:
            return None
        # Incrementing the smallest factor yields the largest capacity gain for
        # one additional virtual token and keeps levels balanced.
        index = min(choices, key=lambda item: (factors[item], item))
        factors[index] += 1
    return tuple(factors)


def _candidate_score(plan: tuple[int, ...], priority: str, target: int) -> tuple[Any, ...]:
    levels = len(plan)
    virtual_tokens = sum(plan)
    capacity = _product(plan)
    if priority == "latency":
        return levels, virtual_tokens, capacity - target, plan
    if priority == "vocabulary":
        return virtual_tokens, levels, capacity - target, plan
    if priority != "balanced":
        raise CodePlanError(
            "latency_priority must be latency, balanced, or vocabulary"
        )
    # Sixteen virtual tokens are treated as roughly one generated level for
    # planning only. The value makes a one-token plan attractive for small
    # catalogs while avoiding an N-token vocabulary expansion for larger sets.
    return levels * 16 + virtual_tokens, levels, virtual_tokens, capacity - target, plan


def plan_codes(candidate_count: int, config: Mapping[str, Any]) -> CodePlan:
    """Create a frozen manual or automatically optimized CodePlan."""

    if isinstance(candidate_count, bool) or candidate_count < 1:
        raise CodePlanError("candidate_count must be positive")
    mode = str(config.get("mode") or "auto")
    priority = str(config.get("latency_priority") or "balanced")
    max_virtual_tokens = int(config.get("max_virtual_tokens") or 512)
    configured_max_branching = int(config.get("max_branching_factor") or 256)
    if max_virtual_tokens < 1 or configured_max_branching < 1:
        raise CodePlanError("code planning limits must be positive")
    limit = min(candidate_count, configured_max_branching)
    ratio = float(config.get("spare_capacity_ratio") or 1.0)
    if not math.isfinite(ratio) or ratio < 1:
        raise CodePlanError("spare_capacity_ratio must be finite and >= 1")
    target = max(candidate_count, math.ceil(candidate_count * ratio))

    if mode == "manual":
        raw = config.get("branching_factors")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise CodePlanError("manual code mode requires branching_factors")
        factors = tuple(int(value) for value in raw)
        if any(value < 1 for value in factors):
            raise CodePlanError("branching_factors must be positive")
        if any(value > candidate_count for value in factors):
            raise CodePlanError(
                "each branching factor must not exceed candidate_count"
            )
        configured_levels = config.get("num_levels")
        if configured_levels is not None and int(configured_levels) != len(factors):
            raise CodePlanError(
                "num_levels must equal len(branching_factors)"
            )
        capacity = _product(factors)
        if capacity < target:
            raise CodePlanError(
                f"manual code capacity {capacity} is below target capacity {target}"
            )
        if sum(factors) > max_virtual_tokens:
            raise CodePlanError(
                "manual plan exceeds code.max_virtual_tokens"
            )
    elif mode == "auto":
        requested_levels = config.get("num_levels")
        level_values = (
            [int(requested_levels)]
            if requested_levels is not None
            else list(range(1, 9))
        )
        candidates: list[tuple[int, ...]] = []
        for levels in level_values:
            if levels < 1:
                raise CodePlanError("num_levels must be positive")
            factors = _balanced_factors(target=target, levels=levels, limit=limit)
            if factors is not None and sum(factors) <= max_virtual_tokens:
                candidates.append(factors)
        if not candidates:
            raise CodePlanError(
                "no code plan satisfies capacity, branching, and virtual-token limits"
            )
        factors = min(
            candidates,
            key=lambda value: _candidate_score(value, priority, target),
        )
        capacity = _product(factors)
    else:
        raise CodePlanError("code.mode must be auto or manual")

    return CodePlan(
        schema_version=1,
        candidate_count=candidate_count,
        mode=mode,
        latency_priority=priority,
        num_levels=len(factors),
        branching_factors=factors,
        capacity=capacity,
        virtual_token_count=sum(factors),
        spare_capacity=capacity - candidate_count,
        target_capacity=target,
        max_branching_factor=configured_max_branching,
        max_virtual_tokens=max_virtual_tokens,
    )
