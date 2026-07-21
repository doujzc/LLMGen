"""Configuration for hierarchical skill tokenizers.

The configuration deliberately owns token rendering.  Strategy implementations
work with integer code indices and cannot silently invent a different token
namespace.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from string import Formatter
from typing import Any, Literal, Mapping


TokenizerStrategy = Literal["interpretable", "balanced"]
OverflowPolicy = Literal["error", "allow"]
BalanceScope = Literal["all", "last"]


class ConfigValidationError(ValueError):
    """Raised when a tokenizer configuration is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Versioned configuration shared by all tokenizer strategies.

    ``num_levels`` is not capped: every implementation must loop over the
    configured levels rather than special-casing a two- or three-level code.
    """

    strategy: TokenizerStrategy = "balanced"
    num_levels: int = 2
    branching_factors: tuple[int, ...] = (64, 64)
    codebook_version: str = "skills-v1"
    token_format: str = "<SK_L{level}_{index}>"
    random_seed: int = 7
    bucket_capacity: int | None = None
    overflow_policy: OverflowPolicy = "error"
    balance_scope: BalanceScope = "all"
    sinkhorn_temperature: float = 0.05
    sinkhorn_iterations: int = 50
    clustering_iterations: int = 20
    collaborative_weight: float = 0.25
    dynamic_balance_weight: float = 0.1

    def __post_init__(self) -> None:
        # Accept JSON-friendly lists at the boundary while exposing an immutable
        # tuple everywhere else.
        try:
            factors = tuple(self.branching_factors)
        except TypeError as exc:
            raise ConfigValidationError(
                "branching_factors must be an iterable of positive integers"
            ) from exc
        object.__setattr__(self, "branching_factors", factors)

        if self.strategy not in ("interpretable", "balanced"):
            raise ConfigValidationError(
                "strategy must be either 'interpretable' or 'balanced'"
            )
        self._require_int("num_levels", self.num_levels, minimum=1)
        if len(factors) != self.num_levels:
            raise ConfigValidationError(
                "len(branching_factors) must equal num_levels: "
                f"got {len(factors)} and {self.num_levels}"
            )
        for level, factor in enumerate(factors, start=1):
            self._require_int(
                f"branching_factors[{level - 1}]", factor, minimum=1
            )

        if not isinstance(self.codebook_version, str) or not self.codebook_version.strip():
            raise ConfigValidationError("codebook_version must be a non-empty string")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ConfigValidationError("random_seed must be an integer")
        if self.bucket_capacity is not None:
            self._require_int("bucket_capacity", self.bucket_capacity, minimum=1)
        if self.overflow_policy not in ("error", "allow"):
            raise ConfigValidationError(
                "overflow_policy must be 'error' or 'allow'"
            )
        if self.balance_scope not in ("all", "last"):
            raise ConfigValidationError("balance_scope must be 'all' or 'last'")

        self._require_positive_float(
            "sinkhorn_temperature", self.sinkhorn_temperature
        )
        self._require_int(
            "sinkhorn_iterations", self.sinkhorn_iterations, minimum=1
        )
        self._require_int(
            "clustering_iterations", self.clustering_iterations, minimum=1
        )
        self._require_probability(
            "collaborative_weight", self.collaborative_weight
        )
        self._require_nonnegative_float(
            "dynamic_balance_weight", self.dynamic_balance_weight
        )
        self._validate_token_format()

    @staticmethod
    def _require_int(name: str, value: object, *, minimum: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ConfigValidationError(
                f"{name} must be an integer greater than or equal to {minimum}"
            )

    @staticmethod
    def _require_positive_float(name: str, value: object) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ConfigValidationError(f"{name} must be a positive number")

    @staticmethod
    def _require_nonnegative_float(name: str, value: object) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ConfigValidationError(f"{name} must be a non-negative number")

    @classmethod
    def _require_probability(cls, name: str, value: object) -> None:
        cls._require_nonnegative_float(name, value)
        if float(value) > 1.0:
            raise ConfigValidationError(f"{name} must be between 0 and 1")

    def _validate_token_format(self) -> None:
        if not isinstance(self.token_format, str) or not self.token_format:
            raise ConfigValidationError("token_format must be a non-empty string")
        try:
            parsed = tuple(Formatter().parse(self.token_format))
        except ValueError as exc:
            raise ConfigValidationError(f"invalid token_format: {exc}") from exc

        fields = [field for _, field, _, _ in parsed if field is not None]
        unknown = set(fields) - {"level", "index"}
        if unknown:
            raise ConfigValidationError(
                "token_format contains unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        if "level" not in fields or "index" not in fields:
            raise ConfigValidationError(
                "token_format must contain both {level} and {index}"
            )

        rendered: set[str] = set()
        for level_index, size in enumerate(self.branching_factors):
            for index in range(size):
                try:
                    token = self.token_format.format(
                        level=level_index + 1, index=index
                    )
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    raise ConfigValidationError(
                        f"token_format cannot be rendered with integer values: {exc}"
                    ) from exc
                if not token or any(char.isspace() for char in token):
                    raise ConfigValidationError(
                        "token_format must render a non-empty token without whitespace"
                    )
                if token in rendered:
                    raise ConfigValidationError(
                        f"token_format is not unique across levels and indices: {token!r}"
                    )
                rendered.add(token)

    def token_for(self, level: int, index: int) -> str:
        """Render one token.

        ``level`` is a zero-based code position; the value rendered into the
        human-facing ``{level}`` placeholder is one-based (L1, L2, ...).
        """

        self._require_int("level", level, minimum=0)
        if level >= self.num_levels:
            raise ConfigValidationError(
                f"level index {level} exceeds configured num_levels={self.num_levels}"
            )
        self._require_int("index", index, minimum=0)
        limit = self.branching_factors[level]
        if index >= limit:
            raise ConfigValidationError(
                f"index {index} is outside level {level + 1} branching factor {limit}"
            )
        return self.token_format.format(level=level + 1, index=index)

    def tokens_for_level(self, level: int) -> tuple[str, ...]:
        """Return the stable special-token namespace for one level."""

        self._require_int("level", level, minimum=0)
        if level >= self.num_levels:
            raise ConfigValidationError(
                f"level index {level} exceeds configured num_levels={self.num_levels}"
            )
        return tuple(
            self.token_for(level, index)
            for index in range(self.branching_factors[level])
        )

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return tuple(
            token
            for level in range(self.num_levels)
            for token in self.tokens_for_level(level)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["branching_factors"] = list(self.branching_factors)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenizerConfig":
        if not isinstance(payload, Mapping):
            raise ConfigValidationError("config snapshot must be a mapping")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise ConfigValidationError(f"invalid config fields: {exc}") from exc

    @property
    def structural_signature(self) -> tuple[object, ...]:
        """Fields whose change invalidates router outputs/checkpoints."""

        return (
            self.strategy,
            self.num_levels,
            self.branching_factors,
            self.codebook_version,
            self.token_format,
        )
