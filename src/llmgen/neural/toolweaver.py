"""Trainable adapter around the vendored ToolWeaver RQ-VAE implementation.

The four upstream model modules live under :mod:`llmgen.vendor.toolweaver` so a
fresh clone needs no second repository. This wrapper adds strict configuration,
sparse collaborative loss, reproducible checkpointing, and code diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:  # Keep error messages actionable when only the lightweight package is used.
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in minimal installs
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


CHECKPOINT_SCHEMA_VERSION = 1
TOOLWEAVER_UPSTREAM_REPO = "https://github.com/Fwibo/ToolWeaver"
TOOLWEAVER_UPSTREAM_REVISION = "3a102bad2d85f9674a7febdbaed0235d137e7222"
TOOLWEAVER_SOURCE_CHECKOUT_REVISION = "a7684edaf2bb3af7ff6928c34e27a324599deda0"
TOOLWEAVER_UPSTREAM_HASHES = {
    "index/models/layers.py": "fc2301382fe570668adc90566e8512194107c7390039384b4dec9c65149d4976",
    "index/models/vq.py": "f8cad8e400cdf0b84d698e5e35115a7ab7d6e31b6b9b2977a76131a890dc93f5",
    "index/models/rq.py": "47a3d777bfc2dc0e8c79f0fd0f320eb9e3f9ce8f01167427a73c0be764f2375c",
    "index/models/rqvae.py": "2e619a8fd66cb8174e221a08a020a911fcdbb201ebce1b94d9bc20062e067fdc",
}
_VENDORED_FILES = {
    logical: Path(__file__).resolve().parents[1]
    / "vendor"
    / "toolweaver"
    / Path(logical).name
    for logical in TOOLWEAVER_UPSTREAM_HASHES
}
_VENDORED_HASHES = {
    "index/models/layers.py": "51aaec170b61bfa8c4a5952f93e4cb2f86bd51122a6055e8290d0008a75335aa",
    "index/models/vq.py": "731800f93e811105c598928b46a7cd031e62b952670d21706dd94a6407f7dab1",
    "index/models/rq.py": "3b037aa3c785e5e8f9e69ce0618bd65ac9c5d90987a24da2217c112dd08f8d67",
    "index/models/rqvae.py": "127bf96829a6d018e6fdbb86b354d4258fe266050128f34e143b6696219470f3",
}


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "neural ToolWeaver training requires PyTorch; install the project's "
            "training dependencies"
        )


def _tuple_of(values: Sequence[Any], caster: type) -> tuple[Any, ...]:
    try:
        return tuple(caster(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid sequence values: {values!r}") from exc


@dataclass(frozen=True, slots=True)
class ToolWeaverModelConfig:
    """Complete, JSON-safe RQ-VAE construction configuration."""

    in_dim: int
    num_levels: int
    num_emb_list: tuple[int, ...]
    e_dim: int = 64
    layers: tuple[int, ...] = (512, 256, 128)
    dropout_prob: float = 0.0
    bn: bool = False
    loss_type: str = "mse"
    quant_loss_weight: float = 1.0
    beta: float = 0.25
    kmeans_init: bool = True
    kmeans_iters: int = 100
    sk_epsilons: tuple[float, ...] = (0.01, 0.01)
    sk_iters: int = 50
    graph_lambda: float = 0.001
    token_format: str = "<SK_L{level}_{index}>"
    codebook_version: str = "skillret-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_emb_list", _tuple_of(self.num_emb_list, int))
        object.__setattr__(self, "layers", _tuple_of(self.layers, int))
        object.__setattr__(self, "sk_epsilons", _tuple_of(self.sk_epsilons, float))
        if self.in_dim < 1 or self.num_levels < 1 or self.e_dim < 1:
            raise ValueError("in_dim, num_levels, and e_dim must be positive")
        if len(self.num_emb_list) != self.num_levels:
            raise ValueError("len(num_emb_list) must equal num_levels")
        if len(self.sk_epsilons) != self.num_levels:
            raise ValueError("len(sk_epsilons) must equal num_levels")
        if any(size < 1 for size in self.num_emb_list):
            raise ValueError("every codebook size must be positive")
        if any(width < 1 for width in self.layers):
            raise ValueError("every MLP layer width must be positive")
        if any(epsilon < 0 or not math.isfinite(epsilon) for epsilon in self.sk_epsilons):
            raise ValueError("Sinkhorn epsilons must be finite and non-negative")
        if self.sk_iters < 1 or self.kmeans_iters < 1:
            raise ValueError("sk_iters and kmeans_iters must be positive")
        for name, value in (
            ("dropout_prob", self.dropout_prob),
            ("quant_loss_weight", self.quant_loss_weight),
            ("beta", self.beta),
            ("graph_lambda", self.graph_lambda),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.dropout_prob < 1:
            raise ValueError("dropout_prob must be in [0, 1)")
        if self.loss_type not in {"mse", "l1"}:
            raise ValueError("loss_type must be mse or l1")
        if not self.codebook_version:
            raise ValueError("codebook_version must be non-empty")
        rendered: set[str] = set()
        for level, size in enumerate(self.num_emb_list, start=1):
            for index in range(size):
                try:
                    token = self.token_format.format(level=level, index=index)
                except (KeyError, IndexError, ValueError) as exc:
                    raise ValueError(f"invalid token_format: {exc}") from exc
                if not token or any(char.isspace() for char in token) or token in rendered:
                    raise ValueError("token_format must produce unique, non-whitespace tokens")
                rendered.add(token)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["num_emb_list"] = list(self.num_emb_list)
        value["layers"] = list(self.layers)
        value["sk_epsilons"] = list(self.sk_epsilons)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolWeaverModelConfig":
        return cls(**dict(value))

    @property
    def virtual_tokens(self) -> tuple[str, ...]:
        return tuple(
            self.token_format.format(level=level + 1, index=index)
            for level, size in enumerate(self.num_emb_list)
            for index in range(size)
        )


@dataclass(frozen=True, slots=True)
class Stage1TrainingConfig:
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    gradient_clip: float = 1.0
    amp_dtype: str = "none"
    seed: int = 2024
    eval_every: int = 1
    max_graph_edges_per_batch: int | None = None
    edge_aware_batches: bool = True
    normalize_embeddings: bool = False
    selection_metric: str = "collision_rate"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.eval_every < 1:
            raise ValueError("epochs, batch_size, and eval_every must be positive")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if self.weight_decay < 0 or self.gradient_clip < 0:
            raise ValueError("weight_decay and gradient_clip must be non-negative")
        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer must be adam, adamw, or sgd")
        if self.scheduler not in {"constant", "linear", "cosine"}:
            raise ValueError("scheduler must be constant, linear, or cosine")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.amp_dtype not in {"none", "fp16", "bf16"}:
            raise ValueError("amp_dtype must be none, fp16, or bf16")
        if self.max_graph_edges_per_batch is not None and self.max_graph_edges_per_batch < 1:
            raise ValueError("max_graph_edges_per_batch must be positive")
        if self.selection_metric not in {"collision_rate", "loss"}:
            raise ValueError("selection_metric must be collision_rate or loss")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SparseCollaborationGraph:
    """Coalesced positive weighted edges without an N x N allocation."""

    def __init__(self, src: Any, dst: Any, weight: Any, num_nodes: int) -> None:
        src_array = np.asarray(src, dtype=np.int64).reshape(-1)
        dst_array = np.asarray(dst, dtype=np.int64).reshape(-1)
        weight_array = np.asarray(weight, dtype=np.float32).reshape(-1)
        if len(src_array) != len(dst_array) or len(src_array) != len(weight_array):
            raise ValueError("src, dst, and weight must have the same length")
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if len(src_array) and (
            src_array.min() < 0
            or dst_array.min() < 0
            or src_array.max() >= num_nodes
            or dst_array.max() >= num_nodes
        ):
            raise ValueError("graph endpoints are outside [0, num_nodes)")
        if np.any(~np.isfinite(weight_array)) or np.any(weight_array < 0):
            raise ValueError("graph weights must be finite and non-negative")
        keep = (src_array != dst_array) & (weight_array > 0)
        src_array, dst_array, weight_array = (
            src_array[keep],
            dst_array[keep],
            weight_array[keep],
        )
        if len(src_array):
            keys = src_array * int(num_nodes) + dst_array
            order = np.argsort(keys, kind="stable")
            keys, src_array, dst_array, weight_array = (
                keys[order], src_array[order], dst_array[order], weight_array[order]
            )
            starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
            src_array = src_array[starts]
            dst_array = dst_array[starts]
            weight_array = np.add.reduceat(weight_array, starts).astype(np.float32)
        self.src = src_array
        self.dst = dst_array
        self.weight = weight_array
        self.num_nodes = int(num_nodes)
        neighbors: list[list[int]] = [[] for _ in range(self.num_nodes)]
        neighbor_weights: list[list[float]] = [[] for _ in range(self.num_nodes)]
        for source, target, edge_weight in zip(self.src, self.dst, self.weight):
            neighbors[int(source)].append(int(target))
            neighbor_weights[int(source)].append(float(edge_weight))
            neighbors[int(target)].append(int(source))
            neighbor_weights[int(target)].append(float(edge_weight))
        self._neighbors = tuple(np.asarray(row, dtype=np.int64) for row in neighbors)
        self._neighbor_weights = tuple(np.asarray(row, dtype=np.float64) for row in neighbor_weights)

    @classmethod
    def from_npz(cls, path: str | Path) -> "SparseCollaborationGraph":
        with np.load(path, allow_pickle=False) as data:
            missing = {"src", "dst", "weight", "num_nodes"} - set(data.files)
            if missing:
                raise ValueError(f"graph npz is missing keys: {sorted(missing)}")
            num_nodes = int(np.asarray(data["num_nodes"]).reshape(-1)[0])
            return cls(data["src"], data["dst"], data["weight"], num_nodes)

    def sample_neighbor(self, node: int, rng: np.random.Generator) -> int | None:
        neighbors = self._neighbors[node]
        if not len(neighbors):
            return None
        weights = self._neighbor_weights[node]
        probabilities = weights / weights.sum() if weights.sum() > 0 else None
        return int(rng.choice(neighbors, p=probabilities))

    def induced_edges(
        self,
        nodes: Sequence[int],
        *,
        max_edges: int | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.full(self.num_nodes, -1, dtype=np.int64)
        node_array = np.asarray(nodes, dtype=np.int64)
        positions[node_array] = np.arange(len(node_array), dtype=np.int64)
        local_src = positions[self.src]
        local_dst = positions[self.dst]
        keep = (local_src >= 0) & (local_dst >= 0)
        indices = np.flatnonzero(keep)
        if max_edges is not None and len(indices) > max_edges:
            indices = rng.choice(indices, size=max_edges, replace=False)
        return local_src[indices], local_dst[indices], self.weight[indices]


class _EdgeAwareBatchSampler:
    def __init__(
        self,
        num_nodes: int,
        batch_size: int,
        graph: SparseCollaborationGraph | None,
        seed: int,
        edge_aware: bool,
    ) -> None:
        self.num_nodes = num_nodes
        self.batch_size = min(batch_size, num_nodes)
        self.graph = graph
        self.seed = seed
        self.edge_aware = edge_aware and graph is not None and len(graph.src) > 0
        self.base_size = max(1, self.batch_size // 2) if self.edge_aware else self.batch_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.num_nodes / self.base_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        permutation = rng.permutation(self.num_nodes)
        for start in range(0, self.num_nodes, self.base_size):
            base = permutation[start : start + self.base_size]
            selected = list(int(value) for value in base)
            seen = set(selected)
            if self.edge_aware and self.graph is not None:
                for node in base:
                    neighbor = self.graph.sample_neighbor(int(node), rng)
                    if neighbor is not None and neighbor not in seen:
                        selected.append(neighbor)
                        seen.add(neighbor)
                    if len(selected) >= self.batch_size:
                        break
            if len(selected) < self.batch_size:
                for node in rng.permutation(self.num_nodes):
                    integer = int(node)
                    if integer not in seen:
                        selected.append(integer)
                        seen.add(integer)
                    if len(selected) >= self.batch_size:
                        break
            yield selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_vendored_hashes() -> dict[str, str]:
    return {logical: _sha256(path) for logical, path in _VENDORED_FILES.items()}


def _assert_vendored_integrity() -> None:
    actual = _current_vendored_hashes()
    if actual != _VENDORED_HASHES:
        changed = sorted(
            logical
            for logical in set(actual) | set(_VENDORED_HASHES)
            if actual.get(logical) != _VENDORED_HASHES.get(logical)
        )
        raise RuntimeError(
            "vendored ToolWeaver sources failed integrity validation: "
            + ", ".join(changed)
        )


def _validate_checkpoint_toolweaver_source(checkpoint: Mapping[str, Any]) -> None:
    _assert_vendored_integrity()
    recorded_source = checkpoint.get("toolweaver_source")
    if not isinstance(recorded_source, Mapping) or not recorded_source:
        raise ValueError("checkpoint is missing ToolWeaver source provenance")
    mode = recorded_source.get("mode")
    if mode not in {"dynamic_load", "vendored"}:
        raise ValueError(f"unsupported checkpoint ToolWeaver source mode: {mode!r}")

    recorded_hashes = recorded_source.get("source_files_sha256")
    if recorded_hashes is not None and (
        not isinstance(recorded_hashes, Mapping)
        or dict(recorded_hashes) != TOOLWEAVER_UPSTREAM_HASHES
    ):
        raise ValueError(
            "checkpoint was built from an incompatible ToolWeaver source revision"
        )
    if mode == "vendored":
        recorded_vendored_hashes = recorded_source.get("vendored_files_sha256")
        if (
            not isinstance(recorded_hashes, Mapping)
            or not isinstance(recorded_vendored_hashes, Mapping)
            or dict(recorded_vendored_hashes) != _VENDORED_HASHES
        ):
            raise ValueError("vendored checkpoint has incomplete or invalid source hashes")
    elif recorded_hashes is None:
        # Compatibility for checkpoints created before source-file hashes were
        # added. Only the two revisions whose model files were verified against
        # the bundled snapshot are accepted; the obsolete absolute root is ignored.
        compatible_revisions = {
            TOOLWEAVER_UPSTREAM_REVISION,
            TOOLWEAVER_SOURCE_CHECKOUT_REVISION,
        }
        if recorded_source.get("git_revision") not in compatible_revisions:
            raise ValueError(
                "legacy dynamic checkpoint has no verifiable ToolWeaver revision"
            )


def load_toolweaver_rqvae_class() -> type:
    """Return the self-contained vendored ToolWeaver ``RQVAE`` class."""

    _require_torch()
    _assert_vendored_integrity()
    from ..vendor.toolweaver.rqvae import RQVAE

    return RQVAE


def create_toolweaver_rqvae(config: ToolWeaverModelConfig) -> Any:
    rqvae_class = load_toolweaver_rqvae_class()
    model = rqvae_class(
        in_dim=config.in_dim,
        num_emb_list=list(config.num_emb_list),
        e_dim=config.e_dim,
        layers=list(config.layers),
        dropout_prob=config.dropout_prob,
        bn=config.bn,
        loss_type=config.loss_type,
        quant_loss_weight=config.quant_loss_weight,
        beta=config.beta,
        kmeans_init=config.kmeans_init,
        kmeans_iters=config.kmeans_iters,
        sk_epsilons=list(config.sk_epsilons),
        sk_iters=config.sk_iters,
        graph_lambda=config.graph_lambda,
    )
    quantizers = tuple(model.rq.vq_layers)
    if len(quantizers) != config.num_levels:
        raise RuntimeError("ToolWeaver silently constructed the wrong number of VQ levels")
    actual_sizes = tuple(int(layer.n_e) for layer in quantizers)
    actual_epsilons = tuple(float(layer.sk_epsilon) for layer in quantizers)
    if actual_sizes != config.num_emb_list or actual_epsilons != config.sk_epsilons:
        raise RuntimeError("ToolWeaver VQ construction does not match the validated config")
    return model


def _mark_codebooks_initialized(model: Any) -> None:
    for quantizer in model.rq.vq_layers:
        quantizer.initted = True


def _torch_load(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    _require_torch()
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, dict):
        raise ValueError("checkpoint must contain a mapping")
    return value


def load_toolweaver_rqvae(
    checkpoint_path: str | Path,
    device: str | Any = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load a Stage-1 artifact using the vendored ToolWeaver implementation."""

    checkpoint = _torch_load(checkpoint_path)
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("checkpoint is missing model_config/model_state")
    config = ToolWeaverModelConfig.from_dict(checkpoint["model_config"])
    _validate_checkpoint_toolweaver_source(checkpoint)
    model = create_toolweaver_rqvae(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    _mark_codebooks_initialized(model)
    model.to(device)
    model.eval()
    return model, checkpoint


def code_assignment_metrics(codes: Any, codebook_sizes: Sequence[int]) -> dict[str, Any]:
    matrix = np.asarray(codes, dtype=np.int64)
    sizes = tuple(int(value) for value in codebook_sizes)
    if matrix.ndim != 2 or matrix.shape[1] != len(sizes):
        raise ValueError("codes must be [num_skills, num_levels]")
    level_metrics = []
    for level, size in enumerate(sizes):
        column = matrix[:, level]
        if len(column) and (column.min() < 0 or column.max() >= size):
            raise ValueError(f"code outside level {level} range")
        counts = np.bincount(column, minlength=size).astype(np.int64)
        probabilities = counts[counts > 0] / max(int(counts.sum()), 1)
        entropy = float(-np.sum(probabilities * np.log(probabilities))) if len(probabilities) else 0.0
        normalized_entropy = entropy / math.log(size) if size > 1 else 1.0
        mean = float(np.mean(counts))
        level_metrics.append(
            {
                "level": level + 1,
                "codebook_size": size,
                "used_codes": int(np.count_nonzero(counts)),
                "utilization": float(np.count_nonzero(counts) / size),
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "coefficient_of_variation": float(np.std(counts) / mean) if mean else 0.0,
                "min_usage": int(counts.min()) if len(counts) else 0,
                "max_usage": int(counts.max()) if len(counts) else 0,
            }
        )
    _, bucket_counts = np.unique(matrix, axis=0, return_counts=True)
    unique_codes = len(bucket_counts)
    sample_count = len(matrix)
    return {
        "num_skills": sample_count,
        "num_unique_codes": unique_codes,
        "collision_count": sample_count - unique_codes,
        "collision_rate": float((sample_count - unique_codes) / max(sample_count, 1)),
        "max_bucket_size": int(bucket_counts.max()) if len(bucket_counts) else 0,
        "mean_bucket_size": float(bucket_counts.mean()) if len(bucket_counts) else 0.0,
        "levels": level_metrics,
    }


def residual_nearest_codes(
    encoded: Any,
    codebooks: Sequence[Any],
) -> np.ndarray:
    """Return ordinary RQ nearest-neighbour codes from encoded skill vectors."""

    residual = np.asarray(encoded, dtype=np.float32).copy()
    if residual.ndim != 2 or not len(residual):
        raise ValueError("encoded vectors must be a non-empty [N, D] matrix")
    assignments = np.empty((len(residual), len(codebooks)), dtype=np.int64)
    for level, raw_codebook in enumerate(codebooks):
        centers = np.asarray(raw_codebook, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != residual.shape[1] or not len(centers):
            raise ValueError("every codebook must be a non-empty [K, D] matrix")
        distances = (
            np.sum(residual * residual, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * residual @ centers.T
        )
        indices = np.argmin(distances, axis=1).astype(np.int64, copy=False)
        assignments[:, level] = indices
        residual -= centers[indices]
    return assignments


def _balanced_capacities(costs: np.ndarray) -> np.ndarray:
    """Choose deterministic floor/ceil capacities using nearest-code demand."""

    sample_count, code_count = costs.shape
    base, remainder = divmod(sample_count, code_count)
    capacities = np.full(code_count, base, dtype=np.int64)
    if remainder:
        demand = np.bincount(np.argmin(costs, axis=1), minlength=code_count)
        # Stable sorting makes the code index the deterministic tie-breaker.
        extra = np.argsort(-demand, kind="stable")[:remainder]
        capacities[extra] += 1
    return capacities


def _deferred_balanced_assignment(
    costs: np.ndarray,
    capacities: np.ndarray,
) -> np.ndarray:
    """Scalable hard assignment with exact capacities and deterministic ties."""

    sample_count, code_count = costs.shape
    preferences = np.argsort(costs, axis=1, kind="stable")
    next_preference = np.zeros(sample_count, dtype=np.int64)
    accepted: list[list[int]] = [[] for _ in range(code_count)]
    pending = list(range(sample_count))
    while pending:
        proposals: list[list[int]] = [[] for _ in range(code_count)]
        for sample in pending:
            position = int(next_preference[sample])
            if position >= code_count:
                raise RuntimeError("balanced assignment exhausted every code preference")
            proposals[int(preferences[sample, position])].append(sample)
        pending = []
        for code, new_samples in enumerate(proposals):
            if not new_samples:
                continue
            candidates = accepted[code] + new_samples
            candidates.sort(key=lambda sample: (float(costs[sample, code]), sample))
            capacity = int(capacities[code])
            accepted[code] = candidates[:capacity]
            rejected = candidates[capacity:]
            for sample in rejected:
                next_preference[sample] += 1
            pending.extend(rejected)

    assignments = np.full(sample_count, -1, dtype=np.int64)
    for code, samples in enumerate(accepted):
        assignments[np.asarray(samples, dtype=np.int64)] = code
    if np.any(assignments < 0):
        raise RuntimeError("balanced assignment left samples unassigned")
    return assignments


def _balanced_group_assignment(
    costs: np.ndarray,
    *,
    exact_group_size: int,
) -> np.ndarray:
    """Min-cost balanced assignment for one hierarchical prefix group."""

    from scipy.optimize import linear_sum_assignment

    sample_count, code_count = costs.shape
    if sample_count <= code_count:
        rows, codes = linear_sum_assignment(costs)
        assignments = np.empty(sample_count, dtype=np.int64)
        assignments[rows] = codes
        return assignments

    capacities = _balanced_capacities(costs)
    if sample_count <= exact_group_size:
        slots = np.repeat(np.arange(code_count, dtype=np.int64), capacities)
        rows, slot_indices = linear_sum_assignment(costs[:, slots])
        assignments = np.empty(sample_count, dtype=np.int64)
        assignments[rows] = slots[slot_indices]
        return assignments
    return _deferred_balanced_assignment(costs, capacities)


def _globally_balanced_prefix_assignment(
    distances: np.ndarray,
    prefix_groups: Sequence[np.ndarray],
) -> np.ndarray:
    """Balance a level globally while keeping codes unique in every prefix."""

    from scipy.optimize import linear_sum_assignment

    sample_count, code_count = distances.shape
    capacities = _balanced_capacities(distances)
    remaining = capacities.copy()
    assignments = np.full(sample_count, -1, dtype=np.int64)
    # Larger groups are the most constrained and are therefore allocated first.
    ordered_groups = sorted(
        prefix_groups,
        key=lambda members: (-len(members), tuple(int(value) for value in members)),
    )
    for group_index, members in enumerate(ordered_groups):
        groups_left = len(ordered_groups) - group_index
        available = np.flatnonzero(remaining > 0)
        if len(available) < len(members):
            raise RuntimeError("global hierarchical balance has too few available codes")
        required = set(np.flatnonzero(remaining == groups_left).tolist())
        if len(required) > len(members):
            raise RuntimeError("global hierarchical balance became infeasible")
        group_costs = distances[np.ix_(members, available)].astype(np.float64, copy=True)
        if required:
            scale = max(float(np.max(np.abs(group_costs))), 1.0)
            required_columns = [
                column for column, code in enumerate(available) if int(code) in required
            ]
            group_costs[:, required_columns] -= scale * (len(members) + 1)
        rows, columns = linear_sum_assignment(group_costs)
        selected = available[columns]
        if not required.issubset(set(int(value) for value in selected)):
            raise RuntimeError("global hierarchical balance skipped a required code")
        assignments[members[rows]] = selected
        remaining[selected] -= 1
    if np.any(assignments < 0) or np.any(remaining != 0):
        raise RuntimeError("global hierarchical balance did not consume exact capacities")
    return assignments


def balanced_hierarchical_codes(
    encoded: Any,
    codebooks: Sequence[Any],
    *,
    exact_group_size: int = 2048,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign balanced hierarchical codes while preserving learned distances.

    Every level receives exact floor/ceil global usage.  From the second level
    onward, codes are also unique inside each existing prefix whenever that is
    feasible.  Consequently, when the product of codebook sizes is at least
    the number of skills, final paths are collision-free without adding a third
    fallback token.  Small groups use an exact Hungarian assignment; large
    groups use a capacity-constrained deferred assignment to avoid an N x N
    cost matrix.
    """

    residual = np.asarray(encoded, dtype=np.float32).copy()
    if residual.ndim != 2 or not len(residual):
        raise ValueError("encoded vectors must be a non-empty [N, D] matrix")
    if exact_group_size < 1:
        raise ValueError("exact_group_size must be positive")
    normalized_codebooks = [np.asarray(value, dtype=np.float32) for value in codebooks]
    if not normalized_codebooks:
        raise ValueError("at least one codebook is required")
    for centers in normalized_codebooks:
        if centers.ndim != 2 or centers.shape[1] != residual.shape[1] or not len(centers):
            raise ValueError("every codebook must be a non-empty [K, D] matrix")

    sample_count = len(residual)
    assignments = np.empty((sample_count, len(normalized_codebooks)), dtype=np.int64)
    prefix_groups = [np.arange(sample_count, dtype=np.int64)]
    level_diagnostics: list[dict[str, Any]] = []
    for level, centers in enumerate(normalized_codebooks):
        distances = (
            np.sum(residual * residual, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * residual @ centers.T
        )
        nearest = np.argmin(distances, axis=1)
        if len(prefix_groups) > 1 and all(
            len(members) <= len(centers) for members in prefix_groups
        ):
            assignments[:, level] = _globally_balanced_prefix_assignment(
                distances, prefix_groups
            )
        else:
            for members in prefix_groups:
                assignments[members, level] = _balanced_group_assignment(
                    distances[members], exact_group_size=exact_group_size
                )
        selected = assignments[:, level]
        selected_distance = distances[np.arange(sample_count), selected]
        nearest_distance = distances[np.arange(sample_count), nearest]
        mean_nearest_distance = float(np.mean(nearest_distance))
        mean_assigned_distance = float(np.mean(selected_distance))
        residual -= centers[selected]

        regrouped: dict[tuple[int, ...], list[int]] = {}
        for sample, prefix in enumerate(assignments[:, : level + 1]):
            regrouped.setdefault(tuple(int(value) for value in prefix), []).append(sample)
        prefix_groups = [
            np.asarray(members, dtype=np.int64)
            for _, members in sorted(regrouped.items())
        ]
        level_diagnostics.append(
            {
                "level": level + 1,
                "mean_nearest_squared_distance": mean_nearest_distance,
                "mean_assigned_squared_distance": mean_assigned_distance,
                "distance_inflation": (
                    mean_assigned_distance / max(mean_nearest_distance, 1e-12)
                ),
                "reassigned_fraction": float(np.mean(selected != nearest)),
                "num_prefixes": len(prefix_groups),
                "max_prefix_size": max(len(group) for group in prefix_groups),
            }
        )

    return assignments, {
        "mode": "balanced_hierarchical",
        "exact_group_size": exact_group_size,
        "levels": level_diagnostics,
    }


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _source_metadata() -> dict[str, Any]:
    _assert_vendored_integrity()
    return {
        "upstream_repo": TOOLWEAVER_UPSTREAM_REPO,
        "git_revision": TOOLWEAVER_UPSTREAM_REVISION,
        "source_checkout_revision": TOOLWEAVER_SOURCE_CHECKOUT_REVISION,
        "mode": "vendored",
        "source_files_sha256": dict(TOOLWEAVER_UPSTREAM_HASHES),
        "vendored_files_sha256": _current_vendored_hashes(),
    }


class ToolWeaverStage1Trainer:
    """Full neural Stage-1 trainer with sparse collaborative regularization."""

    def __init__(
        self,
        model_config: ToolWeaverModelConfig,
        training_config: Stage1TrainingConfig,
        embeddings: np.ndarray,
        graph: SparseCollaborationGraph | None,
        output_dir: str | Path,
        *,
        device: str = "cpu",
        data_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        _require_torch()
        matrix = np.asarray(embeddings)
        if matrix.ndim != 2 or matrix.shape[1] != model_config.in_dim or not len(matrix):
            raise ValueError("embeddings must be non-empty [N, model_config.in_dim]")
        if not np.issubdtype(matrix.dtype, np.floating):
            raise ValueError("embeddings must have a floating dtype")
        if graph is not None and graph.num_nodes != len(matrix):
            raise ValueError("graph num_nodes must equal the number of embeddings")
        if len(matrix) < max(model_config.num_emb_list):
            raise ValueError("number of skills must be at least the largest codebook size")
        if training_config.batch_size < max(model_config.num_emb_list):
            raise ValueError("batch_size must be at least the largest codebook size")
        self.model_config = model_config
        self.training_config = training_config
        self.embeddings = matrix
        self.graph = graph
        self.output_dir = Path(output_dir)
        self.data_provenance = dict(data_provenance or {})
        self.device = torch.device(device)
        self.model = create_toolweaver_rqvae(model_config).to(self.device)
        self.sampler = _EdgeAwareBatchSampler(
            len(matrix), training_config.batch_size, graph, training_config.seed,
            training_config.edge_aware_batches,
        )
        self.global_step = 0
        self.start_epoch = 0
        self.best_score: tuple[float, float] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self._seed_everything(training_config.seed)
        self.optimizer = self._build_optimizer()
        self.total_steps = training_config.epochs * len(self.sampler)
        self.scheduler = self._build_scheduler()
        scaler_enabled = training_config.amp_dtype == "fp16" and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self) -> Any:
        cfg = self.training_config
        if cfg.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        return torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def _build_scheduler(self) -> Any:
        cfg = self.training_config
        warmup_steps = int(self.total_steps * cfg.warmup_ratio)

        def scale(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(step, 1) / warmup_steps
            progress = (step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            if cfg.scheduler == "constant":
                return 1.0
            if cfg.scheduler == "linear":
                return 1.0 - progress
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, scale)

    def _autocast(self) -> Any:
        enabled = self.training_config.amp_dtype != "none"
        dtype = torch.float16 if self.training_config.amp_dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def _batch_tensor(self, indices: Sequence[int]) -> Any:
        values = np.asarray(self.embeddings[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
        if self.training_config.normalize_embeddings:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
        return torch.from_numpy(np.ascontiguousarray(values)).to(self.device)

    def _forward_loss(
        self,
        values: Any,
        local_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[Any, dict[str, float]]:
        encoded = self.model.encoder(values)
        residual = encoded
        quantized_sum = torch.zeros_like(encoded)
        quant_losses, raw_levels = [], []
        for quantizer in self.model.rq.vq_layers:
            # The vendored KMeans initializer crosses the PyTorch/NumPy boundary.
            # Autocast produces bfloat16 latents, which NumPy cannot represent, so
            # initialize each codebook once from an explicitly detached fp32 view.
            if not quantizer.initted and quantizer.training:
                quantizer.init_emb(residual.detach().float())
            quantized, quant_loss, indices = quantizer(residual, use_sk=True)
            raw_levels.append(quantizer.embedding(indices).view_as(residual))
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            quant_losses.append(quant_loss)
        output = self.model.decoder(quantized_sum)
        if self.model_config.loss_type == "mse":
            reconstruction = F.mse_loss(output, values)
        else:
            reconstruction = F.l1_loss(output, values)
        quantization = torch.stack(quant_losses).mean()
        edge_src, edge_dst, edge_weight = local_edges
        if len(edge_src):
            source = torch.as_tensor(edge_src, dtype=torch.long, device=self.device)
            target = torch.as_tensor(edge_dst, dtype=torch.long, device=self.device)
            weights = torch.as_tensor(edge_weight, dtype=encoded.dtype, device=self.device)
            graph_levels = [
                (weights * torch.sum((level[source] - level[target]) ** 2, dim=-1)).sum()
                / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
                for level in raw_levels
            ]
            graph_loss = torch.stack(graph_levels).mean()
        else:
            graph_loss = encoded.sum() * 0.0
        total = reconstruction + self.model_config.quant_loss_weight * (
            quantization + self.model_config.graph_lambda * graph_loss
        )
        return total, {
            "loss": float(total.detach()),
            "reconstruction_loss": float(reconstruction.detach()),
            "quantization_loss": float(quantization.detach()),
            "graph_loss": float(graph_loss.detach()),
            "graph_edges": float(len(edge_src)),
        }

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.sampler.set_epoch(epoch)
        graph_rng = np.random.default_rng(self.training_config.seed + 1_000_003 * epoch)
        totals: dict[str, float] = {}
        batches = 0
        for batch_indices in self.sampler:
            if self.graph is None:
                local_edges = (
                    np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float32),
                )
            else:
                local_edges = self.graph.induced_edges(
                    batch_indices,
                    max_edges=self.training_config.max_graph_edges_per_batch,
                    rng=graph_rng,
                )
            values = self._batch_tensor(batch_indices)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss, metrics = self._forward_loss(values, local_edges)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.training_config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            batches += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
        result = {key: value / batches for key, value in totals.items()}
        result["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        return result

    @torch.no_grad()
    def encode_all(self, batch_size: int | None = None) -> np.ndarray:
        self.model.eval()
        size = batch_size or self.training_config.batch_size
        rows = []
        for start in range(0, len(self.embeddings), size):
            indices = range(start, min(start + size, len(self.embeddings)))
            values = self._batch_tensor(indices)
            with self._autocast():
                codes = self.model.get_indices(values, use_sk=False)
            rows.append(codes.reshape(-1, self.model_config.num_levels).cpu().numpy())
        return np.concatenate(rows).astype(np.int64, copy=False)

    @torch.no_grad()
    def evaluate(self, train_metrics: Mapping[str, float] | None = None) -> dict[str, Any]:
        metrics = code_assignment_metrics(self.encode_all(), self.model_config.num_emb_list)
        if train_metrics:
            metrics.update({key: float(value) for key, value in train_metrics.items()})
        return metrics

    def _checkpoint(self, epoch: int, metrics: Mapping[str, Any], *, resumable: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "artifact_type": "llmgen.toolweaver.stage1",
            "epoch": epoch,
            "global_step": self.global_step,
            "model_config": self.model_config.to_dict(),
            "model_state": self.model.state_dict(),
            "training_config": self.training_config.to_dict(),
            "metrics": dict(metrics),
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "toolweaver_source": _source_metadata(),
            "data_provenance": self.data_provenance,
        }
        if resumable:
            payload.update(
                {
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "scaler_state": self.scaler.state_dict(),
                    "rng_state": _rng_state(),
                }
            )
        return payload

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = _torch_load(checkpoint_path)
        _validate_checkpoint_toolweaver_source(checkpoint)
        restored = ToolWeaverModelConfig.from_dict(checkpoint["model_config"])
        if restored != self.model_config:
            raise ValueError("resume checkpoint model_config does not match")
        restored_training = dict(checkpoint.get("training_config", {}))
        current_training = self.training_config.to_dict()
        # Extending the epoch budget and changing evaluation cadence are safe;
        # optimizer, batching, precision, and objective changes are not.
        for flexible_key in ("epochs", "eval_every"):
            restored_training.pop(flexible_key, None)
            current_training.pop(flexible_key, None)
        if restored_training != current_training:
            raise ValueError("resume checkpoint training_config does not match")
        restored_provenance = checkpoint.get("data_provenance", {})
        for key, expected in self.data_provenance.items():
            if key.endswith("sha256") and restored_provenance.get(key) != expected:
                raise ValueError(f"resume checkpoint data provenance differs for {key}")
        for key in ("optimizer_state", "scheduler_state", "rng_state"):
            if key not in checkpoint:
                raise ValueError(f"resume checkpoint is missing {key}")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        _mark_codebooks_initialized(self.model)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        score = checkpoint.get("best_score")
        self.best_score = tuple(float(value) for value in score) if score is not None else None
        self.best_metrics = checkpoint.get("best_metrics")
        _restore_rng_state(checkpoint["rng_state"])

    def _score(self, metrics: Mapping[str, Any]) -> tuple[float, float]:
        loss = float(metrics["loss"])
        collision = float(metrics["collision_rate"])
        return (collision, loss) if self.training_config.selection_metric == "collision_rate" else (loss, collision)

    def fit(self, resume_from: str | Path | None = None) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "model_config.json").write_text(
            json.dumps(self.model_config.to_dict(), indent=2), encoding="utf-8"
        )
        if resume_from is not None:
            self.resume(resume_from)
        history_path = self.output_dir / "history.jsonl"
        latest_metrics: dict[str, Any] = {}
        for epoch in range(self.start_epoch, self.training_config.epochs):
            train_metrics = self._train_epoch(epoch)
            should_evaluate = (
                (epoch + 1) % self.training_config.eval_every == 0
                or epoch + 1 == self.training_config.epochs
            )
            latest_metrics = self.evaluate(train_metrics) if should_evaluate else dict(train_metrics)
            row = {"epoch": epoch, "global_step": self.global_step, **latest_metrics}
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if should_evaluate:
                print(
                    json.dumps(
                        {
                            "event": "stage1_progress",
                            "epoch": epoch + 1,
                            "epochs": self.training_config.epochs,
                            "global_step": self.global_step,
                            "loss": latest_metrics.get("loss"),
                            "collision_rate": latest_metrics.get("collision_rate"),
                            "levels": latest_metrics.get("levels"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                score = self._score(latest_metrics)
                if self.best_score is None or score < self.best_score:
                    self.best_score = score
                    self.best_metrics = dict(latest_metrics)
                    _atomic_torch_save(
                        self._checkpoint(epoch, latest_metrics, resumable=False),
                        self.output_dir / "best.pt",
                    )
            _atomic_torch_save(
                self._checkpoint(epoch, latest_metrics, resumable=True),
                self.output_dir / "last.pt",
            )
        return {
            "best_checkpoint": str(self.output_dir / "best.pt"),
            "last_checkpoint": str(self.output_dir / "last.pt"),
            "best_metrics": self.best_metrics,
            "last_metrics": latest_metrics,
        }
