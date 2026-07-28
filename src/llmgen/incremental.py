"""Frozen-codebook candidate updates and tiny incremental-router datasets.

An incremental candidate state is deliberately an overlay: it stores only the
active decode map and virtual-token namespace.  Model weights remain in the
source router directory, while inference rebuilds its trie from the overlay.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .router import (
    RouterDataError,
    build_memorization_examples,
    build_retrieval_examples,
    load_virtual_tokens,
    skill_document_text,
    write_jsonl,
)
from .router_bundle import (
    BUNDLED_VIRTUAL_TOKENS_FILENAME,
    DECODE_MAP_FILENAME,
    load_skill_decode_map,
    validate_skill_decode_map,
)
from .skillret import code_token, sha256_file


INCREMENTAL_STATE_SCHEMA_VERSION = 1
OPERATION_FILENAME = "operation.json"

_SKILL_METADATA_FIELDS = (
    "name",
    "display_name",
    "description",
    "summary",
    "text",
    "document_text",
    "domain",
    "capability_zh",
    "mobile_fit",
    "unsafe_action",
    "roles",
    "rank",
    "source_url",
    "canonical_url",
    "owner",
    "slug",
    "skill_md",
    "body",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
    )


def load_candidate_state(
    directory: str | Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Load a model bundle or a previous incremental candidate overlay."""

    root = Path(directory).expanduser().resolve()
    decode_path = root / DECODE_MAP_FILENAME
    token_path = root / BUNDLED_VIRTUAL_TOKENS_FILENAME
    if not decode_path.is_file() or not token_path.is_file():
        raise RouterDataError(
            f"candidate state {root} must contain {DECODE_MAP_FILENAME} and "
            f"{BUNDLED_VIRTUAL_TOKENS_FILENAME}"
        )
    payload = load_skill_decode_map(decode_path)
    file_tokens = load_virtual_tokens(token_path)
    if tuple(payload["virtual_tokens"]) != file_tokens:
        raise RouterDataError(
            "candidate state's virtual_tokens.txt disagrees with its decode map"
        )
    return payload, decode_path, token_path


def incremental_ancestor_hashes(payload: Mapping[str, Any]) -> frozenset[str]:
    """Return hashes that an incremental state explicitly descends from."""

    state = payload.get("incremental_state")
    if not isinstance(state, Mapping):
        return frozenset()
    if state.get("schema_version") != INCREMENTAL_STATE_SCHEMA_VERSION:
        raise RouterDataError("unsupported incremental candidate-state schema")
    raw_hashes = state.get("ancestor_decode_map_sha256")
    if not isinstance(raw_hashes, list):
        raise RouterDataError("incremental candidate state has no ancestor hash list")
    hashes = {
        str(value)
        for value in raw_hashes
        if isinstance(value, str) and len(value) == 64
    }
    if len(hashes) != len(raw_hashes):
        raise RouterDataError("incremental candidate state has an invalid ancestor hash")
    return frozenset(hashes)


def _next_incremental_state(
    source: Mapping[str, Any],
    *,
    source_sha256: str,
    operation: Mapping[str, Any],
) -> dict[str, Any]:
    previous = source.get("incremental_state")
    if isinstance(previous, Mapping):
        ancestors = list(incremental_ancestor_hashes(source))
        base_sha256 = str(
            previous.get("base_decode_map_sha256") or source_sha256
        )
        try:
            revision = int(previous.get("revision", 0)) + 1
        except (TypeError, ValueError) as exc:
            raise RouterDataError(
                "incremental candidate state has an invalid revision"
            ) from exc
        history = [
            dict(value)
            for value in previous.get("operations", ())
            if isinstance(value, Mapping)
        ]
    else:
        ancestors = []
        base_sha256 = source_sha256
        revision = 1
        history = []
    if source_sha256 not in ancestors:
        ancestors.append(source_sha256)
    normalized_operation = dict(operation)
    history.append(normalized_operation)
    return {
        "schema_version": INCREMENTAL_STATE_SCHEMA_VERSION,
        "base_decode_map_sha256": base_sha256,
        "parent_decode_map_sha256": source_sha256,
        "ancestor_decode_map_sha256": sorted(set(ancestors)),
        "revision": revision,
        "last_operation": normalized_operation,
        "operations": history,
    }


def _skill_id(skill: Mapping[str, Any]) -> str:
    raw = skill.get("skill_id", skill.get("id"))
    if not isinstance(raw, str) or not raw.strip():
        raise RouterDataError("new skill must define a non-empty skill_id")
    return raw.strip()


def candidate_metadata(
    skill: Mapping[str, Any],
    *,
    document_text: str | None = None,
) -> dict[str, Any]:
    """Keep useful human-readable fields while excluding embedding vectors."""

    skill_id = _skill_id(skill)
    metadata: dict[str, Any] = {"skill_id": skill_id}
    for field in _SKILL_METADATA_FIELDS:
        value = skill.get(field)
        if value is not None and field not in {"document_text"}:
            metadata[field] = deepcopy(value)
    name = next(
        (
            str(skill.get(field)).strip()
            for field in ("name", "display_name", "slug")
            if isinstance(skill.get(field), str) and str(skill.get(field)).strip()
        ),
        skill_id,
    )
    metadata["name"] = name
    if not isinstance(metadata.get("description"), str) or not str(
        metadata["description"]
    ).strip():
        summary = skill.get("summary")
        if isinstance(summary, str) and summary.strip():
            metadata["description"] = summary.strip()
    rendered = (document_text or skill_document_text(skill)).strip()
    if not rendered:
        raise RouterDataError(f"new skill {skill_id!r} has no document text")
    metadata["text"] = rendered
    metadata.pop("train_target_count", None)
    return metadata


def _rebuild_decode_map(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute every trie-facing field from the one active skill set."""

    skills = payload.get("skills")
    skill_to_code = payload.get("skill_to_code")
    virtual_tokens = payload.get("virtual_tokens")
    num_levels = payload.get("num_levels")
    if not isinstance(skills, dict) or not skills:
        raise RouterDataError("candidate state cannot have an empty active skill set")
    if not isinstance(skill_to_code, dict) or set(skill_to_code) != set(skills):
        raise RouterDataError("candidate state skills and codes do not match")
    if not isinstance(virtual_tokens, list) or not virtual_tokens:
        raise RouterDataError("candidate state has no virtual token namespace")
    if len(set(virtual_tokens)) != len(virtual_tokens):
        raise RouterDataError("candidate-state virtual tokens are not unique")
    if not isinstance(num_levels, int) or num_levels < 1:
        raise RouterDataError("candidate state has an invalid num_levels")

    token_set = set(map(str, virtual_tokens))
    path_members: dict[tuple[str, ...], list[str]] = {}
    token_members: dict[str, list[str]] = {
        str(token): [] for token in virtual_tokens
    }
    normalized_skills: dict[str, dict[str, Any]] = {}
    normalized_codes: dict[str, dict[str, Any]] = {}
    for skill_id in sorted(skills):
        metadata = skills[skill_id]
        details = skill_to_code[skill_id]
        if not isinstance(metadata, Mapping) or not isinstance(details, Mapping):
            raise RouterDataError("candidate metadata and code entries must be objects")
        raw_tokens = details.get("tokens")
        if not isinstance(raw_tokens, list) or len(raw_tokens) != num_levels:
            raise RouterDataError(
                f"candidate {skill_id!r} has the wrong number of code tokens"
            )
        tokens = tuple(map(str, raw_tokens))
        unknown = set(tokens).difference(token_set)
        if unknown:
            raise RouterDataError(
                f"candidate {skill_id!r} uses unknown token {min(unknown)!r}"
            )
        raw_indices = details.get("indices")
        indices = (
            [int(value) for value in raw_indices]
            if isinstance(raw_indices, (list, tuple))
            else None
        )
        if indices is not None and len(indices) != num_levels:
            raise RouterDataError(
                f"candidate {skill_id!r} has the wrong number of code indices"
            )
        normalized_metadata = deepcopy(dict(metadata))
        normalized_metadata["skill_id"] = skill_id
        raw_name = normalized_metadata.get("name")
        normalized_metadata["name"] = (
            raw_name.strip()
            if isinstance(raw_name, str) and raw_name.strip()
            else skill_id
        )
        normalized_metadata.pop("train_target_count", None)
        normalized_skills[skill_id] = normalized_metadata
        normalized_codes[skill_id] = {
            "tokens": list(tokens),
            "code_text": "".join(tokens),
            "indices": indices,
        }
        path_members.setdefault(tokens, []).append(skill_id)
        for token in tokens:
            token_members[token].append(skill_id)

    def candidates(skill_ids: Sequence[str]) -> list[dict[str, str]]:
        return [
            {
                "skill_id": skill_id,
                "name": str(normalized_skills[skill_id]["name"]),
            }
            for skill_id in skill_ids
        ]

    paths = []
    for tokens, raw_members in sorted(path_members.items()):
        members = sorted(raw_members)
        paths.append(
            {
                "code_text": "".join(tokens),
                "tokens": list(tokens),
                "skill_ids": members,
                "candidates": candidates(members),
            }
        )
    token_to_skill_ids = {
        token: sorted(token_members[token]) for token in map(str, virtual_tokens)
    }
    payload.update(
        {
            "created_at": utc_now(),
            "num_skills": len(normalized_skills),
            "num_paths": len(paths),
            "skills": normalized_skills,
            "skill_to_code": normalized_codes,
            "paths": paths,
            "token_to_skill_ids": token_to_skill_ids,
            "token_to_candidates": {
                token: candidates(members)
                for token, members in token_to_skill_ids.items()
            },
            # Full-set train counts are no longer meaningful after an online
            # add/remove.  The candidate state remains valid for inference.
            "supervision": None,
        }
    )
    validate_skill_decode_map(payload)
    return payload


def add_candidate(
    source: Mapping[str, Any],
    *,
    skill: Mapping[str, Any],
    indices: Sequence[int],
    tokens: Sequence[str],
    source_sha256: str,
    assignment_mode: str,
    update_mode: str,
    document_text: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add one active candidate without changing any existing assignment."""

    skill_id = _skill_id(skill)
    if skill_id in source.get("skills", {}):
        raise RouterDataError(f"candidate already exists: {skill_id!r}")
    num_levels = int(source.get("num_levels", 0))
    normalized_indices = [int(value) for value in indices]
    normalized_tokens = [str(value) for value in tokens]
    if (
        len(normalized_indices) != num_levels
        or len(normalized_tokens) != num_levels
    ):
        raise RouterDataError("new candidate code length differs from the frozen index")
    token_set = set(map(str, source.get("virtual_tokens", ())))
    if not set(normalized_tokens) <= token_set:
        raise RouterDataError("new candidate code is outside the frozen token namespace")
    existing_paths = {
        tuple(details["tokens"])
        for details in source.get("skill_to_code", {}).values()
    }
    collision = tuple(normalized_tokens) in existing_paths
    if assignment_mode == "nearest_available" and collision:
        raise RouterDataError("nearest_available assignment returned an occupied code")

    payload = deepcopy(dict(source))
    payload["skills"][skill_id] = candidate_metadata(
        skill,
        document_text=document_text,
    )
    payload["skill_to_code"][skill_id] = {
        "tokens": normalized_tokens,
        "indices": normalized_indices,
        "code_text": "".join(normalized_tokens),
    }
    operation = {
        "type": "add",
        "created_at": utc_now(),
        "skill_ids": [skill_id],
        "indices": normalized_indices,
        "tokens": normalized_tokens,
        "code_text": "".join(normalized_tokens),
        "assignment_mode": assignment_mode,
        "update_mode": update_mode,
        "collision": collision,
    }
    payload["incremental_state"] = _next_incremental_state(
        source,
        source_sha256=source_sha256,
        operation=operation,
    )
    return _rebuild_decode_map(payload), operation


def remove_candidate(
    source: Mapping[str, Any],
    *,
    skill_id: str,
    source_sha256: str,
    disable_shared_path: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove one candidate, optionally disabling its entire shared path."""

    skill_id = skill_id.strip()
    if skill_id not in source.get("skills", {}):
        raise RouterDataError(f"unknown active candidate: {skill_id!r}")
    tokens = tuple(source["skill_to_code"][skill_id]["tokens"])
    members = sorted(
        candidate_id
        for candidate_id, details in source["skill_to_code"].items()
        if tuple(details["tokens"]) == tokens
    )
    removed = members if disable_shared_path else [skill_id]
    payload = deepcopy(dict(source))
    for candidate_id in removed:
        payload["skills"].pop(candidate_id)
        payload["skill_to_code"].pop(candidate_id)
    operation = {
        "type": "remove",
        "created_at": utc_now(),
        "requested_skill_id": skill_id,
        "skill_ids": removed,
        "tokens": list(tokens),
        "code_text": "".join(tokens),
        "shared_path": len(members) > 1,
    }
    payload["incremental_state"] = _next_incremental_state(
        source,
        source_sha256=source_sha256,
        operation=operation,
    )
    return _rebuild_decode_map(payload), operation


def write_candidate_state(
    output_dir: str | Path,
    payload: Mapping[str, Any],
    operation: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically materialize a small candidate overlay directory."""

    destination = Path(output_dir).expanduser().resolve()
    decode_path = destination / DECODE_MAP_FILENAME
    token_path = destination / BUNDLED_VIRTUAL_TOKENS_FILENAME
    operation_path = destination / OPERATION_FILENAME
    existing = [path for path in (decode_path, token_path, operation_path) if path.exists()]
    if existing and not overwrite:
        raise RouterDataError(
            f"candidate state already exists at {destination}; use --force to replace it"
        )
    destination.mkdir(parents=True, exist_ok=True)
    validate_skill_decode_map(payload)
    _atomic_json(decode_path, payload)
    _atomic_text(token_path, "\n".join(map(str, payload["virtual_tokens"])) + "\n")
    result = {
        **dict(operation),
        "output_dir": str(destination),
        "decode_map": str(decode_path),
        "decode_map_sha256": sha256_file(decode_path),
        "virtual_tokens": str(token_path),
        "virtual_tokens_sha256": sha256_file(token_path),
        "revision": payload.get("incremental_state", {}).get("revision"),
        "num_skills": int(payload["num_skills"]),
        "num_paths": int(payload["num_paths"]),
    }
    _atomic_json(operation_path, result)
    return result


def frozen_rq_beam_paths(
    encoded: Sequence[float] | np.ndarray,
    codebooks: Sequence[np.ndarray],
    *,
    beam_size: int,
) -> list[tuple[tuple[int, ...], float]]:
    """Rank residual-quantizer paths while leaving every codebook frozen."""

    if beam_size < 1:
        raise RouterDataError("assignment beam size must be positive")
    vector = np.asarray(encoded, dtype=np.float32)
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.ndim != 1 or not len(vector):
        raise RouterDataError("encoded skill must be a non-empty vector")
    # (current residual norm, path, residual).  The score at each level is the
    # remaining reconstruction error, which is the RQ objective used for
    # deterministic pruning.
    beams: list[tuple[float, tuple[int, ...], np.ndarray]] = [
        (float(np.dot(vector, vector)), (), vector.copy())
    ]
    for raw_codebook in codebooks:
        centers = np.asarray(raw_codebook, dtype=np.float32)
        if (
            centers.ndim != 2
            or centers.shape[1] != vector.shape[0]
            or not len(centers)
        ):
            raise RouterDataError("frozen codebook has an incompatible shape")
        expanded: list[tuple[float, tuple[int, ...], np.ndarray]] = []
        per_parent = min(beam_size, len(centers))
        for _, path, residual in beams:
            distances = (
                np.sum(residual * residual)
                + np.sum(centers * centers, axis=1)
                - 2.0 * centers @ residual
            )
            nearest = np.argsort(distances, kind="stable")[:per_parent]
            for raw_index in nearest:
                index = int(raw_index)
                next_residual = residual - centers[index]
                expanded.append(
                    (float(distances[index]), (*path, index), next_residual)
                )
        expanded.sort(key=lambda item: (item[0], item[1]))
        beams = expanded[:beam_size]
    return [(path, score) for score, path, _ in beams]


def select_frozen_code(
    encoded: Sequence[float] | np.ndarray,
    codebooks: Sequence[np.ndarray],
    *,
    occupied_paths: set[tuple[int, ...]],
    assignment_mode: str,
    beam_size: int,
) -> tuple[tuple[int, ...], float, bool]:
    """Select the raw-nearest path or the nearest currently unused path."""

    if assignment_mode not in {"nearest", "nearest_available"}:
        raise RouterDataError(
            "assignment_mode must be 'nearest' or 'nearest_available'"
        )
    effective_beam = 1 if assignment_mode == "nearest" else beam_size
    paths = frozen_rq_beam_paths(
        encoded,
        codebooks,
        beam_size=effective_beam,
    )
    for path, score in paths:
        collision = path in occupied_paths
        if assignment_mode == "nearest" or not collision:
            return path, score, collision
    raise RouterDataError(
        "no unused code found in the assignment beam; increase "
        "--assignment-beam-size or use --assignment-mode nearest to allow a collision"
    )


def compute_frozen_skill_code(
    *,
    skill: Mapping[str, Any],
    stage1_checkpoint: str | Path,
    source_decode_map: Mapping[str, Any],
    embedding: Sequence[float] | np.ndarray,
    assignment_mode: str,
    assignment_beam_size: int,
    device: str,
) -> dict[str, Any]:
    """Encode one skill through the frozen ToolWeaver encoder and codebooks."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RouterDataError(
            "computing an incremental code requires the training dependencies"
        ) from exc
    from .neural.toolweaver import load_toolweaver_rqvae

    checkpoint_path = Path(stage1_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RouterDataError(f"Stage-1 checkpoint does not exist: {checkpoint_path}")
    expected_checkpoint_sha256 = source_decode_map.get("provenance", {}).get(
        "stage1_checkpoint_sha256"
    )
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        expected_checkpoint_sha256
        and expected_checkpoint_sha256 != actual_checkpoint_sha256
    ):
        raise RouterDataError(
            "Stage-1 checkpoint differs from the candidate state's frozen codebook"
        )
    model, checkpoint = load_toolweaver_rqvae(checkpoint_path, device=device)
    model_config = checkpoint["model_config"]
    branching_factors = tuple(int(value) for value in model_config["num_emb_list"])
    if len(branching_factors) != int(source_decode_map["num_levels"]):
        raise RouterDataError("Stage-1 checkpoint and candidate state use different levels")
    token_format = str(
        model_config.get("token_format", "<SK_L{level}_{index}>")
    )
    values = np.asarray(embedding, dtype=np.float32)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    expected_dim = int(model_config["in_dim"])
    if values.ndim != 1 or values.shape[0] != expected_dim:
        raise RouterDataError(
            f"new skill embedding has dimension {values.shape}, expected {expected_dim}"
        )
    if not np.all(np.isfinite(values)):
        raise RouterDataError("new skill embedding contains non-finite values")
    # The OpenAI embedding pipeline stores unit-normalized rows.  Preserve the
    # same contract for a manually supplied vector as well.
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        raise RouterDataError("new skill embedding has zero norm")
    values = values / norm
    tensor = torch.from_numpy(values[None, :].copy()).to(device)
    if checkpoint.get("training_config", {}).get("normalize_embeddings", False):
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
    with torch.inference_mode():
        encoded = model.encoder(tensor).detach().float().cpu().numpy()[0]
    codebooks = [
        quantizer.embedding.weight.detach().float().cpu().numpy()
        for quantizer in model.rq.vq_layers
    ]
    occupied_paths = {
        tuple(int(value) for value in details["indices"])
        for details in source_decode_map["skill_to_code"].values()
        if isinstance(details.get("indices"), list)
    }
    if len(occupied_paths) != int(source_decode_map["num_paths"]):
        # A collision is fine, but every active code must expose its numerical
        # indices so online assignment uses the same namespace.
        if any(
            not isinstance(details.get("indices"), list)
            for details in source_decode_map["skill_to_code"].values()
        ):
            raise RouterDataError(
                "candidate state lacks numerical code indices required for online add"
            )
    indices, score, collision = select_frozen_code(
        encoded,
        codebooks,
        occupied_paths=occupied_paths,
        assignment_mode=assignment_mode,
        beam_size=assignment_beam_size,
    )
    tokens = [
        code_token(level, index, token_format)
        for level, index in enumerate(indices, start=1)
    ]
    unknown_tokens = set(tokens).difference(source_decode_map["virtual_tokens"])
    if unknown_tokens:
        raise RouterDataError(
            "frozen checkpoint produced tokens outside the router namespace"
        )
    return {
        "skill_id": _skill_id(skill),
        "indices": list(indices),
        "tokens": tokens,
        "code_text": "".join(tokens),
        "distance": score,
        "collision": collision,
        "assignment_mode": assignment_mode,
        "assignment_beam_size": assignment_beam_size,
        "stage1_checkpoint_sha256": actual_checkpoint_sha256,
    }


def build_incremental_training_rows(
    decode_map: Mapping[str, Any],
    *,
    skill_id: str,
    queries: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build exactly one memorization row plus direct single-skill queries."""

    if skill_id not in decode_map.get("skills", {}):
        raise RouterDataError(f"incremental skill is not active: {skill_id!r}")
    normalized_queries: list[str] = []
    seen: set[str] = set()
    for index, raw_query in enumerate(queries, start=1):
        if not isinstance(raw_query, str):
            raise RouterDataError(f"incremental query {index} is not a string")
        query = " ".join(raw_query.split())
        if not 2 <= len(query) <= 180:
            raise RouterDataError(
                f"incremental query {index} length must be in [2, 180]"
            )
        key = query.casefold()
        if key in seen:
            raise RouterDataError(f"duplicate incremental query: {query!r}")
        if skill_id.casefold() in key and any(
            marker in skill_id for marker in ("@", "/", "_")
        ):
            raise RouterDataError(
                f"incremental query {index} leaks the opaque skill_id"
            )
        if any(
            marker in key
            for marker in ("调用工具", "使用skill", "目标候选", "路由训练")
        ):
            raise RouterDataError(
                f"incremental query {index} contains routing/dataset language"
            )
        seen.add(key)
        normalized_queries.append(query)
    if not normalized_queries:
        raise RouterDataError("at least one incremental retrieval query is required")

    code = list(decode_map["skill_to_code"][skill_id]["tokens"])
    memorization = build_memorization_examples(
        [decode_map["skills"][skill_id]],
        {skill_id: code},
    )
    query_rows = [
        {
            "id": f"incremental-{skill_id}-{index:03d}",
            "query": query,
            "skill_ids": [skill_id],
        }
        for index, query in enumerate(normalized_queries, start=1)
    ]
    retrieval = build_retrieval_examples(
        query_rows,
        {skill_id: code},
    )
    return memorization, retrieval


def write_incremental_training_data(
    output_dir: str | Path,
    *,
    decode_map: Mapping[str, Any],
    decode_map_path: str | Path,
    skill_id: str,
    queries: Sequence[str],
    query_provenance: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist the tiny two-phase dataset consumed by ``train_router.py``."""

    destination = Path(output_dir).expanduser().resolve()
    memorization_path = destination / "memorization_train.jsonl"
    retrieval_path = destination / "retrieval_train.jsonl"
    manifest_path = destination / "manifest.json"
    if (
        any(path.exists() for path in (memorization_path, retrieval_path, manifest_path))
        and not overwrite
    ):
        raise RouterDataError(
            f"incremental training data already exists at {destination}; "
            "use --force to replace it"
        )
    memorization, retrieval = build_incremental_training_rows(
        decode_map,
        skill_id=skill_id,
        queries=queries,
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(memorization_path, memorization)
    write_jsonl(retrieval_path, retrieval)
    source_decode_path = Path(decode_map_path).expanduser().resolve()
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "mode": "incremental_new_candidate_only",
        "skill_id": skill_id,
        "target_tokens": list(decode_map["skill_to_code"][skill_id]["tokens"]),
        "examples": {
            "memorization": len(memorization),
            "retrieval": len(retrieval),
        },
        "query_provenance": dict(query_provenance),
        "sources": {
            "candidate_decode_map": {
                "path": str(source_decode_path),
                "sha256": sha256_file(source_decode_path),
            },
            # Keep train_router's existing Stage-1 provenance extraction path.
            "index_manifest": {
                "checkpoint_sha256": decode_map.get("provenance", {}).get(
                    "stage1_checkpoint_sha256"
                ),
                "sha256": decode_map.get("provenance", {}).get(
                    "index_manifest_sha256"
                ),
            },
        },
        "artifacts": {
            memorization_path.name: {
                "sha256": sha256_file(memorization_path),
                "rows": len(memorization),
            },
            retrieval_path.name: {
                "sha256": sha256_file(retrieval_path),
                "rows": len(retrieval),
            },
        },
    }
    _atomic_json(manifest_path, manifest)
    return {
        **manifest,
        "output_dir": str(destination),
        "memorization_train": str(memorization_path),
        "retrieval_train": str(retrieval_path),
        "manifest": str(manifest_path),
    }
