"""Self-contained decoder artifacts stored beside a trained router."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from .router import (
    RouterDataError,
    active_skill_ids_from_registry,
    load_virtual_tokens,
    normalize_code_rows,
    validate_registry_assignments,
)
from .skillret import sha256_file


DECODE_MAP_FILENAME = "skill_decode_map.json"
BUNDLED_VIRTUAL_TOKENS_FILENAME = "virtual_tokens.txt"
DECODE_MAP_SCHEMA_VERSION = 1

_METADATA_FIELDS = (
    "name",
    "description",
    "text",
    "domain",
    "capability_zh",
    "mobile_fit",
    "roles",
    "rank",
    "source_url",
)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RouterDataError(f"expected a JSON object: {path}")
    return payload


def _catalog_by_id(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        raw_id = row.get("skill_id", row.get("id"))
        if not isinstance(raw_id, str) or not raw_id:
            raise RouterDataError(f"catalog row {row_number} has no skill_id")
        if raw_id in catalog:
            raise RouterDataError(f"duplicate catalog skill: {raw_id!r}")
        metadata: dict[str, Any] = {"skill_id": raw_id}
        for field in _METADATA_FIELDS:
            value = row.get(field)
            if value is not None:
                metadata[field] = value
        raw_name = metadata.get("name")
        metadata["name"] = (
            raw_name.strip()
            if isinstance(raw_name, str) and raw_name.strip()
            else raw_id
        )
        catalog[raw_id] = metadata
    return catalog


def build_skill_decode_map(
    *,
    catalog_rows: Sequence[Mapping[str, Any]],
    code_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    virtual_tokens: Sequence[str],
    provenance: Mapping[str, Any] | None = None,
    supervision_rows: Sequence[Mapping[str, Any]] | None = None,
    supervision_phase: str | None = None,
) -> dict[str, Any]:
    """Build exact path decoding plus token-level candidate audit mappings."""

    validate_registry_assignments(registry, code_rows)
    skill_to_tokens, num_levels = normalize_code_rows(code_rows)
    active_ids = tuple(active_skill_ids_from_registry(registry))
    catalog = _catalog_by_id(catalog_rows)
    missing_catalog = sorted(set(active_ids).difference(catalog))
    if missing_catalog:
        raise RouterDataError(
            "active code registry contains skills missing from catalog: "
            + ", ".join(missing_catalog[:10])
        )
    missing_codes = sorted(set(active_ids).difference(skill_to_tokens))
    if missing_codes:
        raise RouterDataError(
            "active registry contains skills missing from codes: "
            + ", ".join(missing_codes[:10])
        )
    token_namespace = tuple(str(token) for token in virtual_tokens)
    if not token_namespace or len(set(token_namespace)) != len(token_namespace):
        raise RouterDataError("virtual token namespace must be non-empty and unique")
    token_set = set(token_namespace)
    used_tokens = {
        token for skill_id in active_ids for token in skill_to_tokens[skill_id]
    }
    unknown_tokens = sorted(used_tokens.difference(token_set))
    if unknown_tokens:
        raise RouterDataError(
            "code rows use tokens absent from virtual token namespace: "
            + ", ".join(unknown_tokens[:10])
        )

    indices_by_skill: dict[str, list[int] | None] = {}
    for row in code_rows:
        skill_id = row.get("skill_id", row.get("id"))
        if not isinstance(skill_id, str):
            continue
        raw_indices = row.get("indices")
        indices_by_skill[skill_id] = (
            [int(value) for value in raw_indices]
            if isinstance(raw_indices, (list, tuple))
            else None
        )

    path_members: dict[tuple[str, ...], list[str]] = defaultdict(list)
    token_members: dict[str, list[str]] = defaultdict(list)
    skill_to_code: dict[str, dict[str, Any]] = {}
    for skill_id in active_ids:
        tokens = tuple(skill_to_tokens[skill_id])
        if len(tokens) != num_levels:
            raise RouterDataError(f"skill {skill_id!r} has the wrong code length")
        path_members[tokens].append(skill_id)
        for token in tokens:
            token_members[token].append(skill_id)
        skill_to_code[skill_id] = {
            "tokens": list(tokens),
            "code_text": "".join(tokens),
            "indices": indices_by_skill.get(skill_id),
        }

    def summaries(skill_ids: Sequence[str]) -> list[dict[str, str]]:
        return [
            {"skill_id": skill_id, "name": str(catalog[skill_id]["name"])}
            for skill_id in skill_ids
        ]

    paths = []
    for tokens, members in sorted(path_members.items()):
        ordered_members = sorted(members)
        paths.append(
            {
                "code_text": "".join(tokens),
                "tokens": list(tokens),
                "skill_ids": ordered_members,
                "candidates": summaries(ordered_members),
            }
        )
    token_to_skill_ids = {
        token: sorted(token_members.get(token, ()))
        for token in token_namespace
    }
    token_to_candidates = {
        token: summaries(skill_ids)
        for token, skill_ids in token_to_skill_ids.items()
    }
    supervision = None
    if supervision_rows is not None:
        target_counts: Counter[str] = Counter()
        for row_number, row in enumerate(supervision_rows, start=1):
            raw_targets = row.get("target_skill_ids", row.get("positive_skill_ids"))
            if raw_targets is None and isinstance(row.get("skill_id"), str):
                raw_targets = [row["skill_id"]]
            if not isinstance(raw_targets, (list, tuple)) or not raw_targets:
                raise RouterDataError(
                    f"supervision row {row_number} has no target skill IDs"
                )
            unique_targets = {
                str(skill_id).strip()
                for skill_id in raw_targets
                if str(skill_id).strip()
            }
            unknown_targets = sorted(unique_targets.difference(active_ids))
            if unknown_targets:
                raise RouterDataError(
                    "supervision references skills outside the active registry: "
                    + ", ".join(unknown_targets[:10])
                )
            target_counts.update(unique_targets)
        supervised_ids = sorted(target_counts)
        for skill_id in active_ids:
            catalog[skill_id]["train_target_count"] = target_counts[skill_id]
            catalog[skill_id]["has_train_target"] = target_counts[skill_id] > 0
        supervision = {
            "phase": supervision_phase,
            "num_examples": len(supervision_rows),
            "num_supervised_skills": len(supervised_ids),
            "num_unsupervised_skills": len(active_ids) - len(supervised_ids),
            "supervised_skill_ids": supervised_ids,
            "target_counts": {
                skill_id: target_counts[skill_id] for skill_id in supervised_ids
            },
        }
    return {
        "schema_version": DECODE_MAP_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_levels": num_levels,
        "num_skills": len(active_ids),
        "num_paths": len(paths),
        "virtual_tokens": list(token_namespace),
        "skills": {
            skill_id: catalog[skill_id]
            for skill_id in sorted(active_ids)
        },
        "skill_to_code": {
            skill_id: skill_to_code[skill_id]
            for skill_id in sorted(skill_to_code)
        },
        "paths": paths,
        "token_to_skill_ids": token_to_skill_ids,
        "token_to_candidates": token_to_candidates,
        "supervision": supervision,
        "provenance": dict(provenance or {}),
    }


def validate_skill_decode_map(payload: Mapping[str, Any]) -> None:
    """Reject incomplete or internally inconsistent bundled decoder maps."""

    if payload.get("schema_version") != DECODE_MAP_SCHEMA_VERSION:
        raise RouterDataError("unsupported skill decode map schema")
    num_levels = payload.get("num_levels")
    skills = payload.get("skills")
    skill_to_code = payload.get("skill_to_code")
    paths = payload.get("paths")
    tokens = payload.get("virtual_tokens")
    if not isinstance(num_levels, int) or num_levels < 1:
        raise RouterDataError("decode map has invalid num_levels")
    if not isinstance(skills, dict) or not skills:
        raise RouterDataError("decode map has no skills")
    if not isinstance(skill_to_code, dict) or set(skill_to_code) != set(skills):
        raise RouterDataError("decode map skill_to_code does not match skills")
    if not isinstance(paths, list) or not paths:
        raise RouterDataError("decode map has no paths")
    if not isinstance(tokens, list) or len(set(tokens)) != len(tokens):
        raise RouterDataError("decode map has an invalid virtual token namespace")
    token_set = set(tokens)
    reconstructed: dict[str, tuple[str, ...]] = {}
    for path in paths:
        if not isinstance(path, dict):
            raise RouterDataError("decode map path must be an object")
        raw_path = path.get("tokens")
        members = path.get("skill_ids")
        if not isinstance(raw_path, list) or len(raw_path) != num_levels:
            raise RouterDataError("decode map path has the wrong code length")
        if not set(raw_path) <= token_set:
            raise RouterDataError("decode map path contains an unknown token")
        if not isinstance(members, list) or not members:
            raise RouterDataError("decode map path has no skill_ids")
        for skill_id in members:
            if skill_id in reconstructed:
                raise RouterDataError(f"skill {skill_id!r} appears in multiple paths")
            reconstructed[str(skill_id)] = tuple(str(value) for value in raw_path)
    if set(reconstructed) != set(skills):
        raise RouterDataError("decode map paths do not cover every skill")
    for skill_id, details in skill_to_code.items():
        if not isinstance(details, dict):
            raise RouterDataError("decode map skill_to_code value must be an object")
        raw_tokens = details.get("tokens")
        if tuple(raw_tokens or ()) != reconstructed[skill_id]:
            raise RouterDataError(f"decode map code mismatch for {skill_id!r}")


def load_skill_decode_map(path: str | Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    validate_skill_decode_map(payload)
    return payload


def dump_router_decoder_artifacts(
    *,
    output_dir: str | Path,
    catalog_path: str | Path,
    codes_path: str | Path,
    registry_path: str | Path,
    virtual_tokens_path: str | Path,
    training_data_path: str | Path | None = None,
    supervision_phase: str | None = None,
) -> dict[str, Any]:
    """Write a portable decoder map and token namespace beside a model dump."""

    from .router import read_jsonl

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    catalog_path = Path(catalog_path).resolve()
    codes_path = Path(codes_path).resolve()
    registry_path = Path(registry_path).resolve()
    virtual_tokens_path = Path(virtual_tokens_path).resolve()
    training_data_path = (
        Path(training_data_path).resolve() if training_data_path is not None else None
    )
    required_paths = [catalog_path, codes_path, registry_path, virtual_tokens_path]
    if training_data_path is not None:
        required_paths.append(training_data_path)
    for path in required_paths:
        if not path.is_file():
            raise RouterDataError(f"decoder artifact input does not exist: {path}")

    index_manifest_path = codes_path.parent / "manifest.json"
    stage1_checkpoint_sha256 = None
    index_manifest_sha256 = None
    if index_manifest_path.is_file():
        index_manifest = _load_json_object(index_manifest_path)
        stage1_checkpoint_sha256 = index_manifest.get("checkpoint_sha256")
        index_manifest_sha256 = sha256_file(index_manifest_path)
    provenance = {
        "catalog_sha256": sha256_file(catalog_path),
        "codes_sha256": sha256_file(codes_path),
        "registry_sha256": sha256_file(registry_path),
        "virtual_tokens_sha256": sha256_file(virtual_tokens_path),
        "index_manifest_sha256": index_manifest_sha256,
        "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
        "training_data_sha256": (
            sha256_file(training_data_path) if training_data_path is not None else None
        ),
    }
    supervision_rows = (
        read_jsonl(training_data_path) if training_data_path is not None else None
    )
    payload = build_skill_decode_map(
        catalog_rows=read_jsonl(catalog_path),
        code_rows=read_jsonl(codes_path),
        registry=_load_json_object(registry_path),
        virtual_tokens=load_virtual_tokens(virtual_tokens_path),
        provenance=provenance,
        supervision_rows=supervision_rows,
        supervision_phase=supervision_phase,
    )
    decode_path = destination / DECODE_MAP_FILENAME
    temporary = decode_path.with_suffix(decode_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, decode_path)
    bundled_tokens_path = destination / BUNDLED_VIRTUAL_TOKENS_FILENAME
    if virtual_tokens_path != bundled_tokens_path.resolve():
        shutil.copyfile(virtual_tokens_path, bundled_tokens_path)
    return {
        "decode_map": DECODE_MAP_FILENAME,
        "decode_map_sha256": sha256_file(decode_path),
        "virtual_tokens": BUNDLED_VIRTUAL_TOKENS_FILENAME,
        "virtual_tokens_sha256": sha256_file(bundled_tokens_path),
        "num_skills": payload["num_skills"],
        "num_paths": payload["num_paths"],
        "num_levels": payload["num_levels"],
        "supervision": (
            {
                key: payload["supervision"][key]
                for key in (
                    "phase",
                    "num_examples",
                    "num_supervised_skills",
                    "num_unsupervised_skills",
                )
            }
            if payload["supervision"] is not None
            else None
        ),
        "provenance": provenance,
    }
