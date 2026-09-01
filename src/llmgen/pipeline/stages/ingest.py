"""Candidate-input Stage adapter."""

from __future__ import annotations

from pathlib import Path

from ..io import atomic_write_bytes, atomic_write_json, atomic_write_jsonl, read_json, sha256_file, utc_now
from ..schema import catalog_rows, read_candidate_file
from .base import ArtifactOutput, StageContext, StageResult


def ingest(context: StageContext) -> StageResult:
    """Freeze, normalize, and catalog the configured candidate JSONL."""

    configured_source = Path(str(context.config.require("input.candidates"))).expanduser()
    configured_source = (configured_source if configured_source.is_absolute() else context.repo_root / configured_source).resolve()
    frozen_fingerprint = read_json(context.run_dir / "config" / "candidate_input.json")
    frozen_path = context.run_dir / str(frozen_fingerprint.get("frozen_path") or "source/candidates.input.jsonl")
    source = frozen_path if frozen_path.is_file() else configured_source
    expected_hash = str(frozen_fingerprint.get("sha256") or "")
    actual_hash = sha256_file(source)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("candidate input changed after Run creation; create a new Run or fork with the intended candidate file")
    candidates = read_candidate_file(source, id_policy=str(context.config.require("input.id_policy")), preserve_metadata=bool(context.config.require("input.preserve_metadata")))
    if len(candidates) == 1 and context.config.require("input.single_candidate_policy") == "error":
        raise ValueError("a single-candidate run cannot construct multi-Skill retrieval data; the current generic adapter requires at least two candidates")
    source_dir = context.run_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    manual_fingerprint = read_json(
        context.run_dir / "config" / "manual_alignment_input.json"
    )
    manual_alignment = context.run_dir / str(
        manual_fingerprint.get("frozen_path")
        or "source/manual_alignment.input.jsonl"
    )
    manual_expected_hash = str(manual_fingerprint.get("sha256") or "")
    if (
        not manual_alignment.is_file()
        or not manual_expected_hash
        or sha256_file(manual_alignment) != manual_expected_hash
    ):
        raise ValueError(
            "manual alignment input changed after Run creation; create a new "
            "Run or fork with the intended curated file"
        )
    input_copy = source_dir / "candidates.input.jsonl"
    normalized_path = source_dir / "candidates.normalized.jsonl"
    catalog_path = source_dir / "catalog.jsonl"
    manifest_path = source_dir / "candidate_manifest.json"
    atomic_write_bytes(input_copy, source.read_bytes())
    atomic_write_jsonl(normalized_path, candidates)
    atomic_write_jsonl(catalog_path, catalog_rows(candidates, source="source/candidates.input.jsonl"))
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source": str(configured_source),
        "candidate_count": len(candidates),
        "unique_id_count": len({row["skill_id"] for row in candidates}),
        "unique_name_count": len({row["name"] for row in candidates}),
        "ordered_skill_ids": [row["skill_id"] for row in candidates],
        "execution_mode": "alignment_only" if len(candidates) == 1 else "retrieval",
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in (input_copy, normalized_path, catalog_path)},
    }
    atomic_write_json(manifest_path, manifest)
    context.update_progress(completed=len(candidates), total=len(candidates))
    return StageResult(
        artifacts=(
            ArtifactOutput("candidates.input", input_copy, "candidate_input/v1"),
            ArtifactOutput("candidates.normalized", normalized_path, "candidate/v1"),
            ArtifactOutput("candidates.catalog", catalog_path, "candidate_catalog/v1"),
            ArtifactOutput("candidates.manifest", manifest_path, "candidate_manifest/v1"),
            ArtifactOutput(
                "inputs.manual_alignment",
                manual_alignment,
                "manual_alignment_input/v1",
            ),
        ),
        progress={"candidate_count": len(candidates)},
    )
