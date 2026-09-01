"""Validated checkpoint discovery and pipeline-lineage sidecars.

The pipeline has two checkpoint formats: HuggingFace Trainer directories named
``checkpoint-N`` and ToolWeaver's resumable ``last.pt``.  This module keeps the
format-specific checks in one place and refuses a checkpoint whose provenance
does not match the current stage by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .io import atomic_write_json, read_json, utc_now


class CheckpointError(RuntimeError):
    """Raised for an incomplete, unsafe, or incompatible checkpoint."""


LINEAGE_FILENAME = "pipeline_lineage.json"
CODEBOOK_LINEAGE_SUFFIX = ".pipeline_lineage.json"
_ROUTER_NAME = re.compile(r"^checkpoint-([0-9]+)$")


@dataclass(frozen=True)
class CheckpointSelection:
    """The checkpoint chosen for a training attempt."""

    path: Path
    kind: str
    global_step: int
    explicit: bool
    legacy_without_sidecar: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "global_step": self.global_step,
            "explicit": self.explicit,
            "legacy_without_sidecar": self.legacy_without_sidecar,
        }


def router_sidecar_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint) / LINEAGE_FILENAME


def codebook_sidecar_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_name(path.name + CODEBOOK_LINEAGE_SUFFIX)


def _mapping(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"invalid JSON checkpoint metadata: {path}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint metadata must be an object: {path}")
    return value


def _global_step(value: Any, *, source: Path) -> int:
    if isinstance(value, bool):
        raise CheckpointError(f"checkpoint global_step is invalid: {source}")
    try:
        step = int(value)
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"checkpoint has no integer global_step: {source}") from error
    if step < 0:
        raise CheckpointError(f"checkpoint global_step must be non-negative: {source}")
    return step


def _read_codebook_payload(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"codebook checkpoint is missing: {path}")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - training extra regression
        raise CheckpointError("PyTorch is required to validate a codebook checkpoint") from error
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch before weights_only
            payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise CheckpointError(f"cannot load codebook checkpoint: {path}") from error
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"codebook checkpoint is not a mapping: {path}")
    return payload


def _validate_lineage(
    sidecar: Path,
    expected: Mapping[str, Any],
    *,
    kind: str,
    global_step: int,
    allow_legacy: bool,
) -> bool:
    """Validate sidecar and return whether a legacy exception was used."""

    if not sidecar.is_file():
        if allow_legacy:
            return True
        raise CheckpointError(
            f"checkpoint has no {LINEAGE_FILENAME} sidecar: {sidecar}; "
            "pass --allow-legacy-checkpoint only for a trusted legacy checkpoint"
        )
    payload = _mapping(sidecar)
    if payload.get("schema_version") != 1:
        raise CheckpointError(f"unsupported checkpoint lineage schema: {sidecar}")
    if payload.get("checkpoint_kind") != kind:
        raise CheckpointError(f"checkpoint lineage kind differs: {sidecar}")
    if _global_step(payload.get("global_step"), source=sidecar) != global_step:
        raise CheckpointError(f"checkpoint sidecar global_step disagrees: {sidecar}")
    for key in ("run_id", "stage", "stage_config_hash", "input_artifacts"):
        if payload.get(key) != expected.get(key):
            raise CheckpointError(f"checkpoint lineage differs for {key}: {sidecar}")
    if payload.get("code_plan_sha256") != expected.get("code_plan_sha256"):
        raise CheckpointError(f"checkpoint lineage differs for code_plan_sha256: {sidecar}")
    return False


def validate_router_checkpoint(
    checkpoint: str | Path,
    *,
    expected_lineage: Mapping[str, Any],
    allow_legacy: bool = False,
) -> CheckpointSelection:
    path = Path(checkpoint).expanduser().resolve()
    match = _ROUTER_NAME.fullmatch(path.name)
    if match is None or not path.is_dir():
        raise CheckpointError("router checkpoint must be a checkpoint-<global_step> directory")
    step = int(match.group(1))
    trainer_state = path / "trainer_state.json"
    state = _mapping(trainer_state)
    if _global_step(state.get("global_step"), source=trainer_state) != step:
        raise CheckpointError(
            "router checkpoint directory and trainer_state.json disagree on global_step"
        )
    legacy = _validate_lineage(
        router_sidecar_path(path), expected_lineage, kind="router", global_step=step,
        allow_legacy=allow_legacy,
    )
    return CheckpointSelection(path, "router", step, explicit=False, legacy_without_sidecar=legacy)


def validate_codebook_checkpoint(
    checkpoint: str | Path,
    *,
    expected_lineage: Mapping[str, Any],
    allow_legacy: bool = False,
) -> CheckpointSelection:
    path = Path(checkpoint).expanduser().resolve()
    payload = _read_codebook_payload(path)
    for key in ("model_state", "optimizer_state", "scheduler_state", "rng_state"):
        if key not in payload:
            raise CheckpointError(f"codebook checkpoint is not resumable; missing {key}: {path}")
    step = _global_step(payload.get("global_step"), source=path)
    legacy = _validate_lineage(
        codebook_sidecar_path(path), expected_lineage, kind="codebook", global_step=step,
        allow_legacy=allow_legacy,
    )
    return CheckpointSelection(path, "codebook", step, explicit=False, legacy_without_sidecar=legacy)


def validate_checkpoint(
    checkpoint: str | Path,
    *,
    kind: str,
    expected_lineage: Mapping[str, Any],
    allow_legacy: bool = False,
) -> CheckpointSelection:
    if kind == "router":
        return validate_router_checkpoint(
            checkpoint, expected_lineage=expected_lineage, allow_legacy=allow_legacy
        )
    if kind == "codebook":
        return validate_codebook_checkpoint(
            checkpoint, expected_lineage=expected_lineage, allow_legacy=allow_legacy
        )
    raise CheckpointError(f"unsupported checkpoint kind: {kind}")


def select_checkpoint(
    root: str | Path,
    *,
    kind: str,
    expected_lineage: Mapping[str, Any],
    explicit: str | Path | None = None,
    allow_legacy: bool = False,
) -> CheckpointSelection | None:
    """Validate an explicit checkpoint or choose the newest compatible one."""

    if explicit is not None:
        selection = validate_checkpoint(
            explicit, kind=kind, expected_lineage=expected_lineage,
            allow_legacy=allow_legacy,
        )
        return CheckpointSelection(
            selection.path, selection.kind, selection.global_step, True,
            selection.legacy_without_sidecar,
        )

    directory = Path(root).expanduser().resolve()
    candidates = (
        sorted(directory.glob("checkpoint-*"), key=lambda item: item.name)
        if kind == "router" and directory.is_dir()
        else sorted(directory.rglob("last.pt"), key=lambda item: item.as_posix())
        if kind == "codebook" and directory.is_dir()
        else []
    )
    valid: list[CheckpointSelection] = []
    for candidate in candidates:
        try:
            valid.append(
                validate_checkpoint(
                    candidate, kind=kind, expected_lineage=expected_lineage,
                    allow_legacy=allow_legacy,
                )
            )
        except CheckpointError:
            continue
    if not valid:
        return None
    return max(valid, key=lambda value: value.global_step)


def write_checkpoint_sidecar(
    checkpoint: str | Path,
    *,
    kind: str,
    lineage: Mapping[str, Any],
    global_step: int,
) -> Path:
    """Write provenance only after the checkpoint itself is complete."""

    if kind not in {"router", "codebook"}:
        raise CheckpointError(f"unsupported checkpoint kind: {kind}")
    target = router_sidecar_path(checkpoint) if kind == "router" else codebook_sidecar_path(checkpoint)
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "checkpoint_kind": kind,
        "run_id": lineage.get("run_id"),
        "stage": lineage.get("stage"),
        "stage_config_hash": lineage.get("stage_config_hash"),
        "input_artifacts": dict(lineage.get("input_artifacts") or {}),
        "code_plan_sha256": lineage.get("code_plan_sha256"),
        "global_step": global_step,
    }
    atomic_write_json(target, payload)
    return target


def write_sidecar_from_lineage_file(
    checkpoint: str | Path,
    *,
    kind: str,
    lineage_file: str | Path,
    global_step: int,
) -> Path:
    """Small subprocess-friendly entry point used by training save hooks."""

    lineage = _mapping(Path(lineage_file))
    return write_checkpoint_sidecar(
        checkpoint, kind=kind, lineage=lineage, global_step=global_step
    )
