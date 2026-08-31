"""Atomic Run and Stage state management."""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence

from .io import atomic_write_json, file_lock, read_json, utc_now


class PipelineStateError(RuntimeError):
    """Raised for corrupt state or invalid lifecycle transitions."""


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"
    SKIPPED = "skipped"


class RunStateStore:
    SCHEMA_VERSION = 1

    def __init__(self, run_dir: str | Path, stage_directories: Mapping[str, str]) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_manifest_path = self.run_dir / "run_manifest.json"
        self.lock_path = self.run_dir / ".pipeline.lock"
        self.stage_directories = dict(stage_directories)

    def initialize(
        self,
        *,
        run_id: str,
        name: str,
        config_hash: str,
        config_path: str,
        source_config_path: str,
        repo_root: str,
        stage_order: Sequence[str],
        git_commit: str | None,
        parent_run_id: str | None = None,
        reused_artifacts: Sequence[str] = (),
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.run_manifest_path.exists():
            raise PipelineStateError(f"run already exists: {self.run_dir}")
        now = utc_now()
        atomic_write_json(
            self.run_manifest_path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": run_id,
                "name": name,
                "status": "created",
                "created_at": now,
                "updated_at": now,
                "config_hash": config_hash,
                "config_path": config_path,
                "source_config_path": source_config_path,
                "repo_root": repo_root,
                "git_commit": git_commit,
                "parent_run_id": parent_run_id,
                "reused_artifacts": list(reused_artifacts),
                "stage_order": list(stage_order),
                "current_stage": None,
                "last_error": None,
            },
        )
        for stage in stage_order:
            self.write_stage(
                stage,
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "stage": stage,
                    "status": StageStatus.PENDING.value,
                    "attempt": 0,
                    "created_at": now,
                    "updated_at": now,
                    "started_at": None,
                    "finished_at": None,
                    "input_artifacts": {},
                    "config_hash": "",
                    "progress": {},
                    "outputs": {},
                    "last_error": None,
                },
            )

    def stage_dir(self, stage: str) -> Path:
        try:
            name = self.stage_directories[stage]
        except KeyError as error:
            raise PipelineStateError(f"unknown stage: {stage}") from error
        stage_root = self.run_dir / "stages"
        resolved_root = stage_root.resolve()
        try:
            resolved_root.relative_to(self.run_dir)
        except ValueError as error:
            raise PipelineStateError(
                f"stage root escapes Run directory: {stage_root}"
            ) from error
        path = (stage_root / name).resolve()
        try:
            path.relative_to(resolved_root)
            path.relative_to(self.run_dir)
        except ValueError as error:
            raise PipelineStateError(
                f"stage directory escapes Run directory: {name!r}"
            ) from error
        return path

    def stage_state_path(self, stage: str) -> Path:
        return self.stage_dir(stage) / "stage_state.json"

    def read_run(self) -> dict[str, Any]:
        if not self.run_manifest_path.is_file():
            raise PipelineStateError(f"run manifest is missing: {self.run_manifest_path}")
        value = read_json(self.run_manifest_path)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise PipelineStateError(f"invalid run manifest: {self.run_manifest_path}")
        return value

    def update_run(self, **changes: Any) -> dict[str, Any]:
        with file_lock(self.run_dir / ".run_manifest.lock"):
            value = self.read_run()
            value.update(changes)
            value["updated_at"] = utc_now()
            atomic_write_json(self.run_manifest_path, value)
            return value

    def read_stage(self, stage: str) -> dict[str, Any]:
        path = self.stage_state_path(stage)
        if not path.is_file():
            raise PipelineStateError(f"stage state is missing: {path}")
        value = read_json(path)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("stage") != stage
        ):
            raise PipelineStateError(f"invalid stage state: {path}")
        return value

    def write_stage(self, stage: str, value: Mapping[str, Any]) -> None:
        path = self.stage_state_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(value)
        payload["schema_version"] = self.SCHEMA_VERSION
        payload["stage"] = stage
        payload["updated_at"] = utc_now()
        atomic_write_json(path, payload)

    def update_stage(self, stage: str, **changes: Any) -> dict[str, Any]:
        with file_lock(self.stage_dir(stage) / ".state.lock"):
            value = self.read_stage(stage)
            value.update(changes)
            self.write_stage(stage, value)
            return value

    def start_stage(
        self,
        stage: str,
        *,
        config_hash: str,
        input_artifacts: Mapping[str, str],
    ) -> dict[str, Any]:
        previous = self.read_stage(stage)
        attempt = int(previous.get("attempt") or 0) + 1
        now = utc_now()
        value = {
            **previous,
            "status": StageStatus.RUNNING.value,
            "attempt": attempt,
            "started_at": now,
            "finished_at": None,
            "config_hash": config_hash,
            "input_artifacts": dict(input_artifacts),
            "progress": {},
            "outputs": {},
            "last_error": None,
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        self.write_stage(stage, value)
        return value

    def complete_stage(
        self,
        stage: str,
        *,
        outputs: Mapping[str, str],
        progress: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.read_stage(stage)
        if state.get("status") != StageStatus.RUNNING.value:
            raise PipelineStateError(f"cannot complete non-running stage: {stage}")
        return self.update_stage(
            stage,
            status=StageStatus.COMPLETED.value,
            finished_at=utc_now(),
            outputs=dict(outputs),
            progress=dict(progress or state.get("progress") or {}),
            last_error=None,
        )

    def fail_stage(
        self,
        stage: str,
        *,
        error_type: str,
        error: str,
        traceback_path: str | None,
        elapsed_ms: float | None = None,
    ) -> dict[str, Any]:
        return self.update_stage(
            stage,
            status=StageStatus.FAILED.value,
            finished_at=utc_now(),
            last_error={
                "type": error_type,
                "message": error,
                "traceback_path": traceback_path,
            },
            elapsed_ms=elapsed_ms,
        )

    def invalidate(self, stages: Sequence[str], *, reason: str) -> None:
        now = utc_now()
        for stage in stages:
            state = self.read_stage(stage)
            if state.get("status") == StageStatus.PENDING.value:
                continue
            self.update_stage(
                stage,
                status=StageStatus.INVALIDATED.value,
                invalidated_at=now,
                invalidation_reason=reason,
            )
