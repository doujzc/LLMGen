"""DAG execution and recovery for generic candidate pipeline Runs.

This module deliberately contains no data or training algorithm.  It owns the
durable boundary around :class:`~llmgen.pipeline.stages.base.StageSpec`: state
transitions, artifact lineage, reuse validation, and useful failure records.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactError, ArtifactRecord, ArtifactRegistry
from .config import PipelineConfig, PipelineConfigError, load_pipeline_config
from .io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    file_lock,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
    utc_now,
)
from .logging import PipelineLogger
from .resources import collect_run_provenance
from .state import PipelineStateError, RunStateStore, StageStatus
from .stages import StageContext, StageResult, StageSpec, default_stage_specs
from .stages.base import StageExecutionError, recover_stale_stage_commands


class PipelineRunnerError(RuntimeError):
    """Raised when a requested pipeline action cannot safely proceed."""


class StageDependencyError(PipelineRunnerError):
    """Raised when independently running a stage lacks a verified upstream."""


@dataclass(frozen=True)
class StageExecution:
    """A concise result for one requested stage."""

    stage: str
    action: str
    attempt: int | None = None


def _repo_root(value: str | Path | None) -> Path:
    """Resolve the repository root without treating the shell as a parser."""

    root = Path(value or Path.cwd()).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return root
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()).resolve()
    return root


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_specs(specs: Sequence[StageSpec]) -> tuple[StageSpec, ...]:
    values = tuple(specs)
    names = [spec.name for spec in values]
    if not values:
        raise PipelineRunnerError("pipeline must declare at least one stage")
    if len(names) != len(set(names)):
        raise PipelineRunnerError("pipeline has duplicate stage names")
    directories = [spec.directory for spec in values]
    if len(directories) != len(set(directories)):
        raise PipelineRunnerError("pipeline has duplicate stage directories")
    safe_stage_name = re.compile(r"^[a-z0-9][a-z0-9-]*$")
    for spec in values:
        if not safe_stage_name.fullmatch(spec.name):
            raise PipelineRunnerError(f"unsafe stage name: {spec.name!r}")
        directory = Path(spec.directory)
        if (
            not spec.directory
            or directory.is_absolute()
            or directory == Path(".")
            or any(part in {"", ".", ".."} for part in directory.parts)
        ):
            raise PipelineRunnerError(
                f"unsafe stage directory for {spec.name}: {spec.directory!r}"
            )
    known = set(names)
    for spec in values:
        unknown = set(spec.dependencies).difference(known)
        if unknown:
            raise PipelineRunnerError(
                f"stage {spec.name} has unknown dependencies: {', '.join(sorted(unknown))}"
            )
        if spec.name in spec.dependencies:
            raise PipelineRunnerError(f"stage {spec.name} cannot depend on itself")

    visiting: set[str] = set()
    visited: set[str] = set()
    by_name = {spec.name: spec for spec in values}

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise PipelineRunnerError(f"pipeline dependency cycle includes {name}")
        visiting.add(name)
        for dependency in by_name[name].dependencies:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in names:
        visit(name)
    # Stage numbering is presentation-only, but execution needs a topological
    # order.  Reject a surprising declaration rather than silently reordering it.
    position = {name: index for index, name in enumerate(names)}
    for spec in values:
        if any(position[dependency] >= position[spec.name] for dependency in spec.dependencies):
            raise PipelineRunnerError(
                f"stage {spec.name} must be declared after its dependencies"
            )
    return values


def _secret_values(config: PipelineConfig) -> tuple[str, ...]:
    values: list[str] = []
    for provider in ("generation", "review", "embedding"):
        name = config.get(f"providers.{provider}.api_key_env")
        if isinstance(name, str) and name and os.environ.get(name):
            values.append(os.environ[name])
    return tuple(values)


def _environment_snapshot(config: PipelineConfig) -> dict[str, Any]:
    """Persist useful provenance without serializing environment secrets."""

    referenced_keys = sorted(
        {
            str(config.get(f"providers.{provider}.api_key_env"))
            for provider in ("generation", "review", "embedding")
            if str(config.get(f"providers.{provider}.api_key_env") or "")
        }
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "api_key_environment": {
            name: {"configured": bool(os.environ.get(name))}
            for name in referenced_keys
        },
    }


def _candidate_path(config: PipelineConfig, repo_root: Path) -> Path:
    value = Path(str(config.require("input.candidates"))).expanduser()
    return (value if value.is_absolute() else repo_root / value).resolve()


def _manual_alignment_path(
    config: PipelineConfig,
    repo_root: Path,
) -> Path | None:
    value = str(config.get("data_generation.manual_alignment_path") or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


class PipelineRunner:
    """Execute an immutable Run using injectable stage specifications."""

    def __init__(
        self,
        run_dir: str | Path,
        config: PipelineConfig,
        *,
        stage_specs: Sequence[StageSpec] | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.config = config
        self.specs = _validate_specs(stage_specs or default_stage_specs())
        self._by_name = {spec.name: spec for spec in self.specs}
        self.repo_root = _repo_root(repo_root)
        self.registry = ArtifactRegistry(self.run_dir)
        self.state = RunStateStore(
            self.run_dir,
            {spec.name: spec.directory for spec in self.specs},
        )

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        *,
        stage_specs: Sequence[StageSpec] | None = None,
        repo_root: str | Path | None = None,
    ) -> "PipelineRunner":
        """Open a pre-existing run from its immutable resolved config snapshot."""

        directory = Path(run_dir).expanduser().resolve()
        config_path = directory / "config" / "pipeline.resolved.yaml"
        config = load_pipeline_config(config_path)
        runner = cls(directory, config, stage_specs=stage_specs, repo_root=repo_root)
        manifest = runner.state.read_run()
        if manifest.get("config_hash") != config.hash:
            raise PipelineRunnerError(
                "resolved configuration hash does not match run manifest; "
                "the Run snapshot may have been modified"
            )
        expected_order = [spec.name for spec in runner.specs]
        if manifest.get("stage_order") != expected_order:
            raise PipelineRunnerError("run stage order does not match this pipeline")
        return runner

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def _stage_config_view(self, spec: StageSpec) -> dict[str, Any]:
        """Project config for built-in stages, conservatively hash all for injected ones."""

        try:
            view = self.config.stage_view(spec.name)
        except PipelineConfigError:
            view = {
                "custom_stage": spec.name,
                "pipeline": self.config.to_dict(),
            }
        implementation: dict[str, str] = {}
        for raw_path in spec.implementation_paths:
            path = (self.repo_root / raw_path).resolve()
            try:
                relative = path.relative_to(self.repo_root)
            except ValueError as error:
                raise PipelineRunnerError(
                    f"stage {spec.name} implementation path escapes repository: "
                    f"{raw_path}"
                ) from error
            if not path.is_file():
                raise PipelineRunnerError(
                    f"stage {spec.name} implementation file is missing: {path}"
                )
            implementation[relative.as_posix()] = sha256_file(path)
        if implementation:
            view["stage.implementation"] = implementation
        # ``runtime.*=auto`` is only a request. The resolved topology frozen at
        # Run creation is part of the effective training configuration, while
        # the user-authored config hash remains stable and portable.
        if spec.name in {
            "train-codebook",
            "assign-codes",
            "train-memorization",
            "train-alignment",
            "train-retrieval",
            "evaluate",
            "export",
        }:
            provenance_path = self.run_dir / "config" / "provenance.json"
            if provenance_path.is_file():
                provenance = read_json(provenance_path)
                # Training output depends on the exact code checkout, package
                # stack, accelerator topology, and base-model bytes—not only
                # on the user-authored YAML values. Bind the complete frozen
                # provenance snapshot into the Stage/checkpoint lineage hash.
                view["run.training_provenance"] = provenance
        if spec.name == "export":
            # The exported manifest and report intentionally embed the full
            # resolved-config hash, so any config change changes this Stage's
            # material output even when the trained model bytes are reusable.
            view["run.full_config_sha256"] = self.config.hash
        return view

    def _stage_config_hash(self, spec: StageSpec) -> str:
        return sha256_json(self._stage_config_view(spec))

    def _require_stage(self, stage: str) -> StageSpec:
        try:
            return self._by_name[stage]
        except KeyError as error:
            raise PipelineRunnerError(f"unknown stage: {stage}") from error

    def _selected_specs(
        self,
        *,
        from_stage: str | None = None,
        to_stage: str | None = None,
    ) -> tuple[StageSpec, ...]:
        start = 0 if from_stage is None else self.stage_names.index(self._require_stage(from_stage).name)
        end = len(self.specs) - 1 if to_stage is None else self.stage_names.index(self._require_stage(to_stage).name)
        if start > end:
            raise PipelineRunnerError("--from stage must not be after --to stage")
        return self.specs[start : end + 1]

    def _descendants(self, stage: str) -> tuple[str, ...]:
        affected = {stage}
        changed = True
        while changed:
            changed = False
            for spec in self.specs:
                if spec.name not in affected and any(
                    dependency in affected for dependency in spec.dependencies
                ):
                    affected.add(spec.name)
                    changed = True
        return tuple(spec.name for spec in self.specs if spec.name in affected)

    def _input_hashes(self, spec: StageSpec) -> dict[str, str]:
        values: dict[str, str] = {}
        if spec.name == "ingest":
            for logical_name, filename in (
                ("external.candidates", "candidate_input.json"),
                ("external.manual_alignment", "manual_alignment_input.json"),
            ):
                fingerprint_path = self.run_dir / "config" / filename
                if not fingerprint_path.is_file():
                    raise StageDependencyError(
                        f"Run input fingerprint is missing: {fingerprint_path}"
                    )
                fingerprint = read_json(fingerprint_path)
                expected = str(fingerprint.get("sha256") or "")
                if not expected:
                    raise StageDependencyError(
                        f"Run input fingerprint is invalid: {fingerprint_path}"
                    )
                values[logical_name] = expected
        for logical_name in spec.required_artifacts:
            try:
                values[logical_name] = self.registry.verify(logical_name).sha256
            except ArtifactError as error:
                producer = self._producer_for(logical_name, consumer=spec)
                advice = f"python scripts/train_candidates.py stage {producer} --run-dir {self.run_dir}"
                raise StageDependencyError(
                    f"stage {spec.name} requires artifact {logical_name!r} "
                    f"from {producer}; it is unavailable or invalid ({error}). "
                    f"Run: {advice}"
                ) from error
        return values

    def _producer_for(self, logical_name: str, *, consumer: StageSpec) -> str:
        # An old Registry record is more specific than guessing from the DAG.
        try:
            return self.registry.get(logical_name).producer
        except ArtifactError:
            pass
        for spec in self.specs:
            try:
                outputs = self.state.read_stage(spec.name).get("outputs") or {}
            except PipelineStateError:
                continue
            if logical_name in outputs:
                return spec.name
        # StageSpec intentionally does not repeat output declarations.  A
        # direct dependency is nevertheless the exact next command in the
        # common case, and is more helpful than inventing an algorithmic owner.
        if len(consumer.dependencies) == 1:
            return consumer.dependencies[0]
        return "an upstream stage"

    def _dependency_ready(self, spec: StageSpec) -> None:
        """Check direct dependency states before a stand-alone execution."""

        for dependency in spec.dependencies:
            state = self.state.read_stage(dependency)
            if state.get("status") != StageStatus.COMPLETED.value or not self._can_reuse(
                self._by_name[dependency]
            ):
                raise StageDependencyError(
                    f"stage {spec.name} depends on {dependency}, which is not a "
                    "completed and verified stage. Run: "
                    f"python scripts/train_candidates.py run --run-dir {self.run_dir} "
                    f"--from {dependency} --to {dependency}"
                )

    def _can_reuse(self, spec: StageSpec) -> bool:
        """Return true only for a fully verified completed stage."""

        state = self.state.read_stage(spec.name)
        if state.get("status") != StageStatus.COMPLETED.value:
            return False
        if not (self.state.stage_dir(spec.name) / "COMPLETED").is_file():
            return False
        if state.get("config_hash") != self._stage_config_hash(spec):
            return False
        try:
            inputs = self._input_hashes(spec)
        except StageDependencyError:
            return False
        if state.get("input_artifacts") != inputs:
            return False
        outputs = state.get("outputs")
        if not isinstance(outputs, Mapping):
            return False
        try:
            for logical_name, expected_hash in outputs.items():
                record = self.registry.verify(str(logical_name))
                if record.producer != spec.name or record.sha256 != expected_hash:
                    return False
        except ArtifactError:
            return False
        return True

    def _invalidate(self, stage: str, *, reason: str) -> tuple[str, ...]:
        affected = self._descendants(stage)
        self.state.invalidate(affected, reason=reason)
        self.registry.remove_by_producer(set(affected))
        for name in affected:
            (self.state.stage_dir(name) / "COMPLETED").unlink(missing_ok=True)
        return affected

    def _register_result(
        self,
        spec: StageSpec,
        result: StageResult,
        input_hashes: Mapping[str, str],
    ) -> dict[str, str]:
        records: dict[str, ArtifactRecord] = {}
        for output in result.artifacts:
            if output.logical_name in records:
                raise PipelineRunnerError(
                    f"stage {spec.name} returned duplicate artifact {output.logical_name}"
                )
            records[output.logical_name] = self.registry.build_record(
                logical_name=output.logical_name,
                path=output.path,
                producer=spec.name,
                artifact_schema=output.artifact_schema,
                format=output.format,
                metadata=output.metadata,
                inputs=input_hashes,
                config_hash=self._stage_config_hash(spec),
            )
        self.registry.register_many(records)
        return {name: record.sha256 for name, record in records.items()}

    def _promote_attempt_output(
        self,
        context: StageContext,
        result: StageResult,
        *,
        stage_hash: str,
        input_hashes: Mapping[str, str],
    ) -> StageResult:
        """Atomically make one successful attempt visible to downstream stages."""

        staging = context.output_dir.resolve()
        formal = context.formal_output_dir.resolve()
        staging.mkdir(parents=True, exist_ok=True)
        mapped_artifacts = []
        for output in result.artifacts:
            path = output.path.expanduser().resolve()
            try:
                relative = path.relative_to(staging)
            except ValueError:
                try:
                    path.relative_to(context.attempt_dir.resolve())
                except ValueError:
                    mapped_artifacts.append(output)
                    continue
                raise PipelineRunnerError(
                    f"stage {context.spec.name} returned an attempt-local artifact "
                    f"outside its output directory: {path}"
                )
            mapped_artifacts.append(replace(output, path=formal / relative))

        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "stage": context.spec.name,
                "attempt": context.attempt,
                "created_at": utc_now(),
                "config_hash": stage_hash,
                "input_artifacts": dict(input_hashes),
                "artifacts": [
                    {
                        "logical_name": output.logical_name,
                        "path": str(
                            output.path.resolve().relative_to(self.run_dir)
                        ),
                        "artifact_schema": output.artifact_schema,
                        "format": output.format,
                        "metadata": dict(output.metadata),
                    }
                    for output in mapped_artifacts
                ],
                "progress": dict(result.progress),
            },
        )

        previous = context.attempt_dir / "previous-output"
        if previous.exists():
            raise PipelineRunnerError(
                f"attempt backup already exists and cannot be replaced: {previous}"
            )
        formal.parent.mkdir(parents=True, exist_ok=True)
        if formal.exists():
            os.replace(formal, previous)
        try:
            os.replace(staging, formal)
        except BaseException:
            if previous.exists() and not formal.exists():
                os.replace(previous, formal)
            raise
        return StageResult(
            artifacts=tuple(mapped_artifacts),
            progress=result.progress,
        )

    def _checkpoint_lineage(
        self,
        spec: StageSpec,
        *,
        stage_hash: str,
        input_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        """Build the immutable provenance training processes attach to saves."""

        code_plan_hash: str | None = None
        try:
            code_plan_hash = self.registry.verify("code.plan").sha256
        except ArtifactError:
            pass
        return {
            "schema_version": 1,
            "run_id": str(self.state.read_run()["run_id"]),
            "stage": spec.name,
            "stage_config_hash": stage_hash,
            "input_artifacts": dict(input_hashes),
            "code_plan_sha256": code_plan_hash,
            "run_provenance_sha256": (
                sha256_file(self.run_dir / "config" / "provenance.json")
                if (self.run_dir / "config" / "provenance.json").is_file()
                else None
            ),
        }

    def _execute(
        self,
        spec: StageSpec,
        *,
        resume_checkpoint: str | None,
        allow_legacy_checkpoint: bool,
    ) -> StageExecution:
        input_hashes = self._input_hashes(spec)
        stage_hash = self._stage_config_hash(spec)
        try:
            recovered_commands = recover_stale_stage_commands(
                self.state.stage_dir(spec.name),
                stage_name=spec.name,
            )
        except StageExecutionError as error:
            raise PipelineRunnerError(str(error)) from error
        if recovered_commands:
            recovery_logger = self._logger()
            for record in recovered_commands:
                recovery_logger.event(
                    "subprocess.stale_recovered",
                    stage=spec.name,
                    command_index=record.get("command_index"),
                    command_id=record.get("command_id"),
                    pid=record.get("pid"),
                    pgid=record.get("pgid"),
                    recovered_at=record.get("recovered_at"),
                    reason=(record.get("recovery") or {}).get("reason"),
                )
        (self.state.stage_dir(spec.name) / "COMPLETED").unlink(missing_ok=True)
        running = self.state.start_stage(
            spec.name,
            config_hash=stage_hash,
            input_artifacts=input_hashes,
        )
        attempt = int(running["attempt"])
        logger = self._logger().child(stage=spec.name, attempt=attempt)
        checkpoint_lineage = self._checkpoint_lineage(
            spec, stage_hash=stage_hash, input_hashes=input_hashes
        )
        checkpoint_lineage_path = (
            self.state.stage_dir(spec.name)
            / "attempts"
            / f"{attempt:04d}"
            / "checkpoint_lineage.json"
        )
        atomic_write_json(checkpoint_lineage_path, checkpoint_lineage, mode=0o600)
        context = StageContext(
            repo_root=self.repo_root,
            run_dir=self.run_dir,
            config=self.config,
            registry=self.registry,
            state=self.state,
            spec=spec,
            attempt=attempt,
            logger=logger,
            resume_checkpoint=resume_checkpoint,
            checkpoint_lineage=checkpoint_lineage,
            checkpoint_lineage_path=checkpoint_lineage_path,
            allow_legacy_checkpoint=allow_legacy_checkpoint,
        )
        context.attempt_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            context.stage_dir / "input_manifest.json",
            {
                "schema_version": 1,
                "stage": spec.name,
                "attempt": attempt,
                "created_at": utc_now(),
                "config_hash": stage_hash,
                "config": self._stage_config_view(spec),
                "input_artifacts": input_hashes,
                "resume_checkpoint": resume_checkpoint,
            },
        )
        logger.event("stage.begin", config_hash=stage_hash, resume_checkpoint=resume_checkpoint)
        started = time.monotonic()
        try:
            result = spec.handler(context)
            if not isinstance(result, StageResult):
                raise PipelineRunnerError(
                    f"stage {spec.name} handler must return StageResult"
                )
            result = self._promote_attempt_output(
                context,
                result,
                stage_hash=stage_hash,
                input_hashes=input_hashes,
            )
            outputs = self._register_result(spec, result, input_hashes)
            self.state.complete_stage(spec.name, outputs=outputs, progress=result.progress)
            # This marker is intentionally last: a state file from an interrupted
            # commit is not enough to make a stage reusable.
            atomic_write_text(self.state.stage_dir(spec.name) / "COMPLETED", "completed\n")
            logger.event(
                "stage.complete",
                outputs=len(outputs),
                config_hash=stage_hash,
                elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            )
            return StageExecution(spec.name, "executed", attempt)
        except BaseException as error:
            diagnostic = context.attempt_dir / "traceback.txt"
            atomic_write_text(
                diagnostic,
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                mode=0o600,
            )
            safe_error = logger.redact(str(error))
            self.state.fail_stage(
                spec.name,
                error_type=type(error).__name__,
                error=safe_error,
                traceback_path=str(diagnostic.relative_to(self.run_dir)),
                elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            )
            logger.event(
                "stage.failed",
                level="ERROR",
                error_type=type(error).__name__,
                error=safe_error,
                traceback_path=str(diagnostic.relative_to(self.run_dir)),
            )
            raise PipelineRunnerError(
                f"stage {spec.name} failed: {type(error).__name__}: {safe_error}. "
                f"Diagnostic traceback: {diagnostic}"
            ) from error

    def _logger(self) -> PipelineLogger:
        run = self.state.read_run()
        return PipelineLogger(
            self.run_dir,
            run_id=str(run["run_id"]),
            marker=str(self.config.get("logging.marker") or "[[LLMGEN-PIPELINE]]"),
            console_level=str(self.config.get("logging.console_level") or "INFO"),
            file_level=str(self.config.get("logging.file_level") or "DEBUG"),
            secret_values=_secret_values(self.config),
            config_hash=self.config.hash,
        )

    def run(
        self,
        *,
        from_stage: str | None = None,
        to_stage: str | None = None,
        force_stage: str | Sequence[str] | None = None,
        resume_checkpoint: str | None = None,
        allow_legacy_checkpoint: bool = False,
    ) -> tuple[StageExecution, ...]:
        """Run a contiguous declared stage range, safely reusing valid stages."""

        selected = self._selected_specs(from_stage=from_stage, to_stage=to_stage)
        forced = (
            ()
            if force_stage is None
            else ((force_stage,) if isinstance(force_stage, str) else tuple(force_stage))
        )
        for name in forced:
            self._require_stage(name)
            if name not in {spec.name for spec in selected}:
                raise PipelineRunnerError(
                    "every --force-stage must be inside the requested range"
                )
        if resume_checkpoint is not None and len(selected) != 1:
            raise PipelineRunnerError("--resume-checkpoint requires exactly one selected stage")
        if resume_checkpoint is not None and selected[0].name not in {
            "train-codebook",
            "train-memorization",
            "train-alignment",
            "train-retrieval",
        }:
            raise PipelineRunnerError(
                "--resume-checkpoint is valid only for a training stage"
            )
        try:
            with file_lock(self.state.lock_path, blocking=False):
                self.state.update_run(status="running", current_stage=None, last_error=None)
                logger = self._logger()
                for name in forced:
                    invalidated = self._invalidate(name, reason="forced rerun")
                    logger.event(
                        "stage.invalidated",
                        stage=name,
                        affected=list(invalidated),
                        reason="forced rerun",
                    )
                executions: list[StageExecution] = []
                selected_names = {spec.name for spec in selected}
                try:
                    for spec in selected:
                        self.state.update_run(current_stage=spec.name)
                        if any(dependency not in selected_names for dependency in spec.dependencies):
                            self._dependency_ready(spec)
                        if self._can_reuse(spec):
                            logger.event("stage.reused", stage=spec.name)
                            executions.append(StageExecution(spec.name, "reused"))
                            continue
                        current = self.state.read_stage(spec.name)
                        if current.get("status") == StageStatus.COMPLETED.value:
                            invalidated = self._invalidate(
                                spec.name,
                                reason="artifact or configuration changed",
                            )
                            logger.event(
                                "stage.invalidated",
                                stage=spec.name,
                                affected=list(invalidated),
                                reason="artifact or configuration changed",
                            )
                        executions.append(
                            self._execute(
                                spec,
                                resume_checkpoint=(
                                    resume_checkpoint if len(selected) == 1 else None
                                ),
                                allow_legacy_checkpoint=allow_legacy_checkpoint,
                            )
                        )
                except BaseException as error:
                    safe_error = self._logger().redact(str(error))
                    self.state.update_run(
                        status="failed",
                        current_stage=None,
                        last_error={
                            "type": type(error).__name__,
                            "message": safe_error,
                        },
                    )
                    raise
                complete = all(self._can_reuse(spec) for spec in self.specs)
                self.state.update_run(
                    status="completed" if complete else "partial",
                    current_stage=None,
                    last_error=None,
                )
                return tuple(executions)
        except BlockingIOError as error:
            raise PipelineRunnerError(
                f"another pipeline process holds the Run lock: {self.run_dir}"
            ) from error

    def stage(
        self,
        stage: str,
        *,
        resume_checkpoint: str | None = None,
        allow_legacy_checkpoint: bool = False,
    ) -> StageExecution:
        """Execute only one stage after explicitly checking all upstream state."""

        return self.run(
            from_stage=stage,
            to_stage=stage,
            resume_checkpoint=resume_checkpoint,
            allow_legacy_checkpoint=allow_legacy_checkpoint,
        )[0]

    def status(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable overview without mutating the Run."""

        run = self.state.read_run()
        stages = []
        for spec in self.specs:
            state = self.state.read_stage(spec.name)
            stages.append(
                {
                    "stage": spec.name,
                    "status": state.get("status"),
                    "attempt": state.get("attempt"),
                    "reusable": self._can_reuse(spec),
                    "last_error": state.get("last_error"),
                }
            )
        return {"run": run, "stages": stages, "artifacts": len(self.registry.all())}

    def reuse_from(self, parent: "PipelineRunner") -> tuple[str, ...]:
        """Copy only verified, config-compatible parent artifacts into this Run.

        A fork never references the parent's paths.  Copying keeps each Run
        self-contained and permits later deletion or movement of the parent.
        """

        reused: list[str] = []
        for spec in self.specs:
            parent_state = parent.state.read_stage(spec.name)
            if not parent._can_reuse(spec):
                break
            if parent_state.get("config_hash") != self._stage_config_hash(spec):
                break
            try:
                parent_inputs = parent._input_hashes(spec)
            except StageDependencyError:
                break
            # Every input hash must already have been copied to this Run.  Ingest
            # has no inputs, so it establishes the candidate snapshot boundary.
            try:
                inputs = self._input_hashes(spec)
            except StageDependencyError:
                if spec.required_artifacts:
                    break
                inputs = {}
            if inputs != parent_inputs:
                break
            output_hashes = parent_state.get("outputs")
            if not isinstance(output_hashes, Mapping):
                break
            records: dict[str, ArtifactRecord] = {}
            try:
                for name, expected_hash in output_hashes.items():
                    record = parent.registry.verify(str(name))
                    if record.producer != spec.name or record.sha256 != expected_hash:
                        raise ArtifactError(f"parent output is stale: {name}")
                    source = parent.registry.resolve(str(name))
                    destination = self.run_dir / record.path
                    if source.is_dir():
                        if destination.exists():
                            shutil.rmtree(destination)
                        shutil.copytree(source, destination)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                    records[str(name)] = self.registry.build_record(
                        logical_name=str(name), path=destination, producer=spec.name,
                        artifact_schema=record.artifact_schema, format=record.format,
                        metadata=record.metadata, inputs=inputs,
                        config_hash=self._stage_config_hash(spec),
                    )
                self.registry.register_many(records)
            except (ArtifactError, OSError):
                break
            self.state.update_stage(
                spec.name,
                status=StageStatus.COMPLETED.value,
                attempt=int(parent_state.get("attempt") or 0),
                started_at=parent_state.get("started_at"),
                finished_at=parent_state.get("finished_at"),
                config_hash=self._stage_config_hash(spec),
                input_artifacts=inputs,
                outputs={name: record.sha256 for name, record in records.items()},
                progress=parent_state.get("progress") or {},
                last_error=None,
            )
            atomic_write_text(self.state.stage_dir(spec.name) / "COMPLETED", "reused\n")
            reused.extend(sorted(records))
        self.state.update_run(reused_artifacts=reused)
        return tuple(reused)


def create_pipeline_run(
    config: PipelineConfig,
    *,
    stage_specs: Sequence[StageSpec] | None = None,
    repo_root: str | Path | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
) -> PipelineRunner:
    """Create one Run directory and its immutable configuration snapshot."""

    directory = Path(config.require("run.output_dir")).expanduser().resolve()
    runner = PipelineRunner(directory, config, stage_specs=stage_specs, repo_root=repo_root)
    candidate_path = _candidate_path(config, runner.repo_root)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    candidate_bytes = candidate_path.read_bytes()
    manual_alignment_path = _manual_alignment_path(config, runner.repo_root)
    if manual_alignment_path is not None and not manual_alignment_path.is_file():
        raise FileNotFoundError(manual_alignment_path)
    manual_alignment_bytes = (
        manual_alignment_path.read_bytes()
        if manual_alignment_path is not None
        else b""
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        # mkdir is the atomic reservation: two creators can no longer both
        # pass a check-then-write window and interleave immutable snapshots.
        directory.mkdir()
    except FileExistsError as error:
        raise PipelineRunnerError(f"run directory already exists: {directory}") from error
    snapshot = runner.run_dir / "config"
    snapshot.mkdir(parents=True, exist_ok=True)
    frozen_candidate = runner.run_dir / "source" / "candidates.input.jsonl"
    frozen_manual_alignment = (
        runner.run_dir / "source" / "manual_alignment.input.jsonl"
    )
    atomic_write_bytes(frozen_candidate, candidate_bytes)
    atomic_write_bytes(frozen_manual_alignment, manual_alignment_bytes)
    atomic_write_bytes(snapshot / "pipeline.source.yaml", config.source_path.read_bytes())
    atomic_write_text(snapshot / "pipeline.resolved.yaml", config.to_yaml())
    atomic_write_json(snapshot / "overrides.json", {"overrides": list(config.overrides)})
    environment_snapshot = _environment_snapshot(config)
    provenance = collect_run_provenance(
        repo_root=runner.repo_root,
        runtime=config.require("runtime"),
        base_model=str(config.require("router.base_model")),
    )
    environment_snapshot["resources"] = provenance["resources"]
    atomic_write_json(snapshot / "environment.json", environment_snapshot)
    atomic_write_json(snapshot / "provenance.json", provenance)
    atomic_write_json(
        snapshot / "candidate_input.json",
        {
            "schema_version": 1,
            "path": str(candidate_path),
            "frozen_path": str(frozen_candidate.relative_to(runner.run_dir)),
            "bytes": len(candidate_bytes),
            "sha256": sha256_bytes(candidate_bytes),
        },
    )
    atomic_write_json(
        snapshot / "manual_alignment_input.json",
        {
            "schema_version": 1,
            "enabled": manual_alignment_path is not None,
            "path": (
                str(manual_alignment_path)
                if manual_alignment_path is not None
                else None
            ),
            "frozen_path": str(
                frozen_manual_alignment.relative_to(runner.run_dir)
            ),
            "bytes": len(manual_alignment_bytes),
            "sha256": sha256_bytes(manual_alignment_bytes),
        },
    )
    runner.registry.initialize()
    resolved = (snapshot / "pipeline.resolved.yaml").relative_to(runner.run_dir).as_posix()
    source = (snapshot / "pipeline.source.yaml").relative_to(runner.run_dir).as_posix()
    runner.state.initialize(
        run_id=run_id or directory.name,
        name=str(config.require("run.name")),
        config_hash=config.hash,
        config_path=resolved,
        source_config_path=source,
        repo_root=str(runner.repo_root),
        stage_order=runner.stage_names,
        git_commit=_git_commit(runner.repo_root),
        parent_run_id=parent_run_id,
    )
    return runner
