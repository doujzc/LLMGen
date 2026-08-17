"""Durable run metadata and append-only logging for Top1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import resource
import subprocess
from typing import Any, Iterable, Mapping

from .top1 import Top1DataError, sha256_file, write_json, write_jsonl


TRAINING_RUN_SCHEMA_VERSION = 2
EVALUATION_RUN_SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def compact_utc_now() -> str:
    """Return a sortable timestamp suitable for an artifact directory name."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON mapping with deterministic key ordering."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read one JSON object and reject other JSON shapes."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Top1DataError(f"invalid JSON object: {source}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError(f"JSON value must be an object: {source}")
    return payload


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Append one compact JSON event with a single operating-system write."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def json_safe(value: Any) -> Any:
    """Convert common numeric and framework values to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def git_snapshot(repository: str | Path) -> dict[str, Any]:
    """Describe the source revision without mutating the repository."""

    root = Path(repository)

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *arguments),
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
    }


def system_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    """Collect reproducibility and hardware metadata without environment secrets."""

    versions: dict[str, str] = {}
    for package in ("llmgen", "torch", "transformers", "accelerate", "deepspeed", "peft"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "process": {
            "world_size": _environment_integer("WORLD_SIZE", 1),
            "rank": _environment_integer("RANK", 0),
            "local_rank": _environment_integer("LOCAL_RANK", 0),
        },
    }
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        available = bool(cuda is not None and cuda.is_available())
        devices = []
        if available:
            for index in range(int(cuda.device_count())):
                properties = cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": str(properties.name),
                        "total_memory_bytes": int(properties.total_memory),
                        "capability": list(cuda.get_device_capability(index)),
                    }
                )
        snapshot["cuda"] = {
            "available": available,
            "runtime_version": getattr(getattr(torch_module, "version", None), "cuda", None),
            "devices": devices,
        }
    return snapshot


def _environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RunStore:
    """Filesystem contract for one immutable-config experiment run."""

    root: Path
    kind: str

    @classmethod
    def training(cls, root: str | Path) -> "RunStore":
        return cls(Path(root).expanduser().resolve(), "training")

    @classmethod
    def evaluation(cls, root: str | Path) -> "RunStore":
        return cls(Path(root).expanduser().resolve(), "evaluation")

    @property
    def manifest_path(self) -> Path:
        return self.root / (
            "run_manifest.json" if self.kind == "training" else "eval_manifest.json"
        )

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def events_path(self) -> Path:
        return self.root / "logs" / "events.jsonl"

    def ensure_layout(self) -> None:
        directories = ("logs",)
        if self.kind == "training":
            directories = ("prepared", "logs", "checkpoints", "final")
        for name in directories:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        manifest: Mapping[str, Any],
        *,
        resume: bool = False,
    ) -> None:
        """Create a run manifest once, or verify it before resuming."""

        self.ensure_layout()
        proposed = dict(manifest)
        if "run_signature" not in proposed:
            raise Top1DataError("run manifest must contain run_signature")
        if self.manifest_path.exists():
            existing = read_json_object(self.manifest_path)
            if not resume:
                raise Top1DataError(
                    f"run already exists; choose a new run directory: {self.root}"
                )
            if existing.get("run_signature") != proposed["run_signature"]:
                raise Top1DataError(
                    "resume configuration does not match the immutable run manifest"
                )
            if self.status_path.exists():
                status = read_json_object(self.status_path)
                if status.get("state") == "COMPLETED":
                    raise Top1DataError("a completed run cannot be resumed in place")
        else:
            write_json(self.manifest_path, proposed)
        state = "RESUMING" if resume and self.status_path.exists() else "CREATED"
        self.update_status(state)
        self.event("run_initialized", state=state)

    def update_status(self, state: str, **details: Any) -> None:
        """Atomically update the small mutable run lifecycle record."""

        previous: dict[str, Any] = {}
        if self.status_path.exists():
            previous = read_json_object(self.status_path)
        created_at = previous.get("created_at", utc_now())
        payload = {
            "schema_version": 1,
            "kind": self.kind,
            "state": state,
            "created_at": created_at,
            "updated_at": utc_now(),
            **json_safe(details),
        }
        write_json(self.status_path, payload)

    def event(self, event: str, **details: Any) -> None:
        """Append one structured lifecycle or trainer event."""

        append_jsonl(
            self.events_path,
            {
                "timestamp": utc_now(),
                "event": event,
                **json_safe(details),
            },
        )


def directory_file_manifest(
    root: str | Path,
    *,
    excluded_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Hash all regular files below a model artifact in stable path order."""

    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise Top1DataError(f"artifact directory does not exist: {source}")
    excluded = set(excluded_names)
    entries = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise Top1DataError(f"artifact contains a symbolic link: {path}")
        if not path.is_file() or path.name in excluded:
            continue
        entries.append(
            {
                "path": path.relative_to(source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise Top1DataError(f"artifact directory contains no files: {source}")
    return entries


def write_model_artifact_manifest(
    model_dir: str | Path,
    *,
    training_run_id: str | None,
) -> dict[str, Any]:
    """Fingerprint a deployable model directory once after it is finalized."""

    root = Path(model_dir).expanduser().resolve()
    files = directory_file_manifest(root, excluded_names=("model_artifact.json",))
    identity = {"schema_version": 1, "files": files}
    payload = {
        **identity,
        "model_id": canonical_sha256(identity),
        "created_at": utc_now(),
        "training_run_id": training_run_id,
    }
    write_json(root / "model_artifact.json", payload)
    return payload


def load_and_verify_model_artifact(
    model_dir: str | Path,
    *,
    verify_files: bool,
) -> dict[str, Any]:
    """Load a model identity, optionally rehashing every artifact file."""

    root = Path(model_dir).expanduser().resolve()
    artifact_path = root / "model_artifact.json"
    if artifact_path.is_file():
        payload = read_json_object(artifact_path)
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("files"),
            list,
        ):
            raise Top1DataError("model_artifact.json has an invalid schema")
        expected_identity = {
            "schema_version": payload.get("schema_version"),
            "files": payload.get("files"),
        }
        if payload.get("model_id") != canonical_sha256(expected_identity):
            raise Top1DataError("model_artifact.json has an invalid model_id")
        if verify_files:
            actual_files = directory_file_manifest(
                root,
                excluded_names=("model_artifact.json",),
            )
            if actual_files != payload.get("files"):
                raise Top1DataError("model artifact files changed after finalization")
        return payload

    files = directory_file_manifest(root)
    identity = {"schema_version": 1, "files": files}
    return {
        **identity,
        "model_id": canonical_sha256(identity),
        "created_at": None,
        "training_run_id": None,
    }


class TrainingLogCallback:
    """Callback behavior that persists every rank-zero training event.

    Use :func:`make_training_log_callback` when passing this callback to
    Transformers.  Keeping this implementation independent of Transformers lets
    metadata utilities and the training CLI load without the optional training
    dependencies installed.
    """

    def __init__(
        self,
        store: RunStore,
        torch_module: Any | None = None,
        *,
        memorization_steps: int = 0,
    ) -> None:
        if memorization_steps < 0:
            raise ValueError("memorization_steps cannot be negative")
        self.store = store
        self.torch = torch_module
        self.memorization_steps = memorization_steps

    def _stage_for_next_step(self, step: int) -> str:
        if step < self.memorization_steps:
            return "memorization"
        return "main"

    def _stage_for_completed_step(self, step: int) -> str:
        if self.memorization_steps and step <= self.memorization_steps:
            return "memorization"
        return "main"

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if getattr(state, "is_world_process_zero", True):
            step = int(getattr(state, "global_step", 0))
            stage = self._stage_for_next_step(step)
            self.store.update_status("RUNNING", step=step, stage=stage)
            self.store.event("training_started", step=step, stage=stage)
        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        if not getattr(state, "is_world_process_zero", True):
            return control
        metrics = json_safe(dict(logs or {}))
        non_finite = [
            key
            for key, value in (logs or {}).items()
            if isinstance(value, Real) and not math.isfinite(float(value))
        ]
        step = int(getattr(state, "global_step", 0))
        self.store.event(
            "trainer_log",
            step=step,
            epoch=getattr(state, "epoch", None),
            stage=self._stage_for_completed_step(step),
            metrics=metrics,
            system=_runtime_memory(self.torch),
            numerical_issue=non_finite or None,
        )
        return control

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        step = int(getattr(state, "global_step", 0))
        if self.memorization_steps and step == self.memorization_steps:
            control.should_save = True
            if getattr(state, "is_world_process_zero", True):
                self.store.event("memorization_completed", step=step)
                self.store.event("main_training_started", step=step)
                self.store.update_status("RUNNING", step=step, stage="main")
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del kwargs
        if getattr(state, "is_world_process_zero", True):
            step = int(getattr(state, "global_step", 0))
            checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
            pointer = {
                "schema_version": 1,
                "step": step,
                "stage": self._stage_for_completed_step(step),
                "path": str(checkpoint.resolve()),
                "updated_at": utc_now(),
            }
            write_json(self.store.root / "checkpoints" / "last_checkpoint.json", pointer)
            self.store.update_status(
                "RUNNING",
                step=step,
                stage=self._stage_for_completed_step(step),
                last_checkpoint=pointer["path"],
            )
            self.store.event("checkpoint_saved", **pointer)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if getattr(state, "is_world_process_zero", True):
            step = int(getattr(state, "global_step", 0))
            self.store.event(
                "training_finished",
                step=step,
                stage=self._stage_for_completed_step(step),
            )
        return control


def make_training_log_callback(
    store: RunStore,
    trainer_callback_class: type[Any],
    torch_module: Any | None = None,
    *,
    memorization_steps: int = 0,
) -> TrainingLogCallback:
    """Bind the logging behavior to the installed Transformers callback API.

    ``Trainer`` dispatches every lifecycle event (including ``on_init_end``)
    directly on each callback.  Inheriting from its own ``TrainerCallback``
    supplies the version-appropriate no-op implementations for events that this
    logger does not handle.
    """

    if not isinstance(trainer_callback_class, type):
        raise TypeError("trainer_callback_class must be a class")
    callback_class = type(
        "BoundTrainingLogCallback",
        (TrainingLogCallback, trainer_callback_class),
        {"__module__": __name__},
    )
    return callback_class(
        store,
        torch_module,
        memorization_steps=memorization_steps,
    )


def _runtime_memory(torch_module: Any | None) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result: dict[str, Any] = {"process_max_rss": int(usage.ru_maxrss)}
    cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
    if cuda is not None and cuda.is_available():
        device = int(cuda.current_device())
        result["cuda"] = {
            "device": device,
            "allocated_bytes": int(cuda.memory_allocated(device)),
            "reserved_bytes": int(cuda.memory_reserved(device)),
            "max_allocated_bytes": int(cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(cuda.max_memory_reserved(device)),
        }
    return result


def write_trainer_history(path: str | Path, history: Iterable[Mapping[str, Any]]) -> None:
    """Persist Trainer state history as valid JSONL, including non-finite guards."""

    write_jsonl(path, (json_safe(row) for row in history))
