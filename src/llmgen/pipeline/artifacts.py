"""Content-addressed artifact registry for one pipeline Run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .io import (
    atomic_write_json,
    file_lock,
    jsonl_row_count,
    path_size,
    read_json,
    sha256_path,
    utc_now,
)


class ArtifactError(RuntimeError):
    """Raised when a declared artifact is missing, stale, or unsafe."""


@dataclass(frozen=True)
class ArtifactRecord:
    logical_name: str
    path: str
    format: str
    artifact_schema: str
    producer: str
    sha256: str
    bytes: int
    created_at: str
    rows: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)
    config_hash: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRecord":
        return cls(
            logical_name=str(value["logical_name"]),
            path=str(value["path"]),
            format=str(value["format"]),
            artifact_schema=str(value["artifact_schema"]),
            producer=str(value["producer"]),
            sha256=str(value["sha256"]),
            bytes=int(value["bytes"]),
            created_at=str(value["created_at"]),
            rows=(int(value["rows"]) if value.get("rows") is not None else None),
            metadata=dict(value.get("metadata") or {}),
            inputs=dict(value.get("inputs") or {}),
            config_hash=str(value.get("config_hash") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_format(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffixes = "".join(path.suffixes).casefold()
    if suffixes.endswith(".jsonl"):
        return "jsonl"
    if suffixes.endswith(".json"):
        return "json"
    if suffixes.endswith(".yaml") or suffixes.endswith(".yml"):
        return "yaml"
    if suffixes.endswith(".npy"):
        return "npy"
    if suffixes.endswith(".npz"):
        return "npz"
    if suffixes.endswith(".safetensors"):
        return "safetensors"
    if suffixes.endswith(".pt") or suffixes.endswith(".bin"):
        return "checkpoint"
    if suffixes.endswith(".txt") or suffixes.endswith(".log"):
        return "text"
    return "binary"


class ArtifactRegistry:
    """Atomic logical-name registry rooted at a single Run directory."""

    SCHEMA_VERSION = 1

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.path = self.run_dir / "artifact_registry.json"
        self.lock_path = self.run_dir / ".artifact_registry.lock"

    def initialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_path):
            if not self.path.exists():
                self._write_unlocked({})

    def _load_unlocked(self) -> dict[str, ArtifactRecord]:
        if not self.path.is_file():
            return {}
        payload = read_json(self.path)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ArtifactError(f"invalid artifact registry: {self.path}")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ArtifactError(f"artifact registry has no artifacts map: {self.path}")
        records: dict[str, ArtifactRecord] = {}
        for name, raw in artifacts.items():
            if not isinstance(raw, Mapping):
                raise ArtifactError(f"invalid artifact record: {name}")
            record = ArtifactRecord.from_dict(raw)
            if record.logical_name != name:
                raise ArtifactError(f"artifact name mismatch: {name}")
            records[name] = record
        return records

    def _write_unlocked(self, records: Mapping[str, ArtifactRecord]) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "updated_at": utc_now(),
                "artifacts": {
                    name: record.to_dict()
                    for name, record in sorted(records.items())
                },
            },
        )

    def all(self) -> dict[str, ArtifactRecord]:
        with file_lock(self.lock_path):
            return self._load_unlocked()

    def get(self, logical_name: str) -> ArtifactRecord:
        records = self.all()
        try:
            return records[logical_name]
        except KeyError as error:
            raise ArtifactError(f"required artifact is not registered: {logical_name}") from error

    def resolve(self, logical_name: str, *, verify: bool = True) -> Path:
        record = self.get(logical_name)
        path = (self.run_dir / record.path).resolve()
        self._require_within_run(path)
        if verify:
            self.verify_record(record)
        return path

    def _require_within_run(self, path: Path) -> None:
        try:
            path.relative_to(self.run_dir)
        except ValueError as error:
            raise ArtifactError(
                f"artifact path escapes run directory: {path}"
            ) from error

    def build_record(
        self,
        *,
        logical_name: str,
        path: str | Path,
        producer: str,
        artifact_schema: str,
        format: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        inputs: Mapping[str, str] | None = None,
        config_hash: str = "",
    ) -> ArtifactRecord:
        if not logical_name or any(character.isspace() for character in logical_name):
            raise ArtifactError(f"invalid artifact logical name: {logical_name!r}")
        value = Path(path).expanduser().resolve()
        self._require_within_run(value)
        if not value.exists():
            raise ArtifactError(f"artifact output does not exist: {value}")
        relative = value.relative_to(self.run_dir).as_posix()
        selected_format = format or infer_format(value)
        rows = (
            jsonl_row_count(value)
            if selected_format == "jsonl" and value.is_file()
            else None
        )
        return ArtifactRecord(
            logical_name=logical_name,
            path=relative,
            format=selected_format,
            artifact_schema=artifact_schema,
            producer=producer,
            sha256=sha256_path(value),
            bytes=path_size(value),
            rows=rows,
            created_at=utc_now(),
            metadata=dict(metadata or {}),
            inputs=dict(inputs or {}),
            config_hash=config_hash,
        )

    def register_many(self, records: Mapping[str, ArtifactRecord]) -> None:
        for name, record in records.items():
            if name != record.logical_name:
                raise ArtifactError(f"artifact key/name mismatch: {name}")
        with file_lock(self.lock_path):
            current = self._load_unlocked()
            for name, record in records.items():
                existing = current.get(name)
                if existing is not None and existing.producer != record.producer:
                    raise ArtifactError(
                        f"artifact {name!r} is owned by {existing.producer}; "
                        f"producer {record.producer} may not overwrite it"
                    )
            current.update(records)
            self._write_unlocked(current)

    def remove_by_producer(self, producers: set[str]) -> list[str]:
        with file_lock(self.lock_path):
            current = self._load_unlocked()
            removed = [
                name
                for name, record in current.items()
                if record.producer in producers
            ]
            for name in removed:
                del current[name]
            self._write_unlocked(current)
        return sorted(removed)

    def verify_record(self, record: ArtifactRecord) -> None:
        path = (self.run_dir / record.path).resolve()
        self._require_within_run(path)
        if not path.exists():
            raise ArtifactError(
                f"artifact {record.logical_name} is missing: {path}"
            )
        actual = sha256_path(path)
        if actual != record.sha256:
            raise ArtifactError(
                f"artifact {record.logical_name} hash changed: "
                f"expected {record.sha256}, got {actual}"
            )

    def verify(self, logical_name: str) -> ArtifactRecord:
        record = self.get(logical_name)
        self.verify_record(record)
        return record
