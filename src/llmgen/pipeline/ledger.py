"""Immutable JSONL shard ledgers for resumable provider work.

The pipeline runner owns Stage-level attempts.  This module supplies the
smaller durable unit needed *inside* a long Stage: one request, response, or
embedding batch.  It deliberately has no Provider dependency.  A caller can
schedule records, invoke any Provider for the returned records, then append
the resulting successes or failures.  On the next invocation the ledger
returns only identities that do not already have a successful result.

Every shard is published with exclusive creation and is never modified after
publication.  ``manifest.json`` is the mutable index and is atomically
replaced only after all files referenced by a new batch are durable.  A
manifest/hash scan detects partial manual edits, missing files, duplicate
identities, and orphan shard files before a caller is allowed to resume.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Literal, Mapping, Sequence

from .io import atomic_write_json, canonical_json, file_lock, read_json, read_jsonl, sha256_bytes, sha256_file, utc_now


LEDGER_SCHEMA_VERSION = 1
ShardKind = Literal["requests", "responses", "embeddings"]
ResultStatus = Literal["succeeded", "failed"]
_SHARD_KINDS: tuple[ShardKind, ...] = ("requests", "responses", "embeddings")
_RESULT_STATUSES: tuple[ResultStatus, ...] = ("succeeded", "failed")


class LedgerError(RuntimeError):
    """Base exception for an invalid ledger operation."""


class LedgerIntegrityError(LedgerError):
    """Raised when immutable ledger files or their manifest disagree."""


def stable_text_hash(text: str) -> str:
    """Return the SHA-256 hash used for prompt and embedding-input identity."""

    if not isinstance(text, str):
        raise TypeError("ledger text must be a string")
    return sha256_bytes(text.encode("utf-8"))


def stable_request_id(
    namespace: str,
    prompt: str,
    *,
    request_key: Any = None,
) -> tuple[str, str]:
    """Build a deterministic request identifier and its prompt hash.

    ``namespace`` should distinguish Provider operations such as
    ``generate-queries`` and ``review-queries``.  ``request_key`` may contain
    model and non-secret request parameters; it prevents a changed request
    configuration from being mistaken for the same paid request.
    """

    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        raise ValueError("ledger namespace must be non-empty")
    prompt_hash = stable_text_hash(prompt)
    try:
        identity = canonical_json(
            {
                "namespace": normalized_namespace,
                "prompt_hash": prompt_hash,
                "request_key": request_key,
            }
        )
    except (TypeError, ValueError) as error:
        raise TypeError("request_key must be JSON-serializable") from error
    return f"req-{sha256_bytes(identity.encode('utf-8'))[:32]}", prompt_hash


def stable_embedding_id(
    namespace: str,
    input_text: str,
    *,
    item_key: Any = None,
    model: str | None = None,
) -> tuple[str, str]:
    """Build a deterministic embedding identity and input hash.

    The selected embedding model participates in the identity because vectors
    from distinct models are not interchangeable even when the source text is
    byte-for-byte identical.
    """

    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        raise ValueError("ledger namespace must be non-empty")
    input_hash = stable_text_hash(input_text)
    try:
        identity = canonical_json(
            {
                "namespace": normalized_namespace,
                "input_hash": input_hash,
                "item_key": item_key,
                "model": model or "",
            }
        )
    except (TypeError, ValueError) as error:
        raise TypeError("item_key must be JSON-serializable") from error
    return f"emb-{sha256_bytes(identity.encode('utf-8'))[:32]}", input_hash


def _json_mapping(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    payload = dict(value or {})
    try:
        canonical_json(payload)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be JSON-serializable") from error
    return payload


@dataclass(frozen=True)
class RequestRecord:
    """One immutable LLM request, including its stable identity and prompt."""

    request_id: str
    prompt_hash: str
    namespace: str
    prompt: str
    request_key: Any = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_prompt(
        cls,
        namespace: str,
        prompt: str,
        *,
        request_key: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RequestRecord":
        """Create a request whose identity is stable across process restarts."""

        request_id, prompt_hash = stable_request_id(
            namespace, prompt, request_key=request_key
        )
        return cls(
            request_id=request_id,
            prompt_hash=prompt_hash,
            namespace=str(namespace),
            prompt=prompt,
            request_key=request_key,
            metadata=_json_mapping(metadata, field_name="request metadata"),
        )

    def to_row(self) -> dict[str, Any]:
        """Serialize the record using the versioned request-row contract."""

        expected_id, expected_hash = stable_request_id(
            self.namespace, self.prompt, request_key=self.request_key
        )
        if self.request_id != expected_id or self.prompt_hash != expected_hash:
            raise LedgerIntegrityError(
                f"request identity does not match prompt for {self.request_id!r}"
            )
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "record_type": "request",
            "request_id": self.request_id,
            "prompt_hash": self.prompt_hash,
            "namespace": self.namespace,
            "prompt": self.prompt,
            "request_key": self.request_key,
            "metadata": _json_mapping(self.metadata, field_name="request metadata"),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "RequestRecord":
        """Parse and validate a request-row artifact."""

        if row.get("schema_version") != LEDGER_SCHEMA_VERSION or row.get("record_type") != "request":
            raise LedgerIntegrityError("invalid request ledger row schema")
        try:
            record = cls(
                request_id=str(row["request_id"]),
                prompt_hash=str(row["prompt_hash"]),
                namespace=str(row["namespace"]),
                prompt=str(row["prompt"]),
                request_key=row.get("request_key"),
                metadata=_json_mapping(row.get("metadata"), field_name="request metadata"),
            )
            record.to_row()
            return record
        except (KeyError, TypeError, ValueError) as error:
            raise LedgerIntegrityError("malformed request ledger row") from error


@dataclass(frozen=True)
class ResponseRecord:
    """A terminal result for one request; failures may be appended as retries."""

    request_id: str
    prompt_hash: str
    status: ResultStatus
    response: Any = None
    error: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    attempt: int = 0

    def to_row(self) -> dict[str, Any]:
        """Serialize a response row after validating its terminal status."""

        if not self.request_id or not self.prompt_hash:
            raise LedgerIntegrityError("response requires request_id and prompt_hash")
        if self.status not in _RESULT_STATUSES:
            raise LedgerIntegrityError(f"invalid response status: {self.status!r}")
        if self.attempt < 1:
            raise LedgerIntegrityError("response attempt must be positive")
        try:
            canonical_json(self.response)
        except (TypeError, ValueError) as error:
            raise TypeError("response payload must be JSON-serializable") from error
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "record_type": "response",
            "request_id": self.request_id,
            "prompt_hash": self.prompt_hash,
            "status": self.status,
            "attempt": self.attempt,
            "response": self.response,
            "error": _json_mapping(self.error, field_name="response error"),
            "metadata": _json_mapping(self.metadata, field_name="response metadata"),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ResponseRecord":
        """Parse and validate a response-row artifact."""

        if row.get("schema_version") != LEDGER_SCHEMA_VERSION or row.get("record_type") != "response":
            raise LedgerIntegrityError("invalid response ledger row schema")
        try:
            record = cls(
                request_id=str(row["request_id"]),
                prompt_hash=str(row["prompt_hash"]),
                status=str(row["status"]),  # type: ignore[arg-type]
                response=row.get("response"),
                error=_json_mapping(row.get("error"), field_name="response error"),
                metadata=_json_mapping(row.get("metadata"), field_name="response metadata"),
                attempt=int(row["attempt"]),
            )
            record.to_row()
            return record
        except (KeyError, TypeError, ValueError) as error:
            raise LedgerIntegrityError("malformed response ledger row") from error


@dataclass(frozen=True)
class EmbeddingRecord:
    """One embedding result.  A failed row remains auditable and retryable."""

    embedding_id: str
    input_hash: str
    namespace: str
    input_text: str
    status: ResultStatus
    item_key: Any = None
    model: str | None = None
    vector: Sequence[float] | None = None
    error: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    attempt: int = 0

    @classmethod
    def from_text(
        cls,
        namespace: str,
        input_text: str,
        *,
        status: ResultStatus,
        item_key: Any = None,
        model: str | None = None,
        vector: Sequence[float] | None = None,
        error: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EmbeddingRecord":
        """Create a stable embedding result before its attempt is assigned."""

        embedding_id, input_hash = stable_embedding_id(
            namespace, input_text, item_key=item_key, model=model
        )
        return cls(
            embedding_id=embedding_id,
            input_hash=input_hash,
            namespace=str(namespace),
            input_text=input_text,
            status=status,
            item_key=item_key,
            model=model,
            vector=tuple(vector) if vector is not None else None,
            error=_json_mapping(error, field_name="embedding error"),
            metadata=_json_mapping(metadata, field_name="embedding metadata"),
        )

    def to_row(self) -> dict[str, Any]:
        """Serialize an embedding row and verify the identity/input invariant."""

        expected_id, expected_hash = stable_embedding_id(
            self.namespace,
            self.input_text,
            item_key=self.item_key,
            model=self.model,
        )
        if self.embedding_id != expected_id or self.input_hash != expected_hash:
            raise LedgerIntegrityError(
                f"embedding identity does not match input for {self.embedding_id!r}"
            )
        if self.status not in _RESULT_STATUSES or self.attempt < 1:
            raise LedgerIntegrityError("embedding has invalid status or attempt")
        if self.status == "succeeded" and self.vector is None:
            raise LedgerIntegrityError("successful embedding requires a vector")
        try:
            vector = list(self.vector) if self.vector is not None else None
            canonical_json(vector)
        except (TypeError, ValueError) as error:
            raise TypeError("embedding vector must be JSON-serializable") from error
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "record_type": "embedding",
            "embedding_id": self.embedding_id,
            "input_hash": self.input_hash,
            "namespace": self.namespace,
            "input_text": self.input_text,
            "item_key": self.item_key,
            "model": self.model,
            "status": self.status,
            "attempt": self.attempt,
            "vector": vector,
            "error": _json_mapping(self.error, field_name="embedding error"),
            "metadata": _json_mapping(self.metadata, field_name="embedding metadata"),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "EmbeddingRecord":
        """Parse and validate an embedding-row artifact."""

        if row.get("schema_version") != LEDGER_SCHEMA_VERSION or row.get("record_type") != "embedding":
            raise LedgerIntegrityError("invalid embedding ledger row schema")
        raw_vector = row.get("vector")
        if raw_vector is not None and not isinstance(raw_vector, list):
            raise LedgerIntegrityError("embedding vector must be a JSON list")
        try:
            record = cls(
                embedding_id=str(row["embedding_id"]),
                input_hash=str(row["input_hash"]),
                namespace=str(row["namespace"]),
                input_text=str(row["input_text"]),
                status=str(row["status"]),  # type: ignore[arg-type]
                item_key=row.get("item_key"),
                model=(str(row["model"]) if row.get("model") is not None else None),
                vector=tuple(raw_vector) if raw_vector is not None else None,
                error=_json_mapping(row.get("error"), field_name="embedding error"),
                metadata=_json_mapping(row.get("metadata"), field_name="embedding metadata"),
                attempt=int(row["attempt"]),
            )
            record.to_row()
            return record
        except (KeyError, TypeError, ValueError) as error:
            raise LedgerIntegrityError("malformed embedding ledger row") from error


@dataclass(frozen=True)
class RequestBatch:
    """Provider work selected from a ledger, bounded by its configured size."""

    records: tuple[RequestRecord, ...]
    newly_recorded: int
    retried: int
    skipped_succeeded: int


@dataclass(frozen=True)
class EmbeddingBatch:
    """Embedding work selected from a ledger, bounded by its configured size."""

    records: tuple[EmbeddingRecord, ...]
    retried: int
    skipped_succeeded: int


@dataclass(frozen=True)
class CommitResult:
    """Details of one immutable shard publication."""

    batch_id: int | None
    shard_paths: tuple[Path, ...]
    accepted: int
    skipped_succeeded: int


@dataclass
class _LedgerIndex:
    requests: dict[str, RequestRecord]
    responses: dict[str, ResponseRecord]
    response_attempts: dict[str, int]
    response_successes: set[str]
    embeddings: dict[str, EmbeddingRecord]
    embedding_attempts: dict[str, int]
    embedding_successes: set[str]


class JsonlShardLedger:
    """Durably schedule and record resumable LLM or embedding batches.

    Parameters
    ----------
    root:
        Ledger directory, normally a Stage attempt directory such as
        ``attempts/0001/ledger``.
    batch_size:
        Maximum number of pending records returned by either scheduling method.
        It maps directly to ``checkpointing.llm_batch_records`` or
        ``checkpointing.embedding_batch_records`` through
        :meth:`from_checkpointing`.
    """

    def __init__(self, root: str | Path, *, batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("ledger batch_size must be a positive integer")
        self.root = Path(root).expanduser().resolve()
        self.batch_size = batch_size
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".ledger.lock"

    @classmethod
    def from_checkpointing(
        cls,
        root: str | Path,
        checkpointing: Mapping[str, Any],
        *,
        kind: Literal["llm", "embedding"],
    ) -> "JsonlShardLedger":
        """Construct a ledger from the public pipeline checkpointing settings."""

        field = "llm_batch_records" if kind == "llm" else "embedding_batch_records"
        value = checkpointing.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"checkpointing.{field} must be a positive integer")
        return cls(root, batch_size=value)

    def initialize(self) -> None:
        """Create an empty manifest, refusing to hide pre-existing shard files."""

        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_path):
            if self.manifest_path.exists():
                manifest = self._load_manifest_locked()
                self._recover_orphans_locked(manifest)
                return
            for kind in _SHARD_KINDS:
                if any((self.root / kind).glob("part-*.jsonl")):
                    raise LedgerIntegrityError(
                        "ledger has shard files but no manifest; refusing to adopt them"
                    )
            self._write_manifest_locked(self._empty_manifest())

    def recover(self) -> dict[str, Any]:
        """Adopt a fully written shard left between publication and indexing.

        Immutable shard publication precedes the atomic manifest update.  A
        hard process kill in that narrow interval must not force a paid request
        to be issued again.  Recovery accepts only contiguous, schema-valid
        shards and then verifies the reconstructed index before publishing the
        repaired manifest.
        """

        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_path):
            manifest = self._load_manifest_locked()
            return self._recover_orphans_locked(manifest)

    def successful_response(self, request_id: str) -> ResponseRecord | None:
        """Return the immutable successful response for ``request_id``."""

        self.initialize()
        with file_lock(self.lock_path):
            manifest = self._load_manifest_locked()
            index, _ = self._build_index_locked(manifest)
            if request_id not in index.response_successes:
                return None
            return index.responses[request_id]

    def successful_embedding(self, embedding_id: str) -> EmbeddingRecord | None:
        """Return the immutable successful embedding for ``embedding_id``."""

        self.initialize()
        with file_lock(self.lock_path):
            manifest = self._load_manifest_locked()
            index, _ = self._build_index_locked(manifest)
            if embedding_id not in index.embedding_successes:
                return None
            return index.embeddings[embedding_id]

    def manifest(self) -> dict[str, Any]:
        """Return a schema-validated manifest snapshot without scanning shards.

        Call :meth:`verify` before relying on a manifest supplied by an
        untrusted or externally modified Run directory.
        """

        with file_lock(self.lock_path):
            return self._load_manifest_locked()

    def verify(self) -> dict[str, Any]:
        """Verify hashes, schemas, identities, and aggregate manifest statistics."""

        with file_lock(self.lock_path):
            manifest = self._load_manifest_locked()
            index, calculated = self._build_index_locked(manifest)
            del index
            if manifest.get("stats") != calculated:
                raise LedgerIntegrityError("ledger manifest statistics do not match shards")
            return manifest

    def schedule_requests(self, records: Iterable[RequestRecord]) -> RequestBatch:
        """Persist new requests and return at most one retry-safe provider batch.

        A request is skipped only after a successful response exists.  A failed
        request remains in the returned work set without creating another
        request row, so retries never overwrite or duplicate request evidence.
        """

        values = tuple(records)
        self._reject_duplicate_input(values, lambda value: value.request_id, "request")
        for value in values:
            value.to_row()
        self.initialize()
        with file_lock(self.lock_path):
            manifest = self._load_manifest_locked()
            index, _ = self._build_index_locked(manifest)
            new_records: list[RequestRecord] = []
            selected: list[RequestRecord] = []
            retries = 0
            skipped = 0
            for value in values:
                existing = index.requests.get(value.request_id)
                if existing is not None and existing.to_row() != value.to_row():
                    raise LedgerIntegrityError(
                        f"request ID collision with different content: {value.request_id}"
                    )
                if value.request_id in index.response_successes:
                    skipped += 1
                    continue
                if len(selected) >= self.batch_size:
                    continue
                if existing is None:
                    new_records.append(value)
                    index.requests[value.request_id] = value
                else:
                    retries += 1
                selected.append(value)
            if new_records:
                self._commit_locked(
                    manifest,
                    {"requests": [record.to_row() for record in new_records]},
                    {"new_requests": len(new_records), "scheduled_requests": len(selected)},
                )
            return RequestBatch(
                records=tuple(selected),
                newly_recorded=len(new_records),
                retried=retries,
                skipped_succeeded=skipped,
            )

    def record_responses(self, records: Iterable[ResponseRecord]) -> CommitResult:
        """Append response outcomes; successful identities are permanently deduped."""

        values = tuple(records)
        self._reject_duplicate_input(values, lambda value: value.request_id, "response")
        self.initialize()
        with file_lock(self.lock_path):
            manifest = self._load_manifest_locked()
            index, _ = self._build_index_locked(manifest)
            accepted: list[ResponseRecord] = []
            skipped = 0
            for value in values:
                request = index.requests.get(value.request_id)
                if request is None:
                    raise LedgerIntegrityError(
                        f"response references unknown request: {value.request_id}"
                    )
                if request.prompt_hash != value.prompt_hash:
                    raise LedgerIntegrityError(
                        f"response prompt hash disagrees for {value.request_id}"
                    )
                if value.request_id in index.response_successes:
                    skipped += 1
                    continue
                committed = replace(
                    value,
                    attempt=index.response_attempts.get(value.request_id, 0) + 1,
                )
                committed.to_row()
                accepted.append(committed)
                index.response_attempts[value.request_id] = committed.attempt
                if committed.status == "succeeded":
                    index.response_successes.add(committed.request_id)
            return self._commit_locked(
                manifest,
                {"responses": [record.to_row() for record in accepted]},
                {
                    "response_succeeded": sum(record.status == "succeeded" for record in accepted),
                    "response_failed": sum(record.status == "failed" for record in accepted),
                    "response_skipped_succeeded": skipped,
                },
                accepted=len(accepted),
                skipped_succeeded=skipped,
            )

    def schedule_embeddings(self, records: Iterable[EmbeddingRecord]) -> EmbeddingBatch:
        """Return unsucceeded embedding work without changing immutable shards.

        The caller supplies records with the desired input identity.  It should
        call :meth:`record_embeddings` after each Provider batch, including
        failed results; those failed rows make a later call retryable.
        """

        values = tuple(records)
        self._reject_duplicate_input(values, lambda value: value.embedding_id, "embedding")
        for value in values:
            # A scheduling record may be a synthetic failed placeholder; it is
            # never serialized here, but its stable identity still must verify.
            expected_id, expected_hash = stable_embedding_id(
                value.namespace, value.input_text, item_key=value.item_key, model=value.model
            )
            if value.embedding_id != expected_id or value.input_hash != expected_hash:
                raise LedgerIntegrityError(
                    f"embedding identity does not match input for {value.embedding_id!r}"
                )
        self.initialize()
        with file_lock(self.lock_path):
            manifest = self._load_manifest_locked()
            index, _ = self._build_index_locked(manifest)
            selected: list[EmbeddingRecord] = []
            retries = 0
            skipped = 0
            for value in values:
                previous = index.embeddings.get(value.embedding_id)
                if previous is not None and self._embedding_identity(previous) != self._embedding_identity(value):
                    raise LedgerIntegrityError(
                        f"embedding ID collision with different content: {value.embedding_id}"
                    )
                if value.embedding_id in index.embedding_successes:
                    skipped += 1
                    continue
                if len(selected) >= self.batch_size:
                    continue
                if previous is not None:
                    retries += 1
                selected.append(value)
            return EmbeddingBatch(tuple(selected), retries, skipped)

    def record_embeddings(self, records: Iterable[EmbeddingRecord]) -> CommitResult:
        """Append embedding successes/failures with immutable retry history."""

        values = tuple(records)
        self._reject_duplicate_input(values, lambda value: value.embedding_id, "embedding")
        self.initialize()
        with file_lock(self.lock_path):
            manifest = self._load_manifest_locked()
            index, _ = self._build_index_locked(manifest)
            accepted: list[EmbeddingRecord] = []
            skipped = 0
            for value in values:
                previous = index.embeddings.get(value.embedding_id)
                if previous is not None and self._embedding_identity(previous) != self._embedding_identity(value):
                    raise LedgerIntegrityError(
                        f"embedding ID collision with different content: {value.embedding_id}"
                    )
                if value.embedding_id in index.embedding_successes:
                    skipped += 1
                    continue
                committed = replace(
                    value,
                    attempt=index.embedding_attempts.get(value.embedding_id, 0) + 1,
                )
                committed.to_row()
                accepted.append(committed)
                index.embeddings[committed.embedding_id] = committed
                index.embedding_attempts[committed.embedding_id] = committed.attempt
                if committed.status == "succeeded":
                    index.embedding_successes.add(committed.embedding_id)
            return self._commit_locked(
                manifest,
                {"embeddings": [record.to_row() for record in accepted]},
                {
                    "embedding_succeeded": sum(record.status == "succeeded" for record in accepted),
                    "embedding_failed": sum(record.status == "failed" for record in accepted),
                    "embedding_skipped_succeeded": skipped,
                },
                accepted=len(accepted),
                skipped_succeeded=skipped,
            )

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "batch_size": self.batch_size,
            "shards": [],
            "batches": [],
            "stats": self._empty_stats(),
        }

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "requests": {"rows": 0, "unique": 0},
            "responses": {"rows": 0, "succeeded": 0, "failed": 0, "success_unique": 0},
            "embeddings": {"rows": 0, "succeeded": 0, "failed": 0, "success_unique": 0},
        }

    def _load_manifest_locked(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise LedgerIntegrityError(f"ledger manifest is missing: {self.manifest_path}")
        try:
            manifest = read_json(self.manifest_path)
        except (OSError, ValueError) as error:
            raise LedgerIntegrityError(f"cannot read ledger manifest: {self.manifest_path}") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerIntegrityError("invalid ledger manifest schema")
        if manifest.get("batch_size") != self.batch_size:
            raise LedgerIntegrityError(
                "ledger batch_size differs from immutable manifest; create a new ledger"
            )
        if not isinstance(manifest.get("shards"), list) or not isinstance(manifest.get("batches"), list):
            raise LedgerIntegrityError("ledger manifest has invalid shard or batch table")
        return manifest

    def _write_manifest_locked(self, manifest: Mapping[str, Any]) -> None:
        payload = dict(manifest)
        payload["updated_at"] = utc_now()
        atomic_write_json(self.manifest_path, payload, mode=0o600)

    def _recover_orphans_locked(
        self,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_paths = {
            str(shard.get("path"))
            for shard in manifest["shards"]
            if isinstance(shard, Mapping)
        }
        actual_by_kind: dict[ShardKind, list[Path]] = {}
        for kind in _SHARD_KINDS:
            directory = self.root / kind
            actual_by_kind[kind] = (
                sorted(directory.glob("part-*.jsonl"))
                if directory.is_dir()
                else []
            )
        actual_paths = {
            path.relative_to(self.root).as_posix()
            for paths in actual_by_kind.values()
            for path in paths
        }
        missing = expected_paths.difference(actual_paths)
        if missing:
            raise LedgerIntegrityError(
                f"ledger manifest references missing shard: {min(missing)}"
            )
        orphan_paths = actual_paths.difference(expected_paths)
        if not orphan_paths:
            return dict(manifest)

        recovered_shards: list[dict[str, Any]] = []
        recovery_batch = len(manifest["batches"]) + 1
        for kind in _SHARD_KINDS:
            expected_count = sum(
                1
                for shard in manifest["shards"]
                if isinstance(shard, Mapping) and shard.get("kind") == kind
            )
            kind_orphans = [
                path
                for path in actual_by_kind[kind]
                if path.relative_to(self.root).as_posix() in orphan_paths
            ]
            expected_names = [
                f"part-{sequence:06d}.jsonl"
                for sequence in range(
                    expected_count + 1,
                    expected_count + len(kind_orphans) + 1,
                )
            ]
            if [path.name for path in kind_orphans] != expected_names:
                raise LedgerIntegrityError(
                    f"non-contiguous orphan shards cannot be recovered for {kind}"
                )
            for path in kind_orphans:
                try:
                    rows = read_jsonl(path)
                except (OSError, ValueError) as error:
                    raise LedgerIntegrityError(
                        f"cannot recover corrupt orphan shard: {path}"
                    ) from error
                if not rows:
                    raise LedgerIntegrityError(
                        f"cannot recover empty orphan shard: {path}"
                    )
                recovered_shards.append(
                    {
                        "kind": kind,
                        "path": path.relative_to(self.root).as_posix(),
                        "batch_id": recovery_batch,
                        "created_at": utc_now(),
                        "rows": len(rows),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "recovered": True,
                    }
                )

        candidate = dict(manifest)
        candidate["shards"] = [*manifest["shards"], *recovered_shards]
        candidate["batches"] = [
            *manifest["batches"],
            {
                "batch_id": recovery_batch,
                "created_at": utc_now(),
                "shards": recovered_shards,
                "stats": {"recovered_orphan_shards": len(recovered_shards)},
                "recovered": True,
            },
        ]
        _, stats = self._build_index_locked(candidate)
        candidate["stats"] = stats
        self._write_manifest_locked(candidate)
        return candidate

    def _build_index_locked(self, manifest: Mapping[str, Any]) -> tuple[_LedgerIndex, dict[str, Any]]:
        request_records: dict[str, RequestRecord] = {}
        response_records: dict[str, ResponseRecord] = {}
        response_attempts: dict[str, int] = defaultdict(int)
        response_successes: set[str] = set()
        embeddings: dict[str, EmbeddingRecord] = {}
        embedding_attempts: dict[str, int] = defaultdict(int)
        embedding_successes: set[str] = set()
        seen_paths: set[str] = set()
        response_statuses: Counter[str] = Counter()
        embedding_statuses: Counter[str] = Counter()

        for raw_shard in manifest["shards"]:
            if not isinstance(raw_shard, Mapping):
                raise LedgerIntegrityError("ledger manifest contains a non-object shard")
            kind = raw_shard.get("kind")
            relative = raw_shard.get("path")
            if kind not in _SHARD_KINDS or not isinstance(relative, str) or relative in seen_paths:
                raise LedgerIntegrityError("ledger manifest contains an invalid or duplicate shard")
            seen_paths.add(relative)
            path = (self.root / relative).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as error:
                raise LedgerIntegrityError(f"ledger shard path escapes root: {relative}") from error
            if not path.is_file():
                raise LedgerIntegrityError(f"ledger shard is missing: {relative}")
            if sha256_file(path) != raw_shard.get("sha256"):
                raise LedgerIntegrityError(f"ledger shard hash changed: {relative}")
            try:
                rows = read_jsonl(path)
            except (OSError, ValueError) as error:
                raise LedgerIntegrityError(f"ledger shard is corrupt: {relative}") from error
            if raw_shard.get("rows") != len(rows):
                raise LedgerIntegrityError(f"ledger shard row count changed: {relative}")
            for row in rows:
                if kind == "requests":
                    record = RequestRecord.from_row(row)
                    prior = request_records.get(record.request_id)
                    if prior is not None:
                        raise LedgerIntegrityError(
                            f"duplicate request ID in immutable shards: {record.request_id}"
                        )
                    request_records[record.request_id] = record
                elif kind == "responses":
                    record = ResponseRecord.from_row(row)
                    request = request_records.get(record.request_id)
                    if request is None or request.prompt_hash != record.prompt_hash:
                        raise LedgerIntegrityError(
                            f"response has no matching request: {record.request_id}"
                        )
                    expected_attempt = response_attempts[record.request_id] + 1
                    if record.attempt != expected_attempt:
                        raise LedgerIntegrityError(
                            f"response attempts are not contiguous for {record.request_id}"
                        )
                    if record.request_id in response_successes:
                        raise LedgerIntegrityError(
                            f"response exists after success for {record.request_id}"
                        )
                    response_attempts[record.request_id] = record.attempt
                    response_records[record.request_id] = record
                    response_statuses[record.status] += 1
                    if record.status == "succeeded":
                        response_successes.add(record.request_id)
                else:
                    record = EmbeddingRecord.from_row(row)
                    prior = embeddings.get(record.embedding_id)
                    if prior is not None and self._embedding_identity(prior) != self._embedding_identity(record):
                        raise LedgerIntegrityError(
                            f"embedding ID collision in immutable shards: {record.embedding_id}"
                        )
                    expected_attempt = embedding_attempts[record.embedding_id] + 1
                    if record.attempt != expected_attempt:
                        raise LedgerIntegrityError(
                            f"embedding attempts are not contiguous for {record.embedding_id}"
                        )
                    if record.embedding_id in embedding_successes:
                        raise LedgerIntegrityError(
                            f"embedding exists after success for {record.embedding_id}"
                        )
                    embeddings[record.embedding_id] = record
                    embedding_attempts[record.embedding_id] = record.attempt
                    embedding_statuses[record.status] += 1
                    if record.status == "succeeded":
                        embedding_successes.add(record.embedding_id)

        # A manifest must account for every published immutable shard.  This
        # catches an interruption between exclusive shard publication and the
        # manifest update instead of silently reissuing a paid request.
        for kind in _SHARD_KINDS:
            directory = self.root / kind
            actual = {
                path.relative_to(self.root).as_posix()
                for path in directory.glob("part-*.jsonl")
            } if directory.is_dir() else set()
            expected = {
                str(shard["path"])
                for shard in manifest["shards"]
                if shard.get("kind") == kind
            }
            if actual != expected:
                raise LedgerIntegrityError(
                    f"ledger manifest/shard mismatch for {kind}: orphan or missing shard"
                )

        stats = {
            "requests": {"rows": len(request_records), "unique": len(request_records)},
            "responses": {
                "rows": sum(response_statuses.values()),
                "succeeded": response_statuses["succeeded"],
                "failed": response_statuses["failed"],
                "success_unique": len(response_successes),
            },
            "embeddings": {
                "rows": sum(embedding_statuses.values()),
                "succeeded": embedding_statuses["succeeded"],
                "failed": embedding_statuses["failed"],
                "success_unique": len(embedding_successes),
            },
        }
        return (
            _LedgerIndex(
                requests=request_records,
                responses=response_records,
                response_attempts=dict(response_attempts),
                response_successes=response_successes,
                embeddings=embeddings,
                embedding_attempts=dict(embedding_attempts),
                embedding_successes=embedding_successes,
            ),
            stats,
        )

    def _commit_locked(
        self,
        manifest: Mapping[str, Any],
        rows_by_kind: Mapping[ShardKind, Sequence[Mapping[str, Any]]],
        batch_stats: Mapping[str, Any],
        *,
        accepted: int | None = None,
        skipped_succeeded: int = 0,
    ) -> CommitResult:
        nonempty = {
            kind: tuple(rows)
            for kind, rows in rows_by_kind.items()
            if rows
        }
        if not nonempty:
            return CommitResult(None, (), 0 if accepted is None else accepted, skipped_succeeded)
        batch_id = len(manifest["batches"]) + 1
        shards = list(manifest["shards"])
        created_paths: list[Path] = []
        batch_shards: list[dict[str, Any]] = []
        try:
            for kind, rows in nonempty.items():
                sequence = 1 + sum(1 for shard in shards if shard.get("kind") == kind)
                relative = f"{kind}/part-{sequence:06d}.jsonl"
                path = self.root / relative
                data = b"".join(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
                    for row in rows
                )
                self._exclusive_write(path, data)
                created_paths.append(path)
                shard = {
                    "kind": kind,
                    "path": relative,
                    "batch_id": batch_id,
                    "created_at": utc_now(),
                    "rows": len(rows),
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
                shards.append(shard)
                batch_shards.append(dict(shard))
            next_manifest = dict(manifest)
            next_manifest["shards"] = shards
            batches = list(manifest["batches"])
            batches.append(
                {
                    "batch_id": batch_id,
                    "created_at": utc_now(),
                    "shards": batch_shards,
                    "stats": dict(batch_stats),
                }
            )
            next_manifest["batches"] = batches
            _, next_stats = self._build_index_from_pending(next_manifest, nonempty)
            next_manifest["stats"] = next_stats
            self._write_manifest_locked(next_manifest)
        except BaseException:
            # Existing shards are never removed.  A newly published shard that
            # was not indexed is deliberately left in place; resume will raise
            # a manifest/shard mismatch instead of risking duplicate billing.
            raise
        return CommitResult(
            batch_id=batch_id,
            shard_paths=tuple(created_paths),
            accepted=sum(len(rows) for rows in nonempty.values()) if accepted is None else accepted,
            skipped_succeeded=skipped_succeeded,
        )

    def _build_index_from_pending(
        self,
        manifest: Mapping[str, Any],
        pending: Mapping[ShardKind, Sequence[Mapping[str, Any]]],
    ) -> tuple[_LedgerIndex, dict[str, Any]]:
        """Calculate post-commit indexes without trusting a just-written manifest.

        The helper parses the newly created files through the normal verifier;
        ``pending`` only exists to keep this call's intent explicit and ensure
        every advertised shard kind was supplied by the current commit.
        """

        del pending
        return self._build_index_locked(manifest)

    @staticmethod
    def _embedding_identity(record: EmbeddingRecord) -> tuple[Any, ...]:
        return (
            record.input_hash,
            record.namespace,
            record.input_text,
            record.item_key,
            record.model,
        )

    @staticmethod
    def _reject_duplicate_input(
        values: Sequence[Any],
        identifier: Any,
        label: str,
    ) -> None:
        seen: set[str] = set()
        for value in values:
            key = str(identifier(value))
            if not key:
                raise LedgerIntegrityError(f"{label} record has an empty identity")
            if key in seen:
                raise LedgerIntegrityError(f"duplicate {label} identity in one batch: {key}")
            seen.add(key)

    @staticmethod
    def _exclusive_write(path: Path, data: bytes) -> None:
        """Durably publish one new shard without replacing an older shard."""

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise LedgerIntegrityError(
                    f"refusing to overwrite immutable shard: {path}"
                ) from error
            finally:
                temporary.unlink(missing_ok=True)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
