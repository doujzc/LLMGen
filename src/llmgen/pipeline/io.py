"""Small, dependency-free persistence helpers for pipeline metadata."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashing and manifests."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or directory without following directory symlinks."""

    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for child in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = child.relative_to(root).as_posix()
        if child.is_symlink():
            digest.update(b"L\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(os.readlink(child).encode("utf-8"))
        elif child.is_file():
            digest.update(b"F\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("ascii"))
        elif child.is_dir():
            digest.update(b"D\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def path_size(path: str | Path) -> int:
    value = Path(path)
    if value.is_file():
        return value.stat().st_size
    if value.is_dir():
        return sum(
            child.stat().st_size
            for child in value.rglob("*")
            if child.is_file() and not child.is_symlink()
        )
    raise FileNotFoundError(value)


def jsonl_row_count(path: str | Path) -> int:
    with Path(path).open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def atomic_write_bytes(path: str | Path, data: bytes, *, mode: int = 0o644) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    mode: int = 0o644,
) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    mode: int = 0o644,
) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        mode=mode,
    )


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    mode: int = 0o644,
) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        mode=mode,
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{number}")
            rows.append(value)
    return rows


@contextmanager
def file_lock(path: str | Path, *, blocking: bool = True) -> Iterator[None]:
    """Take an advisory exclusive lock, creating the lock file safely."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        fcntl.flock(descriptor, operation)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
