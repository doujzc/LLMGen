"""Reproducible ClawHub catalog snapshots and skill package downloads."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import shutil
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_BASE_URL = "https://clawhub.ai"
USER_AGENT = "LLMGen-ClawHub-dataset/0.1 (+https://github.com/)"
SKILL_FILENAMES = {"skill.md", "skills.md"}


class ClawHubError(RuntimeError):
    """Raised when a ClawHub request or artifact cannot be validated."""


class AmbiguousSlug(ClawHubError):
    """Carries the owner candidates returned for a non-unique slug."""

    def __init__(self, slug: str, matches: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(f"ambiguous ClawHub slug: {slug}")
        self.slug = slug
        self.matches = list(matches)


@dataclass(frozen=True)
class DownloadConfig:
    output_dir: Path
    limit: int = 1000
    workers: int = 12
    include_suspicious: bool = False
    refresh_snapshot: bool = False
    force: bool = False
    keep_archives: bool = False
    max_archive_bytes: int = 256 * 1024 * 1024
    max_unpacked_bytes: int = 512 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, _json_bytes(value))


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in rows
    )
    atomic_write(path, data)


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    reset = headers.get("RateLimit-Reset") or headers.get("ratelimit-reset")
    if reset:
        try:
            value = float(reset)
            if value > 10_000_000_000:
                value /= 1000
            return max(0.0, value - time.time())
        except ValueError:
            pass
    return min(30.0, 0.75 * (2**attempt)) + random.random() * 0.25


class ClawHubClient:
    """Small stdlib client with bounded retries and rate-limit handling."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 90.0,
        max_retries: int = 8,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        # urllib prefers an explicitly present lowercase variable even when it
        # is empty. Some managed/WSL environments expose an empty
        # ``https_proxy`` alongside a valid ``HTTPS_PROXY``; recover the valid
        # value instead of silently attempting a slow direct connection.
        proxies = urllib.request.getproxies()
        for scheme in ("http", "https"):
            if not proxies.get(scheme):
                value = os.environ.get(f"{scheme.upper()}_PROXY") or os.environ.get("ALL_PROXY")
                if value:
                    proxies[scheme] = value
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))

    def api_url(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        if params:
            values = {key: value for key, value in params.items() if value is not None}
            url += "?" + urllib.parse.urlencode(values)
        return url

    def request_bytes(self, url: str, *, max_bytes: int | None = None) -> tuple[bytes, Mapping[str, str]]:
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    content_length = response.headers.get("Content-Length")
                    if max_bytes and content_length and int(content_length) > max_bytes:
                        raise ClawHubError(f"response exceeds {max_bytes} bytes: {url}")
                    body = response.read(None if max_bytes is None else max_bytes + 1)
                    if max_bytes and len(body) > max_bytes:
                        raise ClawHubError(f"response exceeds {max_bytes} bytes: {url}")
                    return body, dict(response.headers.items())
            except urllib.error.HTTPError as error:
                body = error.read()
                if error.code == 409:
                    payload = json.loads(body)
                    if payload.get("code") == "AMBIGUOUS_SKILL_SLUG":
                        raise AmbiguousSlug(payload.get("slug", ""), payload.get("matches", []))
                if error.code != 429 and not 500 <= error.code < 600:
                    raise ClawHubError(
                        f"HTTP {error.code} for {url}: {body[:500].decode(errors='replace')}"
                    ) from error
                if attempt >= self.max_retries:
                    raise ClawHubError(f"HTTP {error.code} after retries: {url}") from error
                time.sleep(_retry_delay(dict(error.headers.items()), attempt))
            except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.HTTPException) as error:
                if attempt >= self.max_retries:
                    raise ClawHubError(f"request failed after retries: {url}: {error}") from error
                time.sleep(_retry_delay({}, attempt))
        raise AssertionError("unreachable")

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        body, _ = self.request_bytes(self.api_url(path, params), max_bytes=32 * 1024 * 1024)
        try:
            value = json.loads(body)
        except json.JSONDecodeError as error:
            raise ClawHubError(f"invalid JSON from {path}") from error
        if not isinstance(value, dict):
            raise ClawHubError(f"expected JSON object from {path}")
        return value

    def get_detail(self, slug: str, owner: str | None = None) -> dict[str, Any]:
        path = "/api/v1/skills/" + urllib.parse.quote(slug, safe="")
        return self.get_json(path, {"ownerHandle": owner} if owner else None)

    def download_response(
        self,
        slug: str,
        owner: str,
        version: str,
        *,
        max_bytes: int,
    ) -> tuple[bytes, Mapping[str, str]]:
        url = self.api_url(
            "/api/v1/download",
            {"slug": slug, "ownerHandle": owner, "version": version},
        )
        return self.request_bytes(url, max_bytes=max_bytes)


def catalog_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    stats = item.get("stats") or {}
    return (
        -int(stats.get("downloads") or 0),
        -int(stats.get("stars") or 0),
        -int(stats.get("installs") or 0),
        -int(item.get("updatedAt") or 0),
        str(item.get("slug") or ""),
        str(item.get("displayName") or ""),
    )


def fetch_ranked_catalog(
    client: ClawHubClient,
    *,
    limit: int,
    include_suspicious: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch downloads-desc pages and locally break ties by stars descending."""
    if limit < 1:
        raise ValueError("limit must be positive")
    cursor: str | None = None
    items: list[dict[str, Any]] = []
    pages = 0
    boundary_downloads: int | None = None
    while True:
        payload = client.get_json(
            "/api/v1/skills",
            {
                "limit": 200,
                "cursor": cursor,
                "sort": "downloads",
                "nonSuspiciousOnly": None if include_suspicious else "true",
            },
        )
        page = payload.get("items")
        if not isinstance(page, list):
            raise ClawHubError("ClawHub catalog response has no items list")
        items.extend(item for item in page if isinstance(item, dict))
        pages += 1
        if len(items) >= limit and boundary_downloads is None:
            ordered = sorted(items, key=catalog_sort_key)
            boundary_downloads = int((ordered[limit - 1].get("stats") or {}).get("downloads") or 0)
        cursor = payload.get("nextCursor")
        lowest = min(
            (int((item.get("stats") or {}).get("downloads") or 0) for item in page),
            default=-1,
        )
        # Continue through the complete downloads tie at the selection boundary.
        if not cursor or (boundary_downloads is not None and lowest < boundary_downloads):
            break
    ordered = sorted(items, key=catalog_sort_key)
    selected = ordered[:limit]
    if len(selected) < limit:
        raise ClawHubError(f"catalog only returned {len(selected)} skills, requested {limit}")
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
    snapshot = {
        "captured_at": utc_now(),
        "api": client.api_url("/api/v1/skills"),
        "server_sort": "downloads_desc",
        "local_tie_break": ["stars_desc", "installs_desc", "updated_at_desc", "slug_asc"],
        "non_suspicious_only": not include_suspicious,
        "requested_count": limit,
        "selected_count": len(selected),
        "fetched_count": len(items),
        "pages": pages,
        "boundary_downloads": int((selected[-1].get("stats") or {}).get("downloads") or 0),
    }
    return selected, snapshot


def _detail_score(item: Mapping[str, Any], detail: Mapping[str, Any]) -> int:
    skill = detail.get("skill") or {}
    latest = detail.get("latestVersion") or {}
    listed_latest = item.get("latestVersion") or {}
    score = 0
    if item.get("createdAt") == skill.get("createdAt"):
        score += 1_000_000
    if item.get("updatedAt") == skill.get("updatedAt"):
        score += 100_000
    if listed_latest.get("version") == latest.get("version"):
        score += 10_000
    if item.get("displayName") == skill.get("displayName"):
        score += 1_000
    for key in ("downloads", "stars", "installs", "versions"):
        if (item.get("stats") or {}).get(key) == (skill.get("stats") or {}).get(key):
            score += 10
    return score


def resolve_catalog_owners(
    client: ClawHubClient,
    catalog: Sequence[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in catalog:
        groups[str(item["slug"])].append(item)

    def resolve_group(slug: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            details = [client.get_detail(slug)]
        except AmbiguousSlug as ambiguous:
            details = [
                client.get_detail(slug, str(match["ownerHandle"]))
                for match in ambiguous.matches
            ]
        if len(details) < len(rows):
            raise ClawHubError(f"not enough owner matches for slug {slug}")
        remaining = list(details)
        resolved: list[dict[str, Any]] = []
        for row in rows:
            detail = max(remaining, key=lambda candidate: _detail_score(row, candidate))
            if _detail_score(row, detail) < 1_000_000:
                raise ClawHubError(f"could not uniquely match owner for catalog slug {slug}")
            remaining.remove(detail)
            owner = (detail.get("owner") or {}).get("handle")
            if not owner:
                raise ClawHubError(f"owner missing from detail for slug {slug}")
            merged = dict(row)
            merged["owner"] = owner
            merged["detail"] = detail
            resolved.append(merged)
        return resolved

    resolved: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(resolve_group, slug, rows): slug
            for slug, rows in groups.items()
        }
        for future in as_completed(futures):
            resolved.extend(future.result())
    return sorted(resolved, key=lambda item: int(item["rank"]))


def _load_frozen_catalog(config: DownloadConfig) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    path = config.output_dir / "catalog_snapshot.json"
    if config.refresh_snapshot or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClawHubError(f"invalid frozen catalog snapshot: {path}") from error
    snapshot = payload.get("snapshot") or {}
    items = payload.get("items") or []
    expected_filter = not config.include_suspicious
    if len(items) != config.limit:
        raise ClawHubError(
            f"frozen snapshot contains {len(items)} skills, not --limit {config.limit}; "
            "use --refresh-snapshot to replace it"
        )
    if snapshot.get("non_suspicious_only") != expected_filter:
        raise ClawHubError(
            "frozen snapshot suspicious-content filter differs from this invocation; "
            "use --refresh-snapshot to replace it"
        )
    return list(items), dict(snapshot)


def _load_cached_details(output_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    details: dict[tuple[str, int], dict[str, Any]] = {}
    metadata_root = output_dir / "metadata"
    if not metadata_root.is_dir():
        return details
    for path in metadata_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            detail = payload["detail"]
            skill = detail["skill"]
            key = (str(skill["slug"]), int(skill["createdAt"]))
            if (detail.get("owner") or {}).get("handle"):
                details[key] = detail
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return details


def resolve_catalog_owners_resumable(
    client: ClawHubClient,
    catalog: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    workers: int,
    snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reuse the frozen resolved snapshot or per-skill metadata on reruns."""
    resolved_path = output_dir / "catalog_resolved.json"
    if resolved_path.is_file():
        try:
            payload = json.loads(resolved_path.read_text(encoding="utf-8"))
            items = payload.get("items") or []
            if (
                payload.get("catalog_captured_at") == snapshot.get("captured_at")
                and len(items) == len(catalog)
            ):
                return sorted(items, key=lambda item: int(item["rank"]))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    cached = _load_cached_details(output_dir)
    resolved: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for item in catalog:
        key = (str(item["slug"]), int(item["createdAt"]))
        detail = cached.get(key)
        if detail is None:
            pending.append(item)
            continue
        merged = dict(item)
        merged["owner"] = str(detail["owner"]["handle"])
        merged["detail"] = detail
        resolved.append(merged)
    if pending:
        resolved.extend(resolve_catalog_owners(client, pending, workers=workers))
    resolved = sorted(resolved, key=lambda item: int(item["rank"]))
    atomic_json(
        resolved_path,
        {
            "catalog_captured_at": snapshot.get("captured_at"),
            "resolved_at": utc_now(),
            "items": resolved,
        },
    )
    return resolved


def _safe_name(value: str, label: str) -> str:
    if not value or value in {".", ".."} or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in value):
        raise ClawHubError(f"unsafe {label}: {value!r}")
    return value


def _checked_members(
    archive: zipfile.ZipFile,
    *,
    max_unpacked_bytes: int,
) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    total = 0
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ClawHubError(f"unsafe archive path: {info.filename!r}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ClawHubError(f"archive contains a symlink: {info.filename!r}")
        total += info.file_size
        if total > max_unpacked_bytes:
            raise ClawHubError(f"archive expands beyond {max_unpacked_bytes} bytes")
        members.append(info)
    return members


def _hosted_member_paths(members: Sequence[zipfile.ZipInfo]) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    files = [(info, PurePosixPath(info.filename.replace("\\", "/"))) for info in members if not info.is_dir()]
    if not files:
        raise ClawHubError("empty skill archive")
    if not any(path.name.lower() in SKILL_FILENAMES for _, path in files):
        raise ClawHubError("skill archive does not contain SKILL.md")
    # ClawHub packages normally use root-relative paths. Strip one wrapper only
    # when every file shares it and the skill document is below that wrapper.
    first = {path.parts[0] for _, path in files if path.parts}
    strip = 1 if len(first) == 1 and all(len(path.parts) > 1 for _, path in files) else 0
    return [(info, PurePosixPath(*path.parts[strip:])) for info, path in files]


def _github_member_paths(
    members: Sequence[zipfile.ZipInfo],
    source_path: str,
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    wanted = PurePosixPath(source_path.strip("/"))
    if wanted.is_absolute() or not wanted.parts or ".." in wanted.parts:
        raise ClawHubError(f"unsafe GitHub skill path: {source_path!r}")
    source_is_file = wanted.name.lower() in SKILL_FILENAMES
    selected: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for info in members:
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename.replace("\\", "/")).parts
        # GitHub source archives add a repository/commit wrapper directory.
        for start in range(len(parts)):
            if tuple(parts[start : start + len(wanted.parts)]) != wanted.parts:
                continue
            tail = parts[start + len(wanted.parts) :]
            if source_is_file:
                if not tail:
                    selected.append((info, PurePosixPath(wanted.name)))
            elif tail:
                selected.append((info, PurePosixPath(*tail)))
            break
    if not selected:
        raise ClawHubError(f"GitHub archive does not contain path {source_path!r}")
    if not any(path.name.lower() in SKILL_FILENAMES for _, path in selected):
        raise ClawHubError(f"GitHub path {source_path!r} does not contain SKILL.md")
    return selected


def _extract_selected(
    archive_path: Path,
    destination: Path,
    *,
    source_path: str | None,
    max_unpacked_bytes: int,
) -> list[str]:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _checked_members(archive, max_unpacked_bytes=max_unpacked_bytes)
            selected = (
                _github_member_paths(members, source_path)
                if source_path is not None
                else _hosted_member_paths(members)
            )
            for info, relative in selected:
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise ClawHubError(f"unsafe extracted path: {relative}")
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        files = sorted(str(path.relative_to(staging).as_posix()) for path in staging.rglob("*") if path.is_file())
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        return files
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _response_is_json(headers: Mapping[str, str], body: bytes) -> bool:
    content_type = headers.get("Content-Type", headers.get("content-type", ""))
    return "json" in content_type.lower() or body.lstrip().startswith(b"{")


def _artifact_matches(metadata_path: Path, skill_dir: Path, version: str) -> dict[str, Any] | None:
    if not metadata_path.is_file() or not skill_dir.is_dir():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    artifact = metadata.get("artifact") or {}
    if artifact.get("version") != version:
        return None
    files = artifact.get("files") or []
    if files and all((skill_dir / relative).is_file() for relative in files):
        artifact = dict(artifact)
        artifact["resumed"] = True
        return artifact
    return None


def download_skill(
    client: ClawHubClient,
    item: Mapping[str, Any],
    config: DownloadConfig,
) -> dict[str, Any]:
    owner = _safe_name(str(item["owner"]), "owner")
    slug = _safe_name(str(item["slug"]), "slug")
    detail = item["detail"]
    version = str((detail.get("latestVersion") or {}).get("version") or "")
    if not version:
        raise ClawHubError(f"latest version missing for @{owner}/{slug}")
    skill_dir = config.output_dir / "skills" / owner / slug
    metadata_path = config.output_dir / "metadata" / owner / f"{slug}.json"
    if not config.force:
        existing = _artifact_matches(metadata_path, skill_dir, version)
        if existing:
            return existing

    body, headers = client.download_response(
        slug,
        owner,
        version,
        max_bytes=config.max_archive_bytes,
    )
    source_path: str | None = None
    source: dict[str, Any] = {"kind": "clawhub-hosted"}
    if _response_is_json(headers, body):
        try:
            handoff = json.loads(body)
        except json.JSONDecodeError as error:
            raise ClawHubError(f"invalid download handoff for @{owner}/{slug}") from error
        if handoff.get("sourceRef") != "public-github" or not handoff.get("archiveUrl"):
            raise ClawHubError(f"unsupported download response for @{owner}/{slug}: {handoff}")
        archive_url = str(handoff["archiveUrl"])
        parsed = urllib.parse.urlparse(archive_url)
        if parsed.scheme != "https":
            raise ClawHubError(f"non-HTTPS GitHub archive URL for @{owner}/{slug}")
        body, _ = client.request_bytes(archive_url, max_bytes=config.max_archive_bytes)
        source_path = str(handoff.get("path") or "")
        source = {
            "kind": "public-github",
            "repo": handoff.get("repo"),
            "commit": handoff.get("commit"),
            "path": source_path,
            "content_hash": handoff.get("contentHash"),
            "archive_url": archive_url,
        }
    if not body.startswith(b"PK"):
        raise ClawHubError(f"download is not a zip archive for @{owner}/{slug}")

    archive_sha256 = hashlib.sha256(body).hexdigest()
    archive_dir = config.output_dir / "archives" / owner
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{slug}-{version}.zip"
    atomic_write(archive_path, body)
    try:
        files = _extract_selected(
            archive_path,
            skill_dir,
            source_path=source_path,
            max_unpacked_bytes=config.max_unpacked_bytes,
        )
    finally:
        if not config.keep_archives:
            archive_path.unlink(missing_ok=True)
    artifact = {
        "version": version,
        "downloaded_at": utc_now(),
        "archive_sha256": archive_sha256,
        "files": files,
        "source": source,
        "resumed": False,
    }
    atomic_json(metadata_path, {"detail": detail, "artifact": artifact})
    return artifact


def public_catalog_row(item: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    owner = str(item["owner"])
    slug = str(item["slug"])
    latest = item.get("latestVersion") or {}
    return {
        "rank": int(item["rank"]),
        "skill_id": f"@{owner}/{slug}",
        "owner": owner,
        "slug": slug,
        "display_name": item.get("displayName"),
        "summary": item.get("summary"),
        "description": item.get("description"),
        "topics": item.get("topics") or [],
        "tags": item.get("tags") or {},
        "stats": item.get("stats") or {},
        "created_at": item.get("createdAt"),
        "updated_at": item.get("updatedAt"),
        "latest_version": latest.get("version"),
        "latest_version_created_at": latest.get("createdAt"),
        "canonical_url": f"https://clawhub.ai/{urllib.parse.quote(owner)}/skills/{urllib.parse.quote(slug)}",
        "artifact": dict(artifact),
    }


def crawl(client: ClawHubClient, config: DownloadConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = _load_frozen_catalog(config)
    if frozen is None:
        catalog, snapshot = fetch_ranked_catalog(
            client,
            limit=config.limit,
            include_suspicious=config.include_suspicious,
        )
        atomic_json(config.output_dir / "catalog_snapshot.json", {"snapshot": snapshot, "items": catalog})
        # A refreshed selection invalidates its owner-resolution cache.
        (config.output_dir / "catalog_resolved.json").unlink(missing_ok=True)
    else:
        catalog, snapshot = frozen
    resolved = resolve_catalog_owners_resumable(
        client,
        catalog,
        output_dir=config.output_dir,
        workers=config.workers,
        snapshot=snapshot,
    )
    errors: list[dict[str, Any]] = []
    artifacts: dict[int, dict[str, Any]] = {}

    def do_download(item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return int(item["rank"]), download_skill(client, item, config)

    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as executor:
        futures = {executor.submit(do_download, item): item for item in resolved}
        completed = 0
        for future in as_completed(futures):
            item = futures[future]
            try:
                rank, artifact = future.result()
                artifacts[rank] = artifact
            except Exception as error:  # preserve all per-item failures for a resumable rerun
                errors.append(
                    {
                        "rank": item.get("rank"),
                        "owner": item.get("owner"),
                        "slug": item.get("slug"),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            completed += 1
            if completed % 50 == 0 or completed == len(resolved):
                print(f"downloaded/resumed {completed}/{len(resolved)}; failures={len(errors)}", flush=True)

    rows = [public_catalog_row(item, artifacts[int(item["rank"])]) for item in resolved if int(item["rank"]) in artifacts]
    atomic_jsonl(config.output_dir / "catalog.jsonl", rows)
    errors = sorted(errors, key=lambda value: int(value.get("rank") or 0))
    atomic_jsonl(config.output_dir / "errors.jsonl", errors)
    manifest = {
        "format_version": 1,
        "source": "ClawHub public API",
        "source_url": client.base_url,
        "license_note": "ClawHub skill packages are published under MIT-0; inspect per-skill metadata.",
        "created_at": utc_now(),
        "snapshot": snapshot,
        "selection": {
            "primary": "downloads_desc",
            "secondary": "stars_desc",
            "non_suspicious_only": not config.include_suspicious,
        },
        "requested_count": config.limit,
        "resolved_count": len(resolved),
        "downloaded_count": len(rows),
        "failed_count": len(errors),
        "catalog_sha256": sha256_file(config.output_dir / "catalog.jsonl"),
        "snapshot_sha256": sha256_file(config.output_dir / "catalog_snapshot.json"),
    }
    atomic_json(config.output_dir / "manifest.json", manifest)
    if errors:
        raise ClawHubError(
            f"{len(errors)} downloads failed; see {config.output_dir / 'errors.jsonl'} and rerun to resume"
        )
    return manifest
