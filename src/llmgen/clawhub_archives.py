"""Import a verified ClawHub ZIP snapshot into the LLMGen catalog schema."""

from __future__ import annotations

import html
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from llmgen.clawhub import atomic_json, atomic_jsonl, sha256_file, utc_now


class ArchiveImportError(RuntimeError):
    """Raised when an archive snapshot is incomplete or fails integrity checks."""


def _safe_members(
    archive: zipfile.ZipFile,
    *,
    max_members: int,
    max_unpacked_bytes: int,
) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if not members or len(members) > max_members:
        raise ArchiveImportError(
            f"archive member count {len(members)} outside [1, {max_members}]"
        )
    total = 0
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ArchiveImportError(f"unsafe archive member path: {member.filename}")
        total += member.file_size
        if total > max_unpacked_bytes:
            raise ArchiveImportError(
                f"archive expands beyond {max_unpacked_bytes} bytes"
            )
    return members


def _decode(data: bytes, *, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArchiveImportError(f"{name} is not valid UTF-8") from error


def _frontmatter(markdown: str) -> dict[str, str]:
    if not markdown.startswith("---"):
        return {}
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---(?:\s*\r?\n|$)", markdown, re.S)
    if not match:
        return {}
    lines = match.group(1).splitlines()
    values: dict[str, str] = {}
    index = 0
    while index < len(lines):
        found = re.match(r"^([A-Za-z][\w-]*):\s*(.*)$", lines[index])
        if not found:
            index += 1
            continue
        key, value = found.groups()
        if value in {"|", ">", "|-", ">-"}:
            block: list[str] = []
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or lines[index][:1].isspace()
            ):
                block.append(lines[index].strip())
                index += 1
            values[key] = " ".join(part for part in block if part)
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            try:
                value = str(json.loads(value))
            except json.JSONDecodeError:
                value = value[1:-1]
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1].replace("''", "'")
        values[key] = value
        index += 1
    return values


def _clean_markdown(value: str, *, limit: int) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.S)
    value = re.sub(r"```.*?```", " ", value, flags=re.S)
    value = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value, flags=re.M)
    value = re.sub(r"[*_`>|]", " ", value)
    value = html.unescape(value)
    value = " ".join(value.split())
    return value[:limit].strip()


def _card_section(markdown: str, heading: str) -> str:
    match = re.search(
        rf"^##+\s+{re.escape(heading)}\s*:?[^\n]*\n(.*?)(?=^##+\s|\Z)",
        markdown,
        flags=re.I | re.M | re.S,
    )
    return _clean_markdown(match.group(1), limit=2000) if match else ""


def _pick_member(members: Sequence[zipfile.ZipInfo], basename: str) -> zipfile.ZipInfo | None:
    matches = [
        member
        for member in members
        if not member.is_dir()
        and PurePosixPath(member.filename).name.casefold() == basename.casefold()
    ]
    if not matches:
        return None
    return min(matches, key=lambda member: (len(PurePosixPath(member.filename).parts), member.filename))


def _pick_skill_member(members: Sequence[zipfile.ZipInfo]) -> zipfile.ZipInfo | None:
    matches = [
        member
        for member in members
        if not member.is_dir()
        and PurePosixPath(member.filename).name.casefold() in {"skill.md", "skills.md"}
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda member: (
            PurePosixPath(member.filename).name.casefold() != "skill.md",
            len(PurePosixPath(member.filename).parts),
            member.filename,
        ),
    )


def _catalog_row(
    raw: Mapping[str, Any],
    archive_path: Path,
    *,
    max_members: int,
    max_unpacked_bytes: int,
) -> dict[str, Any]:
    expected_bytes = int(raw.get("bytes", -1))
    if expected_bytes != archive_path.stat().st_size:
        raise ArchiveImportError(f"archive size mismatch: {archive_path.name}")
    expected_sha256 = str(raw.get("sha256") or "")
    actual_sha256 = sha256_file(archive_path)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise ArchiveImportError(f"archive SHA-256 mismatch: {archive_path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(
            archive,
            max_members=max_members,
            max_unpacked_bytes=max_unpacked_bytes,
        )
        skill_member = _pick_skill_member(members)
        skill_markdown = ""
        content_status = "missing_skill_markdown"
        if skill_member is not None:
            try:
                skill_markdown = _decode(
                    archive.read(skill_member), name=skill_member.filename
                )
                content_status = "parsed_utf8_skill_markdown"
            except ArchiveImportError:
                content_status = "invalid_utf8_skill_markdown"
        card_member = _pick_member(members, "skill-card.md")
        card_markdown = (
            _decode(archive.read(card_member), name=card_member.filename)
            if card_member is not None
            else ""
        )

    frontmatter = _frontmatter(skill_markdown)
    card_description = _card_section(card_markdown, "Description")
    use_case = _card_section(card_markdown, "Use Case")
    summary = _clean_markdown(
        frontmatter.get("description") or card_description or use_case,
        limit=1200,
    )
    if not summary:
        summary = _clean_markdown(skill_markdown, limit=1200)
    description = _clean_markdown(skill_markdown, limit=6000)
    if not description:
        description = " ".join(part for part in (card_description, use_case) if part)
        content_status += "_fallback_to_skill_card"
    if use_case and use_case.casefold() not in description.casefold():
        description = f"{summary}\n\nUse case: {use_case}\n\n{description}"[:8000]
    owner = str(raw.get("owner") or "").strip()
    slug = str(raw.get("slug") or "").strip()
    if not owner or not slug:
        raise ArchiveImportError(f"manifest identity missing: {archive_path.name}")
    return {
        "rank": int(raw["rank"]),
        "skill_id": f"@{owner}/{slug}",
        "owner": owner,
        "slug": slug,
        "display_name": str(raw.get("displayName") or frontmatter.get("name") or slug),
        "summary": summary,
        "description": description,
        "latest_version": str(raw.get("version") or frontmatter.get("version") or ""),
        "stats": {
            "downloads": int(raw.get("downloads") or 0),
            "stars": int(raw.get("stars") or 0),
        },
        "canonical_url": str(raw.get("url") or ""),
        "artifact": {
            "kind": "verified_zip_snapshot",
            "filename": archive_path.name,
            "bytes": expected_bytes,
            "sha256": actual_sha256,
            "member_count": len(members),
            "skill_markdown_path": skill_member.filename if skill_member else None,
            "content_status": content_status,
        },
    }


def import_archive_catalog(
    archives_dir: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    expected_count: int | None = 1000,
    max_members: int = 5000,
    max_unpacked_bytes: int = 512 * 1024 * 1024,
) -> dict[str, Any]:
    archives_dir = archives_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ArchiveImportError("archive manifest must be a non-empty JSON list")
    if expected_count is not None and len(payload) != expected_count:
        raise ArchiveImportError(
            f"manifest has {len(payload)} rows, expected {expected_count}"
        )
    ranks = [int(row.get("rank", -1)) for row in payload if isinstance(row, dict)]
    filenames = [str(row.get("filename") or "") for row in payload if isinstance(row, dict)]
    identities = [
        (str(row.get("owner") or ""), str(row.get("slug") or ""))
        for row in payload
        if isinstance(row, dict)
    ]
    if len(ranks) != len(payload) or len(set(ranks)) != len(ranks):
        raise ArchiveImportError("manifest has invalid or duplicate ranks")
    if len(set(filenames)) != len(payload) or any(not name for name in filenames):
        raise ArchiveImportError("manifest has missing or duplicate filenames")
    if len(set(identities)) != len(payload):
        raise ArchiveImportError("manifest has duplicate owner/slug identities")

    catalog: list[dict[str, Any]] = []
    for index, raw in enumerate(sorted(payload, key=lambda row: int(row["rank"])), 1):
        archive_path = archives_dir / str(raw["filename"])
        if not archive_path.is_file():
            raise ArchiveImportError(f"archive missing: {archive_path}")
        catalog.append(
            _catalog_row(
                raw,
                archive_path,
                max_members=max_members,
                max_unpacked_bytes=max_unpacked_bytes,
            )
        )
        if index % 100 == 0 or index == len(payload):
            print(f"verified and imported {index}/{len(payload)} archives", flush=True)
    atomic_jsonl(output_path, catalog)
    result = {
        "stage": "archive_catalog_import",
        "created_at": utc_now(),
        "archives_dir": str(archives_dir),
        "source_manifest": str(manifest_path),
        "catalog": str(output_path),
        "candidate_count": len(catalog),
        "catalog_sha256": sha256_file(output_path),
        "archive_bytes": sum(int(row["artifact"]["bytes"]) for row in catalog),
        "integrity": "all archive sizes and SHA-256 digests verified",
    }
    atomic_json(output_path.with_suffix(".manifest.json"), result)
    return result
