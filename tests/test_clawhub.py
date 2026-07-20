from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from llmgen.clawhub import (
    AmbiguousSlug,
    ClawHubError,
    _extract_selected,
    fetch_ranked_catalog,
    resolve_catalog_owners,
)


def _item(slug: str, downloads: int, stars: int, created_at: int) -> dict:
    return {
        "slug": slug,
        "displayName": slug,
        "stats": {"downloads": downloads, "stars": stars},
        "createdAt": created_at,
        "updatedAt": created_at,
        "latestVersion": {"version": "1.0.0"},
    }


class CatalogClient:
    def __init__(self) -> None:
        self.calls = []
        self.pages = [
            [_item("a", 10, 1, 1), _item("b", 9, 2, 2)],
            [_item("c", 9, 8, 3), _item("d", 8, 99, 4)],
        ]

    def get_json(self, path, params):
        self.calls.append((path, params))
        page = 0 if params.get("cursor") is None else 1
        return {
            "items": self.pages[page],
            "nextCursor": "page-2" if page == 0 else None,
        }

    def api_url(self, path):
        return "https://example.test" + path


def test_ranked_catalog_reads_complete_download_tie() -> None:
    client = CatalogClient()
    rows, snapshot = fetch_ranked_catalog(client, limit=2, include_suspicious=False)
    assert [row["slug"] for row in rows] == ["a", "c"]
    assert [row["rank"] for row in rows] == [1, 2]
    assert snapshot["boundary_downloads"] == 9
    assert snapshot["pages"] == 2
    assert client.calls[0][1]["sort"] == "downloads"
    assert client.calls[0][1]["nonSuspiciousOnly"] == "true"


class OwnerClient:
    def get_detail(self, slug, owner=None):
        if owner is None:
            raise AmbiguousSlug(
                slug,
                [
                    {"ownerHandle": "owner-a"},
                    {"ownerHandle": "owner-b"},
                ],
            )
        created_at = 1 if owner == "owner-a" else 2
        return {
            "skill": {
                "slug": slug,
                "displayName": slug,
                "createdAt": created_at,
                "updatedAt": created_at,
                "stats": {"downloads": 10, "stars": 1},
            },
            "latestVersion": {"version": "1.0.0"},
            "owner": {"handle": owner},
        }


def test_owner_resolution_matches_catalog_identity() -> None:
    rows = [_item("same", 10, 1, 2), _item("same", 10, 1, 1)]
    rows[0]["rank"] = 1
    rows[1]["rank"] = 2
    resolved = resolve_catalog_owners(OwnerClient(), rows, workers=2)
    assert [row["owner"] for row in resolved] == ["owner-b", "owner-a"]


def _make_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)


def test_hosted_archive_extracts_without_execution(tmp_path: Path) -> None:
    archive = tmp_path / "skill.zip"
    _make_zip(archive, {"SKILL.md": "# Skill", "scripts/run.sh": "exit 1"})
    destination = tmp_path / "out"
    files = _extract_selected(
        archive,
        destination,
        source_path=None,
        max_unpacked_bytes=1024,
    )
    assert files == ["SKILL.md", "scripts/run.sh"]
    assert (destination / "SKILL.md").read_text() == "# Skill"


def test_github_archive_extracts_only_skill_subtree(tmp_path: Path) -> None:
    archive = tmp_path / "repo.zip"
    _make_zip(
        archive,
        {
            "repo-commit/README.md": "repo",
            "repo-commit/skills/example/SKILL.md": "# Skill",
            "repo-commit/skills/example/reference.md": "reference",
            "repo-commit/skills/other/SKILL.md": "other",
        },
    )
    destination = tmp_path / "out"
    files = _extract_selected(
        archive,
        destination,
        source_path="skills/example",
        max_unpacked_bytes=4096,
    )
    assert files == ["SKILL.md", "reference.md"]
    assert not (destination / "README.md").exists()


def test_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    _make_zip(archive, {"SKILL.md": "# Skill", "../escape": "bad"})
    with pytest.raises(ClawHubError, match="unsafe archive path"):
        _extract_selected(
            archive,
            tmp_path / "out",
            source_path=None,
            max_unpacked_bytes=1024,
        )
