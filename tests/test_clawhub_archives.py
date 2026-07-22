from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from llmgen.clawhub_archives import ArchiveImportError, import_archive_catalog


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    archives = tmp_path / "archives"
    archives.mkdir()
    archive = archives / "0001__owner__weather__1.0.0.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "SKILL.md",
            "---\nname: weather\ndescription: Query current weather and forecasts.\n"
            "version: 1.0.0\n---\n# Weather\nUse a location and date to retrieve forecasts.\n",
        )
        output.writestr(
            "skill-card.md",
            "## Description: <br>\nWeather forecasts. <br>\n\n"
            "## Use Case: <br>\nPlan trips around rain and temperature. <br>\n",
        )
    data = archive.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "owner": "owner",
                    "slug": "weather",
                    "displayName": "Weather",
                    "version": "1.0.0",
                    "downloads": 10,
                    "stars": 2,
                    "filename": archive.name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "url": "https://example.test/weather",
                }
            ]
        )
    )
    return archives, manifest


def test_import_archive_catalog_verifies_and_extracts_metadata(tmp_path: Path) -> None:
    archives, manifest = _snapshot(tmp_path)
    output = tmp_path / "catalog.jsonl"
    result = import_archive_catalog(
        archives,
        manifest,
        output,
        expected_count=1,
    )
    row = json.loads(output.read_text())
    assert result["candidate_count"] == 1
    assert row["skill_id"] == "@owner/weather"
    assert row["summary"] == "Query current weather and forecasts."
    assert "Plan trips around rain" in row["description"]
    assert row["artifact"]["sha256"] == hashlib.sha256(
        (archives / row["artifact"]["filename"]).read_bytes()
    ).hexdigest()


def test_import_archive_catalog_rejects_hash_mismatch(tmp_path: Path) -> None:
    archives, manifest = _snapshot(tmp_path)
    payload = json.loads(manifest.read_text())
    payload[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ArchiveImportError, match="SHA-256"):
        import_archive_catalog(
            archives,
            manifest,
            tmp_path / "catalog.jsonl",
            expected_count=1,
        )
