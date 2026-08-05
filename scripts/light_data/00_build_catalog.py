#!/usr/bin/env python3
"""Convert the self-contained light candidate JSONL into a training catalog."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from llmgen.clawhub import atomic_json, atomic_jsonl, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data_light/candidates.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("data_light/catalog.jsonl"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data_light/catalog_report.json"),
    )
    return parser.parse_args()


def read_candidates(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {number}: invalid JSON ({error})")
            continue
        if not isinstance(raw, dict):
            errors.append(f"line {number}: expected an object")
            continue
        name = str(raw.get("name") or "").strip()
        candidate = {
            # Hand-curated candidate lists commonly use the compact
            # {name, description} schema. A unique name is a stable closed-set
            # identifier when no explicit id is supplied.
            "id": str(raw.get("id") or name).strip(),
            "name": name,
            "desc": str(
                raw.get("desc") or raw.get("description") or ""
            ).strip(),
        }
        missing = [key for key, value in candidate.items() if not value]
        if missing:
            errors.append(f"line {number}: empty {', '.join(missing)}")
            continue
        rows.append(candidate)
    if errors:
        preview = "; ".join(errors[:10])
        suffix = f"; and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise ValueError(f"invalid candidate records: {preview}{suffix}")
    if not rows:
        raise ValueError(f"candidate list is empty: {path}")
    duplicates = sorted(
        candidate_id
        for candidate_id, count in Counter(row["id"] for row in rows).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate candidate IDs: {', '.join(duplicates)}")
    return rows


def build_catalog(
    candidates: list[dict[str, str]],
    *,
    source_path: Path,
) -> list[dict[str, Any]]:
    source_label = source_path.as_posix()
    return [
        {
            "rank": rank,
            "skill_id": row["id"],
            "owner": "data-light",
            "slug": row["id"],
            "display_name": row["name"],
            "summary": None,
            "description": row["desc"],
            "canonical_url": f"jsonl://{source_label}#{row['id']}",
            "description_provenance": {
                "type": "candidates_jsonl",
                "source_skill_id": row["id"],
                "source_url": f"jsonl://{source_label}#{row['id']}",
            },
        }
        for rank, row in enumerate(candidates, 1)
    ]


def main() -> None:
    args = parse_args()
    candidates = read_candidates(args.candidates)
    catalog = build_catalog(candidates, source_path=args.candidates)
    ids_by_name: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        ids_by_name[row["name"]].append(row["id"])
    duplicate_display_names = {
        name: skill_ids
        for name, skill_ids in sorted(ids_by_name.items())
        if len(skill_ids) > 1
    }
    report = {
        "stage": "light_candidate_catalog",
        "created_at": utc_now(),
        "source": str(args.candidates),
        "candidate_count": len(catalog),
        "unique_id_count": len({row["skill_id"] for row in catalog}),
        "unique_name_count": len({row["display_name"] for row in catalog}),
        "empty_description_count": sum(not row["description"] for row in catalog),
        "duplicate_display_names": duplicate_display_names,
    }
    atomic_jsonl(args.output, catalog)
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
