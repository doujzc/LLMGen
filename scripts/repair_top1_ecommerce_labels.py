#!/usr/bin/env python3
"""Apply reviewed EcommerceProduct boundary corrections to generated data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from llmgen.top1 import (
    Top1DataError,
    load_candidate_names,
    read_jsonl,
    sha256_file,
    validate_training_rows,
    write_json,
    write_jsonl,
)


DEFAULT_DATA = "data_top1/generated/top1_controlled_multiturn_v1/train.jsonl"
DEFAULT_REPAIRS = "configs/top1_ecommerce_boundary_repairs_v1.json"
DEFAULT_SUMMARY = "data_top1/generated/top1_controlled_multiturn_v1/summary.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deterministic label-repair arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--repairs", default=DEFAULT_REPAIRS)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_repair_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate an explicit reviewed label-repair specification."""

    repair_path = Path(path)
    try:
        payload = json.loads(repair_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Top1DataError(f"cannot read repair spec {repair_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise Top1DataError("repair spec must be a schema_version=1 object")
    version = payload.get("repair_version")
    repairs = payload.get("repairs")
    if not isinstance(version, str) or not version.strip():
        raise Top1DataError("repair_version must be a non-empty string")
    if not isinstance(repairs, list) or not repairs:
        raise Top1DataError("repairs must be a non-empty list")

    seen_ids: set[str] = set()
    for index, repair in enumerate(repairs, start=1):
        if not isinstance(repair, dict):
            raise Top1DataError(f"repairs[{index}] must be an object")
        required = (
            "id",
            "from_candidate_name",
            "to_candidate_name",
            "reason_code",
        )
        invalid = [
            field
            for field in required
            if not isinstance(repair.get(field), str) or not repair[field].strip()
        ]
        if invalid:
            raise Top1DataError(
                f"repairs[{index}] has invalid fields: {', '.join(invalid)}"
            )
        row_id = str(repair["id"])
        if row_id in seen_ids:
            raise Top1DataError(f"duplicate repair id: {row_id}")
        if repair["from_candidate_name"] == repair["to_candidate_name"]:
            raise Top1DataError(f"repair {row_id} does not change the label")
        seen_ids.add(row_id)
    return payload


def apply_label_repairs(
    rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply explicit ID-based corrections and preserve model judgments."""

    repair_version = str(spec["repair_version"])
    repair_by_id = {str(repair["id"]): repair for repair in spec["repairs"]}
    found: set[str] = set()
    output: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str):
            raise Top1DataError(f"row {row_number}: id must be a string")
        repair = repair_by_id.get(row_id)
        if repair is None:
            output.append(dict(row))
            continue

        found.add(row_id)
        previous = str(repair["from_candidate_name"])
        replacement = str(repair["to_candidate_name"])
        current = row.get("target_candidate_name")
        if current not in {previous, replacement}:
            raise Top1DataError(
                f"repair {row_id}: expected {previous!r} or already-repaired "
                f"{replacement!r}, got {current!r}"
            )

        corrected = dict(row)
        corrected["target_candidate_name"] = replacement
        expected_review = {
            "repair_version": repair_version,
            "previous_target_candidate_name": previous,
            "corrected_target_candidate_name": replacement,
            "reason_code": str(repair["reason_code"]),
        }
        existing_review = corrected.get("label_review_correction")
        if existing_review is not None and existing_review != expected_review:
            raise Top1DataError(f"repair {row_id}: conflicting correction metadata")
        corrected["label_review_correction"] = expected_review
        output.append(corrected)

    missing = sorted(set(repair_by_id) - found)
    if missing:
        raise Top1DataError("repair IDs missing from dataset: " + ", ".join(missing))
    return output


def _update_summary(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    repair_path: Path,
    data_path: Path,
    source_sha256: str,
    repair_version: str,
    repaired_rows: int,
) -> None:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Top1DataError(f"cannot read synthesis summary {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise Top1DataError("synthesis summary must be an object")

    summary["candidate_counts"] = dict(
        sorted(Counter(str(row["target_candidate_name"]) for row in rows).items())
    )
    prior_review = summary.get("post_generation_label_review")
    source_before_review = source_sha256
    if isinstance(prior_review, dict):
        preserved = prior_review.get("source_sha256_before_review")
        if isinstance(preserved, str) and preserved:
            source_before_review = preserved
    summary["post_generation_label_review"] = {
        "repair_version": repair_version,
        "repair_spec_path": str(repair_path),
        "repair_spec_sha256": sha256_file(repair_path),
        "source_sha256_before_review": source_before_review,
        "repaired_rows": repaired_rows,
        "policy": (
            "ordinary ecommerce goods in pre-purchase recommendation, comparison, "
            "price, promotion, performance, or suitability scenarios are "
            "EcommerceProduct"
        ),
    }
    output = summary.setdefault("output", {})
    if not isinstance(output, dict):
        raise Top1DataError("synthesis summary output must be an object")
    output["path"] = str(data_path)
    output["exists"] = data_path.is_file()
    output["sha256"] = sha256_file(data_path)
    write_json(path, summary)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_path = Path(args.data).expanduser().resolve()
    repair_path = Path(args.repairs).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    candidate_path = Path(args.candidate_registry).expanduser().resolve()

    source_sha256 = sha256_file(data_path)
    spec = load_repair_spec(repair_path)
    candidate_names = load_candidate_names(candidate_path)
    repairs = spec["repairs"]
    unknown_candidates = sorted(
        {
            str(repair[field])
            for repair in repairs
            for field in ("from_candidate_name", "to_candidate_name")
            if repair[field] not in candidate_names
        }
    )
    if unknown_candidates:
        raise Top1DataError(
            "repair spec contains unknown candidates: " + ", ".join(unknown_candidates)
        )

    rows = read_jsonl(data_path)
    corrected = apply_label_repairs(rows, spec)
    validate_training_rows(corrected, candidate_names, source=data_path)
    if args.dry_run:
        print(f"[repair] validated {len(repairs)} corrections; no files written")
        return

    write_jsonl(data_path, corrected)
    _update_summary(
        summary_path,
        rows=corrected,
        repair_path=repair_path,
        data_path=data_path,
        source_sha256=source_sha256,
        repair_version=str(spec["repair_version"]),
        repaired_rows=len(repairs),
    )
    print(f"[repair] corrected rows: {len(repairs)}")
    print(f"[repair] training data: {data_path}")
    print(f"[repair] summary: {summary_path}")


if __name__ == "__main__":
    main()
