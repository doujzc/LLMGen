#!/usr/bin/env python3
"""Combine all reviewed Top1 training datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from llmgen.top1 import (
    Top1DataError,
    load_candidate_names,
    normalize_messages,
    read_jsonl,
    sha256_file,
    validate_training_rows,
    write_json,
    write_jsonl,
)


DATASET_VERSION = "top1_train_combined_v1"
DEFAULT_BASE_DATA = "data_top1/top1_train_v1.jsonl"
DEFAULT_AUGMENTATION_DATA = (
    "data_top1/generated/top1_controlled_multiturn_v1/train.jsonl"
)
DEFAULT_RETAIL_BOUNDARY_DATA = "data_top1/top1_retail_boundary_v1.jsonl"
DEFAULT_SHORT_QUERY_DATA = "data_top1/top1_short_queries_v1.jsonl"
DEFAULT_OUTPUT = "data_top1/top1_train_combined_v1.jsonl"
DEFAULT_SUMMARY = "data_top1/top1_train_combined_v1_summary.json"
EXPECTED_SOURCE_VERSIONS = (
    "top1_train_v1",
    "top1_controlled_multiturn_v1",
    "top1_retail_boundary_v1",
    "top1_short_queries_v1",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deterministic combined-dataset arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", default=DEFAULT_BASE_DATA)
    parser.add_argument("--augmentation-data", default=DEFAULT_AUGMENTATION_DATA)
    parser.add_argument(
        "--retail-boundary-data",
        default=DEFAULT_RETAIL_BOUNDARY_DATA,
    )
    parser.add_argument(
        "--short-query-data",
        default=DEFAULT_SHORT_QUERY_DATA,
    )
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    return parser.parse_args(argv)


def _canonical_messages(row: Mapping[str, Any]) -> str:
    messages = normalize_messages(row.get("messages"))
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def combine_training_rows(
    source_rows: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """Concatenate sources while rejecting duplicate IDs and conversations."""

    combined: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    seen_conversations: dict[str, tuple[str, str]] = {}
    for source_name, rows in source_rows:
        for row_number, row in enumerate(rows, start=1):
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id.strip():
                raise Top1DataError(
                    f"{source_name}:{row_number}: id must be a non-empty string"
                )
            if row_id in seen_ids:
                raise Top1DataError(
                    f"duplicate id {row_id!r} in {seen_ids[row_id]} and {source_name}"
                )
            conversation = _canonical_messages(row)
            if conversation in seen_conversations:
                previous_id, previous_source = seen_conversations[conversation]
                raise Top1DataError(
                    "duplicate conversation in "
                    f"{previous_source}:{previous_id} and {source_name}:{row_id}"
                )
            seen_ids[row_id] = source_name
            seen_conversations[conversation] = (row_id, source_name)
            combined.append(dict(row))
    return combined


def _validate_source_version(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected: str,
    source: Path,
) -> None:
    invalid = [
        index
        for index, row in enumerate(rows, start=1)
        if row.get("dataset_version") != expected
    ]
    if invalid:
        raise Top1DataError(
            f"{source}: dataset_version must be {expected!r}; "
            f"invalid rows begin at {invalid[:5]}"
        )


def _count(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def _phenomenon(row: Mapping[str, Any]) -> str:
    value = row.get("conversation_phenomenon")
    if not isinstance(value, str):
        return "base"
    normalized = value.strip().lower()
    if normalized == "intentchange":
        return "intent_change"
    return normalized or "base"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
    output_path: Path,
) -> dict[str, Any]:
    """Build provenance and balance diagnostics for the combined dataset."""

    current_utterances: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        current = normalize_messages(row.get("messages"))[-1]["content"]
        current_utterances[current].append(row)
    reused_current = {
        current: values
        for current, values in current_utterances.items()
        if len(values) > 1
    }
    conflicting_current = {
        current: values
        for current, values in reused_current.items()
        if len({str(row.get("target_candidate_name")) for row in values}) > 1
    }
    message_counts = Counter(len(normalize_messages(row.get("messages"))) for row in rows)
    reviewed_corrections = [
        row["label_review_correction"]
        for row in rows
        if isinstance(row.get("label_review_correction"), dict)
    ]
    return {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "rows": len(rows),
        "candidate_counts": _count(rows, "target_candidate_name"),
        "source_dataset_counts": _count(rows, "dataset_version"),
        "source_type_counts": _count(rows, "source_type"),
        "phenomenon_counts": dict(
            sorted(Counter(_phenomenon(row) for row in rows).items())
        ),
        "single_turn_rows": message_counts[1],
        "multi_turn_rows": len(rows) - message_counts[1],
        "multi_turn_ratio": (len(rows) - message_counts[1]) / len(rows),
        "message_count_distribution": {
            str(count): value for count, value in sorted(message_counts.items())
        },
        "unique_ids": len({str(row.get("id")) for row in rows}),
        "unique_conversations": len({_canonical_messages(row) for row in rows}),
        "unique_current_utterances": len(current_utterances),
        "reused_current_utterance_groups": len(reused_current),
        "current_utterance_label_conflicts": len(conflicting_current),
        "reviewed_label_corrections": {
            "rows": len(reviewed_corrections),
            "repair_version_counts": dict(
                sorted(
                    Counter(
                        str(correction.get("repair_version"))
                        for correction in reviewed_corrections
                    ).items()
                )
            ),
            "reason_code_counts": dict(
                sorted(
                    Counter(
                        str(correction.get("reason_code"))
                        for correction in reviewed_corrections
                    ).items()
                )
            ),
        },
        "sources": [
            {
                "path": _display_path(path),
                "sha256": sha256_file(path),
                "rows": len(source_rows),
            }
            for path, source_rows in sources
        ],
        "output": {
            "path": _display_path(output_path),
            "sha256": sha256_file(output_path),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    base_path = Path(args.base_data).expanduser().resolve()
    augmentation_path = Path(args.augmentation_data).expanduser().resolve()
    retail_boundary_path = Path(args.retail_boundary_data).expanduser().resolve()
    short_query_path = Path(args.short_query_data).expanduser().resolve()
    candidate_path = Path(args.candidate_registry).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    source_paths = {
        base_path,
        augmentation_path,
        retail_boundary_path,
        short_query_path,
    }
    if len(source_paths) != 4:
        raise Top1DataError("combined sources must be distinct")
    if output_path in source_paths:
        raise Top1DataError("combined output cannot overwrite a source dataset")
    if summary_path in {*source_paths, output_path}:
        raise Top1DataError("summary path must be distinct from data inputs and output")

    candidate_names = load_candidate_names(candidate_path)
    base_rows = read_jsonl(base_path)
    augmentation_rows = read_jsonl(augmentation_path)
    retail_boundary_rows = read_jsonl(retail_boundary_path)
    short_query_rows = read_jsonl(short_query_path)
    validate_training_rows(base_rows, candidate_names, source=base_path)
    validate_training_rows(
        augmentation_rows,
        candidate_names,
        source=augmentation_path,
    )
    validate_training_rows(
        retail_boundary_rows,
        candidate_names,
        source=retail_boundary_path,
    )
    validate_training_rows(
        short_query_rows,
        candidate_names,
        source=short_query_path,
    )
    _validate_source_version(
        base_rows,
        expected=EXPECTED_SOURCE_VERSIONS[0],
        source=base_path,
    )
    _validate_source_version(
        augmentation_rows,
        expected=EXPECTED_SOURCE_VERSIONS[1],
        source=augmentation_path,
    )
    _validate_source_version(
        retail_boundary_rows,
        expected=EXPECTED_SOURCE_VERSIONS[2],
        source=retail_boundary_path,
    )
    _validate_source_version(
        short_query_rows,
        expected=EXPECTED_SOURCE_VERSIONS[3],
        source=short_query_path,
    )
    combined = combine_training_rows(
        (
            (str(base_path), base_rows),
            (str(augmentation_path), augmentation_rows),
            (str(retail_boundary_path), retail_boundary_rows),
            (str(short_query_path), short_query_rows),
        )
    )
    report = validate_training_rows(combined, candidate_names, source=output_path)
    expected_rows = (
        len(base_rows)
        + len(augmentation_rows)
        + len(retail_boundary_rows)
        + len(short_query_rows)
    )
    if report["rows"] != expected_rows:
        raise Top1DataError("combined dataset row count mismatch")

    write_jsonl(output_path, combined)
    summary = build_summary(
        combined,
        sources=(
            (base_path, base_rows),
            (augmentation_path, augmentation_rows),
            (retail_boundary_path, retail_boundary_rows),
            (short_query_path, short_query_rows),
        ),
        output_path=output_path,
    )
    write_json(summary_path, summary)
    print(f"[combined] rows: {len(combined)}")
    print(f"[combined] training data: {output_path}")
    print(f"[combined] summary: {summary_path}")


if __name__ == "__main__":
    main()
