#!/usr/bin/env python3
"""Validate user-provided multi-turn Top1 JSONL without transforming it."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from llmgen.direct_router import (
    load_candidate_registry,
    messages_from_row,
    target_candidate_name,
)
from llmgen.router import RouterDataError, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate direct candidate-name Top1 JSONL files."
    )
    parser.add_argument("--candidate-registry", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--test")
    parser.add_argument("--report")
    return parser.parse_args()


def validate_data_files(
    *,
    candidate_registry: str | Path,
    split_paths: dict[str, str | Path | None],
) -> dict[str, Any]:
    """Validate schemas and targets without enforcing class coverage."""

    routes = load_candidate_registry(candidate_registry)
    routes_by_name = {route.name: route for route in routes}
    legal_names = set(routes_by_name)
    report: dict[str, Any] = {
        "routing_mode": "candidate_name_top1",
        "candidate_names": list(routes_by_name),
        "splits": {},
    }

    for split, raw_path in split_paths.items():
        if not raw_path:
            continue
        path = Path(raw_path)
        rows = read_jsonl(path)
        if split == "train" and not rows:
            raise RouterDataError("training JSONL is empty")
        counts: Counter[str] = Counter()
        labeled = 0
        multi_turn = 0
        for row_number, row in enumerate(rows, start=1):
            messages = messages_from_row(row)
            multi_turn += len(messages) > 1

            try:
                name = target_candidate_name(row)
            except RouterDataError:
                if split != "test":
                    raise RouterDataError(
                        f"{path}:{row_number} has no target_candidate_name"
                    ) from None
                continue
            if name not in legal_names:
                raise RouterDataError(
                    f"{path}:{row_number} has unknown target {name!r}"
                )
            labeled += 1
            counts[name] += 1
            if "expected_system_output" in row:
                expected = row["expected_system_output"]
                if expected != routes_by_name[name].intent_label:
                    raise RouterDataError(
                        f"{path}:{row_number} expected_system_output does not "
                        f"match target {name!r}"
                    )

        report["splits"][split] = {
            "path": str(path),
            "rows": len(rows),
            "labeled_rows": labeled,
            "multi_turn_rows": multi_turn,
            "candidate_counts": {
                name: counts[name] for name in routes_by_name if counts[name]
            },
        }

    return report


def main() -> None:
    args = parse_args()
    report = validate_data_files(
        candidate_registry=args.candidate_registry,
        split_paths={
            "train": args.train,
            "validation": args.validation,
            "test": args.test,
        },
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
