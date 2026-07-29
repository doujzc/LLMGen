#!/usr/bin/env python3
"""Append transparent repository-curated single-skill alignment samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_alignment import append_manual_alignment_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument(
        "--curated",
        type=Path,
        default=Path("data_light/manual_alignment.jsonl"),
    )
    args = parser.parse_args()
    result = append_manual_alignment_queries(
        args.profiles,
        args.queries,
        args.reviews,
        args.curated,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
