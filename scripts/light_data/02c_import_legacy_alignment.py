#!/usr/bin/env python3
"""Fill coverage deficits from previously model-reviewed alignment samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_alignment import append_legacy_alignment_queries
from llmgen.clawhub import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--legacy-queries", type=Path, required=True)
    parser.add_argument("--legacy-reviews", type=Path, required=True)
    parser.add_argument("--coverage-failure", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = append_legacy_alignment_queries(
        args.profiles,
        args.queries,
        args.reviews,
        args.legacy_queries,
        args.legacy_reviews,
        args.coverage_failure,
    )
    atomic_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
