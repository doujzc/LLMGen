#!/usr/bin/env python3
"""Stage 04a: export accepted single-skill curriculum data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_alignment import export_alignment_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-queries-per-skill", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export_alignment_dataset(args.catalog, args.queries, args.reviews, args.output_dir, min_queries_per_skill=args.min_queries_per_skill), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
