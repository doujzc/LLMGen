#!/usr/bin/env python3
"""Append train-only workflows for strict-review coverage deficits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.teststyle_data import append_teststyle_coverage_workflows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--workflows", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--min-train-per-skill", type=int, default=100)
    parser.add_argument("--variants-per-workflow", type=int, default=5)
    parser.add_argument("--oversample-factor", type=float, default=2.0)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    result = append_teststyle_coverage_workflows(
        args.profiles,
        args.workflows,
        args.queries,
        args.reviews,
        minimum_train_per_skill=args.min_train_per_skill,
        variants_per_workflow=args.variants_per_workflow,
        oversample_factor=args.oversample_factor,
        round_index=args.round,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
