#!/usr/bin/env python3
"""Stage 03b: append targeted workflows for undercovered candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_dataset import append_coverage_workflows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=Path("data/clawhub_training/skill_profiles.jsonl"))
    parser.add_argument("--workflows", type=Path, default=Path("data/clawhub_training/workflows.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/clawhub_training/queries.generated.jsonl"))
    parser.add_argument("--reviews", type=Path, default=Path("data/clawhub_training/query_reviews.jsonl"))
    parser.add_argument("--alignment-queries", type=Path)
    parser.add_argument("--alignment-reviews", type=Path)
    parser.add_argument("--min-train-positives-per-skill", type=int, default=10)
    parser.add_argument("--variants-per-workflow", type=int, default=3)
    parser.add_argument("--oversample-factor", type=float, default=3.0)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = append_coverage_workflows(
        args.profiles,
        args.workflows,
        args.queries,
        args.reviews,
        alignment_queries_path=args.alignment_queries,
        alignment_reviews_path=args.alignment_reviews,
        min_train_positives_per_skill=args.min_train_positives_per_skill,
        variants_per_workflow=args.variants_per_workflow,
        oversample_factor=args.oversample_factor,
        round_index=args.round,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
