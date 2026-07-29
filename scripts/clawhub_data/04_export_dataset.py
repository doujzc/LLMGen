#!/usr/bin/env python3
"""Stage 04: filter, deduplicate, split, and export router training files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_dataset import export_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/clawhub/catalog.jsonl"))
    parser.add_argument("--profiles", type=Path, default=Path("data/clawhub_training/skill_profiles.jsonl"))
    parser.add_argument("--workflows", type=Path, default=Path("data/clawhub_training/workflows.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/clawhub_training/queries.generated.jsonl"))
    parser.add_argument("--reviews", type=Path, default=Path("data/clawhub_training/query_reviews.jsonl"))
    parser.add_argument("--alignment-queries", type=Path)
    parser.add_argument("--alignment-reviews", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/clawhub_training/final"))
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--min-train-positives-per-skill", type=int, default=10)
    parser.add_argument(
        "--min-augmented-train-queries",
        type=int,
        default=0,
        help="Fail before replacing the dataset when the augmented train split is smaller.",
    )
    parser.add_argument(
        "--target-order-variants",
        type=int,
        default=4,
        help="Maximum deterministic target permutations per accepted train query.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_training_dataset(
        args.catalog,
        args.profiles,
        args.workflows,
        args.queries,
        args.reviews,
        args.output_dir,
        seed=args.seed,
        min_train_positives_per_skill=args.min_train_positives_per_skill,
        min_augmented_train_queries=args.min_augmented_train_queries,
        target_order_variants=args.target_order_variants,
        alignment_queries_path=args.alignment_queries,
        alignment_reviews_path=args.alignment_reviews,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
