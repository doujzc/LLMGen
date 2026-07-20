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
    parser.add_argument("--output-dir", type=Path, default=Path("data/clawhub_training/final"))
    parser.add_argument("--recovery-config", type=Path, default=Path("configs/clawhub_recovery.json"))
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--allow-missing-train-targets", action="store_true")
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
        require_train_target_coverage=not args.allow_missing_train_targets,
        recovery_config_path=args.recovery_config,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
