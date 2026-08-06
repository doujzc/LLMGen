#!/usr/bin/env python3
"""Audit candidate consistency, test leakage, coverage, and style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.teststyle_data import audit_final_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--distribution-profile", type=Path, required=True)
    parser.add_argument("--min-semantic-train-per-skill", type=int, default=100)
    args = parser.parse_args()
    result = audit_final_dataset(
        args.dataset_dir,
        args.candidates,
        args.heldout,
        args.distribution_profile,
        minimum_semantic_train_per_skill=args.min_semantic_train_per_skill,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
