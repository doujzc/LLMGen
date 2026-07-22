#!/usr/bin/env python3
"""Stage 05: validate coverage, implicit intent, and target-order augmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_audit import audit_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_training_dataset(
        args.dataset_dir,
        expected_candidates=args.expected_candidates,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
