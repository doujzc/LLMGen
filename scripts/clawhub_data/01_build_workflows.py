#!/usr/bin/env python3
"""Stage 01: build balanced, non-redundant multi-skill workflow targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_dataset import build_workflow_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=Path("data/clawhub_training/skill_profiles.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/clawhub_training/workflows.jsonl"))
    parser.add_argument("--workflows-per-skill", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_workflow_specs(
        args.profiles,
        args.output,
        workflows_per_skill=args.workflows_per_skill,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
