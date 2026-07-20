#!/usr/bin/env python3
"""Stage 01b: apply audited candidate-only and recovery workflow decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_dataset import apply_recovery_workflows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=Path("data/clawhub_training/skill_profiles.jsonl"))
    parser.add_argument("--workflows", type=Path, default=Path("data/clawhub_training/workflows.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/clawhub_recovery.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = apply_recovery_workflows(args.profiles, args.workflows, args.config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
