#!/usr/bin/env python3
"""Stage 00: classify all ClawHub candidates by domain and workflow role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_dataset import ChatBatchClient, build_skill_profiles, load_api_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/clawhub/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/clawhub_training/skill_profiles.jsonl"))
    parser.add_argument("--api-config", type=Path, default=Path("~/llm_api.txt"))
    parser.add_argument("--model", default="Qwen3.6-Plus")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_api_config(args.api_config, model=args.model)
    client = ChatBatchClient(config, workers=args.workers, temperature=0.1)
    manifest = build_skill_profiles(
        args.catalog,
        args.output,
        client,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
