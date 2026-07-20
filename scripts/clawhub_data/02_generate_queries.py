#!/usr/bin/env python3
"""Stage 02: generate natural Chinese queries for fixed multi-skill targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_dataset import ChatBatchClient, generate_queries, load_api_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflows", type=Path, default=Path("data/clawhub_training/workflows.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/clawhub_training/queries.generated.jsonl"))
    parser.add_argument("--api-config", type=Path, default=Path("~/llm_api.txt"))
    parser.add_argument("--model", default="Qwen3.6-Plus")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_api_config(args.api_config, model=args.model)
    client = ChatBatchClient(config, workers=args.workers, temperature=0.55)
    manifest = generate_queries(
        args.workflows,
        args.output,
        client,
        variants=args.variants,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
