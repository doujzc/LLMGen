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
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--implicit-variants", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=20260722)
    parser.add_argument("--validation-retry-rounds", type=int, default=3)
    parser.add_argument("--min-completion-rate", type=float, default=0.95)
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
        implicit_variants=args.implicit_variants,
        batch_size=args.batch_size,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        validation_retry_rounds=args.validation_retry_rounds,
        min_completion_rate=args.min_completion_rate,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
