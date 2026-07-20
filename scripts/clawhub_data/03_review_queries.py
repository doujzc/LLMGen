#!/usr/bin/env python3
"""Stage 03: independently score naturalness, complexity, and target necessity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_dataset import ChatBatchClient, load_api_config, review_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("data/clawhub_training/queries.generated.jsonl"))
    parser.add_argument("--workflows", type=Path, default=Path("data/clawhub_training/workflows.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/clawhub_training/query_reviews.jsonl"))
    parser.add_argument("--api-config", type=Path, default=Path("~/llm_api.txt"))
    parser.add_argument("--model", default="GLM-5.1")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_api_config(args.api_config, model=args.model)
    client = ChatBatchClient(config, workers=args.workers, temperature=0.0)
    manifest = review_queries(
        args.queries,
        args.workflows,
        args.output,
        client,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
