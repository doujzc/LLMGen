#!/usr/bin/env python3
"""Independently reject semantically unrelated two-Skill conjunctions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llmgen.clawhub_dataset import ChatBatchClient, load_api_config
from llmgen.teststyle_data import strict_review_teststyle_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--workflows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, default=Path("~/deepseek_api_key.txt"))
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--checkpoint-batches", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com")
    client = ChatBatchClient(
        load_api_config(args.api_config, model=args.model),
        workers=args.workers,
        temperature=0.0,
    )
    result = strict_review_teststyle_queries(
        args.queries,
        args.workflows,
        args.output,
        client,
        batch_size=args.batch_size,
        checkpoint_batches=args.checkpoint_batches,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
