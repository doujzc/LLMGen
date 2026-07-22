#!/usr/bin/env python3
"""Stage 03a: independently review direct single-skill curriculum queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_alignment import review_alignment_queries
from llmgen.clawhub_dataset import ChatBatchClient, load_api_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, default=Path("~/llm_api.txt"))
    parser.add_argument("--model", default="GLM-5.1")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    client = ChatBatchClient(load_api_config(args.api_config, model=args.model), workers=args.workers, temperature=0.0)
    print(json.dumps(review_alignment_queries(args.queries, args.profiles, args.output, client, batch_size=args.batch_size, force=args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
