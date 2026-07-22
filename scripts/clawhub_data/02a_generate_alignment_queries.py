#!/usr/bin/env python3
"""Stage 02a: generate direct single-skill curriculum queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_alignment import generate_alignment_queries
from llmgen.clawhub_dataset import ChatBatchClient, load_api_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, default=Path("~/llm_api.txt"))
    parser.add_argument("--model", default="Qwen3.6-Plus")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    client = ChatBatchClient(load_api_config(args.api_config, model=args.model), workers=args.workers, temperature=0.55)
    print(json.dumps(generate_alignment_queries(args.profiles, args.output, client, variants=args.variants, batch_size=args.batch_size, limit=args.limit, force=args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
