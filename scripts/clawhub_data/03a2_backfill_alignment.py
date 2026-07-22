#!/usr/bin/env python3
"""Stage 03a2: add queries only for candidates lacking accepted alignment data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_alignment import append_alignment_backfill_queries
from llmgen.clawhub_dataset import ChatBatchClient, load_api_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, default=Path("~/llm_api.txt"))
    parser.add_argument("--model", default="Qwen3.6-Plus")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--min-passed-per-skill", type=int, default=1)
    parser.add_argument("--multiskill-queries", type=Path)
    parser.add_argument("--multiskill-reviews", type=Path)
    parser.add_argument("--workflows", type=Path)
    parser.add_argument("--min-combined-per-skill", type=int)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    client = ChatBatchClient(load_api_config(args.api_config, model=args.model), workers=args.workers, temperature=0.65)
    result = append_alignment_backfill_queries(
        args.profiles,
        args.queries,
        args.reviews,
        client,
        round_index=args.round,
        min_passed_per_skill=args.min_passed_per_skill,
        multiskill_queries_path=args.multiskill_queries,
        multiskill_reviews_path=args.multiskill_reviews,
        workflows_path=args.workflows,
        min_combined_per_skill=args.min_combined_per_skill,
        variants=args.variants,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
