#!/usr/bin/env python3
"""Generate compact two-candidate queries with DeepSeek Flash."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llmgen.clawhub_dataset import ChatBatchClient, load_api_config
from llmgen.teststyle_data import generate_teststyle_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflows", type=Path, required=True)
    parser.add_argument("--distribution-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, default=Path("~/deepseek_api_key.txt"))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com")
    client = ChatBatchClient(
        load_api_config(args.api_config, model=args.model),
        workers=args.workers,
        temperature=0.65,
    )
    result = generate_teststyle_queries(
        args.workflows,
        args.distribution_profile,
        args.output,
        client,
        heldout_csv=args.heldout,
        batch_size=args.batch_size,
        repair_rounds=args.repair_rounds,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
