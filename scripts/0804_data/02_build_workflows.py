#!/usr/bin/env python3
"""Build balanced two-candidate workflows from reviewed source supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.teststyle_data import build_teststyle_workflows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--distribution-profile", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument(
        "--source-routing-queries",
        "--source-0804-queries",
        dest="source_routing_queries",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-routing-reviews",
        "--source-0804-reviews",
        dest="source_routing_reviews",
        type=Path,
    )
    parser.add_argument("--source-light-queries", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-workflows-per-skill", type=int, default=45)
    parser.add_argument("--max-pair-repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    result = build_teststyle_workflows(
        args.profiles,
        args.distribution_profile,
        args.patches,
        (
            ("0804-final", args.source_routing_queries, args.source_routing_reviews),
            ("light-final", args.source_light_queries, None),
        ),
        args.output,
        heldout_csv=args.heldout,
        minimum_workflows_per_skill=args.min_workflows_per_skill,
        max_pair_repetitions=args.max_pair_repetitions,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
