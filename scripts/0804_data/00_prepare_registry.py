#!/usr/bin/env python3
"""Apply audited metadata corrections while retaining all 0804 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.teststyle_data import prepare_patched_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--output-profiles", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_patched_registry(
        args.catalog,
        args.profiles,
        args.patches,
        args.output_catalog,
        args.output_profiles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
