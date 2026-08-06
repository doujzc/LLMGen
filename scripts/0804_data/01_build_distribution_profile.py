#!/usr/bin/env python3
"""Extract aggregate test style statistics without retaining test rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.teststyle_data import build_distribution_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_distribution_profile(args.heldout, args.profiles, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
