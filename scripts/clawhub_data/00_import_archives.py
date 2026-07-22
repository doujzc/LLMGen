#!/usr/bin/env python3
"""Stage 00a: import a verified local ClawHub ZIP snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_archives import import_archive_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to the manifest.json beside archives/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.archives_dir.parent / "manifest.json"
    result = import_archive_catalog(
        args.archives_dir,
        manifest,
        args.output,
        expected_count=args.expected_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
