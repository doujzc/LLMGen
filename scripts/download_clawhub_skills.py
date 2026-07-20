#!/usr/bin/env python3
"""Download a ranked, reproducible snapshot of public ClawHub skills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub import ClawHubClient, DownloadConfig, crawl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ClawHub skills ranked by downloads DESC, then stars DESC."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/clawhub"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--base-url", default="https://clawhub.ai")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--include-suspicious", action="store_true")
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="replace an existing frozen catalog selection",
    )
    parser.add_argument("--force", action="store_true", help="redownload existing versions")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--max-archive-mib", type=int, default=256)
    parser.add_argument("--max-unpacked-mib", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.workers < 1:
        raise SystemExit("--limit and --workers must be positive")
    client = ClawHubClient(
        args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    config = DownloadConfig(
        output_dir=args.output_dir,
        limit=args.limit,
        workers=args.workers,
        include_suspicious=args.include_suspicious,
        refresh_snapshot=args.refresh_snapshot,
        force=args.force,
        keep_archives=args.keep_archives,
        max_archive_bytes=args.max_archive_mib * 1024 * 1024,
        max_unpacked_bytes=args.max_unpacked_mib * 1024 * 1024,
    )
    print(json.dumps(crawl(client, config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
