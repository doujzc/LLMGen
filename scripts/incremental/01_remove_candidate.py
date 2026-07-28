#!/usr/bin/env python3
"""Disable one candidate code in a new trie/decode-map overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.incremental import (
    load_candidate_state,
    remove_candidate,
    write_candidate_state,
)
from llmgen.skillret import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--disable-shared-path",
        action="store_true",
        help="Also remove every other skill that shares the requested code.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source, decode_path, _ = load_candidate_state(args.source_state_dir)
    updated, operation = remove_candidate(
        source,
        skill_id=args.skill_id,
        source_sha256=sha256_file(decode_path),
        disable_shared_path=args.disable_shared_path,
    )
    result = write_candidate_state(
        args.output_dir,
        updated,
        operation,
        overwrite=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
