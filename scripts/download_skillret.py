#!/usr/bin/env python3
"""Download and validate the pinned official SkillRet snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from llmgen.skillret import (
    SKILLRET_REPO_ID,
    SKILLRET_REVISION,
    sha256_file,
    validate_raw_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/skillret"))
    parser.add_argument("--repo-id", default=SKILLRET_REPO_ID)
    parser.add_argument("--revision", default=SKILLRET_REVISION)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def verify_checksums(output_dir: Path) -> None:
    checksum_file = Path(__file__).resolve().parents[1] / "data/skillret/SHA256SUMS"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"pinned checksum manifest is missing: {checksum_file}")
    prefix = "data/skillret/"
    failures = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip()
        if relative.startswith(prefix):
            relative = relative[len(prefix) :]
        target = output_dir / relative
        if not target.is_file():
            failures.append(f"missing {target}")
            continue
        actual = sha256_file(target)
        if actual != expected:
            failures.append(f"checksum mismatch for {target}: {actual} != {expected}")
    if failures:
        raise RuntimeError("; ".join(failures))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=[
            "data/**",
            "README.md",
            "LICENSE*",
            ".gitattributes",
            "croissant-rai.json",
        ],
        force_download=args.force_download,
    )
    if not args.skip_checksums:
        verify_checksums(args.output_dir)
    result = validate_raw_dataset(args.output_dir, strict_counts=True)
    result["checksums_verified"] = not args.skip_checksums
    # Keep the checked-in provenance manifest immutable; this file records the
    # validation performed by the current local invocation.
    manifest = args.output_dir / "validation_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
