#!/usr/bin/env python3
"""Export held-out train-query validation data for closed-set retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.router import (
    build_closed_set_evaluation_rows,
    normalize_code_rows,
    read_jsonl,
    write_jsonl,
)
from llmgen.skillret import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse retrieval_validation.jsonl into unique held-out queries and "
            "qrels over the train-skill candidate corpus."
        )
    )
    parser.add_argument("--retrieval-validation", required=True)
    parser.add_argument("--codes", required=True, help="Train candidate codes JSONL.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_path = Path(args.retrieval_validation).expanduser().resolve()
    codes_path = Path(args.codes).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    skill_to_code, num_levels = normalize_code_rows(read_jsonl(codes_path))
    queries, qrels = build_closed_set_evaluation_rows(
        read_jsonl(validation_path),
        allowed_skill_ids=set(skill_to_code),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.jsonl"
    qrels_path = output_dir / "qrels.jsonl"
    write_jsonl(queries_path, queries)
    write_jsonl(qrels_path, qrels)

    manifest = {
        "schema_version": 1,
        "protocol": "closed-set-held-out-train-queries",
        "candidate_split": "train",
        "num_levels": num_levels,
        "counts": {
            "queries": len(queries),
            "qrels": len(qrels),
            "candidate_skills": len(skill_to_code),
        },
        "sources": {
            "retrieval_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
            "codes": {"path": str(codes_path), "sha256": sha256_file(codes_path)},
        },
        "outputs": {
            "queries": {"path": str(queries_path), "sha256": sha256_file(queries_path)},
            "qrels": {"path": str(qrels_path), "sha256": sha256_file(qrels_path)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
