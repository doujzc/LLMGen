#!/usr/bin/env python3
"""Build ToolWeaver-style memorization and retrieval SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.router import (
    RouterDataError,
    build_memorization_examples,
    build_retrieval_examples,
    grouped_train_validation_split,
    load_virtual_tokens,
    normalize_code_rows,
    qrels_by_query,
    read_jsonl,
    write_jsonl,
)
from llmgen.skillret import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join SkillRet skills/queries/qrels with Stage-1 codes and create "
            "query-grouped causal-LM supervision."
        )
    )
    parser.add_argument("--catalog", required=True, help="catalog_train.jsonl")
    parser.add_argument("--queries", required=True, help="queries_train.jsonl")
    parser.add_argument("--qrels", required=True, help="qrels_train.jsonl")
    parser.add_argument("--codes", required=True, help="index/train_codes.jsonl")
    parser.add_argument(
        "--virtual-tokens",
        default=None,
        help="Optional virtual_tokens.txt; verifies every target belongs to the namespace.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help=(
            "Legacy override applied to both phases. By default memorization uses "
            "all skills and retrieval holds out 2%% of query groups."
        ),
    )
    parser.add_argument("--memorization-validation-fraction", type=float, default=None)
    parser.add_argument("--retrieval-validation-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-memorization", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    return parser.parse_args()


def _write_split(
    output_dir: Path,
    phase: str,
    rows: list[dict],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, int]:
    if not rows:
        raise RouterDataError(f"{phase} router data is empty")
    train, validation = grouped_train_validation_split(
        rows,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    write_jsonl(output_dir / f"{phase}_train.jsonl", train)
    write_jsonl(output_dir / f"{phase}_validation.jsonl", validation)
    return {
        "all_examples": len(rows),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "all_groups": len({row["group_id"] for row in rows}),
        "train_groups": len({row["group_id"] for row in train}),
        "validation_groups": len({row["group_id"] for row in validation}),
        "max_target_paths": max(
            (
                len(row["target_paths"])
                if isinstance(row.get("target_paths"), list)
                else 1
            )
            for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    if args.skip_memorization and args.skip_retrieval:
        raise RouterDataError("both router-data phases were disabled")

    catalog = read_jsonl(args.catalog)
    queries = read_jsonl(args.queries)
    qrel_rows = read_jsonl(args.qrels)
    skill_to_code, num_levels = normalize_code_rows(read_jsonl(args.codes))

    if args.virtual_tokens:
        namespace = set(load_virtual_tokens(args.virtual_tokens))
        used = {token for code in skill_to_code.values() for token in code}
        missing = sorted(used.difference(namespace))
        if missing:
            raise RouterDataError(
                "code artifact uses tokens absent from virtual_tokens.txt: "
                + ", ".join(missing[:10])
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    memorization_validation_fraction = (
        args.memorization_validation_fraction
        if args.memorization_validation_fraction is not None
        else (args.validation_fraction if args.validation_fraction is not None else 0.0)
    )
    retrieval_validation_fraction = (
        args.retrieval_validation_fraction
        if args.retrieval_validation_fraction is not None
        else (args.validation_fraction if args.validation_fraction is not None else 0.02)
    )

    if not args.skip_memorization:
        memorization = build_memorization_examples(catalog, skill_to_code)
        counts["memorization"] = _write_split(
            output_dir,
            "memorization",
            memorization,
            validation_fraction=memorization_validation_fraction,
            seed=args.seed,
        )

    if not args.skip_retrieval:
        grouped_qrels = qrels_by_query(qrel_rows)
        query_ids = {
            row.get("query_id", row.get("id"))
            for row in queries
            if row.get("query_id", row.get("id"))
        }
        orphan_qrels = sorted(set(grouped_qrels).difference(query_ids))
        if orphan_qrels:
            raise RouterDataError(
                "qrels reference queries missing from the query artifact: "
                + ", ".join(orphan_qrels[:10])
            )
        retrieval = build_retrieval_examples(
            queries,
            skill_to_code,
            grouped_qrels,
        )
        counts["retrieval"] = _write_split(
            output_dir,
            "retrieval",
            retrieval,
            validation_fraction=retrieval_validation_fraction,
            # Use a separate deterministic shuffle from memorization.
            seed=args.seed + 1,
        )

    def source_artifact(path: str | None):
        if path is None:
            return None
        resolved = Path(path).resolve()
        return {"path": str(resolved), "sha256": sha256_file(resolved)}

    index_manifest_path = Path(args.codes).resolve().parent / "manifest.json"
    index_manifest = None
    if index_manifest_path.is_file():
        index_manifest = source_artifact(str(index_manifest_path))
        payload = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        index_manifest["checkpoint_sha256"] = payload.get("checkpoint_sha256")
        index_manifest["num_levels"] = payload.get("num_levels")

    manifest = {
        "schema_version": 2,
        "num_levels": num_levels,
        "retrieval_target_format": "newline_delimited_code_paths",
        "validation_fraction": args.validation_fraction,
        "validation_fractions": {
            "memorization": memorization_validation_fraction,
            "retrieval": retrieval_validation_fraction,
        },
        "seed": args.seed,
        "sources": {
            "catalog": source_artifact(args.catalog),
            "queries": source_artifact(args.queries),
            "qrels": source_artifact(args.qrels),
            "codes": source_artifact(args.codes),
            "virtual_tokens": source_artifact(args.virtual_tokens),
            "index_manifest": index_manifest,
        },
        "counts": counts,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
