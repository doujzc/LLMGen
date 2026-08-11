#!/usr/bin/env python3
"""Convert PromptGen conversations into direct candidate-name router data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from llmgen.direct_router import (
    candidate_registry_payload,
    conversation_query_group,
    load_candidate_registry,
    normalize_conversation_messages,
)
from llmgen.router import RouterDataError, read_jsonl, write_jsonl


# PromptGen's original 13 diagnostic labels are intentionally collapsed into
# the seven deployable names requested by this branch: two real routes and five
# virtual routes.  No hidden diagnostic label becomes a generated candidate.
SOURCE_LABEL_TO_CANDIDATE = {
    "stock_market_information": "StockQuery",
    "ecommerce_product_recommendation": "Ecommerce",
    "no_route_stock_advice": "StockAdvice",
    "no_route_stock_research": "StockOther",
    "no_route_stock_other": "StockOther",
    "no_route_other_finance": "StockOther",
    "no_route_product_information": "ProductOther",
    "no_route_product_other": "ProductOther",
    "no_route_non_retail": "ProductOther",
    "no_route_multi_product": "ProductOther",
    "no_route_chitchat": "ChitChat",
    "no_route_no_request": "NoAvailable",
    "no_route_no_available": "NoAvailable",
    # Forward-compatible aliases used by newer PromptGen prompt revisions.
    "no_route_product_context": "ProductOther",
    "no_route_product_excluded": "ProductOther",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare multi-turn PromptGen data for candidate-name Top1 SFT."
    )
    parser.add_argument("--source", required=True, help="PromptGen JSONL dataset")
    parser.add_argument("--candidate-registry", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_rows(
    rows: list[dict[str, Any]],
    *,
    legal_candidate_names: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Validate, collapse labels, and preserve the source family-level split."""

    split_aliases = {"train": "train", "dev": "validation", "test": "test"}
    prepared = {"train": [], "validation": [], "test": []}
    seen_ids: set[str] = set()
    seen_conversations: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise RouterDataError(f"source row {row_number} has no id")
        row_id = row_id.strip()
        if row_id in seen_ids:
            raise RouterDataError(f"duplicate source id: {row_id!r}")
        seen_ids.add(row_id)

        source_split = row.get("split")
        if source_split not in split_aliases:
            raise RouterDataError(
                f"row {row_id!r} has unsupported split {source_split!r}"
            )
        split = split_aliases[source_split]
        raw_messages = row.get("messages")
        messages = normalize_conversation_messages(raw_messages)
        group_id = conversation_query_group(messages)
        previous_id = seen_conversations.get(group_id)
        if previous_id is not None:
            raise RouterDataError(
                f"rows {previous_id!r} and {row_id!r} have duplicate conversations"
            )
        seen_conversations[group_id] = row_id

        family = row.get("scenario_family")
        if not isinstance(family, str) or not family.strip():
            raise RouterDataError(f"row {row_id!r} has no scenario_family")
        family = family.strip()
        previous_split = family_splits.setdefault(family, split)
        if previous_split != split:
            raise RouterDataError(
                f"scenario_family {family!r} crosses {previous_split}/{split}"
            )

        source_candidate_id = row.get("expected_candidate_id")
        try:
            target_name = SOURCE_LABEL_TO_CANDIDATE[source_candidate_id]
        except (KeyError, TypeError) as exc:
            raise RouterDataError(
                f"row {row_id!r} has unknown expected_candidate_id "
                f"{source_candidate_id!r}"
            ) from exc
        if target_name not in legal_candidate_names:
            raise RouterDataError(
                f"collapsed target {target_name!r} is absent from candidate registry"
            )

        expected_system_output = row.get("expected_system_output")
        expected_for_name = {
            "StockQuery": "SearchStockQuotes",
            "Ecommerce": "RecommendProduct",
        }.get(target_name)
        if expected_system_output != expected_for_name:
            raise RouterDataError(
                f"row {row_id!r} target/output mismatch: {target_name!r} vs "
                f"{expected_system_output!r}"
            )

        prepared[split].append(
            {
                "id": row_id,
                "split": split,
                "group_id": group_id,
                "scenario_family": family,
                "bucket": row.get("bucket"),
                "messages": list(messages),
                "target_candidate_name": target_name,
                "expected_system_output": expected_system_output,
                "source_candidate_id": source_candidate_id,
                "source_id": row.get("source_id"),
            }
        )
    return prepared


def main() -> None:
    args = parse_args()
    routes = load_candidate_registry(args.candidate_registry)
    legal_names = {route.name for route in routes}
    prepared = prepare_rows(read_jsonl(args.source), legal_candidate_names=legal_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    total_counts: Counter[str] = Counter()
    for split, rows in prepared.items():
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        counts = Counter(row["target_candidate_name"] for row in rows)
        total_counts.update(counts)
        artifacts[split] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": _sha256(path),
            "candidate_counts": dict(sorted(counts.items())),
            "multi_turn_rows": sum(len(row["messages"]) > 1 for row in rows),
        }

    registry_path = output_dir / "candidate_registry.json"
    registry_path.write_text(
        json.dumps(candidate_registry_payload(routes), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "routing_mode": "candidate_name_top1",
        # Keep the tracked artifact machine-independent; the content hash is
        # authoritative, while the local import path is intentionally omitted.
        "source": Path(args.source).name,
        "source_sha256": _sha256(args.source),
        "candidate_registry": registry_path.name,
        "candidate_registry_sha256": _sha256(registry_path),
        "candidate_names": [route.name for route in routes],
        "virtual_candidate_names": [route.name for route in routes if route.virtual],
        "rows": sum(len(rows) for rows in prepared.values()),
        "candidate_counts": dict(sorted(total_counts.items())),
        "artifacts": artifacts,
        "source_label_mapping": SOURCE_LABEL_TO_CANDIDATE,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
