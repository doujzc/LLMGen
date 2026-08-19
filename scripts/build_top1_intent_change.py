#!/usr/bin/env python3
"""Build balanced IntentChange augmentation from explicitly labeled seed data."""

from __future__ import annotations

import argparse
from collections import Counter
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from llmgen.top1 import (
    Top1DataError,
    load_candidate_names,
    normalize_messages,
    read_jsonl,
    sha256_file,
    validate_training_rows,
    write_json,
    write_jsonl,
)


GENERATOR_VERSION = "intent_change_v1"
DEFAULT_SEED = 20260817
DEFAULT_PER_PAIR = 10

# Stable dataset-ID components; they are not model candidate names and must not be
# rewritten when a candidate is renamed.
CANDIDATE_SLUGS = {
    "StockAdvice": "stock_advice",
    "StockOther": "stock_other",
    "StockQuery": "stock_query",
    "ProductGeneral": "general_product",
    "ProductEcommerce": "ecommerce_product",
    "ChitChat": "chitchat",
    "NoAvailable": "no_available",
}

# This is an explicit migration between two authored taxonomies. No text is used
# to infer a label.
LEGACY_TO_TOP1 = {
    "stock_market_information": "StockQuery",
    "ecommerce_product_recommendation": "ProductEcommerce",
    "no_route_stock_advice": "StockAdvice",
    "no_route_stock_other": "StockOther",
    "no_route_stock_research": "StockOther",
    "no_route_other_finance": "StockOther",
    "no_route_product_information": "ProductGeneral",
    "no_route_product_other": "ProductGeneral",
    "no_route_non_retail": "ProductGeneral",
    "no_route_chitchat": "ChitChat",
    "no_route_multi_product": "NoAvailable",
    "no_route_no_available": "NoAvailable",
    "no_route_no_request": "NoAvailable",
}

# The source dataset's ChitChat family is entirely held out by its reserved audit
# cohorts. These independent seeds avoid copying evaluation text into training.
CHITCHAT_SEEDS = (
    "嗨，今天过得怎么样",
    "早上好，见到你很高兴",
    "没什么事，就是想随便聊聊",
    "你觉得幽默在聊天里重要吗",
    "你平时喜欢和人聊些什么",
    "我刚忙完，来和你说会儿话",
    "谢谢你一直这么耐心",
    "你会觉得聊天很有意思吗",
    "如果你有性格，会是什么样",
    "今天心情不错，想找你聊聊",
    "你好呀，来和你打个招呼",
    "你觉得朋友之间什么最重要",
    "辛苦啦，和你聊得很开心",
    "晚安，明天再聊",
    "你会怎么介绍自己",
    "我想听听你对生活的感受",
    "陪我聊两句吧",
    "你觉得自己更像助手还是朋友",
    "很高兴又见到你",
    "谢谢陪我聊了这么久",
)

ASSISTANT_BRIDGES = {
    "StockAdvice": (
        "我明白你是在考虑投资判断和后续操作。",
        "可以，我们继续讨论这项投资决策。",
        "好的，我先围绕你的投资选择来分析。",
    ),
    "StockOther": (
        "我可以继续帮你梳理这项金融问题。",
        "好的，我们继续看这项公司或资产信息。",
        "明白，我先沿着这个金融话题说明。",
    ),
    "StockQuery": (
        "我可以继续帮你查询相关公开行情。",
        "可以，我先按公开市场信息帮你梳理。",
        "好的，我们继续看相关证券行情。",
    ),
    "ProductGeneral": (
        "我可以继续说明这个商品或消费问题。",
        "好的，我们接着看这个已经提到的商品。",
        "明白，我先帮你处理这项商品信息。",
    ),
    "ProductEcommerce": (
        "我可以继续按这些条件帮你筛选商品。",
        "好的，我们接着找符合要求的商品。",
        "明白，我继续帮你比较可购买的商品。",
    ),
    "ChitChat": (
        "好呀，我们继续聊。",
        "当然，可以随便聊聊。",
        "我在，想聊什么都可以。",
    ),
    "NoAvailable": (
        "明白，我先处理你刚才提出的事情。",
        "好的，我们继续看这个请求。",
        "我知道了，先沿着刚才的话题继续。",
    ),
}

# These seeds are labeled NoAvailable specifically because they lack a referent.
# A newly attached history could resolve that referent and invalidate the label.
TARGET_EXCLUDED_SCENARIO_FAMILIES = {
    "oos_no_available_insufficient_reference",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the reproducible dataset-build command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--reserved-id-dir", required=True)
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument(
        "--output",
        default="data_top1/top1_intent_change_v1.jsonl",
    )
    parser.add_argument("--summary-output")
    parser.add_argument("--per-pair", type=int, default=DEFAULT_PER_PAIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def load_reserved_ids(directory: str | Path) -> tuple[set[str], tuple[Path, ...]]:
    """Load every frozen audit cohort ID so none can become a training seed."""

    root = Path(directory).expanduser().resolve()
    paths = tuple(sorted(root.glob("xiaoyi_v1_reserved_audit*.txt")))
    if not paths:
        raise Top1DataError(f"no reserved audit ID files found in {root}")
    reserved: set[str] = set()
    for path in paths:
        reserved.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return reserved, paths


def _chitchat_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": f"independent_chitchat_seed_{index:03d}",
            "messages": [{"role": "user", "content": content}],
        }
        for index, content in enumerate(CHITCHAT_SEEDS, start=1)
    ]


def _seed_pools(
    source_rows: Iterable[Mapping[str, Any]],
    *,
    reserved_ids: set[str],
    candidate_names: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    source_pools = {name: [] for name in candidate_names}
    target_pools = {name: [] for name in candidate_names}
    unknown_labels: Counter[str] = Counter()

    for row in source_rows:
        if row.get("split") != "train":
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise Top1DataError("source training row must have a non-empty id")
        if row_id in reserved_ids:
            continue
        legacy_label = row.get("expected_candidate_id")
        if not isinstance(legacy_label, str):
            raise Top1DataError(f"source row {row_id} has no structured candidate label")
        candidate = LEGACY_TO_TOP1.get(legacy_label)
        if candidate is None:
            unknown_labels[legacy_label] += 1
            continue
        messages = [dict(message) for message in normalize_messages(row["messages"])]
        seed = {"id": row_id, "messages": messages}
        source_pools[candidate].append(seed)
        if (
            len(messages) == 1
            and row.get("scenario_family")
            not in TARGET_EXCLUDED_SCENARIO_FAMILIES
        ):
            target_pools[candidate].append(seed)

    if unknown_labels:
        details = ", ".join(
            f"{label}={count}" for label, count in sorted(unknown_labels.items())
        )
        raise Top1DataError(f"unmapped structured source labels: {details}")

    independent_chitchat = _chitchat_rows()
    source_pools["ChitChat"].extend(independent_chitchat)
    target_pools["ChitChat"].extend(independent_chitchat)
    return source_pools, target_pools


def _sample_without_replacement(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    rng: random.Random,
    pool_name: str,
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise Top1DataError(
            f"{pool_name} has {len(rows)} seeds, but {count} are required per pair"
        )
    return rng.sample(list(rows), count)


def build_intent_change_rows(
    source_rows: Iterable[Mapping[str, Any]],
    *,
    reserved_ids: set[str],
    candidate_names: Sequence[str],
    per_pair: int = DEFAULT_PER_PAIR,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Build every directed cross-candidate transition with balanced coverage."""

    if per_pair <= 0:
        raise Top1DataError("per_pair must be positive")
    expected_candidates = tuple(CANDIDATE_SLUGS)
    if tuple(candidate_names) != expected_candidates:
        raise Top1DataError(
            "candidate registry must exactly match the IntentChange generator contract: "
            + ", ".join(expected_candidates)
        )

    source_pools, target_pools = _seed_pools(
        source_rows,
        reserved_ids=reserved_ids,
        candidate_names=candidate_names,
    )
    rng = random.Random(seed)
    generated: list[dict[str, Any]] = []
    canonical_messages: set[tuple[tuple[str, str], ...]] = set()

    for source_candidate in candidate_names:
        for target_candidate in candidate_names:
            if source_candidate == target_candidate:
                continue
            selected_sources = _sample_without_replacement(
                source_pools[source_candidate],
                count=per_pair,
                rng=rng,
                pool_name=f"source pool {source_candidate}",
            )
            selected_targets = _sample_without_replacement(
                target_pools[target_candidate],
                count=per_pair,
                rng=rng,
                pool_name=f"target pool {target_candidate}",
            )
            for index, (source_row, target_row) in enumerate(
                zip(selected_sources, selected_targets, strict=True),
                start=1,
            ):
                target_query = target_row["messages"][-1]["content"]
                bridge = rng.choice(ASSISTANT_BRIDGES[source_candidate])
                messages = [
                    *[dict(message) for message in source_row["messages"]],
                    {"role": "assistant", "content": bridge},
                    {"role": "user", "content": target_query},
                ]
                canonical = tuple(
                    (str(message["role"]), str(message["content"]))
                    for message in messages
                )
                if canonical in canonical_messages:
                    raise Top1DataError("duplicate generated IntentChange conversation")
                canonical_messages.add(canonical)
                generated.append(
                    {
                        "id": (
                            f"{GENERATOR_VERSION}_"
                            f"{CANDIDATE_SLUGS[source_candidate]}_to_"
                            f"{CANDIDATE_SLUGS[target_candidate]}_{index:03d}"
                        ),
                        "source_type": "intent_change_augmentation",
                        "augmentation_version": GENERATOR_VERSION,
                        "transition_behavior": "IntentChange",
                        "transition_style": "direct",
                        "source_candidate_name": source_candidate,
                        "target_candidate_name": target_candidate,
                        "source_seed_id": source_row["id"],
                        "target_seed_id": target_row["id"],
                        "messages": messages,
                    }
                )

    validate_training_rows(generated, candidate_names, source=GENERATOR_VERSION)
    return generated


def _count(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = Path(args.source_data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_name(output_path.stem + "_summary.json")
    )
    candidate_names = load_candidate_names(args.candidate_registry)
    reserved_ids, reserved_paths = load_reserved_ids(args.reserved_id_dir)
    rows = build_intent_change_rows(
        read_jsonl(source_path),
        reserved_ids=reserved_ids,
        candidate_names=candidate_names,
        per_pair=args.per_pair,
        seed=args.seed,
    )
    write_jsonl(output_path, rows)
    pair_counts = Counter(
        f'{row["source_candidate_name"]}->{row["target_candidate_name"]}'
        for row in rows
    )
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "generator_version": GENERATOR_VERSION,
            "seed": args.seed,
            "per_directed_pair": args.per_pair,
            "rows": len(rows),
            "source_data": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "reserved_id_files": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in reserved_paths
            ],
            "reserved_ids_excluded": len(reserved_ids),
            "independent_chitchat_seeds": len(CHITCHAT_SEEDS),
            "target_excluded_scenario_families": sorted(
                TARGET_EXCLUDED_SCENARIO_FAMILIES
            ),
            "source_candidate_counts": _count(rows, "source_candidate_name"),
            "target_candidate_counts": _count(rows, "target_candidate_name"),
            "transition_style_counts": _count(rows, "transition_style"),
            "directed_pair_counts": dict(sorted(pair_counts.items())),
            "output": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            },
        },
    )
    print(f"[top1] generated {len(rows)} IntentChange rows: {output_path}")
    print(f"[top1] generation summary: {summary_path}")


if __name__ == "__main__":
    main()
