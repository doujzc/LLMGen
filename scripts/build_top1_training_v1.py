#!/usr/bin/env python3
"""Build the reviewed, balanced 1,000-row Top1 training dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import os
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


DATASET_VERSION = "top1_train_v1"
DEFAULT_SEED = 20260818
BASE_PER_CANDIDATE = 100
BASE_MULTI_TURN_PER_CANDIDATE = 50
INTENT_CHANGE_PER_PAIR = 7
EXPECTED_ROWS = 1000

EXPECTED_CANDIDATES = (
    "StockAdvice",
    "StockOther",
    "StockQuery",
    "GeneralProduct",
    "EcommerceProduct",
    "ChitChat",
    "NoAvailable",
)

CANDIDATE_SLUGS = {
    "StockAdvice": "stock_advice",
    "StockOther": "stock_other",
    "StockQuery": "stock_query",
    "GeneralProduct": "general_product",
    "EcommerceProduct": "ecommerce_product",
    "ChitChat": "chitchat",
    "NoAvailable": "no_available",
}

# This migration is intentionally narrower than the historical virtual-candidate
# mapping. PromptGen's seven-way policy changed after XiaoYi v1 was frozen, so
# only source groups whose authored semantics still agree with the current policy
# are eligible. This is structured taxonomy migration, not text classification.
APPROVED_LEGACY_CANDIDATES = {
    "no_route_stock_advice": "StockAdvice",
    "no_route_stock_research": "StockOther",
    "no_route_other_finance": "StockOther",
    "stock_market_information": "StockQuery",
    "no_route_product_information": "GeneralProduct",
    "no_route_product_other": "GeneralProduct",
    "no_route_non_retail": "GeneralProduct",
    "ecommerce_product_recommendation": "EcommerceProduct",
    "no_route_chitchat": "ChitChat",
}

# Other historical NoAvailable families include multi-product and product/stock
# statements that the current policy routes differently. These five families were
# reviewed as policy-compatible: unsupported assistant tasks, unresolved
# reference, or no actionable request.
APPROVED_NO_AVAILABLE_FAMILIES = frozenset(
    {
        "oos_no_available_assistant_suggestion_not_selected",
        "oos_no_available_common_task_followup",
        "oos_no_available_insufficient_reference",
        "oos_no_available_xiaoyi_common_tasks",
        "oos_no_request_general_statement",
    }
)

# An insufficient-reference utterance is correctly NoAvailable in isolation but
# a newly attached history could accidentally resolve it. IntentChange targets
# must therefore be independently interpretable and contain no synthetic switch
# cue; the preceding history alone creates the intent change.
APPROVED_INTENT_CHANGE_TARGET_FAMILIES = {
    "NoAvailable": frozenset(
        {
            "oos_no_available_xiaoyi_common_tasks",
            "oos_no_request_general_statement",
        }
    )
}

APPROVED_INTENT_CHANGE_TARGET_IDS = {
    "ChitChat": frozenset(
        {
            "xiaoyi_v1_oos_00027",
            "xiaoyi_v1_oos_00028",
            "xiaoyi_v1_oos_00030",
            "xiaoyi_v1_oos_00032",
            "xiaoyi_v1_oos_00034",
            "xiaoyi_v1_oos_00035",
            "xiaoyi_v1_oos_00036",
            "xiaoyi_v1_oos_00037",
            "xiaoyi_v1_oos_00041",
            "xiaoyi_v1_oos_00044",
            "xiaoyi_v1_oos_00049",
            "xiaoyi_v1_oos_00052",
            *{f"xiaoyi_v1_oos_{index:05d}" for index in range(53, 79)},
        }
    ),
    "NoAvailable": frozenset(
        {
            "xiaoyi_v1_oos_00521",
            "xiaoyi_v1_oos_00522",
            "xiaoyi_v1_oos_00523",
            "xiaoyi_v1_oos_00526",
            "xiaoyi_v1_oos_00527",
            "xiaoyi_v1_oos_00528",
            "xiaoyi_v1_oos_00534",
            "xiaoyi_v1_oos_00535",
            "xiaoyi_v1_oos_00538",
            "xiaoyi_v1_oos_00541",
            "xiaoyi_v1_oos_00545",
            *{f"xiaoyi_v1_oos_{index:05d}" for index in range(573, 599)},
        }
    ),
}

INDEPENDENT_INTENT_CHANGE_TARGETS = {
    "ChitChat": (
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
    ),
    "NoAvailable": (
        "导航到离我最近的地铁站入口",
        "把卧室空调温度调到二十四度",
        "提醒我明早八点带上身份证",
        "给张老师发消息说我会晚到十分钟",
        "播放一首轻松的纯音乐",
        "把这句话翻译成英文：会议提前到九点",
        "查一下明天杭州的天气",
        "帮我写一封简短的请假邮件",
        "把手机屏幕亮度调低一点",
        "计算三百二十五乘以四十八",
    ),
}

# These production regressions were reviewed against the current PromptGen
# system prompt. Ambiguous and policy-drifted rows (for example an ungrounded
# market fragment or a future stock prediction) are deliberately absent.
REVIEWED_PRODUCTION_LABELS = {
    "combined_20260812_002": "EcommerceProduct",
    "combined_20260812_006": "EcommerceProduct",
    "combined_20260812_008": "EcommerceProduct",
    "combined_20260812_010": "EcommerceProduct",
    "combined_20260812_012": "EcommerceProduct",
    "combined_20260812_015": "EcommerceProduct",
    "combined_20260812_016": "EcommerceProduct",
    "combined_20260812_021": "EcommerceProduct",
    "combined_20260812_026": "EcommerceProduct",
    "combined_20260812_027": "EcommerceProduct",
    "combined_20260812_029": "EcommerceProduct",
    "combined_20260812_035": "EcommerceProduct",
    "combined_20260812_037": "EcommerceProduct",
    "combined_20260812_038": "EcommerceProduct",
    "combined_20260812_040": "EcommerceProduct",
    "combined_20260812_045": "EcommerceProduct",
    "combined_20260812_059": "EcommerceProduct",
    "combined_20260812_060": "EcommerceProduct",
    "combined_20260812_071": "EcommerceProduct",
    "combined_20260812_072": "EcommerceProduct",
    "combined_20260812_073": "EcommerceProduct",
    "combined_20260812_074": "EcommerceProduct",
    "combined_20260812_075": "EcommerceProduct",
    "combined_20260812_076": "EcommerceProduct",
    "combined_20260812_077": "EcommerceProduct",
    "combined_20260812_078": "EcommerceProduct",
    "combined_20260812_080": "EcommerceProduct",
    "combined_20260812_081": "EcommerceProduct",
    "combined_20260812_082": "EcommerceProduct",
    "combined_20260812_083": "EcommerceProduct",
    "combined_20260812_084": "EcommerceProduct",
    "combined_20260812_085": "EcommerceProduct",
    "combined_20260812_086": "EcommerceProduct",
    "combined_20260812_087": "EcommerceProduct",
    "combined_20260812_088": "EcommerceProduct",
    "combined_20260812_089": "EcommerceProduct",
    "combined_20260812_011": "StockQuery",
    "combined_20260812_017": "StockQuery",
    "combined_20260812_019": "StockQuery",
    "combined_20260812_024": "StockAdvice",
    "combined_20260812_025": "StockQuery",
    "combined_20260812_048": "StockQuery",
    "combined_20260812_058": "StockQuery",
    "combined_20260812_061": "StockQuery",
    "combined_20260812_079": "StockQuery",
    "combined_20260812_112": "StockQuery",
}

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
    "GeneralProduct": (
        "我可以继续说明这个商品或消费问题。",
        "好的，我们接着看这个已经提到的商品。",
        "明白，我先帮你处理这项商品信息。",
    ),
    "EcommerceProduct": (
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse reproducible dataset builder arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data",
        default="../PromptGen/data/xiaoyi_intent_v1.jsonl",
    )
    parser.add_argument(
        "--hard-cases",
        default="../PromptGen/eval/combined_bad_cases_20260812.jsonl",
    )
    parser.add_argument(
        "--source-system-prompt",
        default="../PromptGen/intent_router/system_prompt.md",
    )
    parser.add_argument(
        "--audit-id-dir",
        default="../PromptGen/eval/cohorts",
    )
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument(
        "--output",
        default="data_top1/top1_train_v1.jsonl",
    )
    parser.add_argument("--summary-output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def reviewed_source_candidate(row: Mapping[str, Any]) -> str | None:
    """Map an explicitly reviewed XiaoYi source group to the current taxonomy."""

    family = row.get("scenario_family")
    if family in APPROVED_NO_AVAILABLE_FAMILIES:
        return "NoAvailable"
    legacy_candidate = row.get("expected_candidate_id")
    if not isinstance(legacy_candidate, str):
        return None
    return APPROVED_LEGACY_CANDIDATES.get(legacy_candidate)


def _canonical_messages(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"])
        for message in normalize_messages(row["messages"])
    )


def _is_multi_turn(row: Mapping[str, Any]) -> bool:
    return len(normalize_messages(row["messages"])) > 1


def _group_name(row: Mapping[str, Any]) -> str:
    family = row.get("scenario_family")
    if isinstance(family, str) and family:
        return family
    return "production_hardcase"


def _round_robin_sample(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    rng: random.Random,
    pool_name: str,
) -> list[dict[str, Any]]:
    """Sample across authored scenario families before repeating a family."""

    if count < 0:
        raise Top1DataError(f"{pool_name} requested a negative sample count")
    if len(rows) < count:
        raise Top1DataError(
            f"{pool_name} contains {len(rows)} rows, but {count} are required"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_name(row)].append(dict(row))
    queues: dict[str, deque[dict[str, Any]]] = {}
    for family, members in grouped.items():
        rng.shuffle(members)
        queues[family] = deque(members)

    selected: list[dict[str, Any]] = []
    active = sorted(queues)
    while len(selected) < count:
        rng.shuffle(active)
        next_active: list[str] = []
        for family in active:
            queue = queues[family]
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                next_active.append(family)
        active = next_active
        if not active and len(selected) < count:
            raise Top1DataError(f"{pool_name} was exhausted during sampling")
    return selected


def _selected_base_row(
    row: Mapping[str, Any],
    *,
    candidate: str,
    source_type: str,
) -> dict[str, Any]:
    source_id = str(row["id"])
    result: dict[str, Any] = {
        "id": f"{DATASET_VERSION}_base_{source_id}",
        "dataset_version": DATASET_VERSION,
        "source_type": source_type,
        "source_id": source_id,
        "messages": [dict(message) for message in normalize_messages(row["messages"])],
        "target_candidate_name": candidate,
    }
    for source_field, output_field in (
        ("scenario_family", "source_scenario_family"),
        ("difficulty", "source_difficulty"),
        ("bucket", "source_bucket"),
    ):
        value = row.get(source_field)
        if isinstance(value, str) and value:
            result[output_field] = value
    return result


def _reviewed_hard_cases(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    indexed = {str(row.get("id")): dict(row) for row in rows}
    missing = sorted(set(REVIEWED_PRODUCTION_LABELS) - set(indexed))
    if missing:
        raise Top1DataError(
            "reviewed production rows are missing: " + ", ".join(missing)
        )
    selected = {candidate: [] for candidate in EXPECTED_CANDIDATES}
    for row_id, candidate in REVIEWED_PRODUCTION_LABELS.items():
        row = indexed[row_id]
        expected_backend = row.get("expected_system_output")
        required_backend = (
            "RecommendProduct" if candidate == "EcommerceProduct" else "SearchStockQuotes"
        )
        if expected_backend != required_backend:
            raise Top1DataError(
                f"reviewed production row {row_id} changed backend label"
            )
        selected[candidate].append(row)
    return selected


def select_base_rows(
    source_rows: Iterable[Mapping[str, Any]],
    hard_case_rows: Iterable[Mapping[str, Any]],
    *,
    candidate_names: Sequence[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Select 100 rows per class with equal single/multi-turn supervision."""

    if tuple(candidate_names) != EXPECTED_CANDIDATES:
        raise Top1DataError(
            "candidate registry differs from the Top1 v1 dataset contract"
        )
    rng = random.Random(seed)
    eligible = {candidate: [] for candidate in candidate_names}
    for original in source_rows:
        row = dict(original)
        if row.get("split") != "train":
            continue
        candidate = reviewed_source_candidate(row)
        if candidate is not None:
            eligible[candidate].append(row)

    reviewed_hard = _reviewed_hard_cases(hard_case_rows)
    selected: list[dict[str, Any]] = []
    for candidate in candidate_names:
        hard_rows = reviewed_hard[candidate]
        hard_multi = sum(_is_multi_turn(row) for row in hard_rows)
        hard_single = len(hard_rows) - hard_multi
        need_multi = BASE_MULTI_TURN_PER_CANDIDATE - hard_multi
        need_single = (
            BASE_PER_CANDIDATE - BASE_MULTI_TURN_PER_CANDIDATE - hard_single
        )
        if need_multi < 0 or need_single < 0:
            raise Top1DataError(
                f"reviewed hard cases exceed the base quota for {candidate}"
            )
        multi_pool = [row for row in eligible[candidate] if _is_multi_turn(row)]
        single_pool = [row for row in eligible[candidate] if not _is_multi_turn(row)]
        chosen_source = [
            *_round_robin_sample(
                multi_pool,
                need_multi,
                rng=rng,
                pool_name=f"{candidate} multi-turn source pool",
            ),
            *_round_robin_sample(
                single_pool,
                need_single,
                rng=rng,
                pool_name=f"{candidate} single-turn source pool",
            ),
        ]
        selected.extend(
            _selected_base_row(
                row,
                candidate=candidate,
                source_type="promptgen_xiaoyi_reviewed",
            )
            for row in chosen_source
        )
        selected.extend(
            _selected_base_row(
                row,
                candidate=candidate,
                source_type="promptgen_production_hardcase_reviewed",
            )
            for row in hard_rows
        )

    rng.shuffle(selected)
    return selected, eligible


def directed_pair_plan(
    candidate_names: Sequence[str],
) -> dict[tuple[str, str], int]:
    """Allocate 300 examples nearly uniformly over all directed intent changes."""

    names = tuple(candidate_names)
    if names != EXPECTED_CANDIDATES:
        raise Top1DataError("unexpected candidate order for directed pair plan")
    counts = {
        (source, target): INTENT_CHANGE_PER_PAIR
        for source in names
        for target in names
        if source != target
    }
    # Seven examples over all 42 pairs produce 294 rows. Add a cyclic matching
    # of six rows, keeping source and target marginals within one example.
    for index in range(len(names) - 1):
        counts[(names[index], names[index + 1])] += 1
    return counts


def _seed_row_id(row: Mapping[str, Any]) -> str:
    source_id = row.get("source_id")
    if isinstance(source_id, str) and source_id:
        return source_id
    return str(row["id"])


def build_intent_change_rows(
    base_rows: Sequence[Mapping[str, Any]],
    eligible_source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    candidate_names: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Attach direct new requests to unrelated reviewed dialogue histories."""

    rng = random.Random(seed + 1)
    pair_counts = directed_pair_plan(candidate_names)
    outgoing = Counter()
    incoming = Counter()
    for (source, target), count in pair_counts.items():
        outgoing[source] += count
        incoming[target] += count

    base_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_source_ids: set[str] = set()
    for raw_row in base_rows:
        row = dict(raw_row)
        candidate = str(row["target_candidate_name"])
        base_by_candidate[candidate].append(row)
        if row.get("source_type") == "promptgen_xiaoyi_reviewed":
            selected_source_ids.add(str(row["source_id"]))

    source_queues: dict[str, deque[dict[str, Any]]] = {}
    for candidate in candidate_names:
        required = outgoing[candidate]
        pool = base_by_candidate[candidate]
        desired_multi = (required + 1) // 2
        chosen = [
            *_round_robin_sample(
                [row for row in pool if _is_multi_turn(row)],
                desired_multi,
                rng=rng,
                pool_name=f"{candidate} augmentation source multi-turn pool",
            ),
            *_round_robin_sample(
                [row for row in pool if not _is_multi_turn(row)],
                required - desired_multi,
                rng=rng,
                pool_name=f"{candidate} augmentation source single-turn pool",
            ),
        ]
        rng.shuffle(chosen)
        source_queues[candidate] = deque(chosen)

    target_queues: dict[str, deque[dict[str, Any]]] = {}
    for candidate in candidate_names:
        required = incoming[candidate]
        single_turn = [
            dict(row)
            for row in eligible_source_rows[candidate]
            if not _is_multi_turn(row)
            and (
                candidate not in APPROVED_INTENT_CHANGE_TARGET_FAMILIES
                or row.get("scenario_family")
                in APPROVED_INTENT_CHANGE_TARGET_FAMILIES[candidate]
            )
            and (
                candidate not in APPROVED_INTENT_CHANGE_TARGET_IDS
                or str(row.get("id"))
                in APPROVED_INTENT_CHANGE_TARGET_IDS[candidate]
            )
        ]
        single_turn.extend(
            {
                "id": f"independent_{CANDIDATE_SLUGS[candidate]}_{index:03d}",
                "scenario_family": f"independent_{CANDIDATE_SLUGS[candidate]}",
                "messages": [{"role": "user", "content": content}],
            }
            for index, content in enumerate(
                INDEPENDENT_INTENT_CHANGE_TARGETS.get(candidate, ()),
                start=1,
            )
        )
        unselected = [
            row for row in single_turn if str(row["id"]) not in selected_source_ids
        ]
        selected = [
            row for row in single_turn if str(row["id"]) in selected_source_ids
        ]
        chosen = _round_robin_sample(
            unselected,
            min(required, len(unselected)),
            rng=rng,
            pool_name=f"{candidate} unused augmentation target pool",
        )
        if len(chosen) < required:
            chosen.extend(
                _round_robin_sample(
                    selected,
                    required - len(chosen),
                    rng=rng,
                    pool_name=f"{candidate} reused augmentation target pool",
                )
            )
        rng.shuffle(chosen)
        target_queues[candidate] = deque(chosen)

    generated: list[dict[str, Any]] = []
    seen_conversations: set[tuple[tuple[str, str], ...]] = set()
    for source_candidate in candidate_names:
        for target_candidate in candidate_names:
            if source_candidate == target_candidate:
                continue
            count = pair_counts[(source_candidate, target_candidate)]
            for pair_index in range(1, count + 1):
                source_row = source_queues[source_candidate].popleft()
                target_row = target_queues[target_candidate].popleft()
                target_messages = normalize_messages(target_row["messages"])
                if len(target_messages) != 1:
                    raise Top1DataError("IntentChange target seed must be single-turn")
                messages = [
                    *[dict(message) for message in normalize_messages(source_row["messages"])],
                    {
                        "role": "assistant",
                        "content": rng.choice(ASSISTANT_BRIDGES[source_candidate]),
                    },
                    {"role": "user", "content": target_messages[0]["content"]},
                ]
                canonical = tuple(
                    (message["role"], message["content"]) for message in messages
                )
                if canonical in seen_conversations:
                    raise Top1DataError("duplicate generated IntentChange conversation")
                seen_conversations.add(canonical)
                generated.append(
                    {
                        "id": (
                            f"{DATASET_VERSION}_intent_change_"
                            f"{CANDIDATE_SLUGS[source_candidate]}_to_"
                            f"{CANDIDATE_SLUGS[target_candidate]}_{pair_index:03d}"
                        ),
                        "dataset_version": DATASET_VERSION,
                        "source_type": "intent_change_augmentation",
                        "conversation_phenomenon": "IntentChange",
                        "transition_style": "direct",
                        "source_candidate_name": source_candidate,
                        "target_candidate_name": target_candidate,
                        "source_seed_id": _seed_row_id(source_row),
                        "target_seed_id": str(target_row["id"]),
                        "messages": messages,
                    }
                )
    if any(source_queues[candidate] for candidate in candidate_names):
        raise Top1DataError("unused IntentChange source seeds remain")
    if any(target_queues[candidate] for candidate in candidate_names):
        raise Top1DataError("unused IntentChange target seeds remain")
    return generated


def _read_audit_ids(directory: Path) -> tuple[set[str], tuple[Path, ...]]:
    paths = tuple(sorted(directory.glob("xiaoyi_v1_reserved_audit*.txt")))
    if not paths:
        raise Top1DataError(f"no historical audit cohorts found in {directory}")
    identifiers: set[str] = set()
    for path in paths:
        identifiers.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return identifiers, paths


def _count(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _portable_path(path: Path) -> str:
    """Record paths relative to the repository instead of the build machine."""

    return Path(os.path.relpath(path, Path.cwd().resolve())).as_posix()


def _validate_final_rows(
    rows: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
) -> dict[str, Any]:
    profile = validate_training_rows(rows, candidate_names, source=DATASET_VERSION)
    if len(rows) != EXPECTED_ROWS:
        raise Top1DataError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    identifiers = [str(row.get("id")) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise Top1DataError("duplicate final dataset id")
    conversations = [_canonical_messages(row) for row in rows]
    if len(set(conversations)) != len(conversations):
        raise Top1DataError("duplicate final conversation")
    candidate_counts = Counter(str(row["target_candidate_name"]) for row in rows)
    if max(candidate_counts.values()) - min(candidate_counts.values()) > 1:
        raise Top1DataError("final candidate counts differ by more than one row")
    return profile


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = Path(args.source_data).expanduser().resolve()
    hard_case_path = Path(args.hard_cases).expanduser().resolve()
    source_prompt_path = Path(args.source_system_prompt).expanduser().resolve()
    registry_path = Path(args.candidate_registry).expanduser().resolve()
    audit_directory = Path(args.audit_id_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_name(output_path.stem + "_summary.json")
    )
    for required in (source_path, hard_case_path, source_prompt_path, registry_path):
        if not required.is_file():
            raise Top1DataError(f"required input does not exist: {required}")

    candidate_names = load_candidate_names(registry_path)
    source_rows = read_jsonl(source_path)
    base_rows, eligible = select_base_rows(
        source_rows,
        read_jsonl(hard_case_path),
        candidate_names=candidate_names,
        seed=args.seed,
    )
    intent_change_rows = build_intent_change_rows(
        base_rows,
        eligible,
        candidate_names=candidate_names,
        seed=args.seed,
    )
    final_rows = [*base_rows, *intent_change_rows]
    random.Random(args.seed + 2).shuffle(final_rows)
    profile = _validate_final_rows(final_rows, candidate_names)
    write_jsonl(output_path, final_rows)

    audit_ids, audit_paths = _read_audit_ids(audit_directory)
    xiaoyi_seed_ids = {
        str(row[field])
        for row in final_rows
        for field in ("source_id", "source_seed_id", "target_seed_id")
        if isinstance(row.get(field), str)
        and str(row[field]).startswith("xiaoyi_v1_")
    }
    candidate_counts = Counter(
        str(row["target_candidate_name"]) for row in final_rows
    )
    base_candidate_counts = Counter(
        str(row["target_candidate_name"]) for row in base_rows
    )
    base_difficulty_counts = Counter(
        str(row.get("source_difficulty", "production_hardcase"))
        for row in base_rows
    )
    base_scenario_families = {
        str(row["source_scenario_family"])
        for row in base_rows
        if isinstance(row.get("source_scenario_family"), str)
    }
    message_counts = Counter(len(normalize_messages(row["messages"])) for row in final_rows)
    current_utterances = {
        normalize_messages(row["messages"])[-1]["content"] for row in final_rows
    }
    reused_targets = sum(
        str(row.get("target_seed_id"))
        in {
            str(base["source_id"])
            for base in base_rows
            if base.get("source_type") == "promptgen_xiaoyi_reviewed"
        }
        for row in intent_change_rows
    )
    pair_counts = Counter(
        f'{row["source_candidate_name"]}->{row["target_candidate_name"]}'
        for row in intent_change_rows
    )
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "dataset_version": DATASET_VERSION,
            "seed": args.seed,
            "rows": len(final_rows),
            "candidate_counts": dict(
                (name, candidate_counts[name]) for name in candidate_names
            ),
            "source_type_counts": _count(final_rows, "source_type"),
            "multi_turn_rows": profile["multi_turn_rows"],
            "multi_turn_ratio": profile["multi_turn_rows"] / len(final_rows),
            "base_profile": {
                "rows": len(base_rows),
                "candidate_counts": {
                    name: base_candidate_counts[name] for name in candidate_names
                },
                "multi_turn_rows": sum(_is_multi_turn(row) for row in base_rows),
                "difficulty_counts": dict(sorted(base_difficulty_counts.items())),
                "authored_scenario_families": len(base_scenario_families),
            },
            "message_count_distribution": {
                str(count): rows for count, rows in sorted(message_counts.items())
            },
            "unique_conversations": len(final_rows),
            "unique_current_utterances": len(current_utterances),
            "intent_change": {
                "rows": len(intent_change_rows),
                "style": "direct",
                "per_directed_pair_base": INTENT_CHANGE_PER_PAIR,
                "directed_pair_counts": dict(sorted(pair_counts.items())),
                "source_candidate_counts": _count(
                    intent_change_rows, "source_candidate_name"
                ),
                "target_candidate_counts": _count(
                    intent_change_rows, "target_candidate_name"
                ),
                "targets_reusing_a_base_current_utterance": reused_targets,
            },
            "selection_contract": {
                "base_per_candidate": BASE_PER_CANDIDATE,
                "base_multi_turn_per_candidate": BASE_MULTI_TURN_PER_CANDIDATE,
                "approved_legacy_candidates": dict(
                    sorted(APPROVED_LEGACY_CANDIDATES.items())
                ),
                "approved_no_available_families": sorted(
                    APPROVED_NO_AVAILABLE_FAMILIES
                ),
                "approved_intent_change_target_families": {
                    candidate: sorted(families)
                    for candidate, families in sorted(
                        APPROVED_INTENT_CHANGE_TARGET_FAMILIES.items()
                    )
                },
                "approved_intent_change_target_ids": {
                    candidate: len(identifiers)
                    for candidate, identifiers in sorted(
                        APPROVED_INTENT_CHANGE_TARGET_IDS.items()
                    )
                },
                "independent_intent_change_target_counts": {
                    candidate: len(contents)
                    for candidate, contents in sorted(
                        INDEPENDENT_INTENT_CHANGE_TARGETS.items()
                    )
                },
                "reviewed_production_rows": len(REVIEWED_PRODUCTION_LABELS),
                "excluded_source_splits": ["dev", "test"],
            },
            "historical_audit_reuse": {
                "policy": (
                    "historical PromptGen audit cohorts are retired as unbiased "
                    "evaluation when their source rows are reused for training"
                ),
                "unique_audit_ids": len(audit_ids),
                "unique_xiaoyi_seed_ids_used": len(xiaoyi_seed_ids),
                "used_seed_ids_overlapping_audits": len(xiaoyi_seed_ids & audit_ids),
                "reviewed_hard_case_rows_now_used_for_training": len(
                    REVIEWED_PRODUCTION_LABELS
                ),
                "cohort_files": [
                    {"path": _portable_path(path), "sha256": sha256_file(path)}
                    for path in audit_paths
                ],
            },
            "sources": {
                "xiaoyi": {
                    "path": _portable_path(source_path),
                    "sha256": sha256_file(source_path),
                },
                "hard_cases": {
                    "path": _portable_path(hard_case_path),
                    "sha256": sha256_file(hard_case_path),
                },
                "semantic_policy": {
                    "path": _portable_path(source_prompt_path),
                    "sha256": sha256_file(source_prompt_path),
                    "candidate_renames": {
                        "ProductOther": "GeneralProduct",
                        "Ecommerce": "EcommerceProduct",
                    },
                    "task_clarifications": {
                        "EcommerceProduct": (
                            "ordinary goods generally available through ecommerce; "
                            "include pre-purchase model comparison, price, promotion, "
                            "performance, and suitability even without a platform cue; "
                            "exclude vehicles, property, medicine, software, and services"
                        ),
                        "StockQuery": (
                            "factual lookup only; future prediction, selection, and "
                            "investment advice are StockAdvice"
                        ),
                    },
                },
                "candidate_registry": {
                    "path": _portable_path(registry_path),
                    "sha256": sha256_file(registry_path),
                },
            },
            "output": {
                "path": _portable_path(output_path),
                "sha256": sha256_file(output_path),
            },
        },
    )
    print(f"[top1] generated {len(final_rows)} reviewed rows: {output_path}")
    print(f"[top1] dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
