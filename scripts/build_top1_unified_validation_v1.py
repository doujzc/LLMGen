#!/usr/bin/env python3
"""Build the balanced, leakage-audited Top1 development validation set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from llmgen.top1 import (
    Top1DataError,
    load_candidate_names,
    normalize_messages,
    read_jsonl,
    sha256_file,
    validate_memorization_rows,
    validate_training_rows,
    write_json,
    write_jsonl,
)


DATASET_VERSION = "top1_validation_unified_v1"
DEFAULT_SEED = 20260821
DEFAULT_SOURCE_DATA = "../PromptGen/data/xiaoyi_intent_v1.jsonl"
DEFAULT_RETAIL_VALIDATION = (
    "data_top1/top1_retail_boundary_v1_validation.jsonl"
)
DEFAULT_SHORT_VALIDATION = "data_top1/top1_short_queries_v1_validation.jsonl"
DEFAULT_STOCK_VALIDATION = (
    "data_top1/top1_stock_prediction_v1_validation.jsonl"
)
DEFAULT_TRAIN_DATA = "data_top1/top1_train_combined_v1.jsonl"
DEFAULT_OUTPUT = "data_top1/top1_validation_unified_v1.jsonl"
DEFAULT_SUMMARY = "data_top1/top1_validation_unified_v1_summary.json"

EXPECTED_CANDIDATES = (
    "StockAdvice",
    "StockOther",
    "StockQuery",
    "ProductGeneral",
    "ProductEcommerce",
    "ChitChat",
    "NoAvailable",
)
CANDIDATE_SLUGS = {
    "StockAdvice": "stock_advice",
    "StockOther": "stock_other",
    "StockQuery": "stock_query",
    "ProductGeneral": "product_general",
    "ProductEcommerce": "product_ecommerce",
    "ChitChat": "chitchat",
    "NoAvailable": "no_available",
}

ROWS_PER_CANDIDATE = 40
SINGLE_TURN_PER_CANDIDATE = 20
CONTEXTUAL_MULTI_TURN_PER_CANDIDATE = 8
CONTEXTUAL_MEDIUM_PER_CANDIDATE = 4
INTENT_CHANGE_PER_DIRECTED_PAIR = 2
INTENT_CHANGE_PER_CANDIDATE = (
    (len(EXPECTED_CANDIDATES) - 1) * INTENT_CHANGE_PER_DIRECTED_PAIR
)
EXPECTED_ROWS = ROWS_PER_CANDIDATE * len(EXPECTED_CANDIDATES)
EXPECTED_PHENOMENON_COUNTS = {
    "single_core": 108,
    "boundary_contrast": 32,
    "contextual_multiturn": 56,
    "intent_change": 84,
}
EXPECTED_DIFFICULTY_COUNTS = {
    "easy": 54,
    "medium": 82,
    "hard": 144,
}
EXPECTED_CONTEXT_REQUIREMENT_COUNTS = {
    "independent": 140,
    "supportive": 56,
    "override": 84,
}
SINGLE_EASY_PER_CANDIDATE = {
    "StockAdvice": 8,
    "StockOther": 10,
    "StockQuery": 8,
    "ProductGeneral": 4,
    "ProductEcommerce": 4,
    "ChitChat": 10,
    "NoAvailable": 10,
}

# XiaoYi v1 used an older virtual-candidate policy. Every migration here is an
# explicit source-group decision against the current LabelDesc definitions. Text
# is never classified or filtered with lexical rules.
SOURCE_CANDIDATE_MIGRATION = {
    "no_route_stock_advice": "StockAdvice",
    "no_route_stock_research": "StockOther",
    "no_route_other_finance": "StockOther",
    "no_route_stock_other": "StockQuery",
    "stock_market_information": "StockQuery",
    "no_route_product_information": "ProductGeneral",
    "no_route_product_other": "ProductGeneral",
    "no_route_non_retail": "ProductGeneral",
    "ecommerce_product_recommendation": "ProductEcommerce",
    "no_route_multi_product": "ProductEcommerce",
    "no_route_chitchat": "ChitChat",
    "no_route_no_available": "NoAvailable",
    "no_route_no_request": "NoAvailable",
}
SOURCE_MIGRATION_NOTES = {
    "no_route_multi_product": (
        "Both ordinary retail goods require fresh shopping results under the "
        "current multi-product policy."
    ),
    "no_route_stock_other": (
        "The held-out families cover securities software or brokerage service "
        "queries, which the current StockQuery definition includes."
    ),
    "no_route_no_available": (
        "The held-out families are unresolved references or general assistant "
        "tasks outside the six domain routes."
    ),
    "no_route_no_request": (
        "The held-out families end without an actionable stock, product, or "
        "chitchat request."
    ),
}
SOURCE_DIFFICULTY_MIGRATION = {
    "core": "easy",
    "core_oos": "easy",
    "edge": "medium",
    "hard": "hard",
    "hard_negative": "hard",
    "adversarial": "hard",
}

ASSISTANT_BRIDGES = {
    "StockAdvice": (
        "可以，我先按你的风险偏好梳理这项投资判断。",
        "明白，我会围绕这项证券决策说明可能的取舍。",
        "好的，我先把这个投资选择需要考虑的因素列清楚。",
    ),
    "StockOther": (
        "可以，我先沿着这项公司或金融研究继续说明。",
        "明白，我会围绕这项非行情金融问题补充信息。",
        "好的，我先把这项公司研究或资产问题梳理一下。",
    ),
    "StockQuery": (
        "可以，我先按公开证券信息继续查询。",
        "明白，我会围绕这项市场行情整理可查事实。",
        "好的，我先继续核对相关证券市场数据。",
    ),
    "ProductGeneral": (
        "可以，我先继续处理这个消费对象或已有商品的问题。",
        "明白，我会沿着当前的商品相关问题继续说明。",
        "好的，我先把这个非购物检索问题梳理清楚。",
    ),
    "ProductEcommerce": (
        "可以，我先按这些条件继续筛选可购买的商品。",
        "明白，我会继续比较符合需求的普通零售商品。",
        "好的，我先整理适合这次购买需求的候选。",
    ),
    "ChitChat": (
        "当然，我们可以接着随便聊聊。",
        "好呀，我在听，你可以继续说。",
        "可以，轻松聊一会儿也很好。",
    ),
    "NoAvailable": (
        "明白了。",
        "好的，我知道了。",
        "可以，有需要再告诉我。",
    ),
}


@dataclass(frozen=True)
class AuthoredRequest:
    """One manually reviewed context-independent request."""

    record_id: str
    content: str
    scenario_family: str
    surface_family: str


AUTHORED_NO_AVAILABLE_SINGLE = (
    AuthoredRequest(
        "authored_no_available_single_001",
        "查一下周六青岛的天气，告诉我是否会下雨。",
        "general_weather_query",
        "direct_information_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_002",
        "提醒我今晚九点半给家里的绿植浇水。",
        "general_reminder",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_single_003",
        "把书房的灯光亮度调到百分之四十。",
        "smart_home_control",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_single_004",
        "导航到离上海图书馆最近的地铁站出口。",
        "navigation_request",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_single_005",
        "把“报告已经发送”翻译成法语。",
        "translation_request",
        "direct_transformation_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_006",
        "计算一下七百二十八除以十六。",
        "calculation_request",
        "direct_information_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_007",
        "帮我写一封推迟会议的简短邮件。",
        "writing_request",
        "direct_creation_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_008",
        "播放一期关于古建筑的中文播客。",
        "media_playback",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_single_009",
        "设置一个明早六点四十五分的闹钟。",
        "alarm_request",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_single_010",
        "给王老师发消息，说材料我明天下午补交。",
        "messaging_request",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_single_011",
        "识别这张照片里的植物种类。",
        "image_understanding",
        "direct_information_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_012",
        "在日历里添加下周三下午的牙科复诊。",
        "calendar_management",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_single_013",
        "查一下明早从南京到苏州的高铁班次。",
        "transport_schedule_query",
        "direct_information_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_014",
        "把这段会议记录整理成三个行动项。",
        "text_summarization",
        "direct_transformation_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_015",
        "生成一张雪山脚下露营的插画。",
        "image_generation",
        "direct_creation_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_016",
        "讲一下勾股定理为什么成立。",
        "general_knowledge_explanation",
        "direct_information_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_017",
        "查询二号线今天末班车几点发车。",
        "public_transit_query",
        "direct_information_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_018",
        "打开蓝牙并连接客厅的音箱。",
        "device_connectivity",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_single_019",
        "把这段录音转成带时间戳的文字。",
        "audio_transcription",
        "direct_transformation_request",
    ),
    AuthoredRequest(
        "authored_no_available_single_020",
        "将三点五英里换算成公里。",
        "unit_conversion",
        "direct_information_request",
    ),
)

AUTHORED_NO_AVAILABLE_TARGETS = (
    AuthoredRequest(
        "authored_no_available_target_001",
        "把明天下午四点的项目会加入日历。",
        "calendar_management",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_target_002",
        "将手机屏幕亮度降低到百分之三十。",
        "device_display_control",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_target_003",
        "把“入口在大楼北侧”翻译成日语。",
        "translation_request",
        "direct_transformation_request",
    ),
    AuthoredRequest(
        "authored_no_available_target_004",
        "导航到杭州植物园的北门。",
        "navigation_request",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_target_005",
        "查一下厦门后天的最高气温。",
        "general_weather_query",
        "direct_information_request",
    ),
    AuthoredRequest(
        "authored_no_available_target_006",
        "写一段确认收到合同的正式回复。",
        "writing_request",
        "direct_creation_request",
    ),
    AuthoredRequest(
        "authored_no_available_target_007",
        "算一下四十五乘以一百二十六。",
        "calculation_request",
        "direct_information_request",
    ),
    AuthoredRequest(
        "authored_no_available_target_008",
        "播放一段二十分钟的白噪音。",
        "media_playback",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_target_009",
        "提醒我周五下班前提交周报。",
        "general_reminder",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_target_010",
        "画一张雨夜城市街道的水彩插画。",
        "image_generation",
        "direct_creation_request",
    ),
    AuthoredRequest(
        "authored_no_available_target_011",
        "给小陈发消息，说我十分钟后到。",
        "messaging_request",
        "direct_device_action",
    ),
    AuthoredRequest(
        "authored_no_available_target_012",
        "把二十三摄氏度换算成华氏度。",
        "unit_conversion",
        "direct_information_request",
    ),
)

AUTHORED_CHITCHAT_TARGETS = (
    AuthoredRequest(
        "authored_chitchat_target_001",
        "今天心情很好，想和你聊聊天。",
        "casual_mood_chat",
        "direct_smalltalk",
    ),
    AuthoredRequest(
        "authored_chitchat_target_002",
        "你觉得雨天适合发呆吗？",
        "casual_opinion_chat",
        "direct_smalltalk",
    ),
    AuthoredRequest(
        "authored_chitchat_target_003",
        "晚上好，陪我说会儿话吧。",
        "casual_companionship",
        "direct_smalltalk",
    ),
    AuthoredRequest(
        "authored_chitchat_target_004",
        "你平时更喜欢安静还是热闹？",
        "casual_preference_chat",
        "direct_smalltalk",
    ),
    AuthoredRequest(
        "authored_chitchat_target_005",
        "认识你很高兴。",
        "casual_greeting",
        "direct_smalltalk",
    ),
    AuthoredRequest(
        "authored_chitchat_target_006",
        "我刚忙完，想随便聊两句。",
        "casual_companionship",
        "direct_smalltalk",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deterministic unified-validation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", default=DEFAULT_SOURCE_DATA)
    parser.add_argument(
        "--retail-validation",
        default=DEFAULT_RETAIL_VALIDATION,
    )
    parser.add_argument("--short-validation", default=DEFAULT_SHORT_VALIDATION)
    parser.add_argument("--stock-validation", default=DEFAULT_STOCK_VALIDATION)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument(
        "--taxonomy-data",
        default="data_top1/top1_labeldesc_paper_v1.jsonl",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def source_candidate(row: Mapping[str, Any]) -> str:
    """Return the explicitly reviewed current candidate for one XiaoYi row."""

    source_candidate_id = row.get("expected_candidate_id")
    if not isinstance(source_candidate_id, str):
        raise Top1DataError("XiaoYi row has no expected_candidate_id")
    try:
        return SOURCE_CANDIDATE_MIGRATION[source_candidate_id]
    except KeyError as exc:
        raise Top1DataError(
            f"unreviewed XiaoYi candidate group: {source_candidate_id!r}"
        ) from exc


def _canonical_messages(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"])
        for message in normalize_messages(row.get("messages"))
    )


def _normalized_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _normalized_conversation(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (role, _normalized_surface(content))
        for role, content in _canonical_messages(row)
    )


def _current_utterance(row: Mapping[str, Any]) -> str:
    return normalize_messages(row.get("messages"))[-1]["content"]


def _is_multi_turn(row: Mapping[str, Any]) -> bool:
    return len(normalize_messages(row.get("messages"))) > 1


def _take_round_robin(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    used_ids: set[str],
    used_current_surfaces: set[str] | None = None,
    rng: random.Random,
    pool_name: str,
) -> list[dict[str, Any]]:
    """Select unused rows across source families before repeating a family."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_row in rows:
        row = dict(raw_row)
        row_id = str(row.get("id"))
        family = row.get("scenario_family")
        if row_id in used_ids:
            continue
        current_surface = _normalized_surface(_current_utterance(row))
        if (
            used_current_surfaces is not None
            and current_surface in used_current_surfaces
        ):
            continue
        if not isinstance(family, str) or not family:
            raise Top1DataError(f"{pool_name}: source row has no scenario_family")
        grouped[family].append(row)
    for members in grouped.values():
        members.sort(key=lambda row: str(row["id"]))
        rng.shuffle(members)

    selected: list[dict[str, Any]] = []
    active = sorted(grouped)
    while len(selected) < count and active:
        rng.shuffle(active)
        remaining: list[str] = []
        for family in active:
            members = grouped[family]
            while members and len(selected) < count:
                row = members.pop()
                current_surface = _normalized_surface(_current_utterance(row))
                if (
                    used_current_surfaces is not None
                    and current_surface in used_current_surfaces
                ):
                    continue
                selected.append(row)
                used_ids.add(str(row["id"]))
                if used_current_surfaces is not None:
                    used_current_surfaces.add(current_surface)
                break
            if members:
                remaining.append(family)
        active = remaining
    if len(selected) != count:
        raise Top1DataError(
            f"{pool_name}: selected {len(selected)} rows, expected {count}"
        )
    return selected


def _source_type(split: str) -> str:
    if split == "dev":
        return "promptgen_xiaoyi_dev_holdout"
    if split == "test":
        return "promptgen_xiaoyi_dev2_holdout"
    raise Top1DataError(f"unsupported XiaoYi development split: {split!r}")


def _convert_promptgen_row(
    row: Mapping[str, Any],
    *,
    phenomenon: str,
    context_requirement: str,
    difficulty: str,
) -> dict[str, Any]:
    source_id = str(row["id"])
    source_difficulty = str(row.get("difficulty"))
    if source_difficulty not in SOURCE_DIFFICULTY_MIGRATION:
        raise Top1DataError(
            f"{source_id}: unsupported source difficulty {source_difficulty!r}"
        )
    split = str(row["split"])
    source_candidate_id = str(row["expected_candidate_id"])
    return {
        "id": f"{DATASET_VERSION}_source_{source_id}",
        "dataset_version": DATASET_VERSION,
        "split": "validation",
        "source_type": _source_type(split),
        "source_record_id": source_id,
        "source_split": "dev2" if split == "test" else split,
        "source_candidate_id": source_candidate_id,
        "source_scenario_family": str(row["scenario_family"]),
        "scenario_family": str(row["scenario_family"]),
        "surface_family": str(row.get("variation_axis", "source_authored")),
        "conversation_phenomenon": phenomenon,
        # XiaoYi difficulty measured the former three-way backend task. The
        # unified difficulty is a separate, balanced seven-way eval stratum;
        # retain the original value alongside it for provenance.
        "difficulty": difficulty,
        "source_difficulty": source_difficulty,
        "context_requirement": context_requirement,
        "review_status": "current_taxonomy_source_group_reviewed",
        "messages": [
            dict(message) for message in normalize_messages(row.get("messages"))
        ],
        "target_candidate_name": source_candidate(row),
    }


def _select_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_field: str,
    selected_pair_ids: Iterable[str],
    source_name: str,
) -> list[dict[str, Any]]:
    selected = set(selected_pair_ids)
    result = [dict(row) for row in rows if str(row.get(pair_field)) in selected]
    grouped = Counter(str(row.get(pair_field)) for row in result)
    if set(grouped) != selected or set(grouped.values()) != {2}:
        raise Top1DataError(f"{source_name}: every selected contrast must have two rows")
    return result


def select_specialty_rows(
    retail_rows: Sequence[Mapping[str, Any]],
    short_rows: Sequence[Mapping[str, Any]],
    stock_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select 32 balanced specialty rows without letting them dominate loss."""

    retail_pair_ids: list[str] = []
    retail_families: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in retail_rows:
        key = (str(row.get("boundary_axis")), str(row.get("boundary_family")))
        retail_families[key].add(str(row.get("boundary_pair_id")))
    axes = sorted({axis for axis, _ in retail_families})
    for axis in axes:
        families = sorted(
            family for candidate_axis, family in retail_families if candidate_axis == axis
        )
        if len(families) != 2:
            raise Top1DataError(
                f"retail validation axis {axis!r} must contain two held-out families"
            )
        for family in families:
            pair_ids = sorted(retail_families[(axis, family)])
            if not pair_ids:
                raise Top1DataError("retail validation family contains no contrast")
            retail_pair_ids.append(pair_ids[0])
    if len(retail_pair_ids) != 8:
        raise Top1DataError("retail specialty slice must contain eight pairs")

    short_pair_ids = sorted(
        {str(row.get("contrast_pair_id")) for row in short_rows}
    )[:4]
    stock_pair_ids = sorted(
        {str(row.get("contrast_pair_id")) for row in stock_rows}
    )[:4]
    selected = [
        *_select_pair_rows(
            retail_rows,
            pair_field="boundary_pair_id",
            selected_pair_ids=retail_pair_ids,
            source_name="retail validation",
        ),
        *_select_pair_rows(
            short_rows,
            pair_field="contrast_pair_id",
            selected_pair_ids=short_pair_ids,
            source_name="short-query validation",
        ),
        *_select_pair_rows(
            stock_rows,
            pair_field="contrast_pair_id",
            selected_pair_ids=stock_pair_ids,
            source_name="stock-prediction validation",
        ),
    ]
    counts = Counter(str(row["target_candidate_name"]) for row in selected)
    expected = Counter(
        {
            "ProductGeneral": 12,
            "ProductEcommerce": 12,
            "StockAdvice": 4,
            "StockQuery": 4,
        }
    )
    if counts != expected:
        raise Top1DataError(
            f"unexpected specialty candidate counts: {dict(sorted(counts.items()))}"
        )
    return selected


def _specialty_family(row: Mapping[str, Any]) -> str:
    for field in ("boundary_family", "contrast_family"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise Top1DataError(f"specialty row {row.get('id')!r} has no family")


def _specialty_surface(row: Mapping[str, Any]) -> str:
    for field in ("boundary_axis", "short_query_intent", "forecast_type"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise Top1DataError(f"specialty row {row.get('id')!r} has no surface axis")


def _specialty_pair(row: Mapping[str, Any]) -> str:
    for field in ("boundary_pair_id", "contrast_pair_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise Top1DataError(f"specialty row {row.get('id')!r} has no pair id")


def _convert_specialty_row(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = str(row["target_candidate_name"])
    confusion = {
        "ProductGeneral": "ProductEcommerce",
        "ProductEcommerce": "ProductGeneral",
        "StockAdvice": "StockQuery",
        "StockQuery": "StockAdvice",
    }[candidate]
    source_id = str(row["id"])
    return {
        "id": f"{DATASET_VERSION}_specialty_{source_id}",
        "dataset_version": DATASET_VERSION,
        "split": "validation",
        "source_type": "reviewed_specialty_validation_slice",
        "source_record_id": source_id,
        "source_dataset_version": str(row["dataset_version"]),
        "source_scenario_family": _specialty_family(row),
        "scenario_family": (
            f"{row['dataset_version']}::{_specialty_family(row)}"
        ),
        "surface_family": _specialty_surface(row),
        "conversation_phenomenon": "boundary_contrast",
        "difficulty": "hard",
        "context_requirement": "independent",
        "primary_confusion_candidate": confusion,
        "contrast_set_id": _specialty_pair(row),
        "review_status": "reviewed_specialty_source",
        "messages": [
            dict(message) for message in normalize_messages(row.get("messages"))
        ],
        "target_candidate_name": candidate,
    }


def _authored_row(
    request: AuthoredRequest,
    candidate: str,
    *,
    difficulty: str,
) -> dict[str, Any]:
    return {
        "id": f"{DATASET_VERSION}_{request.record_id}",
        "dataset_version": DATASET_VERSION,
        "split": "validation",
        "source_type": "reviewed_authored_validation",
        "source_record_id": request.record_id,
        "scenario_family": f"authored::{request.scenario_family}",
        "surface_family": request.surface_family,
        "conversation_phenomenon": "single_core",
        "difficulty": difficulty,
        "context_requirement": "independent",
        "review_status": "manually_reviewed_against_current_labeldesc",
        "messages": [{"role": "user", "content": request.content}],
        "target_candidate_name": candidate,
    }


def _request_seed_from_source(row: Mapping[str, Any]) -> dict[str, str]:
    messages = normalize_messages(row.get("messages"))
    if len(messages) != 1:
        raise Top1DataError("IntentChange target source must be single-turn")
    split = str(row["split"])
    return {
        "record_id": str(row["id"]),
        "source_type": _source_type(split),
        "source_split": "dev2" if split == "test" else split,
        "scenario_family": str(row["scenario_family"]),
        "surface_family": str(row.get("variation_axis", "source_authored")),
        "content": messages[0]["content"],
    }


def _request_seed_from_authored(request: AuthoredRequest) -> dict[str, str]:
    return {
        "record_id": request.record_id,
        "source_type": "reviewed_authored_validation",
        "source_split": "validation_authored",
        "scenario_family": f"authored::{request.scenario_family}",
        "surface_family": request.surface_family,
        "content": request.content,
    }


def _build_intent_change_rows(
    source_histories: Mapping[str, Sequence[Mapping[str, Any]]],
    target_requests: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    source_queues: dict[str, deque[dict[str, Any]]] = {}
    target_queues: dict[str, deque[dict[str, str]]] = {}
    for candidate in EXPECTED_CANDIDATES:
        sources = [dict(row) for row in source_histories[candidate]]
        targets = [dict(row) for row in target_requests[candidate]]
        if len(sources) != INTENT_CHANGE_PER_CANDIDATE:
            raise Top1DataError(f"{candidate}: wrong IntentChange source quota")
        if len(targets) != INTENT_CHANGE_PER_CANDIDATE:
            raise Top1DataError(f"{candidate}: wrong IntentChange target quota")
        rng.shuffle(sources)
        rng.shuffle(targets)
        source_queues[candidate] = deque(sources)
        target_queues[candidate] = deque(targets)

    rows: list[dict[str, Any]] = []
    for source_candidate in EXPECTED_CANDIDATES:
        for target_candidate in EXPECTED_CANDIDATES:
            if source_candidate == target_candidate:
                continue
            for pair_index in range(1, INTENT_CHANGE_PER_DIRECTED_PAIR + 1):
                source = source_queues[source_candidate].popleft()
                target = target_queues[target_candidate].popleft()
                source_messages = [
                    dict(message)
                    for message in normalize_messages(source.get("messages"))
                ]
                bridge_index = (
                    len(rows) + pair_index + DEFAULT_SEED
                ) % len(ASSISTANT_BRIDGES[source_candidate])
                messages = [
                    *source_messages,
                    {
                        "role": "assistant",
                        "content": ASSISTANT_BRIDGES[source_candidate][bridge_index],
                    },
                    {"role": "user", "content": str(target["content"])},
                ]
                source_slug = CANDIDATE_SLUGS[source_candidate]
                target_slug = CANDIDATE_SLUGS[target_candidate]
                rows.append(
                    {
                        "id": (
                            f"{DATASET_VERSION}_intent_change_{source_slug}_to_"
                            f"{target_slug}_{pair_index:02d}"
                        ),
                        "dataset_version": DATASET_VERSION,
                        "split": "validation",
                        "source_type": "controlled_heldout_intent_change",
                        "source_record_id": str(source["id"]),
                        "target_record_id": str(target["record_id"]),
                        "source_split": (
                            "dev2" if source.get("split") == "test" else source["split"]
                        ),
                        "target_split": str(target["source_split"]),
                        "source_scenario_family": str(source["scenario_family"]),
                        "target_scenario_family": str(target["scenario_family"]),
                        "scenario_family": (
                            f"intent_change::{source_slug}_to_{target_slug}"
                        ),
                        "surface_family": "direct_new_request_override",
                        "conversation_phenomenon": "intent_change",
                        "difficulty": "hard",
                        "context_requirement": "override",
                        "transition_style": "direct",
                        "source_candidate_name": source_candidate,
                        "target_candidate_name": target_candidate,
                        "primary_confusion_candidate": source_candidate,
                        "review_status": "structured_sources_and_direct_target_reviewed",
                        "messages": messages,
                    }
                )
    if any(source_queues[candidate] for candidate in EXPECTED_CANDIDATES):
        raise Top1DataError("unused IntentChange source histories remain")
    if any(target_queues[candidate] for candidate in EXPECTED_CANDIDATES):
        raise Top1DataError("unused IntentChange target requests remain")
    return rows


def build_unified_validation_rows(
    source_rows: Sequence[Mapping[str, Any]],
    retail_rows: Sequence[Mapping[str, Any]],
    short_rows: Sequence[Mapping[str, Any]],
    stock_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_names: Sequence[str],
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Build the immutable 280-row validation content from reviewed sources."""

    if tuple(candidate_names) != EXPECTED_CANDIDATES:
        raise Top1DataError("candidate registry differs from unified validation v1")
    rng = random.Random(seed)
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {
        candidate: {"single": [], "multi": []}
        for candidate in candidate_names
    }
    for raw_row in source_rows:
        row = dict(raw_row)
        if row.get("split") not in {"dev", "test"}:
            continue
        candidate = source_candidate(row)
        turn_key = "multi" if _is_multi_turn(row) else "single"
        pools[candidate][turn_key].append(row)

    specialty = [_convert_specialty_row(row) for row in select_specialty_rows(
        retail_rows,
        short_rows,
        stock_rows,
    )]
    specialty_counts = Counter(
        str(row["target_candidate_name"]) for row in specialty
    )
    output_rows: list[dict[str, Any]] = list(specialty)
    output_rows.extend(
        _authored_row(
            request,
            "NoAvailable",
            difficulty=(
                "easy"
                if index <= SINGLE_EASY_PER_CANDIDATE["NoAvailable"]
                else "medium"
            ),
        )
        for index, request in enumerate(AUTHORED_NO_AVAILABLE_SINGLE, start=1)
    )

    used_source_ids: set[str] = set()
    # Current utterances are the independently scored units. Reserve authored
    # IntentChange targets before sampling so no source row can duplicate them.
    used_final_currents = {
        _normalized_surface(_current_utterance(row)) for row in output_rows
    }
    used_final_currents.update(
        _normalized_surface(request.content)
        for request in (
            *AUTHORED_NO_AVAILABLE_TARGETS,
            *AUTHORED_CHITCHAT_TARGETS,
        )
    )
    source_histories: dict[str, list[dict[str, Any]]] = {}
    target_requests: dict[str, list[dict[str, str]]] = {}
    for candidate in candidate_names:
        authored_single_count = (
            len(AUTHORED_NO_AVAILABLE_SINGLE)
            if candidate == "NoAvailable"
            else 0
        )
        promptgen_single_count = (
            SINGLE_TURN_PER_CANDIDATE
            - specialty_counts[candidate]
            - authored_single_count
        )
        selected_single = _take_round_robin(
            pools[candidate]["single"],
            promptgen_single_count,
            used_ids=used_source_ids,
            used_current_surfaces=used_final_currents,
            rng=rng,
            pool_name=f"{candidate} validation single-turn",
        )
        promptgen_easy_count = (
            SINGLE_EASY_PER_CANDIDATE[candidate]
            if candidate != "NoAvailable"
            else 0
        )
        output_rows.extend(
            _convert_promptgen_row(
                row,
                phenomenon="single_core",
                context_requirement="independent",
                difficulty=("easy" if index <= promptgen_easy_count else "medium"),
            )
            for index, row in enumerate(selected_single, start=1)
        )

        selected_contextual = _take_round_robin(
            pools[candidate]["multi"],
            CONTEXTUAL_MULTI_TURN_PER_CANDIDATE,
            used_ids=used_source_ids,
            used_current_surfaces=used_final_currents,
            rng=rng,
            pool_name=f"{candidate} contextual multi-turn",
        )
        output_rows.extend(
            _convert_promptgen_row(
                row,
                phenomenon="contextual_multiturn",
                context_requirement="supportive",
                difficulty=(
                    "medium"
                    if index <= CONTEXTUAL_MEDIUM_PER_CANDIDATE
                    else "hard"
                ),
            )
            for index, row in enumerate(selected_contextual, start=1)
        )

        source_histories[candidate] = _take_round_robin(
            pools[candidate]["multi"],
            INTENT_CHANGE_PER_CANDIDATE,
            used_ids=used_source_ids,
            rng=rng,
            pool_name=f"{candidate} IntentChange source histories",
        )

        authored_targets: tuple[AuthoredRequest, ...] = ()
        source_target_count = INTENT_CHANGE_PER_CANDIDATE
        if candidate == "NoAvailable":
            authored_targets = AUTHORED_NO_AVAILABLE_TARGETS
            source_target_count = 0
        elif candidate == "ChitChat":
            authored_targets = AUTHORED_CHITCHAT_TARGETS
            source_target_count -= len(authored_targets)
        selected_targets = _take_round_robin(
            pools[candidate]["single"],
            source_target_count,
            used_ids=used_source_ids,
            used_current_surfaces=used_final_currents,
            rng=rng,
            pool_name=f"{candidate} IntentChange targets",
        )
        target_requests[candidate] = [
            *(_request_seed_from_source(row) for row in selected_targets),
            *(_request_seed_from_authored(request) for request in authored_targets),
        ]

    output_rows.extend(
        _build_intent_change_rows(
            source_histories,
            target_requests,
            rng=rng,
        )
    )
    rng.shuffle(output_rows)
    return output_rows


def _family_values(row: Mapping[str, Any]) -> set[str]:
    return {
        str(row[field])
        for field in (
            "scenario_family",
            "source_scenario_family",
            "target_scenario_family",
            "boundary_family",
            "contrast_family",
        )
        if isinstance(row.get(field), str) and row.get(field)
    }


def validate_unified_validation_rows(
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
) -> dict[str, Any]:
    """Enforce balance, structure, and normalized train/validation isolation."""

    profile = validate_training_rows(
        rows,
        candidate_names,
        source=DATASET_VERSION,
    )
    if len(rows) != EXPECTED_ROWS:
        raise Top1DataError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    ids = [str(row.get("id")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise Top1DataError("unified validation ids must be non-empty and unique")
    if any(row.get("dataset_version") != DATASET_VERSION for row in rows):
        raise Top1DataError("unified validation dataset_version mismatch")
    if any(row.get("split") != "validation" for row in rows):
        raise Top1DataError("unified validation rows must use split='validation'")

    candidate_counts = Counter(
        str(row["target_candidate_name"]) for row in rows
    )
    expected_candidate_counts = Counter(
        {candidate: ROWS_PER_CANDIDATE for candidate in candidate_names}
    )
    if candidate_counts != expected_candidate_counts:
        raise Top1DataError(
            f"unbalanced unified candidates: {dict(sorted(candidate_counts.items()))}"
        )
    turn_counts: dict[str, Counter[str]] = {
        candidate: Counter() for candidate in candidate_names
    }
    for row in rows:
        messages = normalize_messages(row.get("messages"))
        roles = [message["role"] for message in messages]
        expected_roles = [
            "user" if index % 2 == 0 else "assistant"
            for index in range(len(messages))
        ]
        if roles != expected_roles:
            raise Top1DataError(f"{row.get('id')}: messages must alternate roles")
        key = "multi" if len(messages) > 1 else "single"
        turn_counts[str(row["target_candidate_name"])][key] += 1
    for candidate in candidate_names:
        if turn_counts[candidate] != Counter(
            {
                "single": SINGLE_TURN_PER_CANDIDATE,
                "multi": ROWS_PER_CANDIDATE - SINGLE_TURN_PER_CANDIDATE,
            }
        ):
            raise Top1DataError(
                f"{candidate}: unexpected single/multi distribution "
                f"{dict(turn_counts[candidate])}"
            )

    intent_change_rows = [
        row
        for row in rows
        if row.get("conversation_phenomenon") == "intent_change"
    ]
    expected_intent_rows = (
        len(candidate_names)
        * (len(candidate_names) - 1)
        * INTENT_CHANGE_PER_DIRECTED_PAIR
    )
    if len(intent_change_rows) != expected_intent_rows:
        raise Top1DataError("unexpected IntentChange row count")
    transition_counts = Counter(
        (
            str(row.get("source_candidate_name")),
            str(row.get("target_candidate_name")),
        )
        for row in intent_change_rows
    )
    expected_transitions = Counter(
        {
            (source, target): INTENT_CHANGE_PER_DIRECTED_PAIR
            for source in candidate_names
            for target in candidate_names
            if source != target
        }
    )
    if transition_counts != expected_transitions:
        raise Top1DataError("IntentChange directed-pair coverage mismatch")
    for field, expected_counts in (
        ("conversation_phenomenon", EXPECTED_PHENOMENON_COUNTS),
        ("difficulty", EXPECTED_DIFFICULTY_COUNTS),
        ("context_requirement", EXPECTED_CONTEXT_REQUIREMENT_COUNTS),
    ):
        actual_counts = Counter(str(row.get(field)) for row in rows)
        if actual_counts != Counter(expected_counts):
            raise Top1DataError(
                f"unexpected {field} distribution: "
                f"{dict(sorted(actual_counts.items()))}"
            )

    conversations = [_normalized_conversation(row) for row in rows]
    currents = [
        _normalized_surface(_current_utterance(row)) for row in rows
    ]
    if len(set(conversations)) != len(conversations):
        raise Top1DataError("duplicate normalized conversation in validation")
    if len(set(currents)) != len(currents):
        raise Top1DataError("duplicate normalized current utterance in validation")

    train_ids = {str(row.get("id")) for row in train_rows}
    if set(ids) & train_ids:
        raise Top1DataError("validation id overlaps training")
    train_conversations = {
        _normalized_conversation(row) for row in train_rows
    }
    train_currents = {
        _normalized_surface(_current_utterance(row)) for row in train_rows
    }
    if set(conversations) & train_conversations:
        raise Top1DataError("normalized validation conversation overlaps training")
    if set(currents) & train_currents:
        raise Top1DataError("normalized validation current utterance overlaps training")

    train_source_ids = {
        str(row[field])
        for row in train_rows
        for field in (
            "id",
            "source_id",
            "source_seed_id",
            "target_seed_id",
            "source_record_id",
            "target_record_id",
        )
        if isinstance(row.get(field), str) and row.get(field)
    }
    validation_source_ids = {
        str(row[field])
        for row in rows
        for field in ("source_record_id", "target_record_id")
        if isinstance(row.get(field), str) and row.get(field)
    }
    if validation_source_ids & train_source_ids:
        raise Top1DataError("validation provenance source id overlaps training")

    train_families = {
        family for row in train_rows for family in _family_values(row)
    }
    validation_source_families = {
        str(row[field])
        for row in rows
        for field in ("source_scenario_family", "target_scenario_family")
        if isinstance(row.get(field), str) and row.get(field)
    }
    if validation_source_families & train_families:
        raise Top1DataError("validation source family overlaps training")

    return {
        **profile,
        "unique_ids": len(set(ids)),
        "unique_conversations": len(set(conversations)),
        "unique_current_utterances": len(set(currents)),
        "single_turn_rows": sum(not _is_multi_turn(row) for row in rows),
        "multi_turn_rows": sum(_is_multi_turn(row) for row in rows),
        "intent_change_rows": len(intent_change_rows),
        "intent_change_directed_pairs": len(transition_counts),
        "train_id_overlap": 0,
        "train_conversation_overlap": 0,
        "train_current_utterance_overlap": 0,
        "train_source_id_overlap": 0,
        "train_source_family_overlap": 0,
    }


def _count(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def _portable_path(path: Path) -> str:
    return Path(os.path.relpath(path, Path.cwd().resolve())).as_posix()


def _near_duplicate_audit(
    validation_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report lexical similarity for review without changing labels or rows."""

    train_values = [
        (
            str(row.get("id")),
            str(row.get("target_candidate_name")),
            _normalized_surface(_current_utterance(row)),
        )
        for row in train_rows
    ]
    matches: list[dict[str, Any]] = []
    for row in validation_rows:
        current = _normalized_surface(_current_utterance(row))
        best_id = ""
        best_candidate = ""
        best_ratio = 0.0
        for train_id, train_candidate, train_current in train_values:
            ratio = SequenceMatcher(
                None,
                current,
                train_current,
                autojunk=False,
            ).ratio()
            if ratio > best_ratio:
                best_id = train_id
                best_candidate = train_candidate
                best_ratio = ratio
        matches.append(
            {
                "validation_id": str(row["id"]),
                "validation_candidate": str(row["target_candidate_name"]),
                "nearest_train_id": best_id,
                "nearest_train_candidate": best_candidate,
                "similarity": round(best_ratio, 6),
            }
        )
    matches.sort(key=lambda value: (-float(value["similarity"]), value["validation_id"]))
    return {
        "method": "difflib_sequence_matcher_on_nfkc_casefold_no_whitespace_punctuation",
        "threshold_counts": {
            threshold: sum(
                float(match["similarity"]) >= float(threshold)
                for match in matches
            )
            for threshold in ("0.80", "0.85", "0.90", "0.95")
        },
        "top_matches": matches[:20],
        "policy": "diagnostic_only_no_automatic_filtering_or_labeling",
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    *,
    report: Mapping[str, Any],
    inputs: Sequence[tuple[str, Path, int]],
    output_path: Path,
    seed: int,
) -> dict[str, Any]:
    """Build reproducibility, distribution, and leakage diagnostics."""

    turns_by_candidate: dict[str, dict[str, int]] = {}
    for candidate in EXPECTED_CANDIDATES:
        candidate_rows = [
            row for row in rows if row["target_candidate_name"] == candidate
        ]
        single = sum(not _is_multi_turn(row) for row in candidate_rows)
        turns_by_candidate[candidate] = {
            "single": single,
            "multi": len(candidate_rows) - single,
        }
    transitions = Counter(
        f"{row['source_candidate_name']}->{row['target_candidate_name']}"
        for row in rows
        if row.get("conversation_phenomenon") == "intent_change"
    )
    return {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "purpose": "development_validation_for_checkpoint_selection",
        "blind_test": False,
        "blind_test_note": (
            "PromptGen dev and former test/dev2 were previously used for development; "
            "this artifact must not be reported as an unbiased final test."
        ),
        "seed": seed,
        "rows": len(rows),
        "candidate_counts": _count(rows, "target_candidate_name"),
        "turn_counts_by_candidate": turns_by_candidate,
        "source_type_counts": _count(rows, "source_type"),
        "phenomenon_counts": _count(rows, "conversation_phenomenon"),
        "difficulty_counts": _count(rows, "difficulty"),
        "context_requirement_counts": _count(rows, "context_requirement"),
        "intent_change_pair_counts": dict(sorted(transitions.items())),
        "source_candidate_migration": dict(
            sorted(SOURCE_CANDIDATE_MIGRATION.items())
        ),
        "migration_review_notes": dict(sorted(SOURCE_MIGRATION_NOTES.items())),
        "validation_contract": {
            "rows_per_candidate": ROWS_PER_CANDIDATE,
            "single_turn_per_candidate": SINGLE_TURN_PER_CANDIDATE,
            "contextual_multi_turn_per_candidate": (
                CONTEXTUAL_MULTI_TURN_PER_CANDIDATE
            ),
            "intent_change_per_directed_pair": (
                INTENT_CHANGE_PER_DIRECTED_PAIR
            ),
            "legacy_specialty_rows": 32,
            "legacy_specialty_role": "bounded_slice_not_dominant_loss",
            "freeze_policy": (
                "If any row or family is used to repair training, retire v1 and "
                "publish a new validation version instead of editing in place."
            ),
        },
        "leakage_checks": dict(report),
        "near_duplicate_audit": _near_duplicate_audit(rows, train_rows),
        "inputs": [
            {
                "name": name,
                "path": _portable_path(path),
                "rows": count,
                "sha256": sha256_file(path),
            }
            for name, path, count in inputs
        ],
        "output": {
            "path": _portable_path(output_path),
            "sha256": sha256_file(output_path),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = Path(args.source_data).expanduser().resolve()
    retail_path = Path(args.retail_validation).expanduser().resolve()
    short_path = Path(args.short_validation).expanduser().resolve()
    stock_path = Path(args.stock_validation).expanduser().resolve()
    train_path = Path(args.train_data).expanduser().resolve()
    candidate_path = Path(args.candidate_registry).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy_data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    required_paths = (
        source_path,
        retail_path,
        short_path,
        stock_path,
        train_path,
        candidate_path,
        taxonomy_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise Top1DataError(f"required input does not exist: {path}")
    if output_path in required_paths or summary_path in {*required_paths, output_path}:
        raise Top1DataError("output and summary paths must not overwrite inputs")

    candidate_names = load_candidate_names(candidate_path)
    source_rows = read_jsonl(source_path)
    retail_rows = read_jsonl(retail_path)
    short_rows = read_jsonl(short_path)
    stock_rows = read_jsonl(stock_path)
    train_rows = read_jsonl(train_path)
    taxonomy_rows = read_jsonl(taxonomy_path)
    validate_memorization_rows(
        taxonomy_rows,
        candidate_names,
        source=taxonomy_path,
    )
    rows = build_unified_validation_rows(
        source_rows,
        retail_rows,
        short_rows,
        stock_rows,
        candidate_names=candidate_names,
        seed=args.seed,
    )
    report = validate_unified_validation_rows(rows, train_rows, candidate_names)
    write_jsonl(output_path, rows)
    summary = build_summary(
        rows,
        train_rows,
        report=report,
        inputs=(
            ("promptgen_xiaoyi", source_path, len(source_rows)),
            ("retail_boundary_validation", retail_path, len(retail_rows)),
            ("short_query_validation", short_path, len(short_rows)),
            ("stock_prediction_validation", stock_path, len(stock_rows)),
            ("combined_training", train_path, len(train_rows)),
            ("candidate_registry", candidate_path, 0),
            ("current_labeldesc_taxonomy", taxonomy_path, len(taxonomy_rows)),
        ),
        output_path=output_path,
        seed=args.seed,
    )
    write_json(summary_path, summary)
    print(f"[validation] wrote {len(rows)} rows: {output_path}")
    print(f"[validation] summary: {summary_path}")


if __name__ == "__main__":
    main()
