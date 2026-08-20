#!/usr/bin/env python3
"""Build reviewed short-query retail contrasts for training and validation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from llmgen.top1 import (
    Top1DataError,
    load_candidate_names,
    normalize_messages,
    sha256_file,
    validate_training_rows,
    write_json,
    write_jsonl,
)


DATASET_VERSION = "top1_short_queries_v1"
DEFAULT_TRAIN_OUTPUT = "data_top1/top1_short_queries_v1.jsonl"
DEFAULT_VALIDATION_OUTPUT = "data_top1/top1_short_queries_v1_validation.jsonl"
DEFAULT_SUMMARY_OUTPUT = "data_top1/top1_short_queries_v1_summary.json"
MAX_QUERY_CHARACTERS = 24


@dataclass(frozen=True)
class ShortQueryContrast:
    """One concise retail request and a concise non-retail contrast."""

    family: str
    query_intent: str
    retail_query: str
    general_query: str
    decision_basis: str


TRAIN_CONTRASTS = (
    ShortQueryContrast(
        "sports_shoes_vs_suv",
        "brand_selection",
        "运动鞋哪个牌子好？",
        "七座SUV哪个牌子好？",
        "ordinary retail footwear versus a registered vehicle",
    ),
    ShortQueryContrast(
        "mouse_vs_antivirus_software",
        "brand_selection",
        "鼠标买什么牌子？",
        "杀毒软件买什么牌子？",
        "ordinary computer accessory versus software",
    ),
    ShortQueryContrast(
        "mattress_vs_moving_service",
        "brand_selection",
        "床垫哪个品牌靠谱？",
        "搬家公司哪家靠谱？",
        "ordinary household good versus labor service",
    ),
    ShortQueryContrast(
        "headphones_vs_broadband_service",
        "brand_selection",
        "无线耳机选哪个牌子？",
        "家庭宽带选哪家？",
        "ordinary electronics versus connectivity service",
    ),
    ShortQueryContrast(
        "rice_cooker_vs_renovation_service",
        "brand_selection",
        "电饭煲哪个品牌好？",
        "装修公司哪家好？",
        "ordinary appliance versus contracted service",
    ),
    ShortQueryContrast(
        "backpack_vs_online_course",
        "brand_selection",
        "双肩包买哪个牌子？",
        "英语网课选哪家？",
        "ordinary bag versus education service",
    ),
    ShortQueryContrast(
        "projector_vs_wedding_photography",
        "brand_selection",
        "投影仪哪个牌子好？",
        "婚纱摄影选哪家？",
        "ordinary electronics versus photography service",
    ),
    ShortQueryContrast(
        "robot_vacuum_vs_cleaning_service",
        "brand_selection",
        "扫地机器人哪个品牌好？",
        "家政保洁选哪家？",
        "ordinary appliance versus household service",
    ),
    ShortQueryContrast(
        "running_shoes_vs_family_car",
        "model_selection",
        "跑步鞋买哪双？",
        "家用SUV买哪辆？",
        "ordinary retail footwear versus a registered vehicle",
    ),
    ShortQueryContrast(
        "keyboard_vs_office_software",
        "model_selection",
        "机械键盘选哪款？",
        "办公软件选哪款？",
        "ordinary computer accessory versus software",
    ),
    ShortQueryContrast(
        "air_fryer_vs_personal_training",
        "model_selection",
        "空气炸锅买哪个？",
        "健身私教课买哪个？",
        "ordinary appliance versus coaching service",
    ),
    ShortQueryContrast(
        "picture_book_vs_copyright",
        "model_selection",
        "儿童绘本选哪本？",
        "绘本电子版权买哪种？",
        "physical retail copy versus rights transaction",
    ),
    ShortQueryContrast(
        "commute_headphones_vs_commute_car",
        "recommendation",
        "推荐一款通勤耳机。",
        "推荐一辆通勤汽车。",
        "ordinary electronics versus a registered vehicle",
    ),
    ShortQueryContrast(
        "router_vs_broadband_plan",
        "recommendation",
        "推荐个家用路由器。",
        "推荐个家庭宽带套餐。",
        "ordinary networking device versus connectivity service",
    ),
    ShortQueryContrast(
        "office_chair_vs_office_renovation",
        "recommendation",
        "推荐一把办公椅。",
        "推荐套办公室装修方案。",
        "ordinary furniture versus contracted project",
    ),
    ShortQueryContrast(
        "carry_on_vs_tour_package",
        "recommendation",
        "推荐个登机箱。",
        "推荐个境外旅行团。",
        "ordinary luggage versus travel service",
    ),
    ShortQueryContrast(
        "mouse_price_vs_apartment_price",
        "price",
        "这款鼠标多少钱？",
        "这套房子多少钱？",
        "ordinary retail good versus real estate",
    ),
    ShortQueryContrast(
        "sports_watch_vs_gym_membership",
        "promotion",
        "运动手表有优惠吗？",
        "健身年卡有优惠吗？",
        "ordinary wearable versus service membership",
    ),
    ShortQueryContrast(
        "senior_phone_vs_care_home",
        "suitability",
        "适合老人用的手机？",
        "适合老人的养老院？",
        "ordinary electronics versus residential care service",
    ),
    ShortQueryContrast(
        "air_conditioner_vs_renovation",
        "suitability",
        "小户型用什么空调？",
        "小户型找哪家装修？",
        "ordinary appliance versus contracted service",
    ),
    ShortQueryContrast(
        "baby_wipes_vs_baby_photography",
        "suitability",
        "宝宝用哪款湿巾？",
        "宝宝摄影选哪家？",
        "ordinary consumable versus photography service",
    ),
    ShortQueryContrast(
        "camping_tent_vs_gear_rental",
        "recommendation",
        "露营帐篷推荐一下。",
        "露营装备租赁推荐一下。",
        "ordinary outdoor good versus rental service",
    ),
    ShortQueryContrast(
        "coffee_machine_vs_training_course",
        "selection",
        "咖啡机怎么选？",
        "咖啡培训班怎么选？",
        "ordinary appliance versus education service",
    ),
    ShortQueryContrast(
        "printer_vs_printing_service",
        "recommendation",
        "打印机有什么推荐？",
        "印刷服务有什么推荐？",
        "ordinary electronics versus professional service",
    ),
)


VALIDATION_CONTRASTS = (
    ShortQueryContrast(
        "badminton_shoes_vs_recreational_vehicle",
        "brand_selection",
        "羽毛球鞋买什么牌子？",
        "房车买什么牌子？",
        "ordinary retail footwear versus a registered vehicle",
    ),
    ShortQueryContrast(
        "monitor_vs_image_software",
        "model_selection",
        "显示器选哪款？",
        "修图软件选哪款？",
        "ordinary electronics versus software",
    ),
    ShortQueryContrast(
        "thermos_vs_moving_package",
        "recommendation",
        "推荐个保温杯。",
        "推荐个搬家套餐。",
        "ordinary drinkware versus labor service",
    ),
    ShortQueryContrast(
        "floor_washer_vs_cleaning_service",
        "price",
        "洗地机多少钱？",
        "保洁服务多少钱？",
        "ordinary appliance versus household service",
    ),
    ShortQueryContrast(
        "dorm_fridge_vs_broadband_plan",
        "suitability",
        "适合宿舍的小冰箱？",
        "适合宿舍的宽带套餐？",
        "ordinary appliance versus connectivity service",
    ),
    ShortQueryContrast(
        "hair_dryer_vs_haircut_service",
        "promotion",
        "吹风机有优惠吗？",
        "理发服务有优惠吗？",
        "ordinary appliance versus personal service",
    ),
    ShortQueryContrast(
        "stroller_vs_nanny_service",
        "brand_selection",
        "婴儿车哪个品牌好？",
        "月嫂公司哪家好？",
        "ordinary child product versus care service",
    ),
    ShortQueryContrast(
        "paper_book_vs_adaptation_rights",
        "model_selection",
        "纸质绘本买哪本？",
        "绘本改编权怎么买？",
        "physical retail copy versus rights transaction",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deterministic short-query dataset arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--validation-output", default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args(argv)


def build_short_query_rows(split: str) -> list[dict[str, Any]]:
    """Build one-turn rows from explicitly reviewed semantic contrasts."""

    if split not in {"train", "validation"}:
        raise Top1DataError("short-query split must be 'train' or 'validation'")
    contrasts = TRAIN_CONTRASTS if split == "train" else VALIDATION_CONTRASTS
    rows: list[dict[str, Any]] = []
    for index, contrast in enumerate(contrasts, start=1):
        pair_id = f"{DATASET_VERSION}_{split}_{index:03d}"
        for side, query, target, eligible in (
            ("retail", contrast.retail_query, "ProductEcommerce", True),
            ("general", contrast.general_query, "ProductGeneral", False),
        ):
            rows.append(
                {
                    "id": f"{pair_id}_{side}",
                    "dataset_version": DATASET_VERSION,
                    "source_type": "reviewed_short_query_contrast",
                    "messages": [{"role": "user", "content": query}],
                    "target_candidate_name": target,
                    "short_query_intent": contrast.query_intent,
                    "contrast_family": contrast.family,
                    "contrast_pair_id": pair_id,
                    "contrast_side": side,
                    "retail_eligibility": {
                        "eligible": eligible,
                        "decision_basis": contrast.decision_basis,
                    },
                }
            )
    return rows


def _canonical_messages(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"])
        for message in normalize_messages(row.get("messages"))
    )


def validate_short_query_rows(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate concise phrasing, balanced pairs, and disjoint holdout families."""

    all_rows = [*train_rows, *validation_rows]
    ids = [str(row.get("id")) for row in all_rows]
    if len(ids) != len(set(ids)):
        raise Top1DataError("short-query data contains duplicate IDs")
    conversations = [_canonical_messages(row) for row in all_rows]
    if len(conversations) != len(set(conversations)):
        raise Top1DataError("short-query data contains duplicate conversations")

    train_families = {str(row.get("contrast_family")) for row in train_rows}
    validation_families = {
        str(row.get("contrast_family")) for row in validation_rows
    }
    overlap = sorted(train_families & validation_families)
    if overlap:
        raise Top1DataError(
            "train and validation short-query families overlap: "
            + ", ".join(overlap)
        )

    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            messages = normalize_messages(row.get("messages"))
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise Top1DataError(f"{split}: short query must be one user message")
            if len(messages[0]["content"]) > MAX_QUERY_CHARACTERS:
                raise Top1DataError(
                    f"{split}:{row.get('id')}: query exceeds "
                    f"{MAX_QUERY_CHARACTERS} characters"
                )
            pair_id = row.get("contrast_pair_id")
            if not isinstance(pair_id, str):
                raise Top1DataError(f"{split}: contrast_pair_id must be a string")
            grouped.setdefault(pair_id, []).append(row)
        for pair_id, pair_rows in grouped.items():
            if len(pair_rows) != 2:
                raise Top1DataError(f"{split}:{pair_id}: expected exactly two rows")
            if {str(row.get("contrast_side")) for row in pair_rows} != {
                "retail",
                "general",
            }:
                raise Top1DataError(f"{split}:{pair_id}: invalid contrast sides")
            if {str(row.get("target_candidate_name")) for row in pair_rows} != {
                "ProductEcommerce",
                "ProductGeneral",
            }:
                raise Top1DataError(f"{split}:{pair_id}: invalid contrast labels")


def _count(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def build_summary(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    train_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    """Build provenance and balance diagnostics for both data splits."""

    def split_summary(
        rows: Sequence[Mapping[str, Any]],
        path: Path,
    ) -> dict[str, Any]:
        return {
            "rows": len(rows),
            "pairs": len(rows) // 2,
            "candidate_counts": _count(rows, "target_candidate_name"),
            "query_intent_counts": _count(rows, "short_query_intent"),
            "families": len({str(row.get("contrast_family")) for row in rows}),
            "max_query_characters": max(
                len(normalize_messages(row.get("messages"))[0]["content"])
                for row in rows
            ),
            "path": str(path),
            "sha256": sha256_file(path),
        }

    return {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "construction": "reviewed_balanced_short_query_contrasts",
        "policy": (
            "short wording does not determine the route; classify the exact "
            "requested object by ordinary-retail eligibility"
        ),
        "train": split_summary(train_rows, train_path),
        "validation": {
            **split_summary(validation_rows, validation_path),
            "family_overlap_with_train": 0,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_path = Path(args.candidate_registry).expanduser().resolve()
    train_path = Path(args.train_output).expanduser().resolve()
    validation_path = Path(args.validation_output).expanduser().resolve()
    summary_path = Path(args.summary_output).expanduser().resolve()
    if len({train_path, validation_path, summary_path}) != 3:
        raise Top1DataError("train, validation, and summary paths must be distinct")

    candidate_names = load_candidate_names(candidate_path)
    train_rows = build_short_query_rows("train")
    validation_rows = build_short_query_rows("validation")
    validate_training_rows(train_rows, candidate_names, source=train_path)
    validate_training_rows(
        validation_rows,
        candidate_names,
        source=validation_path,
    )
    validate_short_query_rows(train_rows, validation_rows)

    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    write_json(
        summary_path,
        build_summary(
            train_rows,
            validation_rows,
            train_path=train_path,
            validation_path=validation_path,
        ),
    )
    print(f"[short-query] training rows: {len(train_rows)} -> {train_path}")
    print(
        "[short-query] validation rows: "
        f"{len(validation_rows)} -> {validation_path}"
    )
    print(f"[short-query] summary: {summary_path}")


if __name__ == "__main__":
    main()
