#!/usr/bin/env python3
"""Build paired retail-eligibility training and held-out validation data."""

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


DATASET_VERSION = "top1_retail_boundary_v1"
DEFAULT_TRAIN_OUTPUT = "data_top1/top1_retail_boundary_v1.jsonl"
DEFAULT_VALIDATION_OUTPUT = "data_top1/top1_retail_boundary_v1_validation.jsonl"
DEFAULT_SUMMARY_OUTPUT = "data_top1/top1_retail_boundary_v1_summary.json"


@dataclass(frozen=True)
class BoundaryObjectPair:
    """One unsupported object and its ordinary-retail minimal contrast."""

    family: str
    unsupported_object: str
    retail_object: str


@dataclass(frozen=True)
class BoundaryAxis:
    """An abstract retail-eligibility dimension with authored query frames."""

    name: str
    decision_basis: str
    templates: tuple[str, ...]
    train_pairs: tuple[BoundaryObjectPair, ...]
    validation_pairs: tuple[BoundaryObjectPair, ...]


BOUNDARY_AXES = (
    BoundaryAxis(
        name="registered_asset_vs_retail_replica",
        decision_basis=(
            "registered or titled assets are not ordinary retail SKUs; toys, models, "
            "and replicas are ordinary retail goods"
        ),
        templates=(
            "送他{object}大概要多少钱？",
            "我想买{object}，你们这里能直接下单吗？",
            "预算有限，{object}有没有合适的选择？",
            "帮我看看{object}一般什么价位，有没有推荐的？",
        ),
        train_pairs=(
            BoundaryObjectPair(
                "helicopter",
                "一架小型直升机",
                "一架遥控玩具直升机",
            ),
            BoundaryObjectPair(
                "yacht",
                "一艘小型游艇",
                "一艘遥控游艇模型",
            ),
            BoundaryObjectPair(
                "suv",
                "一辆七座SUV整车",
                "一辆七座SUV合金车模",
            ),
            BoundaryObjectPair(
                "apartment",
                "一套海景公寓",
                "一套海景公寓积木模型",
            ),
            BoundaryObjectPair(
                "private_jet",
                "一架私人喷气式飞机",
                "一架喷气式飞机拼装模型",
            ),
            BoundaryObjectPair(
                "speedboat",
                "一艘钓鱼快艇",
                "一艘儿童遥控快艇",
            ),
        ),
        validation_pairs=(
            BoundaryObjectPair(
                "tourist_train",
                "一列私人观光火车",
                "一套电动火车模型",
            ),
            BoundaryObjectPair(
                "hot_air_balloon",
                "一架载人热气球",
                "一个热气球拼装模型",
            ),
        ),
    ),
    BoundaryAxis(
        name="rights_license_vs_retail_copy",
        decision_basis=(
            "copyright, licensing, and usage rights require a rights transaction; "
            "physical copies and merchandise are ordinary retail goods"
        ),
        templates=(
            "想买{object}，能直接下单吗？",
            "{object}一般需要多少钱？",
            "公司准备采购{object}，有没有合适的推荐？",
            "帮我看看{object}现在有什么优惠。",
        ),
        train_pairs=(
            BoundaryObjectPair(
                "picture_book_copyright",
                "这本绘本的电子版权",
                "这本绘本的纸质版",
            ),
            BoundaryObjectPair(
                "music_commercial_rights",
                "一首歌的商业使用权",
                "这首歌的正版音乐专辑",
            ),
            BoundaryObjectPair(
                "anime_merchandise_license",
                "某动漫形象的周边授权",
                "某动漫形象的正版手办",
            ),
            BoundaryObjectPair(
                "trademark_license",
                "一个品牌的商标使用权",
                "这个品牌的联名T恤",
            ),
            BoundaryObjectPair(
                "patent_license",
                "一项发明专利的独家许可",
                "一本专利入门书",
            ),
            BoundaryObjectPair(
                "movie_streaming_rights",
                "一部电影的网络播放权",
                "这部电影的正版蓝光碟",
            ),
        ),
        validation_pairs=(
            BoundaryObjectPair(
                "photo_copyright",
                "一张摄影作品的商业版权",
                "这张摄影作品的装饰画",
            ),
            BoundaryObjectPair(
                "comic_adaptation_rights",
                "一部漫画的影视改编权",
                "这部漫画的实体典藏版",
            ),
        ),
    ),
    BoundaryAxis(
        name="service_vs_retail_tool",
        decision_basis=(
            "labor and professional services are not retail goods; the tools and "
            "supplies used for the same goal can be ordinary retail SKUs"
        ),
        templates=(
            "最近需要{object}，大概多少钱？",
            "想买{object}，这里能直接下单吗？",
            "{object}有没有适合普通预算的？",
            "帮我推荐{object}，我想先比较一下价格。",
        ),
        train_pairs=(
            BoundaryObjectPair(
                "home_renovation",
                "一套全屋装修施工服务",
                "一套家用电钻工具箱",
            ),
            BoundaryObjectPair(
                "wedding_photography",
                "一次婚礼摄影跟拍服务",
                "一台入门微单相机",
            ),
            BoundaryObjectPair(
                "moving_service",
                "一次搬家打包服务",
                "一套加厚搬家纸箱",
            ),
            BoundaryObjectPair(
                "air_conditioner_cleaning",
                "一次空调上门清洗服务",
                "一瓶空调清洗剂",
            ),
            BoundaryObjectPair(
                "pet_transport",
                "一套宠物托运服务",
                "一个宠物航空箱",
            ),
            BoundaryObjectPair(
                "personal_training",
                "一套私人健身教练课程",
                "一套家用可调哑铃",
            ),
        ),
        validation_pairs=(
            BoundaryObjectPair(
                "website_development",
                "一套网站定制开发服务",
                "一台办公笔记本电脑",
            ),
            BoundaryObjectPair(
                "legal_consulting",
                "一套企业法律顾问服务",
                "一本企业法律实务书",
            ),
        ),
    ),
    BoundaryAxis(
        name="custom_project_vs_household_product",
        decision_basis=(
            "bespoke industrial systems and contracted projects are not ordinary "
            "retail SKUs; standardized household products are"
        ),
        templates=(
            "想采购{object}，能直接下单吗？",
            "{object}一般要多少钱？",
            "预算有限，{object}有没有合适的选择？",
            "帮我推荐{object}，我想比较一下价格和配置。",
        ),
        train_pairs=(
            BoundaryObjectPair(
                "automation_line",
                "一条定制自动化生产线",
                "一台家用3D打印机",
            ),
            BoundaryObjectPair(
                "commercial_cold_storage",
                "一套商用冷库工程",
                "一台家用冰柜",
            ),
            BoundaryObjectPair(
                "industrial_water_treatment",
                "一套工厂污水处理系统",
                "一台家用净水器",
            ),
            BoundaryObjectPair(
                "stage_lighting_project",
                "一套大型舞台灯光工程",
                "一盏家用氛围灯",
            ),
            BoundaryObjectPair(
                "commercial_kitchen",
                "一套商用中央厨房工程",
                "一台家用烤箱",
            ),
            BoundaryObjectPair(
                "prefabricated_warehouse",
                "一座装配式仓库",
                "一个家用金属收纳柜",
            ),
        ),
        validation_pairs=(
            BoundaryObjectPair(
                "hotel_access_control",
                "一套酒店门禁改造工程",
                "一把家用智能门锁",
            ),
            BoundaryObjectPair(
                "mall_security_system",
                "一套商场安防监控工程",
                "一台家用监控摄像头",
            ),
        ),
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deterministic retail-boundary dataset arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--validation-output", default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args(argv)


def _slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")


def build_boundary_rows(split: str) -> list[dict[str, Any]]:
    """Build paired rows from explicit semantic axes for one data split."""

    if split not in {"train", "validation"}:
        raise Top1DataError("boundary split must be 'train' or 'validation'")
    rows: list[dict[str, Any]] = []
    for axis in BOUNDARY_AXES:
        pairs = axis.train_pairs if split == "train" else axis.validation_pairs
        for pair_index, pair in enumerate(pairs, start=1):
            for template_index, template in enumerate(axis.templates, start=1):
                pair_id = (
                    f"{DATASET_VERSION}_{split}_{_slug(axis.name)}_"
                    f"{pair_index:02d}_{template_index:02d}"
                )
                for side, object_text, target, eligible in (
                    (
                        "unsupported",
                        pair.unsupported_object,
                        "GeneralProduct",
                        False,
                    ),
                    ("retail", pair.retail_object, "EcommerceProduct", True),
                ):
                    rows.append(
                        {
                            "id": f"{pair_id}_{side}",
                            "dataset_version": DATASET_VERSION,
                            "source_type": "reviewed_structured_retail_boundary",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": template.format(object=object_text),
                                }
                            ],
                            "target_candidate_name": target,
                            "boundary_axis": axis.name,
                            "boundary_family": pair.family,
                            "boundary_pair_id": pair_id,
                            "boundary_side": side,
                            "retail_eligibility": {
                                "eligible": eligible,
                                "decision_basis": axis.decision_basis,
                            },
                        }
                    )
    return rows


def _canonical_messages(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"])
        for message in normalize_messages(row.get("messages"))
    )


def validate_boundary_rows(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate pair completeness and family-disjoint holdout semantics."""

    all_rows = [*train_rows, *validation_rows]
    ids = [str(row.get("id")) for row in all_rows]
    if len(ids) != len(set(ids)):
        raise Top1DataError("retail-boundary data contains duplicate IDs")
    conversations = [_canonical_messages(row) for row in all_rows]
    if len(conversations) != len(set(conversations)):
        raise Top1DataError("retail-boundary data contains duplicate conversations")

    train_families = {str(row.get("boundary_family")) for row in train_rows}
    validation_families = {
        str(row.get("boundary_family")) for row in validation_rows
    }
    overlap = sorted(train_families & validation_families)
    if overlap:
        raise Top1DataError(
            "train and validation boundary families overlap: " + ", ".join(overlap)
        )

    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            pair_id = row.get("boundary_pair_id")
            if not isinstance(pair_id, str):
                raise Top1DataError(f"{split}: boundary_pair_id must be a string")
            grouped.setdefault(pair_id, []).append(row)
        for pair_id, pair_rows in grouped.items():
            if len(pair_rows) != 2:
                raise Top1DataError(f"{split}:{pair_id}: expected exactly two rows")
            sides = {str(row.get("boundary_side")) for row in pair_rows}
            labels = {str(row.get("target_candidate_name")) for row in pair_rows}
            if sides != {"unsupported", "retail"}:
                raise Top1DataError(f"{split}:{pair_id}: invalid pair sides")
            if labels != {"GeneralProduct", "EcommerceProduct"}:
                raise Top1DataError(f"{split}:{pair_id}: invalid pair labels")


def _count(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def build_summary(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    train_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    """Build balance, pairing, and holdout provenance for both splits."""

    return {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "construction": "reviewed_structured_minimal_pairs",
        "policy": (
            "classify the exact requested transaction object by ordinary-retail "
            "eligibility instead of enumerating product names"
        ),
        "train": {
            "rows": len(train_rows),
            "pairs": len(train_rows) // 2,
            "candidate_counts": _count(train_rows, "target_candidate_name"),
            "axis_counts": _count(train_rows, "boundary_axis"),
            "families": len(
                {str(row.get("boundary_family")) for row in train_rows}
            ),
            "path": str(train_path),
            "sha256": sha256_file(train_path),
        },
        "validation": {
            "rows": len(validation_rows),
            "pairs": len(validation_rows) // 2,
            "candidate_counts": _count(validation_rows, "target_candidate_name"),
            "axis_counts": _count(validation_rows, "boundary_axis"),
            "families": len(
                {str(row.get("boundary_family")) for row in validation_rows}
            ),
            "family_overlap_with_train": 0,
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
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
    train_rows = build_boundary_rows("train")
    validation_rows = build_boundary_rows("validation")
    validate_training_rows(train_rows, candidate_names, source=train_path)
    validate_training_rows(
        validation_rows,
        candidate_names,
        source=validation_path,
    )
    validate_boundary_rows(train_rows, validation_rows)

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
    print(f"[retail-boundary] training rows: {len(train_rows)} -> {train_path}")
    print(
        "[retail-boundary] validation rows: "
        f"{len(validation_rows)} -> {validation_path}"
    )
    print(f"[retail-boundary] summary: {summary_path}")


if __name__ == "__main__":
    main()
