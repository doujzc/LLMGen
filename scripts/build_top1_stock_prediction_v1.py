#!/usr/bin/env python3
"""Build reviewed short stock-prediction and factual-query contrasts."""

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


DATASET_VERSION = "top1_stock_prediction_v1"
DEFAULT_TRAIN_OUTPUT = "data_top1/top1_stock_prediction_v1.jsonl"
DEFAULT_VALIDATION_OUTPUT = "data_top1/top1_stock_prediction_v1_validation.jsonl"
DEFAULT_SUMMARY_OUTPUT = "data_top1/top1_stock_prediction_v1_summary.json"
MAX_QUERY_CHARACTERS = 24
DECISION_BASIS = (
    "future price direction, level, rebound, or breakout requires a forecast and "
    "is StockAdvice; already observed price or return is a factual StockQuery"
)


@dataclass(frozen=True)
class StockTemporalContrast:
    """One future forecast and one observed-market factual query."""

    family: str
    forecast_type: str
    horizon: str
    prediction_query: str
    factual_query: str


TRAIN_CONTRASTS = (
    StockTemporalContrast(
        "kweichow_moutai",
        "direction",
        "next_week",
        "贵州茅台下周会不会涨？",
        "贵州茅台今天涨了多少？",
    ),
    StockTemporalContrast(
        "catl",
        "direction",
        "next_week",
        "宁德时代下周会涨还是跌？",
        "宁德时代今天涨还是跌？",
    ),
    StockTemporalContrast(
        "byd",
        "direction",
        "tomorrow",
        "比亚迪明天会不会跌？",
        "比亚迪今天跌了吗？",
    ),
    StockTemporalContrast(
        "zijin_mining",
        "continuation",
        "next_month",
        "紫金矿业下月还能涨吗？",
        "紫金矿业本月涨了多少？",
    ),
    StockTemporalContrast(
        "ping_an",
        "direction",
        "next_week",
        "中国平安下周会走强吗？",
        "中国平安本周涨了多少？",
    ),
    StockTemporalContrast(
        "china_merchants_bank",
        "rebound",
        "tomorrow",
        "招商银行明天能反弹吗？",
        "招商银行今天反弹了吗？",
    ),
    StockTemporalContrast(
        "citic_securities",
        "pullback",
        "next_week",
        "中信证券下周会回调吗？",
        "中信证券今天回调了吗？",
    ),
    StockTemporalContrast(
        "longi_green_energy",
        "new_high",
        "next_week",
        "隆基绿能下周会创新高吗？",
        "隆基绿能今天创新高了吗？",
    ),
    StockTemporalContrast(
        "szse_component",
        "direction",
        "next_week",
        "深证成指下周会不会涨？",
        "深证成指今天涨了多少？",
    ),
    StockTemporalContrast(
        "sse_composite",
        "direction",
        "tomorrow",
        "上证指数明天会涨吗？",
        "上证指数今天涨了吗？",
    ),
    StockTemporalContrast(
        "chinext",
        "direction",
        "next_week",
        "创业板下周会不会涨？",
        "创业板本周涨了多少？",
    ),
    StockTemporalContrast(
        "csi_300",
        "direction",
        "next_month",
        "沪深300下月会走强吗？",
        "沪深300本月涨了多少？",
    ),
    StockTemporalContrast(
        "star_50",
        "pullback",
        "tomorrow",
        "科创50明天会回调吗？",
        "科创50今天回调了吗？",
    ),
    StockTemporalContrast(
        "hang_seng_index",
        "rebound",
        "next_week",
        "恒生指数下周能反弹吗？",
        "恒生指数本周反弹了吗？",
    ),
    StockTemporalContrast(
        "tencent",
        "continuation",
        "next_week",
        "腾讯控股下周还能涨吗？",
        "腾讯控股这周涨了多少？",
    ),
    StockTemporalContrast(
        "alibaba",
        "rebound",
        "tomorrow",
        "阿里巴巴明天会反弹吗？",
        "阿里巴巴今天反弹了吗？",
    ),
    StockTemporalContrast(
        "meituan",
        "direction",
        "next_month",
        "美团下月会不会跌？",
        "美团本月跌了多少？",
    ),
    StockTemporalContrast(
        "xiaomi",
        "continuation",
        "next_week",
        "小米集团下周会继续涨吗？",
        "小米集团本周涨了多少？",
    ),
    StockTemporalContrast(
        "apple",
        "direction",
        "tomorrow",
        "苹果股票明天会涨吗？",
        "苹果股票今天涨了吗？",
    ),
    StockTemporalContrast(
        "nvidia",
        "pullback",
        "next_week",
        "英伟达下周会回调吗？",
        "英伟达这周回调了吗？",
    ),
    StockTemporalContrast(
        "tesla",
        "continuation",
        "next_month",
        "特斯拉下月还能涨吗？",
        "特斯拉本月涨了多少？",
    ),
    StockTemporalContrast(
        "microsoft",
        "target_price",
        "year_end",
        "微软年底能到多少？",
        "微软现在股价多少？",
    ),
    StockTemporalContrast(
        "nasdaq_composite",
        "direction",
        "next_week",
        "纳斯达克下周会跌吗？",
        "纳斯达克今天跌了吗？",
    ),
    StockTemporalContrast(
        "sp_500",
        "new_high",
        "tomorrow",
        "标普500明天能创新高吗？",
        "标普500今天创新高了吗？",
    ),
)


VALIDATION_CONTRASTS = (
    StockTemporalContrast(
        "wuliangye",
        "direction",
        "next_week",
        "五粮液下周会不会涨？",
        "五粮液今天涨了多少？",
    ),
    StockTemporalContrast(
        "smic",
        "direction",
        "tomorrow",
        "中芯国际明天会跌吗？",
        "中芯国际今天跌了吗？",
    ),
    StockTemporalContrast(
        "jd_com",
        "rebound",
        "next_week",
        "京东集团下周能反弹吗？",
        "京东集团本周反弹了吗？",
    ),
    StockTemporalContrast(
        "meta",
        "continuation",
        "next_month",
        "Meta下月还能涨吗？",
        "Meta本月涨了多少？",
    ),
    StockTemporalContrast(
        "dow_jones",
        "direction",
        "tomorrow",
        "道琼斯明天会涨吗？",
        "道琼斯今天涨了吗？",
    ),
    StockTemporalContrast(
        "hang_seng_tech",
        "new_high",
        "next_week",
        "恒生科技下周会创新高吗？",
        "恒生科技本周创新高了吗？",
    ),
    StockTemporalContrast(
        "healthcare_etf",
        "rebound",
        "next_week",
        "医药ETF下周能反弹吗？",
        "医药ETF本周反弹了吗？",
    ),
    StockTemporalContrast(
        "new_energy_etf",
        "direction",
        "next_month",
        "新能源ETF下月会走强吗？",
        "新能源ETF本月涨了多少？",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deterministic stock-prediction dataset arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--validation-output", default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args(argv)


def build_stock_prediction_rows(split: str) -> list[dict[str, Any]]:
    """Build paired future-prediction and observed-fact rows."""

    if split not in {"train", "validation"}:
        raise Top1DataError(
            "stock-prediction split must be 'train' or 'validation'"
        )
    contrasts = TRAIN_CONTRASTS if split == "train" else VALIDATION_CONTRASTS
    rows: list[dict[str, Any]] = []
    for index, contrast in enumerate(contrasts, start=1):
        pair_id = f"{DATASET_VERSION}_{split}_{index:03d}"
        for side, query, target, future in (
            ("future_prediction", contrast.prediction_query, "StockAdvice", True),
            ("observed_fact", contrast.factual_query, "StockQuery", False),
        ):
            rows.append(
                {
                    "id": f"{pair_id}_{side}",
                    "dataset_version": DATASET_VERSION,
                    "source_type": "reviewed_stock_temporal_contrast",
                    "messages": [{"role": "user", "content": query}],
                    "target_candidate_name": target,
                    "forecast_type": contrast.forecast_type,
                    "forecast_horizon": contrast.horizon if future else None,
                    "contrast_family": contrast.family,
                    "contrast_pair_id": pair_id,
                    "contrast_side": side,
                    "temporal_decision": {
                        "requires_future_market_judgment": future,
                        "decision_basis": DECISION_BASIS,
                    },
                }
            )
    return rows


def _canonical_messages(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"])
        for message in normalize_messages(row.get("messages"))
    )


def validate_stock_prediction_rows(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate concise phrasing, temporal pairs, and disjoint instruments."""

    all_rows = [*train_rows, *validation_rows]
    ids = [str(row.get("id")) for row in all_rows]
    if len(ids) != len(set(ids)):
        raise Top1DataError("stock-prediction data contains duplicate IDs")
    conversations = [_canonical_messages(row) for row in all_rows]
    if len(conversations) != len(set(conversations)):
        raise Top1DataError(
            "stock-prediction data contains duplicate conversations"
        )

    train_families = {str(row.get("contrast_family")) for row in train_rows}
    validation_families = {
        str(row.get("contrast_family")) for row in validation_rows
    }
    overlap = sorted(train_families & validation_families)
    if overlap:
        raise Top1DataError(
            "train and validation stock-prediction families overlap: "
            + ", ".join(overlap)
        )

    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            messages = normalize_messages(row.get("messages"))
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise Top1DataError(
                    f"{split}: stock temporal query must be one user message"
                )
            if len(messages[0]["content"]) > MAX_QUERY_CHARACTERS:
                raise Top1DataError(
                    f"{split}:{row.get('id')}: query exceeds "
                    f"{MAX_QUERY_CHARACTERS} characters"
                )
            pair_id = row.get("contrast_pair_id")
            if not isinstance(pair_id, str):
                raise Top1DataError(f"{split}: contrast_pair_id must be a string")
            side = row.get("contrast_side")
            temporal = row.get("temporal_decision")
            if not isinstance(temporal, Mapping):
                raise Top1DataError(
                    f"{split}:{row.get('id')}: temporal_decision must be an object"
                )
            if side == "future_prediction":
                if row.get("target_candidate_name") != "StockAdvice":
                    raise Top1DataError(
                        f"{split}:{row.get('id')}: future prediction must be StockAdvice"
                    )
                if not isinstance(row.get("forecast_horizon"), str):
                    raise Top1DataError(
                        f"{split}:{row.get('id')}: future horizon must be a string"
                    )
                if temporal.get("requires_future_market_judgment") is not True:
                    raise Top1DataError(
                        f"{split}:{row.get('id')}: future signal must be true"
                    )
            elif side == "observed_fact":
                if row.get("target_candidate_name") != "StockQuery":
                    raise Top1DataError(
                        f"{split}:{row.get('id')}: observed fact must be StockQuery"
                    )
                if row.get("forecast_horizon") is not None:
                    raise Top1DataError(
                        f"{split}:{row.get('id')}: observed fact cannot have a horizon"
                    )
                if temporal.get("requires_future_market_judgment") is not False:
                    raise Top1DataError(
                        f"{split}:{row.get('id')}: observed-fact signal must be false"
                    )
            grouped.setdefault(pair_id, []).append(row)
        for pair_id, pair_rows in grouped.items():
            if len(pair_rows) != 2:
                raise Top1DataError(f"{split}:{pair_id}: expected exactly two rows")
            if {str(row.get("contrast_side")) for row in pair_rows} != {
                "future_prediction",
                "observed_fact",
            }:
                raise Top1DataError(f"{split}:{pair_id}: invalid temporal sides")
            if {str(row.get("target_candidate_name")) for row in pair_rows} != {
                "StockAdvice",
                "StockQuery",
            }:
                raise Top1DataError(f"{split}:{pair_id}: invalid temporal labels")


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
            "forecast_type_counts": _count(rows, "forecast_type"),
            "future_horizon_counts": _count(
                [
                    row
                    for row in rows
                    if row.get("contrast_side") == "future_prediction"
                ],
                "forecast_horizon",
            ),
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
        "construction": "reviewed_stock_temporal_minimal_contrasts",
        "policy": DECISION_BASIS,
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
    train_rows = build_stock_prediction_rows("train")
    validation_rows = build_stock_prediction_rows("validation")
    validate_training_rows(train_rows, candidate_names, source=train_path)
    validate_training_rows(
        validation_rows,
        candidate_names,
        source=validation_path,
    )
    validate_stock_prediction_rows(train_rows, validation_rows)

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
    print(f"[stock-prediction] training rows: {len(train_rows)} -> {train_path}")
    print(
        "[stock-prediction] validation rows: "
        f"{len(validation_rows)} -> {validation_path}"
    )
    print(f"[stock-prediction] summary: {summary_path}")


if __name__ == "__main__":
    main()
