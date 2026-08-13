from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from llmgen.router import RouterDataError
from scripts.top1.visualize_threshold import (
    ThresholdExample,
    compute_curve,
    load_examples,
    main,
)


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_compute_curve_tracks_positive_and_negative_accuracy() -> None:
    examples = [
        ThresholdExample(0.8, "RecommendProduct", "RecommendProduct"),
        ThresholdExample(0.9, "RecommendProduct", "SearchStockQuotes"),
        ThresholdExample(0.4, "RecommendProduct", None),
        ThresholdExample(0.95, None, None),
    ]

    points = compute_curve(examples, num_thresholds=5)

    assert points[0].threshold == 0.0
    assert points[0].positive_accuracy == 0.5
    assert points[0].negative_accuracy == 0.5
    assert points[2].threshold == 0.5
    assert points[2].positive_accuracy == 0.5
    assert points[2].negative_accuracy == 1.0
    assert points[-1].threshold == 1.0
    assert points[-1].positive_accuracy == 0.0
    assert points[-1].negative_accuracy == 1.0


def test_load_examples_prefers_raw_intent_and_accepts_scalar_label_formats(
    tmp_path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "candidate_confidence": 0.7,
                "raw_intent_label": "RecommendProduct",
                "intent_label": None,
            },
            {"candidate_confidence": 0.2, "raw_intent_label": None},
        ],
    )
    labels.write_text('"RecommendProduct"\nnull\n', encoding="utf-8")

    examples = load_examples(predictions, labels)

    assert examples == [
        ThresholdExample(0.7, "RecommendProduct", "RecommendProduct"),
        ThresholdExample(0.2, None, None),
    ]


def test_load_examples_rejects_row_count_mismatch(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(
        predictions,
        [{"candidate_confidence": 0.7, "raw_intent_label": None}],
    )
    labels.write_text("null\nRecommendProduct\n", encoding="utf-8")

    with pytest.raises(RouterDataError, match="row count mismatch"):
        load_examples(predictions, labels)


def test_main_writes_svg_and_curve_values(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "threshold.svg"
    _write_jsonl(
        predictions,
        [
            {
                "candidate_confidence": 0.8,
                "raw_intent_label": "RecommendProduct",
            },
            {"candidate_confidence": 0.4, "raw_intent_label": None},
        ],
    )
    labels.write_text("RecommendProduct\nnull\n", encoding="utf-8")

    main(
        [
            "--predictions",
            str(predictions),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--num-thresholds",
            "3",
        ]
    )

    svg = output.read_text(encoding="utf-8")
    ET.parse(output)
    metrics = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert "Positive accuracy" in svg
    assert "Negative accuracy" in svg
    assert metrics["positive_examples"] == 1
    assert metrics["negative_examples"] == 1
    assert len(metrics["points"]) == 3
