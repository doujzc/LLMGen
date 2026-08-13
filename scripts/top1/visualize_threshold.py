#!/usr/bin/env python3
"""Visualize positive and negative Top1 routing accuracy by threshold."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Sequence

from llmgen.router import RouterDataError, read_jsonl


@dataclass(frozen=True)
class ThresholdExample:
    """One raw router prediction paired with its expected intent label."""

    confidence: float
    predicted_intent: str | None
    expected_intent: str | None


@dataclass(frozen=True)
class ThresholdPoint:
    """Class-specific accuracy after applying one route threshold."""

    threshold: float
    positive_accuracy: float
    negative_accuracy: float
    balanced_accuracy: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot positive- and negative-sample routing accuracy as the route "
            "threshold changes."
        )
    )
    parser.add_argument(
        "--predictions",
        "--evaluate-results",
        dest="predictions",
        required=True,
        help="JSONL output produced by scripts/top1/02_evaluate.sh.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="One intent-label JSON scalar per line; use null for negative samples.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination SVG curve.",
    )
    parser.add_argument(
        "--metrics-output",
        help="Curve values as JSON; defaults to the output path with a .json suffix.",
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=201,
        help="Number of evenly spaced thresholds in [0, 1] (default: 201).",
    )
    parser.add_argument(
        "--title",
        default="Top1 routing accuracy by confidence threshold",
    )
    return parser.parse_args(argv)


def _read_labels(path: str | Path) -> list[str | None]:
    """Read JSONL scalar labels, also accepting unquoted intent names."""

    source = Path(path)
    labels: list[str | None] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                raise RouterDataError(f"empty label at {source}:{line_number}")
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text
            if value is not None and not isinstance(value, str):
                raise RouterDataError(
                    f"label at {source}:{line_number} must be a string or null"
                )
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    raise RouterDataError(
                        f"label at {source}:{line_number} cannot be empty"
                    )
            labels.append(value)
    if not labels:
        raise RouterDataError("label input is empty")
    return labels


def _raw_intent(row: dict[str, Any], *, row_number: int) -> str | None:
    field = "raw_intent_label" if "raw_intent_label" in row else "intent_label"
    if field not in row:
        raise RouterDataError(
            f"prediction row {row_number} has no raw_intent_label or intent_label"
        )
    value = row[field]
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise RouterDataError(
            f"prediction row {row_number} has an invalid {field}"
        )
    return value.strip() if isinstance(value, str) else None


def _confidence(row: dict[str, Any], *, row_number: int) -> float:
    value = row.get("candidate_confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouterDataError(
            f"prediction row {row_number} has no numeric candidate_confidence"
        )
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RouterDataError(
            f"prediction row {row_number} candidate_confidence must be in [0, 1]"
        )
    return confidence


def load_examples(
    predictions_path: str | Path,
    labels_path: str | Path,
) -> list[ThresholdExample]:
    predictions = read_jsonl(predictions_path)
    labels = _read_labels(labels_path)
    if len(predictions) != len(labels):
        raise RouterDataError(
            "prediction/label row count mismatch: "
            f"{len(predictions)} predictions != {len(labels)} labels"
        )
    examples = [
        ThresholdExample(
            confidence=_confidence(row, row_number=index),
            predicted_intent=_raw_intent(row, row_number=index),
            expected_intent=label,
        )
        for index, (row, label) in enumerate(
            zip(predictions, labels, strict=True), start=1
        )
    ]
    positive_count = sum(row.expected_intent is not None for row in examples)
    negative_count = len(examples) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise RouterDataError(
            "threshold visualization requires at least one positive and one negative label"
        )
    return examples


def compute_curve(
    examples: Sequence[ThresholdExample],
    *,
    num_thresholds: int,
) -> list[ThresholdPoint]:
    if num_thresholds < 2:
        raise RouterDataError("num_thresholds must be at least 2")
    positive = tuple(row for row in examples if row.expected_intent is not None)
    negative = tuple(row for row in examples if row.expected_intent is None)
    if not positive or not negative:
        raise RouterDataError(
            "threshold visualization requires at least one positive and one negative label"
        )

    points: list[ThresholdPoint] = []
    for index in range(num_thresholds):
        threshold = index / (num_thresholds - 1)

        def thresholded_intent(row: ThresholdExample) -> str | None:
            if row.predicted_intent is None or row.confidence < threshold:
                return None
            return row.predicted_intent

        positive_accuracy = sum(
            thresholded_intent(row) == row.expected_intent for row in positive
        ) / len(positive)
        negative_accuracy = sum(
            thresholded_intent(row) is None for row in negative
        ) / len(negative)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                positive_accuracy=positive_accuracy,
                negative_accuracy=negative_accuracy,
                balanced_accuracy=(positive_accuracy + negative_accuracy) / 2,
            )
        )
    return points


def _step_path(
    points: Sequence[ThresholdPoint],
    *,
    value_field: str,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    coordinates = [
        (
            left + point.threshold * width,
            top + (1.0 - float(getattr(point, value_field))) * height,
        )
        for point in points
    ]
    commands = [f"M {coordinates[0][0]:.2f} {coordinates[0][1]:.2f}"]
    for x, y in coordinates[1:]:
        commands.extend((f"H {x:.2f}", f"V {y:.2f}"))
    return " ".join(commands)


def render_svg(
    points: Sequence[ThresholdPoint],
    *,
    positive_count: int,
    negative_count: int,
    title: str,
) -> str:
    """Render a dependency-free SVG chart."""

    canvas_width, canvas_height = 1000, 640
    left, top, plot_width, plot_height = 100.0, 90.0, 840.0, 450.0
    bottom = top + plot_height
    positive_path = _step_path(
        points,
        value_field="positive_accuracy",
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
    )
    negative_path = _step_path(
        points,
        value_field="negative_accuracy",
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
    )
    grid: list[str] = []
    for index in range(6):
        value = index / 5
        x = left + value * plot_width
        y = bottom - value * plot_height
        grid.extend(
            (
                f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" '
                f'x2="{x:.2f}" y2="{bottom:.2f}"/>',
                f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" '
                f'x2="{left + plot_width:.2f}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{x:.2f}" y="{bottom + 28:.2f}" '
                f'text-anchor="middle">{value:.1f}</text>',
                f'<text class="tick" x="{left - 14:.2f}" y="{y + 5:.2f}" '
                f'text-anchor="end">{value:.1f}</text>',
            )
        )
    safe_title = escape(title)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}"
     viewBox="0 0 {canvas_width} {canvas_height}" role="img">
  <title>{safe_title}</title>
  <style>
    text {{ font-family: Inter, "Noto Sans", Arial, sans-serif; fill: #263042; }}
    .title {{ font-size: 24px; font-weight: 650; }}
    .subtitle {{ font-size: 14px; fill: #667085; }}
    .grid {{ stroke: #e4e7ec; stroke-width: 1; }}
    .axis {{ stroke: #667085; stroke-width: 1.5; }}
    .tick {{ font-size: 13px; fill: #667085; }}
    .axis-label {{ font-size: 15px; font-weight: 600; }}
    .legend {{ font-size: 14px; font-weight: 600; }}
    .curve {{ fill: none; stroke-width: 3; stroke-linejoin: round; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text class="title" x="{left:.0f}" y="38">{safe_title}</text>
  <text class="subtitle" x="{left:.0f}" y="63">Positive n={positive_count} · Negative n={negative_count}</text>
  {''.join(grid)}
  <line class="axis" x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{bottom}"/>
  <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>
  <path class="curve" stroke="#2563eb" d="{positive_path}"/>
  <path class="curve" stroke="#ea580c" d="{negative_path}"/>
  <text class="axis-label" x="{left + plot_width / 2:.0f}" y="606" text-anchor="middle">Threshold</text>
  <text class="axis-label" x="28" y="{top + plot_height / 2:.0f}" text-anchor="middle"
        transform="rotate(-90 28 {top + plot_height / 2:.0f})">Accuracy</text>
  <line x1="{left + 545}" y1="55" x2="{left + 577}" y2="55" stroke="#2563eb" stroke-width="3"/>
  <text class="legend" x="{left + 586}" y="60">Positive accuracy</text>
  <line x1="{left + 710}" y1="55" x2="{left + 742}" y2="55" stroke="#ea580c" stroke-width="3"/>
  <text class="legend" x="{left + 751}" y="60">Negative accuracy</text>
</svg>
'''


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output).expanduser()
    if output_path.suffix.lower() != ".svg":
        raise RouterDataError("--output must use the .svg extension")
    examples = load_examples(args.predictions, args.labels)
    points = compute_curve(examples, num_thresholds=args.num_thresholds)
    positive_count = sum(row.expected_intent is not None for row in examples)
    negative_count = len(examples) - positive_count
    best = max(points, key=lambda point: point.balanced_accuracy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_svg(
            points,
            positive_count=positive_count,
            negative_count=negative_count,
            title=args.title,
        ),
        encoding="utf-8",
    )
    metrics_path = (
        Path(args.metrics_output).expanduser()
        if args.metrics_output
        else output_path.with_suffix(".json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "examples": len(examples),
                "positive_examples": positive_count,
                "negative_examples": negative_count,
                "best_balanced_point": asdict(best),
                "points": [asdict(point) for point in points],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[threshold] examples={len(examples)}")
    print(
        "[threshold] best balanced accuracy="
        f"{best.balanced_accuracy:.4f} at threshold={best.threshold:.4f}"
    )
    print(f"[threshold] plot={output_path}")
    print(f"[threshold] metrics={metrics_path}")


if __name__ == "__main__":
    main()
