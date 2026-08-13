#!/usr/bin/env python3
"""Visualize positive and negative Top1 routing accuracy by threshold."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
        help="Destination interactive HTML curve.",
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


def write_interactive_plot(
    points: Sequence[ThresholdPoint],
    *,
    positive_count: int,
    negative_count: int,
    title: str,
    best: ThresholdPoint,
    destination: str | Path,
) -> None:
    """Write a standalone Plotly chart with shared hover values."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise SystemExit(
            "Interactive threshold plots require plotly. Install with "
            "python -m pip install 'plotly>=6,<7'."
        ) from exc

    thresholds = [point.threshold for point in points]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=thresholds,
            y=[point.positive_accuracy for point in points],
            mode="lines",
            name=f"Positive accuracy (n={positive_count})",
            line={"color": "#2563eb", "width": 3, "shape": "hv"},
            hovertemplate=(
                "Threshold: %{x:.3f}<br>Positive accuracy: %{y:.2%}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=thresholds,
            y=[point.negative_accuracy for point in points],
            mode="lines",
            name=f"Negative accuracy (n={negative_count})",
            line={"color": "#ea580c", "width": 3, "shape": "hv"},
            hovertemplate=(
                "Threshold: %{x:.3f}<br>Negative accuracy: %{y:.2%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"{title}<br><sup>Best balanced accuracy "
                f"{best.balanced_accuracy:.2%} at threshold "
                f"{best.threshold:.3f}</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
        },
        xaxis={
            "title": "Threshold",
            "range": [0, 1],
            "tickformat": ".1f",
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
        },
        yaxis={
            "title": "Accuracy",
            "range": [0, 1.01],
            "tickformat": ".0%",
        },
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 80, "r": 40, "t": 120, "b": 75},
        height=650,
    )
    figure.write_html(
        str(destination),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output).expanduser()
    if output_path.suffix.lower() != ".html":
        raise RouterDataError("--output must use the .html extension")
    examples = load_examples(args.predictions, args.labels)
    points = compute_curve(examples, num_thresholds=args.num_thresholds)
    positive_count = sum(row.expected_intent is not None for row in examples)
    negative_count = len(examples) - positive_count
    best = max(points, key=lambda point: point.balanced_accuracy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_interactive_plot(
        points,
        positive_count=positive_count,
        negative_count=negative_count,
        title=args.title,
        best=best,
        destination=output_path,
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
