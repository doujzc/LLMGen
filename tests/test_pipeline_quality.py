"""Tests for portable evaluation and export report evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmgen.pipeline.quality import (
    QualityValidationError,
    audit_evaluation,
    build_run_lineage_payload,
    model_file_manifest,
    write_export_reports,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _decode_map(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "num_levels": 2,
            "skills": {"a": {"name": "A"}, "b": {"name": "B"}},
            "skill_to_code": {
                "a": {"tokens": ["<0>", "<0>"]},
                "b": {"tokens": ["<1>", "<1>"]},
            },
            "paths": [
                {"tokens": ["<0>", "<0>"], "skill_ids": ["a"]},
                {"tokens": ["<1>", "<1>"], "skill_ids": ["b"]},
            ],
            "virtual_tokens": ["<0>", "<1>"],
        },
    )


def test_quality_report_validates_paths_and_writes_export_evidence(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    decode_map = tmp_path / "skill_decode_map.json"
    _write_json(metrics, {"metrics": {"recall@1": 0.8}})
    _write_jsonl(
        predictions,
        [
            {
                "query_id": "q1",
                "paths": [{"code_tokens": ["<0>", "<0>"], "skill_ids": ["a"]}],
            }
        ],
    )
    _decode_map(decode_map)

    audit = audit_evaluation(
        metrics_path=metrics,
        predictions_path=predictions,
        decode_map_path=decode_map,
        metric_thresholds={"recall@1": 0.75},
    )
    assert audit.quality_gates["passed"] is True
    assert audit.failed_examples == ()

    model = tmp_path / "model"
    (model / "nested").mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "nested" / "model.safetensors").write_bytes(b"weights")
    manifest = model_file_manifest(model)
    assert [item["path"] for item in manifest["files"]] == [
        "config.json",
        "nested/model.safetensors",
    ]
    lineage = build_run_lineage_payload(
        run_id="run-1",
        candidate_input_sha256="candidate-hash",
        config_sha256="config-hash",
        model_manifest=manifest,
        artifact_lineage={"model.retrieval": {"sha256": "model-hash"}},
        git_commit="abc123",
    )
    paths = write_export_reports(
        tmp_path / "report",
        audit=audit,
        model_manifest=manifest,
        run_lineage=lineage,
    )
    assert json.loads(paths["evaluation"].read_text())["prediction_count"] == 1
    assert json.loads(paths["run_lineage"].read_text())["model_files"]["file_count"] == 2
    assert paths["failed_examples"].read_text() == ""


def test_quality_report_marks_unknown_or_mismatched_decode_paths_failed(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    decode_map = tmp_path / "skill_decode_map.json"
    _write_json(metrics, {"metrics": {"recall@1": 0.9}})
    _write_jsonl(
        predictions,
        [
            {"query_id": "unknown", "paths": [{"skill_ids": ["missing"]}]},
            {
                "query_id": "mismatch",
                "paths": [{"code_tokens": ["<0>", "<0>"], "skill_ids": ["b"]}],
            },
        ],
    )
    _decode_map(decode_map)

    audit = audit_evaluation(
        metrics_path=metrics,
        predictions_path=predictions,
        decode_map_path=decode_map,
        required_format_valid_rate=1.0,
    )
    assert audit.quality_gates["passed"] is False
    assert audit.quality_gates["gates"]["decode_map_paths"] is False
    assert audit.evaluation["format_valid_rate"] == 0.0
    assert "not a decode-map path" in audit.failed_examples[0]["failure_reasons"][0]
    assert "disagree with decode-map path" in audit.failed_examples[1]["failure_reasons"][0]


def test_quality_report_rejects_malformed_metric_document(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    decode_map = tmp_path / "skill_decode_map.json"
    _write_json(metrics, {"recall@1": 0.9})
    _write_jsonl(predictions, [{"query_id": "q1", "paths": [{"skill_ids": ["a"]}]}])
    _decode_map(decode_map)

    with pytest.raises(QualityValidationError, match="metrics object"):
        audit_evaluation(
            metrics_path=metrics,
            predictions_path=predictions,
            decode_map_path=decode_map,
        )
