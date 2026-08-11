from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from llmgen.router import RouterDataError, write_jsonl
from scripts.top1.validate_data import validate_data_files


CANDIDATE_NAMES = (
    "StockAdvice",
    "StockOther",
    "StockQuery",
    "ProductOther",
    "Ecommerce",
    "ChitChat",
    "NoAvailable",
)


def _row(index: int, target: str) -> dict:
    return {
        "id": f"q-{index}",
        "messages": [
            {"role": "user", "content": f"context {index}"},
            {"role": "assistant", "content": "请继续"},
            {"role": "user", "content": f"request {index}"},
        ],
        "target_candidate_name": target,
    }


def test_validator_accepts_user_jsonl_and_unlabeled_test_rows(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, [_row(index, name) for index, name in enumerate(CANDIDATE_NAMES)])
    write_jsonl(validation, [_row(20, "StockQuery")])
    write_jsonl(
        test,
        [
            {
                "id": "q-30",
                "messages": [{"role": "user", "content": "unlabeled request"}],
            }
        ],
    )

    report = validate_data_files(
        candidate_registry="configs/top1_candidates.json",
        split_paths={"train": train, "validation": validation, "test": test},
    )

    assert report["splits"]["train"]["rows"] == 7
    assert report["splits"]["train"]["multi_turn_rows"] == 7
    assert report["splits"]["test"]["labeled_rows"] == 0
    assert set(report["splits"]["train"]["candidate_counts"]) == set(
        CANDIDATE_NAMES
    )


def test_validator_ignores_missing_invalid_and_duplicate_ids(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    rows = [_row(index, name) for index, name in enumerate(CANDIDATE_NAMES)]
    rows[0].pop("id")
    rows[1]["id"] = "duplicate"
    rows[2]["id"] = "duplicate"
    rows[3]["id"] = {"arbitrary": "metadata"}
    rows[4]["query_id"] = 42
    write_jsonl(train, rows)

    report = validate_data_files(
        candidate_registry="configs/top1_candidates.json",
        split_paths={"train": train},
    )

    assert report["splits"]["train"]["rows"] == len(CANDIDATE_NAMES)


def test_validator_rejects_conversation_leakage_across_splits(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train_rows = [_row(index, name) for index, name in enumerate(CANDIDATE_NAMES)]
    leaked = dict(train_rows[0])
    leaked["id"] = "different-id"
    write_jsonl(train, train_rows)
    write_jsonl(validation, [leaked])

    with pytest.raises(RouterDataError, match="also occurs in train"):
        validate_data_files(
            candidate_registry="configs/top1_candidates.json",
            split_paths={"train": train, "validation": validation},
        )


def test_validator_requires_training_coverage_for_every_candidate(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    write_jsonl(train, [_row(0, "StockQuery")])

    with pytest.raises(RouterDataError, match="no supervision"):
        validate_data_files(
            candidate_registry="configs/top1_candidates.json",
            split_paths={"train": train},
        )


def test_top1_validation_shell_reads_user_data_paths_directly(tmp_path) -> None:
    train = tmp_path / "custom-multiturn.jsonl"
    write_jsonl(
        train,
        [_row(index, name) for index, name in enumerate(CANDIDATE_NAMES)],
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON": sys.executable,
            "TOP1_TRAIN_DATA": str(train),
            "TOP1_VALIDATION_DATA": "",
            "TOP1_TEST_DATA": "",
            "TOP1_CANDIDATE_REGISTRY": str(
                Path("configs/top1_candidates.json").resolve()
            ),
        }
    )

    completed = subprocess.run(
        ["bash", "scripts/top1/00_validate.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"rows": 7' in completed.stdout
    assert str(train) in completed.stdout
