from __future__ import annotations

from pathlib import Path

from scripts.prepare_clawhub import validate_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_versioned_clawhub_dataset_is_directly_trainable() -> None:
    validated = validate_dataset(REPO_ROOT / "data/clawhub_training/final")
    assert validated["counts"] == {
        "skills": 1000,
        "queries_train": 3353,
        "qrels_train": 9707,
        "queries_validation": 448,
        "qrels_validation": 1344,
        "queries_test": 399,
        "qrels_test": 1197,
    }
    assert len(validated["skill_ids"]) == len(set(validated["skill_ids"]))
