from __future__ import annotations

import json
from pathlib import Path

from llmgen.direct_router import load_candidate_registry
from llmgen.router import read_jsonl


DATA_DIR = Path("data_promptgen/final")


def test_tracked_promptgen_dataset_uses_only_the_seven_deployable_names() -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    routes = load_candidate_registry(DATA_DIR / "candidate_registry.json")
    names = {route.name for route in routes}

    assert manifest["rows"] == 5000
    assert names == {
        "StockAdvice",
        "StockOther",
        "StockQuery",
        "ProductOther",
        "Ecommerce",
        "ChitChat",
        "NoAvailable",
    }
    assert set(manifest["candidate_counts"]) == names
    assert set(manifest["virtual_candidate_names"]) == {
        "StockAdvice",
        "StockOther",
        "ProductOther",
        "ChitChat",
        "NoAvailable",
    }


def test_promptgen_splits_preserve_messages_and_family_isolation() -> None:
    family_split: dict[str, str] = {}
    total = 0
    multi_turn = 0
    for split in ("train", "validation", "test"):
        rows = read_jsonl(DATA_DIR / f"{split}.jsonl")
        total += len(rows)
        multi_turn += sum(len(row["messages"]) > 1 for row in rows)
        for row in rows:
            assert row["split"] == split
            assert row["messages"][-1]["role"] == "user"
            assert "input_text" not in row
            family = row["scenario_family"]
            assert family_split.setdefault(family, split) == split

    assert total == 5000
    assert multi_turn == 2086
