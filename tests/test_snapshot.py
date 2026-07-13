from copy import deepcopy

import pytest

from llmgen import (
    SerializationError,
    SkillRecord,
    TokenizerConfig,
    tokenizer_from_snapshot,
)
from llmgen.tokenization import (
    BalancedSkillTokenizer,
    InterpretableSkillTokenizer,
)


def test_interpretable_snapshot_round_trip() -> None:
    tokenizer = InterpretableSkillTokenizer(
        TokenizerConfig(
            strategy="interpretable",
            num_levels=3,
            branching_factors=(4, 4, 4),
            codebook_version="taxonomy-v3",
        )
    )
    tokenizer.fit(
        [
            SkillRecord(
                skill_id="calendar.read",
                name="Read calendar",
                hierarchy=("productivity", "calendar", "read"),
            )
        ]
    )

    restored = tokenizer_from_snapshot(tokenizer.snapshot())

    assert restored.encode("calendar.read") == tokenizer.encode("calendar.read")
    added = restored.add(
        SkillRecord(
            skill_id="calendar.create",
            name="Create event",
            hierarchy=("productivity", "calendar", "mutate"),
        )
    )
    assert restored.decode(added.indices) == ("calendar.create",)


def test_balanced_snapshot_round_trip() -> None:
    records = [
        SkillRecord(skill_id=f"s{i}", name=f"s{i}", embedding=(i, i * i, 1.0))
        for i in range(6)
    ]
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=2,
            branching_factors=(2, 3),
            codebook_version="balanced-v3",
            random_seed=17,
        )
    )
    tokenizer.fit(records)

    restored = tokenizer_from_snapshot(tokenizer.snapshot())

    assert all(
        restored.encode(record.skill_id) == tokenizer.encode(record.skill_id)
        for record in records
    )
    added = restored.add(
        SkillRecord(skill_id="new", name="new", embedding=(3.2, 7.1, 1.0))
    )
    assert "new" in restored.decode(added.indices)


def test_json_file_round_trip(tmp_path) -> None:
    tokenizer = InterpretableSkillTokenizer(
        TokenizerConfig(
            strategy="interpretable",
            num_levels=2,
            branching_factors=(2, 2),
        )
    )
    tokenizer.fit(
        [
            SkillRecord(
                skill_id="calendar.read",
                hierarchy=("productivity", "calendar"),
            )
        ]
    )
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    restored = InterpretableSkillTokenizer.load(path)

    assert restored.snapshot() == tokenizer.snapshot()


def test_balanced_snapshot_rejects_usage_that_disagrees_with_registry() -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=2,
            branching_factors=(2, 2),
        )
    )
    tokenizer.fit(
        [
            SkillRecord("a", embedding=(1.0, 0.0)),
            SkillRecord("b", embedding=(0.0, 1.0)),
        ]
    )
    snapshot = deepcopy(tokenizer.snapshot())
    snapshot["strategy_state"]["usage"][0][0] += 1

    with pytest.raises(SerializationError, match="does not match registry"):
        tokenizer_from_snapshot(snapshot)


def test_balanced_snapshot_does_not_truncate_float_usage() -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=1,
            branching_factors=(2,),
        )
    )
    tokenizer.fit(
        [
            SkillRecord("a", embedding=(1.0, 0.0)),
            SkillRecord("b", embedding=(0.0, 1.0)),
        ]
    )
    snapshot = deepcopy(tokenizer.snapshot())
    snapshot["strategy_state"]["usage"][0][0] = 1.5

    with pytest.raises(SerializationError, match="integer counts"):
        tokenizer_from_snapshot(snapshot)


def test_interpretable_snapshot_cross_checks_taxonomy_and_skill_cards() -> None:
    tokenizer = InterpretableSkillTokenizer(
        TokenizerConfig(
            strategy="interpretable",
            num_levels=2,
            branching_factors=(2, 2),
        )
    )
    tokenizer.fit(
        [
            SkillRecord(
                "calendar.read",
                hierarchy=("productivity", "calendar"),
            )
        ]
    )
    snapshot = deepcopy(tokenizer.snapshot())
    snapshot["strategy_state"]["nodes"][0]["labels"][0]["label"] = "wrong"

    with pytest.raises(SerializationError, match="does not encode"):
        tokenizer_from_snapshot(snapshot)
