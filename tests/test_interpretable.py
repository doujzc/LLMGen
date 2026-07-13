import pytest

from llmgen import SkillRecord, TokenizerConfig
from llmgen.tokenization import InterpretableSkillTokenizer


def skill(skill_id: str, *hierarchy: str) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=skill_id,
        hierarchy=hierarchy,
    )


@pytest.mark.parametrize("levels", [1, 2, 4])
def test_configurable_number_of_levels(levels: int) -> None:
    config = TokenizerConfig(
        strategy="interpretable",
        num_levels=levels,
        branching_factors=(8,) * levels,
    )
    tokenizer = InterpretableSkillTokenizer(config)
    record = skill("calendar.create", *(f"level-{i}" for i in range(levels)))

    tokenizer.fit([record])
    code = tokenizer.encode(record.skill_id)

    assert len(code.indices) == levels
    assert len(code.tokens) == levels


def test_taxonomy_is_explainable_and_prefix_decodable() -> None:
    config = TokenizerConfig(
        strategy="interpretable",
        num_levels=3,
        branching_factors=(4, 4, 4),
        codebook_version="taxonomy-v1",
    )
    tokenizer = InterpretableSkillTokenizer(config)
    tokenizer.fit(
        [
            skill("calendar.create", "productivity", "calendar", "mutate"),
            skill("calendar.read", "productivity", "calendar", "read"),
            skill("git.pr", "engineering", "source-control", "pull-request"),
        ]
    )

    create_code = tokenizer.encode("calendar.create")
    read_code = tokenizer.encode("calendar.read")

    assert create_code.indices[:2] == read_code.indices[:2]
    assert create_code.indices[2] != read_code.indices[2]
    assert tokenizer.explain(read_code) == (
        "productivity",
        "calendar",
        "read",
    )
    assert tokenizer.decode(read_code.indices[:2]) == (
        "calendar.create",
        "calendar.read",
    )
    assert tokenizer.valid_next_tokens(read_code.indices[:2]) == (
        create_code.tokens[2],
        read_code.tokens[2],
    )


def test_add_and_remove_do_not_recode_existing_skills() -> None:
    config = TokenizerConfig(
        strategy="interpretable",
        num_levels=3,
        branching_factors=(4, 4, 4),
    )
    tokenizer = InterpretableSkillTokenizer(config)
    tokenizer.fit(
        [skill("calendar.read", "productivity", "calendar", "read")]
    )
    original = tokenizer.encode("calendar.read")

    added = tokenizer.add(
        skill("calendar.create", "productivity", "calendar", "mutate")
    )

    assert tokenizer.encode("calendar.read") == original
    assert tokenizer.decode(added.indices) == ("calendar.create",)
    assert tokenizer.remove("calendar.create") is True
    assert tokenizer.remove("calendar.create") is False
    assert tokenizer.decode(added.indices) == ()
    assert added.tokens[2] not in tokenizer.valid_next_tokens(added.indices[:2])

    readded = tokenizer.add(
        skill("calendar.reschedule", "productivity", "calendar", "mutate")
    )
    assert readded.indices == added.indices


def test_same_path_is_a_bucket_not_an_encoding_error() -> None:
    config = TokenizerConfig(
        strategy="interpretable",
        num_levels=2,
        branching_factors=(2, 2),
    )
    tokenizer = InterpretableSkillTokenizer(config)
    tokenizer.fit(
        [
            skill("calendar.create", "productivity", "calendar"),
            skill("calendar.update", "productivity", "calendar"),
        ]
    )

    code = tokenizer.encode("calendar.create")
    assert tokenizer.decode(code.indices) == (
        "calendar.create",
        "calendar.update",
    )


def test_path_shorter_than_num_levels_is_rejected() -> None:
    config = TokenizerConfig(
        strategy="interpretable",
        num_levels=3,
        branching_factors=(2, 2, 2),
    )
    tokenizer = InterpretableSkillTokenizer(config)

    with pytest.raises(ValueError, match="hierarchy"):
        tokenizer.fit([skill("too-short", "one", "two")])


def test_optional_bucket_capacity_can_be_a_hard_limit() -> None:
    config = TokenizerConfig(
        strategy="interpretable",
        num_levels=2,
        branching_factors=(2, 2),
        bucket_capacity=1,
        overflow_policy="error",
    )
    tokenizer = InterpretableSkillTokenizer(config)
    tokenizer.fit([skill("first", "productivity", "calendar")])

    with pytest.raises(RuntimeError, match="capacity"):
        tokenizer.add(skill("second", "productivity", "calendar"))

    assert tokenizer.decode(tokenizer.encode("first").indices) == ("first",)


def test_previewing_a_new_skill_does_not_reserve_taxonomy_slots() -> None:
    tokenizer = InterpretableSkillTokenizer(
        TokenizerConfig(
            strategy="interpretable",
            num_levels=2,
            branching_factors=(4, 4),
        )
    )
    tokenizer.fit([skill("existing", "productivity", "calendar")])
    before = tokenizer.taxonomy

    tokenizer.encode(skill("preview", "engineering", "source-control"))

    assert tokenizer.taxonomy == before


def test_failed_refit_restores_previous_taxonomy_and_registry() -> None:
    tokenizer = InterpretableSkillTokenizer(
        TokenizerConfig(
            strategy="interpretable",
            num_levels=2,
            branching_factors=(2, 2),
            bucket_capacity=1,
            overflow_policy="error",
        )
    )
    tokenizer.fit([skill("old", "old-domain", "old-action")])
    old_code = tokenizer.encode("old")
    old_taxonomy = tokenizer.taxonomy

    with pytest.raises(RuntimeError, match="capacity"):
        tokenizer.fit(
            [
                skill("new-a", "new-domain", "same"),
                skill("new-b", "new-domain", "same"),
            ]
        )

    assert tokenizer.encode("old") == old_code
    assert tokenizer.explain(old_code) == ("old-domain", "old-action")
    assert tokenizer.taxonomy == old_taxonomy
