from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from llmgen import SkillRecord, TokenizerConfig
from llmgen.tokenization import BalancedSkillTokenizer


def embedded_skill(
    skill_id: str,
    embedding: tuple[float, ...],
    collaborative_embedding: tuple[float, ...] = (),
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=skill_id,
        embedding=embedding,
        collaborative_embedding=collaborative_embedding,
    )


def corpus(size: int = 12) -> list[SkillRecord]:
    # Deliberately non-symmetric values avoid unstable distance ties.
    return [
        embedded_skill(
            f"skill-{i:02d}",
            (float(i), float((i * i + 3) % 11), float((i * 7 + 1) % 13)),
        )
        for i in range(size)
    ]


@pytest.mark.parametrize("levels", [1, 2, 4])
def test_balanced_tokenizer_supports_arbitrary_levels(levels: int) -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=levels,
            branching_factors=(2,) * levels,
            random_seed=11,
            clustering_iterations=8,
        )
    )
    tokenizer.fit(corpus())

    assert all(
        len(tokenizer.encode(record.skill_id).indices) == levels
        for record in corpus()
    )
    assert len(tokenizer.codebooks) == levels


def test_sinkhorn_balances_each_configured_level() -> None:
    records = corpus(12)
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=3,
            branching_factors=(3, 3, 3),
            balance_scope="all",
            random_seed=5,
            clustering_iterations=10,
        )
    )
    tokenizer.fit(records)

    codes = [tokenizer.encode(record.skill_id).indices for record in records]
    for level in range(3):
        counts = Counter(code[level] for code in codes)
        assert set(counts) == {0, 1, 2}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_fit_is_deterministic_for_a_fixed_seed() -> None:
    records = corpus()
    config = TokenizerConfig(
        strategy="balanced",
        num_levels=3,
        branching_factors=(3, 2, 2),
        random_seed=42,
        clustering_iterations=8,
    )
    first = BalancedSkillTokenizer(config)
    second = BalancedSkillTokenizer(config)

    first.fit(records)
    second.fit(records)

    assert {
        record.skill_id: first.encode(record.skill_id).indices for record in records
    } == {
        record.skill_id: second.encode(record.skill_id).indices for record in records
    }


def test_dynamic_add_uses_frozen_codebooks_and_delete_updates_trie() -> None:
    records = corpus()
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=2,
            branching_factors=(3, 3),
            random_seed=3,
            dynamic_balance_weight=0.05,
        )
    )
    tokenizer.fit(records)
    old_codes = {
        record.skill_id: tokenizer.encode(record.skill_id) for record in records
    }
    old_codebooks = [codebook.copy() for codebook in tokenizer.codebooks]

    added = tokenizer.add(
        embedded_skill("skill-new", (12.5, 3.25, 8.75))
    )

    assert all(
        tokenizer.encode(skill_id) == code for skill_id, code in old_codes.items()
    )
    for before, after in zip(old_codebooks, tokenizer.codebooks):
        assert (before == after).all()
    assert "skill-new" in tokenizer.decode(added.indices)

    assert tokenizer.remove("skill-new") is True
    assert "skill-new" not in tokenizer.decode(added.indices)


def test_collaborative_embedding_dimensions_are_validated() -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=2,
            branching_factors=(2, 2),
            collaborative_weight=0.5,
        )
    )

    with pytest.raises(ValueError, match="collaborative_embedding"):
        tokenizer.fit(
            [
                embedded_skill("a", (1.0, 0.0), (1.0, 2.0)),
                embedded_skill("b", (0.0, 1.0), (1.0, 2.0, 3.0)),
            ]
        )


def test_full_collaborative_weight_requires_collaborative_data() -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=1,
            branching_factors=(2,),
            collaborative_weight=1.0,
        )
    )

    with pytest.raises(ValueError, match="requires collaborative_embedding"):
        tokenizer.fit(corpus(4))


def test_codebook_can_reserve_more_codes_than_initial_skills() -> None:
    records = corpus(3)
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=2,
            branching_factors=(8, 5),
            random_seed=9,
            clustering_iterations=4,
        )
    )

    tokenizer.fit(records)

    assert tokenizer.codebooks[0].shape[0] == 8
    assert tokenizer.codebooks[1].shape[0] == 5
    assert all(len(tokenizer.encode(record.skill_id).indices) == 2 for record in records)


def test_previewing_a_new_skill_does_not_change_usage() -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=2,
            branching_factors=(2, 2),
        )
    )
    tokenizer.fit(corpus(6))
    before = tokenizer.usage_counts

    tokenizer.encode(embedded_skill("preview", (4.2, 1.3, 9.1)))

    assert tokenizer.usage_counts == before


def test_failed_refit_restores_previous_codebook_and_registry() -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=1,
            branching_factors=(1,),
            bucket_capacity=1,
            overflow_policy="error",
        )
    )
    tokenizer.fit([embedded_skill("old", (1.0, 0.0, 0.0))])
    old_code = tokenizer.encode("old")
    old_codebook = tokenizer.codebooks[0]
    old_usage = tokenizer.usage_counts

    with pytest.raises(RuntimeError, match="capacity"):
        tokenizer.fit(
            [
                embedded_skill("new-a", (0.0, 1.0, 0.0)),
                embedded_skill("new-b", (0.0, 0.0, 1.0)),
            ]
        )

    assert tokenizer.encode("old") == old_code
    assert (tokenizer.codebooks[0] == old_codebook).all()
    assert tokenizer.usage_counts == old_usage


def test_concurrent_double_remove_is_one_atomic_transaction() -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=2,
            branching_factors=(2, 2),
        )
    )
    records = corpus(6)
    tokenizer.fit(records)
    barrier = Barrier(3)

    def remove_once() -> bool:
        barrier.wait()
        return tokenizer.remove(records[0].skill_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(remove_once) for _ in range(2)]
        barrier.wait()
        results = sorted(future.result() for future in futures)

    assert results == [False, True]
    assert records[0].skill_id not in tokenizer.registry
    assert all(sum(level) == len(records) - 1 for level in tokenizer.usage_counts)
