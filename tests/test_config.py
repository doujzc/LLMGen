import pytest

from llmgen import TokenizerConfig, create_tokenizer
from llmgen.tokenization import BalancedSkillTokenizer, InterpretableSkillTokenizer


def test_num_levels_must_match_branching_factors() -> None:
    with pytest.raises(ValueError, match="branching_factors"):
        TokenizerConfig(
            strategy="interpretable",
            num_levels=3,
            branching_factors=(8, 8),
        )


def test_token_format_must_encode_level_and_index() -> None:
    with pytest.raises(ValueError, match="token_format"):
        TokenizerConfig(
            strategy="interpretable",
            num_levels=2,
            branching_factors=(8, 8),
            token_format="<SK_{index}>",
        )


def test_safe_default_rejects_silent_overflow() -> None:
    config = TokenizerConfig(
        strategy="interpretable",
        num_levels=1,
        branching_factors=(2,),
    )

    assert config.overflow_policy == "error"


def test_config_round_trip() -> None:
    config = TokenizerConfig(
        strategy="balanced",
        num_levels=4,
        branching_factors=(3, 4, 5, 6),
        codebook_version="test-v2",
        balance_scope="last",
        collaborative_weight=0.2,
    )

    restored = TokenizerConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.token_for(3, 4) == "<SK_L4_4>"


@pytest.mark.parametrize(
    ("strategy", "expected_type"),
    [
        ("interpretable", InterpretableSkillTokenizer),
        ("balanced", BalancedSkillTokenizer),
    ],
)
def test_strategy_factory(strategy, expected_type) -> None:
    config = TokenizerConfig(
        strategy=strategy,
        num_levels=2,
        branching_factors=(2, 2),
    )

    assert isinstance(create_tokenizer(config), expected_type)
