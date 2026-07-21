"""Small executable example for both hierarchical tokenizer strategies."""

from llmgen import SkillRecord, TokenizerConfig
from llmgen.tokenization import (
    BalancedSkillTokenizer,
    InterpretableSkillTokenizer,
)


def interpretable_example() -> None:
    tokenizer = InterpretableSkillTokenizer(
        TokenizerConfig(
            strategy="interpretable",
            num_levels=2,
            branching_factors=(8, 8),
            overflow_policy="error",
        )
    )
    tokenizer.fit(
        [
            SkillRecord(
                skill_id="calendar.read",
                hierarchy=("productivity", "calendar-read"),
            ),
            SkillRecord(
                skill_id="calendar.create",
                hierarchy=("productivity", "calendar-mutate"),
            ),
        ]
    )
    code = tokenizer.encode("calendar.read")
    print("interpretable:", code.tokens, "->", tokenizer.decode(code.indices))


def balanced_example() -> None:
    tokenizer = BalancedSkillTokenizer(
        TokenizerConfig(
            strategy="balanced",
            num_levels=2,
            branching_factors=(2, 2),
            balance_scope="all",
            random_seed=7,
        )
    )
    tokenizer.fit(
        [
            SkillRecord("calendar.read", embedding=(0.9, 0.1, 0.0)),
            SkillRecord("calendar.create", embedding=(0.8, 0.2, 0.0)),
            SkillRecord("git.read", embedding=(0.0, 0.2, 0.8)),
            SkillRecord("git.create-pr", embedding=(0.0, 0.1, 0.9)),
        ]
    )
    new_code = tokenizer.add(
        SkillRecord("calendar.reschedule", embedding=(0.85, 0.15, 0.0))
    )
    print("balanced:", new_code.tokens, "->", tokenizer.decode(new_code.indices))


if __name__ == "__main__":
    interpretable_example()
    balanced_example()
