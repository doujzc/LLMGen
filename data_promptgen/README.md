# PromptGen Top1 Training Data

`final/` contains 5,000 PromptGen conversations converted for direct
candidate-name routing. The original family-level split is preserved:
3,934 train, 533 validation, and 533 test rows. Of these, 2,086 are multi-turn.

Each row keeps the structured `messages` array and adds
`target_candidate_name`. The generated output space contains two real routes
(`StockQuery`, `Ecommerce`) and exactly five virtual routes (`StockAdvice`,
`StockOther`, `ProductOther`, `ChitChat`, `NoAvailable`). PromptGen's finer
diagnostic labels are retained as `source_candidate_id` but are never model
outputs. The deterministic collapse is recorded in `final/manifest.json`.

The source snapshot SHA-256 is
`8b46dccc67bdbdbc9dd4e3ab975560290852e4f397c34b6914a9d82bd2540584`.
To rebuild from a PromptGen checkout:

```bash
PROMPTGEN_SOURCE=../PromptGen/data/xiaoyi_intent_v1.jsonl \
  bash scripts/promptgen/00_prepare.sh
```
