# Multi-turn Top1 Data

Place user-provided JSONL files in this directory, or point `TOP1_TRAIN_DATA`,
`TOP1_VALIDATION_DATA`, and `TOP1_TEST_DATA` at files elsewhere. No source-data
conversion step is required.

Each line must contain structured messages and one direct candidate-name target:

```json
{"messages":[{"role":"user","content":"推荐一款耳机"},{"role":"assistant","content":"预算是多少？"},{"role":"user","content":"500元以内"}],"target_candidate_name":"Ecommerce"}
```

`messages` supports `user`, `assistant`, and `tool`; the final non-system message
must be `user`. Training requires `target_candidate_name`; inference does not.
The target must exist in `configs/top1_candidates.json`. Optional metadata fields,
including `id`, `query_id`, and `scenario_family`, are ignored by validation and
training. Splits may contain repeated conversations or shared scenario families.
The training split does not need to contain supervision for every registered
candidate.

Validate the files before loading a model:

```bash
TOP1_TRAIN_DATA=/data/router/train.jsonl \
TOP1_VALIDATION_DATA=/data/router/validation.jsonl \
  bash scripts/top1/00_validate.sh
```
