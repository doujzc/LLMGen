# Repository Guidelines

## Scope and structure

This repository only trains direct candidate-name Top1 routers. Pure data and prompt
logic lives in `src/llmgen/top1.py`; experiment lifecycle and diagnostics live in
`src/llmgen/experiment.py`, `src/llmgen/diagnostics.py`, and
`src/llmgen/evaluation.py`. The Python training entry point is
`scripts/train_top1.py`, `scripts/train_top1.sh` provides the configured launcher,
and `scripts/evaluate_top1.py` records independent Top1 evaluation runs. Runtime
settings live in `configs/`, user datasets in `data_top1/`, and tests in `tests/`.
Treat `data_top1/` and `runs/` as datasets or generated artifacts.

## Development

Use four-space Python indentation, type hints on public interfaces, concise docstrings,
and `snake_case` names. Shell scripts must use `set -euo pipefail` and quote expansions.
Every commit subject must begin with `[training]`, `[data]`, or `[docs]`.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv run --no-sync python -m unittest discover -s tests -v
bash -n scripts/train_top1.sh
uv run --no-sync python -m compileall -q src scripts tests
```

Full training is expensive; test data validation and encoding independently. Never
silently relax prompt fitting or candidate-token quality gates.

## Modeling constraint

Never implement routing, relevance, filtering, labeling, evaluation, or fallback
behavior with keyword lists or regular expressions. Use explicit structured signals
or learned/model-based methods.

## Safety

Do not commit API keys, local model paths, weights, unreviewed/raw training data, or
`runs/` outputs. A reviewed, versioned training dataset may be committed when the
user explicitly requests it and its provenance and validation summary are included.
Keep raw LLM responses, rejected attempts, caches, and other intermediate generation
artifacts ignored.
