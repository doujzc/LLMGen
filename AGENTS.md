# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/llmgen/`; adapted ToolWeaver code is isolated in
`vendor/toolweaver/`. Stage-oriented entry points live in `scripts/`, including
dataset flows in `scripts/skillret/`, `scripts/clawhub_data/`, and
`scripts/incremental/`. Runtime settings belong in `configs/`. The API and browser
UI are in `web_server/`; documentation is in `docs/`. Tests and small fixtures live
in `tests/` and `tests/fixtures/`. Treat `data/`, `data_light/`, `data_top1/`, and `runs/` as
datasets or generated artifacts, not application modules.

## Development Threads & Commit Coordination

Every commit subject must start with exactly one thread prefix:

- `[training]`: tokenizer/router optimization, checkpoints, DeepSpeed, and metrics.
- `[inference]`: decoding, model export, CLI/API inference, and runtime performance.
- `[data]` (data construction): catalogs, query generation, validation, and splits.
- `[docs]`: README files, design notes, diagrams, and contributor guidance.
- `[frontend]`: Web console markup, styling, and browser-side behavior.

Continue the history's short, imperative summaries, for example
`[inference] Add constrained batch decoding`. For cross-thread changes, select the
primary owner and explain secondary effects in the body. Keep commits scoped.

## Build, Test, and Development Commands

```bash
python -m pip install -e '.[train,test]'       # editable development install
pytest                                         # complete test suite
pytest tests/test_router.py -q                 # focused test module
bash scripts/router_pipeline.sh --help         # list pipeline stages
bash scripts/router_pipeline.sh light paths    # inspect resolved paths/config
```

Full training is expensive; run the smallest relevant stage while iterating. Never
silently relax tokenizer quality gates.

## Coding Style & Naming Conventions

Use four-space Python indentation, type hints on public interfaces, and concise
docstrings for non-obvious behavior. Follow `snake_case` for functions/files,
`PascalCase` for classes, and `UPPER_SNAKE_CASE` for environment variables. Shell
scripts should use `set -euo pipefail`, quote expansions, and expose defaults through
environment variables. Match the existing two-space style in JavaScript and CSS.

## Modeling Constraints

Never implement project behavior or inference methods using keyword lists or regular-
expression matching. This permanent constraint applies to routing, intent detection,
conversation relevance, filtering, labeling, evaluation, and fallback heuristics. Use
explicit structured signals or learned/model-based methods instead. If neither is
available, state the limitation and request a design decision rather than adding a
keyword or regex approximation.

## Testing Guidelines

Pytest is configured through `pyproject.toml`. Name files `test_<area>.py` and tests
`test_<behavior>`. Add regression tests for bug fixes and fixture-based tests for
data transformations. There is no fixed coverage threshold, but changed branches
should be exercised. Preserve the invariant that training, evaluation, export, and
Web decoding share one candidate registry.

## Pull Requests & Configuration Safety

PRs should state the thread, motivation, commands run, and compatibility impact.
Include screenshots for frontend changes and key metrics for training changes. Do
not commit API keys, local model paths, weights, or `runs/` outputs; use environment
overrides.
