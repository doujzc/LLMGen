# ClawHub multi-skill query data

This directory contains generated routing data for one frozen 568-skill ClawHub candidate set. The set was filtered from a Top-1,000 source crawl; filtered source records never enter the training catalog or inference decoder. Intermediate generation artifacts are excluded from Git, while the reviewed files under `final/` are versioned and immediately trainable after clone.

Create `~/llm_api.txt` outside the repository:

```yaml
base_url: http://host:port/v1
api_key: YOUR_KEY
model: Qwen3.6-Plus
```

Run the complete pipeline:

```bash
bash scripts/run_clawhub_data.sh
```

Or rerun individual stages:

```bash
.venv/bin/python scripts/clawhub_data/00_profile_skills.py
.venv/bin/python scripts/clawhub_data/01_build_workflows.py
.venv/bin/python scripts/clawhub_data/01b_apply_recovery_workflows.py
.venv/bin/python scripts/clawhub_data/02a_generate_alignment_queries.py --help
.venv/bin/python scripts/clawhub_data/03a_review_alignment_queries.py --help
.venv/bin/python scripts/clawhub_data/03a2_backfill_alignment.py --help
.venv/bin/python scripts/clawhub_data/02_generate_queries.py
.venv/bin/python scripts/clawhub_data/03_review_queries.py
.venv/bin/python scripts/clawhub_data/03b_build_coverage_workflows.py
.venv/bin/python scripts/clawhub_data/04_export_dataset.py
.venv/bin/python scripts/clawhub_data/04a_export_alignment.py --help
.venv/bin/python scripts/clawhub_data/05_validate_dataset.py --help
```

Defaults use Qwen3.6-Plus for profiling/generation and GLM-5.1 for independent review, with model thinking disabled. The API key is never copied into output metadata.

The checked-in `final/` snapshot remains unchanged until a complete rebuild passes its gates. During a rebuild, every Skill in the input catalog is retained; `mobile_fit` is metadata and never filters candidates. The pipeline adds targeted coverage workflows for undercovered Skills and, by default, requires at least 10 independently reviewed train positives per candidate. A failed gate writes `coverage_failure.json` without replacing the existing final dataset.

Coverage controls for a future rebuild:

```bash
WORKFLOWS_PER_SKILL=3 \
MIN_TRAIN_POSITIVES_PER_SKILL=10 \
COVERAGE_ROUNDS=3 \
COVERAGE_OVERSAMPLE_FACTOR=3.0 \
  bash scripts/run_clawhub_data.sh
```

Final files under `final/`:

- `skills.jsonl`: the single shared 568-skill candidate registry.
- `queries_{train,validation,test}.jsonl`: query text and multi-skill targets.
- `queries_alignment.jsonl`: independently reviewed one-query-to-one-skill curriculum data.
- `qrels_{train,validation,test}.jsonl`: one positive relevance row per query/skill pair.
- `queries.jsonl`: accepted examples with evidence spans, split, and review scores.
- `manifest.json`: counts, coverage, domain distribution, and quality audit.

Train directly from the versioned files:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
bash scripts/run_clawhub_full.sh
```

The default configuration is `configs/clawhub.env`. Each training stage can be rerun independently through `scripts/clawhub_train/01_prepare.sh` to `07_evaluate.sh`.
