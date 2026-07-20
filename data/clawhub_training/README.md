# ClawHub multi-skill query data

This directory contains generated routing data for the frozen 1,000-skill ClawHub candidate set. Intermediate generation artifacts are excluded from Git; the reviewed files under `final/` are versioned and immediately usable for training after clone.

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
.venv/bin/python scripts/clawhub_data/02_generate_queries.py
.venv/bin/python scripts/clawhub_data/03_review_queries.py
.venv/bin/python scripts/clawhub_data/04_export_dataset.py
```

Defaults use Qwen3.6-Plus for profiling/generation and GLM-5.1 for independent review, with model thinking disabled. The API key is never copied into output metadata.

The 1,000 skills remain the closed candidate set. Complex queries target only skills classified as high/medium mobile fit by default; forcing low-fit infrastructure skills into personal-assistant requests produces artificial labels. Stage 1 can be rerun with `--min-mobile-fit low` when full positive-target coverage is explicitly required. Stage 1b applies the audited decisions in `configs/clawhub_recovery.json`: internal meta-skills remain candidates without artificial positives, while user-visible skills that failed the first review receive coherent recovery workflows.

Final files under `final/`:

- `skills.jsonl`: the shared 1,000-skill candidate registry.
- `queries_{train,validation,test}.jsonl`: query text and multi-skill targets.
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
