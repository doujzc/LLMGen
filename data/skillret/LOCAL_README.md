# SkillRet data snapshot

This directory contains the complete public SkillRet v1.0 snapshot from the
official Hugging Face repository, `ThakiCloud/SKILLRET`, pinned to revision
`7cae7cfbad2b0e1ebc9170892f568993aae543b0`.

The paper authors are from ThakiCloud and the repository links directly to
arXiv:2605.05726. The similarly named `anonymous-ed-benchmark/SKILLRET` is the
older anonymous review release; `thaki-AI/SkillRetBench` is a different,
501-skill benchmark.

## Layout

- `data/skills/{train,test}.jsonl`: disjoint train/test skill corpora.
- `data/queries/{train,test}.jsonl`: train/test user queries.
- `data/qrels/{train,test}.jsonl`: binary query-skill relevance pairs.
- `data/skills.jsonl`: full 17,810-skill catalog, including 1,027 skills outside
  the train/test corpora.
- `data/taxonomy.json`: the two-level taxonomy.
- `README.md` and `croissant-rai.json`: the upstream data card and metadata.

## Reproduce and verify

```bash
hf download ThakiCloud/SKILLRET \
  --repo-type dataset \
  --revision 7cae7cfbad2b0e1ebc9170892f568993aae543b0 \
  --local-dir data/skillret
sha256sum --check data/skillret/SHA256SUMS
```

Benchmark metadata, queries, and taxonomy are Apache-2.0. Individual skill
documents retain their recorded MIT or Apache-2.0 source licenses. See the
upstream `README.md` for intended use, limitations, and citation details.
