# ClawHub skill snapshot

The downloaded corpus is intentionally excluded from Git. Recreate or resume it from the repository root:

```bash
python scripts/download_clawhub_skills.py
```

The default snapshot selects 1,000 non-suspicious skills by `downloads DESC`, breaking equal-download ties by `stars DESC`. It records the catalog before downloading so that the downloads performed by the script cannot alter the selection.

Outputs:

- `catalog.jsonl`: one normalized record per selected skill, in rank order.
- `catalog_snapshot.json`: the raw pre-download API snapshot and selection metadata.
- `manifest.json`: counts, sorting/filter settings, timestamps, and checksums.
- `metadata/<owner>/<slug>.json`: raw skill detail plus artifact provenance.
- `skills/<owner>/<slug>/`: safely extracted, unexecuted skill package.
- `errors.jsonl`: failed records; rerunning resumes completed versions.

Reruns reuse the frozen catalog and completed packages. Pass `--refresh-snapshot` only when you intentionally want a new ranking snapshot.

Skill packages are untrusted third-party content. Do not execute their scripts without review. ClawHub documents the published package license as MIT-0; preserve the per-skill metadata and canonical URL when redistributing derived data.
