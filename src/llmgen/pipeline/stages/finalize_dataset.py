"""Dataset export, qrels normalization, and embedding-preparation adapter."""

from __future__ import annotations

from ..io import read_jsonl
from ..providers import ledger_outputs
from ..schema import ensure_ordered_qrels, validate_ordered_qrels
from .base import ArtifactOutput, StageContext, StageResult
from .common import paths, python, router_pipeline


def finalize_dataset(context: StageContext) -> StageResult:
    """Export data, validate ordered qrels, and prepare processed embeddings."""

    stage_paths = paths(context)
    stage_paths["dataset"].mkdir(parents=True, exist_ok=True)
    minimum = int(context.config.require("data_generation.retrieval_positives_per_skill"))
    split = context.config.require("data_generation.split")
    common = ["--catalog", str(stage_paths["catalog"]), "--profiles", str(stage_paths["profiles"]), "--workflows", str(stage_paths["review_workflows"]), "--queries", str(stage_paths["review_queries"]), "--reviews", str(stage_paths["reviews"]), "--alignment-queries", str(stage_paths["review_alignment_queries"]), "--alignment-reviews", str(stage_paths["alignment_reviews"])]
    context.run_command(
        [python(context), "scripts/clawhub_data/04_export_dataset.py", *common, "--output-dir", str(stage_paths["dataset"]), "--seed", str(int(context.config.require("run.seed"))), "--min-train-positives-per-skill", str(minimum), "--min-augmented-train-queries", str(int(context.config.require("data_generation.min_augmented_train_queries"))), "--target-order-variants", str(int(context.config.require("data_generation.order_variants"))), "--train-fraction", str(float(split["train"])), "--validation-fraction", str(float(split["validation"])), "--test-fraction", str(float(split["test"]))],
        label="export-training-dataset",
    )
    context.run_command(
        [python(context), "scripts/clawhub_data/04a_export_alignment.py", "--catalog", str(stage_paths["catalog"]), "--queries", str(stage_paths["review_alignment_queries"]), "--reviews", str(stage_paths["alignment_reviews"]), "--output-dir", str(stage_paths["dataset"]), "--min-queries-per-skill", str(int(context.config.require("data_generation.alignment_queries_per_skill")))],
        label="export-alignment-dataset",
    )
    ensure_ordered_qrels(stage_paths["dataset"])
    validate_ordered_qrels(stage_paths["dataset"])
    count = len(read_jsonl(stage_paths["dataset"] / "skills.jsonl"))
    context.run_command([python(context), "scripts/clawhub_data/05_validate_dataset.py", "--dataset-dir", str(stage_paths["dataset"]), "--expected-candidates", str(count)], label="audit-final-dataset")
    router_pipeline(context, "prepare")
    artifacts = [
        ArtifactOutput("dataset.directory", stage_paths["dataset"], "closedset_dataset/v3"),
        ArtifactOutput("dataset.manifest", stage_paths["dataset"] / "manifest.json", "closedset_manifest/v3"),
        ArtifactOutput("processed.directory", stage_paths["processed"], "processed_closedset/v1"),
        ArtifactOutput("processed.manifest", stage_paths["processed"] / "manifest.json", "processed_manifest/v1"),
        ArtifactOutput("embeddings.directory", stage_paths["embeddings"], "embedding_bundle/v1"),
        ArtifactOutput("embeddings.manifest", stage_paths["embeddings"] / "manifest.json", "embedding_manifest/v1"),
    ]
    for split_name in ("train", "validation", "test", "alignment"):
        for kind in ("queries", "qrels"):
            path = stage_paths["dataset"] / f"{kind}_{split_name}.jsonl"
            if path.is_file():
                artifacts.append(ArtifactOutput(f"dataset.{kind}.{split_name}", path, f"router_{kind.rstrip('s')}/1"))
    ledger_artifacts, ledger_progress = ledger_outputs(context, "embedding")
    artifacts.extend(ledger_artifacts)
    return StageResult(artifacts=tuple(artifacts), progress={"candidate_count": count, "provider_ledgers": ledger_progress})
