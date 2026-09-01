"""Default Stage DAG for the generic candidate-to-model pipeline."""

from .assign_codes import assign_codes as _assign_codes
from .base import ArtifactOutput, StageContext, StageResult, StageSpec
from .build_sft import build_sft as _build_sft
from .enrich import enrich as _enrich
from .evaluate import evaluate as _evaluate
from .export import export as _export
from .finalize_dataset import finalize_dataset as _finalize_dataset
from .generate_queries import generate_queries as _generate_queries
from .ingest import ingest as _ingest
from .plan_queries import plan_queries as _plan_queries
from .review_queries import review_queries as _review_queries
from .train_codebook import train_codebook as _train_codebook
from .train_router import (
    train_alignment as _train_alignment,
    train_memorization as _train_memorization,
    train_retrieval as _train_retrieval,
)


_PIPELINE_RUNTIME_COMMON = (
    "src/llmgen/pipeline/stages/base.py",
    "src/llmgen/pipeline/artifacts.py",
    "src/llmgen/pipeline/checkpoints.py",
    "src/llmgen/pipeline/config.py",
    "src/llmgen/pipeline/io.py",
    "src/llmgen/pipeline/logging.py",
    "src/llmgen/pipeline/resources.py",
    "src/llmgen/pipeline/state.py",
)
_STAGE_COMMON = (
    "src/llmgen/pipeline/stages/common.py",
    "src/llmgen/pipeline/providers.py",
    "src/llmgen/pipeline/ledger.py",
    *_PIPELINE_RUNTIME_COMMON,
)
_CLAWHUB_DATASET_COMMON = (
    "src/llmgen/clawhub.py",
    "src/llmgen/clawhub_dataset.py",
)
_CLAWHUB_ALIGNMENT_COMMON = (
    "src/llmgen/clawhub_alignment.py",
    *_CLAWHUB_DATASET_COMMON,
)
_CLAWHUB_AUDIT_COMMON = (
    "src/llmgen/clawhub_audit.py",
    *_CLAWHUB_ALIGNMENT_COMMON,
)
_SKILLRET_COMMON = (
    "configs/generic.env",
    "scripts/skillret/common.sh",
    "src/llmgen/skillret.py",
)
_TOOLWEAVER_COMMON = (
    "src/llmgen/neural/toolweaver.py",
    "src/llmgen/vendor/toolweaver/layers.py",
    "src/llmgen/vendor/toolweaver/rq.py",
    "src/llmgen/vendor/toolweaver/rqvae.py",
    "src/llmgen/vendor/toolweaver/vq.py",
)
_ROUTER_COMMON = (
    "scripts/train_router.py",
    "src/llmgen/router.py",
    "src/llmgen/router_bundle.py",
)
_ROUTER_INFERENCE_COMMON = (
    "scripts/infer_router.py",
    "src/llmgen/incremental.py",
    *_TOOLWEAVER_COMMON,
)


def default_stage_specs() -> tuple[StageSpec, ...]:
    """Return the stable, independently importable Stage DAG."""

    return (
        StageSpec(
            "ingest",
            "00_ingest",
            (),
            (),
            _ingest,
            "validate and freeze candidates",
            implementation_paths=(
                "src/llmgen/pipeline/stages/ingest.py",
                "src/llmgen/pipeline/schema.py",
                *_PIPELINE_RUNTIME_COMMON,
            ),
        ),
        StageSpec(
            "enrich",
            "01_enrich",
            ("ingest",),
            ("candidates.catalog",),
            _enrich,
            "profile candidate capabilities",
            implementation_paths=(
                "src/llmgen/pipeline/stages/enrich.py",
                "scripts/clawhub_data/00_profile_skills.py",
                *_CLAWHUB_DATASET_COMMON,
                *_STAGE_COMMON,
            ),
        ),
        StageSpec(
            "plan-queries",
            "02_plan_queries",
            ("enrich",),
            ("candidates.manifest", "data.profiles"),
            _plan_queries,
            "plan multi-Skill workflows",
            implementation_paths=(
                "src/llmgen/pipeline/stages/plan_queries.py",
                "scripts/clawhub_data/01_build_workflows.py",
                *_CLAWHUB_DATASET_COMMON,
                *_STAGE_COMMON,
            ),
        ),
        StageSpec(
            "generate-queries",
            "03_generate_queries",
            ("plan-queries",),
            ("candidates.manifest", "data.profiles", "data.workflows"),
            _generate_queries,
            "generate alignment and retrieval queries",
            implementation_paths=(
                "src/llmgen/pipeline/stages/generate_queries.py",
                "scripts/clawhub_data/02_generate_queries.py",
                "scripts/clawhub_data/02a_generate_alignment_queries.py",
                *_CLAWHUB_ALIGNMENT_COMMON,
                *_STAGE_COMMON,
            ),
        ),
        StageSpec(
            "review-queries",
            "04_review_queries",
            ("generate-queries",),
            (
                "candidates.manifest",
                "data.profiles",
                "data.workflows",
                "data.queries.generated",
                "data.queries.alignment.generated",
                "inputs.manual_alignment",
            ),
            _review_queries,
            "review queries and backfill coverage",
            implementation_paths=(
                "src/llmgen/pipeline/stages/review_queries.py",
                "scripts/clawhub_data/02_generate_queries.py",
                "scripts/clawhub_data/03_review_queries.py",
                "scripts/clawhub_data/03a_review_alignment_queries.py",
                "scripts/clawhub_data/03a2_backfill_alignment.py",
                "scripts/clawhub_data/03b_build_coverage_workflows.py",
                "scripts/light_data/02b_apply_manual_alignment.py",
                *_CLAWHUB_ALIGNMENT_COMMON,
                *_STAGE_COMMON,
            ),
        ),
        StageSpec(
            "finalize-dataset",
            "05_finalize_dataset",
            ("review-queries",),
            (
                "candidates.manifest",
                "candidates.catalog",
                "data.profiles",
                "data.workflows.reviewed",
                "data.queries.reviewed",
                "data.reviews",
                "data.queries.alignment.reviewed",
                "data.reviews.alignment",
            ),
            _finalize_dataset,
            "export ordered qrels and prepare embeddings",
            implementation_paths=(
                "src/llmgen/pipeline/stages/finalize_dataset.py",
                "scripts/clawhub_data/04_export_dataset.py",
                "scripts/clawhub_data/04a_export_alignment.py",
                "scripts/clawhub_data/05_validate_dataset.py",
                "scripts/prepare_closedset.py",
                "scripts/skillret/01_prepare.sh",
                *_CLAWHUB_AUDIT_COMMON,
                "src/llmgen/embeddings.py",
                "src/llmgen/pipeline/schema.py",
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
            ),
        ),
        StageSpec(
            "train-codebook",
            "06_train_codebook",
            ("finalize-dataset",),
            (
                "candidates.manifest",
                "dataset.directory",
                "processed.directory",
                "processed.manifest",
                "embeddings.directory",
                "embeddings.manifest",
            ),
            _train_codebook,
            "plan and train the hierarchical codebook",
            implementation_paths=(
                "src/llmgen/pipeline/stages/train_codebook.py",
                "src/llmgen/pipeline/code_plan.py",
                "scripts/skillret/02_train_tokenizer.sh",
                "scripts/train_tokenizer.py",
                *_TOOLWEAVER_COMMON,
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
            ),
        ),
        StageSpec(
            "assign-codes",
            "07_assign_codes",
            ("train-codebook",),
            (
                "code.plan",
                "codebook.best",
                "candidates.manifest",
                "processed.directory",
                "embeddings.directory",
            ),
            _assign_codes,
            "assign codes and run quality gates",
            implementation_paths=(
                "src/llmgen/pipeline/stages/assign_codes.py",
                "scripts/skillret/03_export_codes.sh",
                "scripts/export_skill_codes.py",
                *_TOOLWEAVER_COMMON,
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
            ),
        ),
        StageSpec(
            "build-sft",
            "08_build_sft",
            ("assign-codes",),
            (
                "code.plan",
                "candidates.manifest",
                "processed.directory",
                "codes.train",
                "codes.registry",
                "codes.virtual_tokens",
            ),
            _build_sft,
            "build target-only Router SFT data",
            implementation_paths=(
                "src/llmgen/pipeline/stages/build_sft.py",
                "scripts/skillret/04_build_router_data.sh",
                "scripts/build_router_data.py",
                "src/llmgen/router.py",
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
            ),
        ),
        StageSpec(
            "train-memorization",
            "09_train_memorization",
            ("build-sft",),
            (
                "code.plan",
                "candidates.manifest",
                "processed.directory",
                "codes.train",
                "codes.registry",
                "codes.virtual_tokens",
                "sft.directory",
            ),
            _train_memorization,
            "train Skill document memorization",
            implementation_paths=(
                "src/llmgen/pipeline/stages/train_router.py",
                "scripts/skillret/05_train_memorization.sh",
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
                *_ROUTER_COMMON,
            ),
        ),
        StageSpec(
            "train-alignment",
            "10_train_alignment",
            ("train-memorization",),
            (
                "code.plan",
                "candidates.manifest",
                "model.memorization",
                "processed.directory",
                "codes.train",
                "codes.registry",
                "codes.virtual_tokens",
                "sft.directory",
            ),
            _train_alignment,
            "train single-Skill alignment",
            implementation_paths=(
                "src/llmgen/pipeline/stages/train_router.py",
                "scripts/skillret/06a_train_alignment.sh",
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
                *_ROUTER_COMMON,
            ),
        ),
        StageSpec(
            "train-retrieval",
            "11_train_retrieval",
            ("train-alignment",),
            (
                "code.plan",
                "candidates.manifest",
                "model.memorization",
                "model.alignment",
                "processed.directory",
                "codes.train",
                "codes.registry",
                "codes.virtual_tokens",
                "sft.directory",
            ),
            _train_retrieval,
            "train multi-Skill retrieval",
            implementation_paths=(
                "src/llmgen/pipeline/stages/train_router.py",
                "scripts/skillret/06b_train_retrieval.sh",
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
                *_ROUTER_COMMON,
            ),
        ),
        StageSpec(
            "evaluate",
            "12_evaluate",
            ("train-retrieval",),
            (
                "code.plan",
                "candidates.manifest",
                "model.retrieval",
                "processed.directory",
                "codes.train",
                "codes.registry",
                "codes.virtual_tokens",
                "sft.directory",
            ),
            _evaluate,
            "run constrained closed-set evaluation",
            implementation_paths=(
                "src/llmgen/pipeline/stages/evaluate.py",
                "src/llmgen/pipeline/quality.py",
                "scripts/skillret/07_evaluate.sh",
                "scripts/export_closedset_validation.py",
                *_ROUTER_INFERENCE_COMMON,
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
                *_ROUTER_COMMON,
            ),
        ),
        StageSpec(
            "export",
            "13_export",
            ("evaluate",),
            (
                "code.plan",
                "candidates.manifest",
                "model.retrieval",
                "evaluation.directory",
                "processed.directory",
                "codes.train",
                "codes.registry",
                "codes.virtual_tokens",
                "sft.directory",
            ),
            _export,
            "export a self-contained deployment model",
            implementation_paths=(
                "src/llmgen/pipeline/stages/export.py",
                "src/llmgen/pipeline/quality.py",
                "scripts/skillret/10_export_web_bundle.sh",
                "scripts/export_router_bundle.py",
                "scripts/merge_router_adapter.py",
                *_ROUTER_INFERENCE_COMMON,
                *_STAGE_COMMON,
                *_SKILLRET_COMMON,
                *_ROUTER_COMMON,
            ),
        ),
    )

__all__ = [
    "ArtifactOutput",
    "StageContext",
    "StageResult",
    "StageSpec",
    "default_stage_specs",
]
