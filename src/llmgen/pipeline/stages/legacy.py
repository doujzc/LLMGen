"""Compatibility imports for the pre-refactor Stage adapter module.

The default DAG is implemented by the dedicated modules in this package. The
private aliases remain temporarily available for downstream tests and local
extensions that imported the original monolithic adapter.
"""

from __future__ import annotations

from ..providers import (
    ledger_outputs as _ledger_outputs,
    provider_api_config as _provider_api_config,
    provider_config as _provider,
    provider_environment as _provider_environment,
    workers as _workers,
)
from . import default_stage_specs
from .assign_codes import assign_codes as _assign_codes
from .build_sft import build_sft as _build_sft
from .common import (
    alignment_only as _alignment_only,
    checkpoint_environment as _checkpoint_environment,
    configured as _configured,
    copy as _copy,
    device_count as _device_count,
    legacy_environment as _legacy_environment,
    paths as _paths,
    python as _python,
    router_pipeline as _router_pipeline,
    verify_training_provenance as _verify_training_provenance,
)
from .enrich import enrich as _enrich
from .evaluate import evaluate as _evaluate
from .export import (
    copy_model_tree as _copy_model_tree,
    export as _export,
    export_quality as _export_quality,
    full_weights_are_present as _full_weights_are_present,
    root_weight_files as _root_weight_files,
    run_export_model_smoke as _run_export_model_smoke,
)
from .finalize_dataset import finalize_dataset as _finalize_dataset
from .generate_queries import generate_queries as _generate_queries
from .ingest import ingest as _ingest
from .plan_queries import plan_queries as _plan_queries
from .review_queries import (
    _review_alignment,
    _review_multiskill,
    review_queries as _review_queries,
)
from .train_codebook import train_codebook as _train_codebook
from .train_router import (
    train_alignment as _train_alignment,
    train_memorization as _train_memorization,
    train_retrieval as _train_retrieval,
)


__all__ = ["default_stage_specs"]
