"""Resumable orchestration for candidate-to-router training runs.

The pipeline package deliberately owns orchestration only.  Existing dataset
construction, tokenizer training, Router training, evaluation, and export
entry points remain the algorithmic implementation and are invoked by stage
adapters.
"""

from .artifacts import ArtifactRecord, ArtifactRegistry
from .config import PipelineConfig, PipelineConfigError, load_pipeline_config
from .runner import PipelineRunner, create_pipeline_run
from .state import PipelineStateError, StageStatus

__all__ = [
    "ArtifactRecord",
    "ArtifactRegistry",
    "PipelineConfig",
    "PipelineConfigError",
    "PipelineRunner",
    "PipelineStateError",
    "StageStatus",
    "create_pipeline_run",
    "load_pipeline_config",
]
