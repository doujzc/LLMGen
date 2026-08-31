"""Default Stage DAG and legacy algorithm adapters."""

from .base import ArtifactOutput, StageContext, StageResult, StageSpec
from .legacy import default_stage_specs

__all__ = [
    "ArtifactOutput",
    "StageContext",
    "StageResult",
    "StageSpec",
    "default_stage_specs",
]
