import numpy as np

from llmgen.neural.toolweaver import code_assignment_metrics
from scripts.export_skill_codes import _quality_violations


def test_quality_gate_checks_raw_metrics_before_balanced_post_assignment():
    raw = code_assignment_metrics(
        np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int64),
        (2, 2),
    )
    assigned = code_assignment_metrics(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        (2, 2),
    )
    violations = _quality_violations(
        metrics=assigned,
        raw_metrics=raw,
        max_collision_rate=0.01,
        max_raw_collision_rate=0.15,
        max_bucket_size=1,
        min_level_utilization=0.9,
        min_normalized_entropy=0.9,
        min_raw_level_utilization=0.75,
        min_raw_normalized_entropy=0.8,
    )

    assert any(value.startswith("raw collision_rate=") for value in violations)
    assert not any(value.startswith("collision_rate=") for value in violations)
