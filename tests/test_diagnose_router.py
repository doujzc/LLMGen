from scripts.diagnose_router import (
    analyze_codes,
    analyze_dataset,
    analyze_predictions,
    build_findings,
)


def _artifacts():
    rows = [
        {
            "skill_id": f"s{index + 1}",
            "indices": [index // 2, index % 2],
            "tokens": [f"<L1_{index // 2}>", f"<L2_{index % 2}>"],
        }
        for index in range(4)
    ]
    registry = {
        "num_levels": 2,
        "branching_factors": [2, 2],
        "buckets": {
            f"{row['indices'][0]}/{row['indices'][1]}": [row["skill_id"]]
            for row in rows
        },
    }
    return rows, registry


def test_static_diagnostics_recompute_metrics_and_baselines():
    code_rows, registry = _artifacts()
    train_relevance = {
        "train-1": ("s1", "s2"),
        "train-2": ("s1", "s3"),
    }
    eval_relevance = {"eval-1": ("s1", "s2")}
    router_rows = [
        {
            "positive_skill_ids": ["s1", "s2"],
            "target_paths": [
                ["<L1_0>", "<L2_0>"],
                ["<L1_0>", "<L2_1>"],
            ]
        }
    ]
    dataset, train_frequency, _ = analyze_dataset(
        train_relevance=train_relevance,
        eval_relevance=eval_relevance,
        active_skill_ids=("s1", "s2", "s3", "s4"),
        router_train_rows=router_rows,
        cutoffs=(1, 5),
        source_train_queries=None,
    )
    codes, skill_to_code, buckets = analyze_codes(
        code_rows, registry, train_frequency
    )
    predictions = analyze_predictions(
        predictions=[
            {
                "query_id": "eval-1",
                "paths": [
                    {"code_tokens": ["<L1_0>", "<L2_0>"]},
                    {"code_tokens": ["<L1_0>", "<L2_1>"]},
                ],
                "candidates": [
                    {"skill_id": "s1"},
                    {"skill_id": "s2"},
                ],
            }
        ],
        eval_relevance=eval_relevance,
        train_frequency=train_frequency,
        skill_to_code=skill_to_code,
        buckets=buckets,
        cutoffs=(1, 5),
    )

    assert dataset["unseen_eval_association_count"] == 0
    assert dataset["popularity_baseline"]["recall@1"] == 0.5
    assert codes["collision_rate"] == 0.0
    assert predictions["metrics"]["recall@5"] == 1.0
    assert predictions["prefix_recall"] == {"level_1": 1.0, "level_2": 1.0}

    report = {"dataset": dataset, "codes": codes, "predictions": predictions}
    findings = build_findings(report, (1, 5))
    assert any(row["code"] == "below_popularity_baseline" for row in findings)


def test_code_diagnostics_expose_collisions():
    code_rows, registry = _artifacts()
    code_rows[-1]["indices"] = [1, 0]
    code_rows[-1]["tokens"] = ["<L1_1>", "<L2_0>"]
    registry["buckets"] = {
        "0/0": ["s1"],
        "0/1": ["s2"],
        "1/0": ["s3", "s4"],
    }
    codes, _, _ = analyze_codes(
        code_rows,
        registry,
        {"s1": 2, "s2": 1, "s3": 1, "s4": 0},
    )
    assert codes["num_active_paths"] == 3
    assert codes["collision_count"] == 1
    assert codes["collision_rate"] == 0.25
    assert codes["max_bucket_size"] == 2
