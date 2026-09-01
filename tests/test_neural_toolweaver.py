from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from llmgen.neural.toolweaver import (
    SparseCollaborationGraph,
    Stage1TrainingConfig,
    TOOLWEAVER_UPSTREAM_HASHES,
    ToolWeaverModelConfig,
    ToolWeaverStage1Trainer,
    balanced_hierarchical_codes,
    code_assignment_metrics,
    create_toolweaver_rqvae,
    load_toolweaver_rqvae,
    residual_nearest_codes,
)


def model_config(**overrides):
    values = {
        "in_dim": 6,
        "num_levels": 2,
        "num_emb_list": (4, 3),
        "e_dim": 4,
        "layers": (8,),
        "kmeans_init": False,
        "sk_epsilons": (0.0, 0.02),
        "sk_iters": 3,
        "graph_lambda": 0.01,
    }
    values.update(overrides)
    return ToolWeaverModelConfig(**values)


def test_model_config_strictly_validates_level_sequences():
    with pytest.raises(ValueError, match="num_emb_list"):
        model_config(num_levels=3)
    with pytest.raises(ValueError, match="sk_epsilons"):
        model_config(sk_epsilons=(0.01,))


def test_vendored_adapter_supports_heterogeneous_codebooks(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("TOOLWEAVER_ROOT", str(tmp_path / "does-not-exist"))
    config = model_config(num_levels=3, num_emb_list=(4, 3, 2), sk_epsilons=(0.0, 0.0, 0.02))
    model = create_toolweaver_rqvae(config)
    assert [quantizer.n_e for quantizer in model.rq.vq_layers] == [4, 3, 2]
    indices = model.get_indices(torch.randn(5, 6), use_sk=False)
    assert tuple(indices.shape) == (5, 3)


def test_sparse_graph_coalesces_edges_and_returns_local_indices(tmp_path):
    graph_path = tmp_path / "graph.npz"
    np.savez(
        graph_path,
        src=np.array([0, 0, 0, 1, 2]),
        dst=np.array([1, 1, 0, 2, 3]),
        weight=np.array([1.0, 2.0, 9.0, 1.0, 1.0]),
        num_nodes=np.array(4),
    )
    graph = SparseCollaborationGraph.from_npz(graph_path)
    assert graph.src.tolist() == [0, 1, 2]
    assert graph.weight.tolist() == [3.0, 1.0, 1.0]
    src, dst, weight = graph.induced_edges(
        [0, 1, 3], max_edges=None, rng=np.random.default_rng(2)
    )
    assert src.tolist() == [0]
    assert dst.tolist() == [1]
    assert weight.tolist() == [3.0]


def test_code_metrics_report_collision_utilization_entropy_and_cv():
    metrics = code_assignment_metrics(
        np.array([[0, 0], [0, 0], [1, 1], [1, 2]]), (2, 3)
    )
    assert metrics["collision_rate"] == pytest.approx(0.25)
    assert metrics["max_bucket_size"] == 2
    assert metrics["levels"][0]["utilization"] == 1.0
    assert metrics["levels"][1]["utilization"] == 1.0
    assert metrics["levels"][0]["coefficient_of_variation"] == 0.0


def test_stage1_uses_sinkhorn_for_training_but_nearest_after_training(tmp_path):
    embeddings = np.array([[0.0], [0.05], [0.1], [0.15]], dtype=np.float32)
    config = model_config(
        in_dim=1,
        num_levels=1,
        num_emb_list=(2,),
        e_dim=1,
        layers=(),
        sk_epsilons=(0.1,),
        sk_iters=50,
    )
    trainer = ToolWeaverStage1Trainer(
        config,
        Stage1TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            scheduler="constant",
        ),
        embeddings,
        None,
        tmp_path,
        device="cpu",
    )
    linear = next(
        module for module in trainer.model.encoder.modules()
        if isinstance(module, torch.nn.Linear)
    )
    linear.weight.data.fill_(1.0)
    linear.bias.data.zero_()
    trainer.model.rq.vq_layers[0].embedding.weight.data.copy_(
        torch.tensor([[0.0], [1.0]])
    )

    encoded = trainer.model.encoder(torch.from_numpy(embeddings))
    _, _, training_codes = trainer.model.rq(encoded, use_sk=True)
    one_at_a_time = trainer.encode_all(batch_size=1)
    all_at_once = trainer.encode_all(batch_size=4)

    assert training_codes[:, 0].tolist() == [0, 0, 1, 1]
    np.testing.assert_array_equal(one_at_a_time, all_at_once)
    assert all_at_once[:, 0].tolist() == [0, 0, 0, 0]


def test_stage1_trains_a_real_one_candidate_one_token_codebook(tmp_path):
    embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    trainer = ToolWeaverStage1Trainer(
        model_config(
            in_dim=3,
            num_levels=1,
            num_emb_list=(1,),
            e_dim=2,
            layers=(),
            sk_epsilons=(0.0,),
            graph_lambda=0.0,
        ),
        Stage1TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            scheduler="constant",
            eval_every=1,
        ),
        embeddings,
        None,
        tmp_path,
        device="cpu",
    )

    result = trainer.fit()

    assert Path(result["best_checkpoint"]).is_file()
    assert result["best_metrics"]["collision_rate"] == 0.0
    assert result["best_metrics"]["levels"][0]["utilization"] == 1.0
    np.testing.assert_array_equal(trainer.encode_all(batch_size=1), [[0]])


def test_balanced_hierarchical_assignment_eliminates_avoidable_collisions():
    encoded = np.zeros((4, 2), dtype=np.float32)
    codebooks = (
        np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float32),
    )
    raw = residual_nearest_codes(encoded, codebooks)
    balanced, diagnostics = balanced_hierarchical_codes(
        encoded, codebooks, exact_group_size=16
    )

    assert code_assignment_metrics(raw, (2, 2))["collision_rate"] == 0.75
    metrics = code_assignment_metrics(balanced, (2, 2))
    assert metrics["collision_rate"] == 0.0
    assert metrics["max_bucket_size"] == 1
    assert [level["utilization"] for level in metrics["levels"]] == [1.0, 1.0]
    assert diagnostics["mode"] == "balanced_hierarchical"


def test_balanced_hierarchical_assignment_supports_configurable_levels():
    rng = np.random.default_rng(23)
    encoded = rng.normal(size=(17, 3)).astype(np.float32)
    codebooks = tuple(
        rng.normal(size=(size, 3)).astype(np.float32)
        for size in (3, 3, 2)
    )
    balanced, diagnostics = balanced_hierarchical_codes(encoded, codebooks)

    assert balanced.shape == (17, 3)
    assert code_assignment_metrics(balanced, (3, 3, 2))["collision_rate"] == 0.0
    assert len(diagnostics["levels"]) == 3


def test_full_training_checkpoint_loader_and_resume(tmp_path):
    embeddings = np.random.default_rng(7).normal(size=(16, 6)).astype(np.float32)
    graph = SparseCollaborationGraph(
        np.arange(15), np.arange(1, 16), np.ones(15), num_nodes=16
    )
    config = model_config()
    first_training = Stage1TrainingConfig(
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        scheduler="constant",
        eval_every=1,
    )
    trainer = ToolWeaverStage1Trainer(
        config,
        first_training,
        embeddings,
        graph,
        tmp_path,
        device="cpu",
        data_provenance={"ordered_skill_ids_sha256": "fixture"},
    )
    result = trainer.fit()
    assert Path(result["best_checkpoint"]).is_file()
    assert Path(result["last_checkpoint"]).is_file()
    model, checkpoint = load_toolweaver_rqvae(result["best_checkpoint"], "cpu")
    assert "model_config" in checkpoint and "model_state" in checkpoint
    assert checkpoint["toolweaver_source"]["mode"] == "vendored"
    assert "root" not in checkpoint["toolweaver_source"]
    assert checkpoint["toolweaver_source"]["source_files_sha256"] == TOOLWEAVER_UPSTREAM_HASHES
    assert tuple(model.get_indices(torch.from_numpy(embeddings[:3]), use_sk=False).shape) == (3, 2)

    legacy = dict(checkpoint)
    legacy["toolweaver_source"] = {
        "root": "/checkout/that/no/longer/exists",
        "git_revision": "a7684edaf2bb3af7ff6928c34e27a324599deda0",
        "mode": "dynamic_load",
        "source_files_sha256": dict(TOOLWEAVER_UPSTREAM_HASHES),
    }
    legacy_path = tmp_path / "legacy-dynamic.pt"
    torch.save(legacy, legacy_path)
    legacy_model, _ = load_toolweaver_rqvae(legacy_path, "cpu")
    assert tuple(
        legacy_model.get_indices(torch.from_numpy(embeddings[:3]), use_sk=False).shape
    ) == (3, 2)

    unhashed_legacy = dict(checkpoint)
    unhashed_legacy["toolweaver_source"] = {
        "root": "/another/missing/checkout",
        "git_revision": "a7684edaf2bb3af7ff6928c34e27a324599deda0",
        "mode": "dynamic_load",
    }
    unhashed_legacy_path = tmp_path / "legacy-unhashed.pt"
    torch.save(unhashed_legacy, unhashed_legacy_path)
    load_toolweaver_rqvae(unhashed_legacy_path, "cpu")

    invalid_sources = [
        None,
        {"mode": "unknown"},
        {
            **checkpoint["toolweaver_source"],
            "vendored_files_sha256": {
                **checkpoint["toolweaver_source"]["vendored_files_sha256"],
                "index/models/rqvae.py": "0" * 64,
            },
        },
    ]
    for offset, source in enumerate(invalid_sources):
        invalid = dict(checkpoint)
        if source is None:
            invalid.pop("toolweaver_source")
        else:
            invalid["toolweaver_source"] = source
        invalid_path = tmp_path / f"invalid-source-{offset}.pt"
        torch.save(invalid, invalid_path)
        with pytest.raises(ValueError, match="source|ToolWeaver"):
            load_toolweaver_rqvae(invalid_path, "cpu")

    resumed = ToolWeaverStage1Trainer(
        config,
        Stage1TrainingConfig(
            epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            scheduler="constant",
            eval_every=1,
        ),
        embeddings,
        graph,
        tmp_path,
        device="cpu",
        data_provenance={"ordered_skill_ids_sha256": "fixture"},
    )
    resumed_result = resumed.fit(result["last_checkpoint"])
    assert resumed_result["last_metrics"]["num_skills"] == 16


def test_bfloat16_amp_initializes_kmeans_codebooks_in_float32(tmp_path):
    embeddings = np.random.default_rng(11).normal(size=(8, 6)).astype(np.float32)
    trainer = ToolWeaverStage1Trainer(
        model_config(
            kmeans_init=True,
            kmeans_iters=2,
            num_emb_list=(2, 2),
            sk_epsilons=(0.0, 0.0),
        ),
        Stage1TrainingConfig(
            epochs=1,
            batch_size=8,
            amp_dtype="bf16",
            scheduler="constant",
        ),
        embeddings,
        None,
        tmp_path,
        device="cpu",
    )

    result = trainer.fit()

    assert Path(result["last_checkpoint"]).is_file()
    assert all(quantizer.initted for quantizer in trainer.model.rq.vq_layers)


def test_resume_rejects_changed_data_fingerprint(tmp_path):
    embeddings = np.random.default_rng(9).normal(size=(8, 6)).astype(np.float32)
    config = model_config(num_emb_list=(2, 2))
    training = Stage1TrainingConfig(
        epochs=1, batch_size=4, learning_rate=1e-3, scheduler="constant"
    )
    trainer = ToolWeaverStage1Trainer(
        config,
        training,
        embeddings,
        None,
        tmp_path,
        data_provenance={"embedding_file_sha256": "old"},
    )
    result = trainer.fit()
    changed = ToolWeaverStage1Trainer(
        config,
        Stage1TrainingConfig(
            epochs=2, batch_size=4, learning_rate=1e-3, scheduler="constant"
        ),
        embeddings,
        None,
        tmp_path,
        data_provenance={"embedding_file_sha256": "new"},
    )
    with pytest.raises(ValueError, match="provenance"):
        changed.resume(result["last_checkpoint"])
