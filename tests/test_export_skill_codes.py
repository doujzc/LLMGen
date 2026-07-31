from types import SimpleNamespace

import numpy as np
import torch

from llmgen.neural.toolweaver import code_assignment_metrics
from llmgen.skillret import read_jsonl, write_jsonl
from scripts.export_skill_codes import _export_split, _quality_violations
from scripts.train_tokenizer import encode_embeddings


def _identity_rqvae() -> SimpleNamespace:
    quantizer = SimpleNamespace(
        embedding=torch.nn.Embedding.from_pretrained(
            torch.tensor([[0.0], [1.0]]),
            freeze=False,
        ),
        sk_epsilon=0.1,
        sk_iters=50,
    )
    return SimpleNamespace(
        eval=lambda: None,
        encoder=torch.nn.Identity(),
        rq=SimpleNamespace(vq_layers=[quantizer]),
        get_indices=lambda tensor, use_sk=False: torch.zeros(
            (len(tensor), 1),
            dtype=torch.long,
            device=tensor.device,
        ),
    )


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
        min_raw_level_utilization=(0.75, 0.65),
        min_raw_normalized_entropy=0.8,
    )

    assert any(value.startswith("raw collision_rate=") for value in violations)
    assert not any(value.startswith("collision_rate=") for value in violations)


def test_train_tokenizer_exports_per_sample_nearest_codes():
    embeddings = np.array([[0.0], [0.05], [0.1], [0.15]], dtype=np.float32)

    small_batches = encode_embeddings(
        _identity_rqvae(),
        embeddings,
        device="cpu",
        batch_size=1,
        num_levels=1,
        normalize=False,
    )
    large_batch = encode_embeddings(
        _identity_rqvae(),
        embeddings,
        device="cpu",
        batch_size=4,
        num_levels=1,
        normalize=False,
    )

    np.testing.assert_array_equal(small_batches, large_batch)
    assert large_batch[:, 0].tolist() == [0, 0, 0, 0]


def test_export_split_uses_nearest_for_raw_codes_and_quality_metrics(tmp_path):
    embeddings = np.array([[0.0], [0.05], [0.1], [0.15]], dtype=np.float32)
    embeddings_path = tmp_path / "train.npy"
    catalog_path = tmp_path / "catalog_train.jsonl"
    output_dir = tmp_path / "index"
    output_dir.mkdir()
    np.save(embeddings_path, embeddings)
    write_jsonl(
        catalog_path,
        [{"skill_id": f"skill-{index}"} for index in range(len(embeddings))],
    )

    artifact = _export_split(
        split="train",
        model=_identity_rqvae(),
        embeddings_path=embeddings_path,
        catalog_path=catalog_path,
        output_dir=output_dir,
        branching_factors=[2],
        token_format="<SK_L{level}_{index}>",
        device="cpu",
        batch_size=1,
        normalize_embeddings=False,
        expected_order_hash=None,
        expected_embedding_sha256=None,
        assignment_mode="balanced_hierarchical",
        assignment_exact_group_size=16,
        enforce_quality_gate=True,
        max_collision_rate=1.0,
        max_raw_collision_rate=1.0,
        max_bucket_size=None,
        min_level_utilization=1.0,
        min_normalized_entropy=1.0,
        min_raw_level_utilization=(0.5,),
        min_raw_normalized_entropy=0.0,
    )

    exported = read_jsonl(output_dir / "train_codes.jsonl")
    assert [row["indices"] for row in exported] == [[0], [0], [1], [1]]
    assert artifact["assignment_diagnostics"]["mode"] == "balanced_hierarchical"
    assert artifact["metrics"]["levels"][0]["utilization"] == 1.0
    assert artifact["raw_nearest_metrics"]["levels"][0]["utilization"] == 0.5
    assert artifact["quality_gate"]["passed"] is True
