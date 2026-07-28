#!/usr/bin/env python3
"""Train ToolWeaver Stage 1 on a normalized skill catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llmgen.neural.toolweaver import (  # noqa: E402
    SparseCollaborationGraph,
    Stage1TrainingConfig,
    ToolWeaverModelConfig,
    ToolWeaverStage1Trainer,
    code_assignment_metrics,
    load_toolweaver_rqvae,
    sinkhorn_residual_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the neural ToolWeaver tokenizer on normalized skill embeddings."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/skillret"))
    parser.add_argument("--embedding-path", type=Path, default=None)
    parser.add_argument("--embedding-manifest-path", type=Path, default=None)
    parser.add_argument("--graph-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Resumable last.pt checkpoint.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")

    parser.add_argument("--num-levels", type=int, default=None)
    parser.add_argument("--branching-factors", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--sk-epsilons", type=float, nargs="+", default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=[512, 256, 128])
    parser.add_argument("--e-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-type", choices=["mse", "l1"], default="mse")
    parser.add_argument("--quant-loss-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--kmeans-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--sk-iters", type=int, default=50)
    parser.add_argument("--graph-lambda", type=float, default=0.001)
    parser.add_argument("--token-format", default="<SK_L{level}_{index}>")
    parser.add_argument("--codebook-version", default="skillret-v1")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["constant", "linear", "cosine"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=["none", "fp16", "bf16"], default="none")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-graph-edges-per-batch", type=int, default=None)
    parser.add_argument("--edge-aware-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selection-metric", choices=["collision_rate", "loss"], default="collision_rate")
    parser.add_argument("--use-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-batch-size", type=int, default=1024)
    return parser.parse_args()


def ordered_ids_sha256(skill_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for skill_id in skill_ids:
        digest.update(skill_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_artifact(data_root: Path, manifest_path: Path, raw_path: str) -> Path:
    value = Path(raw_path).expanduser()
    if value.is_absolute():
        return value
    candidates = (
        Path.cwd() / value,
        REPO_ROOT / value,
        manifest_path.parent / value,
        data_root / value,
        data_root / "processed" / value,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_catalog_ids(
    data_root: Path, manifest_path: Path, expected_count: int
) -> tuple[list[str], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        split = manifest["splits"]["train"]
        catalog_path = _resolve_artifact(data_root, manifest_path, split["files"]["catalog"])
    except (KeyError, TypeError) as exc:
        raise ValueError("manifest must define splits.train.files.catalog") from exc
    skill_ids: list[str] = []
    with catalog_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            skill_id = row.get("skill_id") or row.get("id")
            if not isinstance(skill_id, str) or not skill_id:
                raise ValueError(f"catalog row {line_number} has no skill_id")
            skill_ids.append(skill_id)
    if len(skill_ids) != expected_count:
        raise ValueError(
            f"catalog/embedding row mismatch: {len(skill_ids)} != {expected_count}"
        )
    expected_hash = split.get("hashes", {}).get("ordered_skill_ids_sha256")
    actual_hash = ordered_ids_sha256(skill_ids)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("catalog ordered_skill_ids_sha256 does not match manifest")
    actual_catalog_sha256 = file_sha256(catalog_path)
    expected_catalog_sha256 = split.get("hashes", {}).get("catalog_sha256")
    if expected_catalog_sha256 and expected_catalog_sha256 != actual_catalog_sha256:
        raise ValueError("catalog SHA-256 does not match processed manifest")
    return skill_ids, {
        "manifest": str(manifest_path),
        "catalog": str(catalog_path),
        "catalog_file_sha256": actual_catalog_sha256,
        "ordered_skill_ids_sha256": actual_hash,
    }


def validate_npz_skill_hash(path: Path, expected_hash: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        for key in ("ordered_skill_ids_sha256", "skill_ids_sha256"):
            if key not in data.files:
                continue
            raw = np.asarray(data[key]).reshape(-1)[0]
            actual = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            if actual != expected_hash:
                raise ValueError(f"{path} {key} does not match catalog ordering")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def encode_embeddings(
    model: Any,
    embeddings: np.ndarray,
    *,
    device: str,
    batch_size: int,
    num_levels: int,
    normalize: bool,
) -> np.ndarray:
    model.eval()
    encoded_rows = []
    for start in range(0, len(embeddings), batch_size):
        values = np.array(
            embeddings[start : start + batch_size], dtype=np.float32, copy=True
        )
        if normalize:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
        tensor = torch.from_numpy(np.ascontiguousarray(values)).to(device)
        encoded_rows.append(
            model.encoder(tensor).detach().float().cpu().numpy()
        )
    quantizers = tuple(model.rq.vq_layers)
    if len(quantizers) != num_levels:
        raise ValueError("num_levels does not match the trained RQ-VAE")
    sk_iters = {int(quantizer.sk_iters) for quantizer in quantizers}
    if len(sk_iters) != 1:
        raise ValueError("all RQ-VAE levels must use the same sk_iters")
    return sinkhorn_residual_codes(
        np.concatenate(encoded_rows, axis=0),
        [quantizer.embedding.weight.detach() for quantizer in quantizers],
        sk_epsilons=[float(quantizer.sk_epsilon) for quantizer in quantizers],
        sk_iters=sk_iters.pop(),
        device=device,
    )


def export_assignments(
    output_dir: Path,
    skill_ids: list[str],
    codes: np.ndarray,
    config: ToolWeaverModelConfig,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    np.save(output_dir / "train_codes.npy", codes)
    buckets: dict[str, list[str]] = defaultdict(list)
    with (output_dir / "train_codes.jsonl").open("w", encoding="utf-8") as stream:
        for row_index, (skill_id, indices_array) in enumerate(zip(skill_ids, codes)):
            indices = [int(value) for value in indices_array]
            tokens = [
                config.token_format.format(level=level + 1, index=index)
                for level, index in enumerate(indices)
            ]
            code_text = "".join(tokens)
            buckets[code_text].append(skill_id)
            stream.write(
                json.dumps(
                    {
                        "skill_id": skill_id,
                        "row_index": row_index,
                        "split": "train",
                        "indices": indices,
                        "tokens": tokens,
                        "code_text": code_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    registry = {
        "schema_version": 1,
        "codebook_version": config.codebook_version,
        "num_levels": config.num_levels,
        "branching_factors": list(config.num_emb_list),
        "token_format": config.token_format,
        "assignment_mode": "sinkhorn",
        "assignment_scope": "full_catalog",
        "sk_epsilons": list(config.sk_epsilons),
        "sk_iters": config.sk_iters,
        "ordered_skill_ids_sha256": source["ordered_skill_ids_sha256"],
        "buckets": dict(buckets),
    }
    (output_dir / "registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "virtual_tokens.txt").write_text(
        "".join(f"{token}\n" for token in config.virtual_tokens), encoding="utf-8"
    )
    return registry


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    embedding_path = (args.embedding_path or data_root / "embeddings" / "train.npy").resolve()
    embedding_manifest_path = (
        args.embedding_manifest_path or embedding_path.parent / "manifest.json"
    ).expanduser().resolve()
    graph_path = (args.graph_path or data_root / "processed" / "collab_graph_train.npz").resolve()
    manifest_path = (args.manifest_path or data_root / "processed" / "manifest.json").resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError("train.npy must be a two-dimensional embedding matrix")
    skill_ids, source = load_catalog_ids(data_root, manifest_path, len(embeddings))
    if not embedding_manifest_path.is_file():
        raise ValueError(
            "embedding manifest is required to verify row order and file integrity: "
            f"{embedding_manifest_path}"
        )
    embedding_manifest = json.loads(
        embedding_manifest_path.read_text(encoding="utf-8")
    )
    try:
        expected_embedding_order = embedding_manifest["ordered_skill_ids_sha256"]["train"]
        expected_embedding_sha256 = embedding_manifest["sha256"]["train"]
        expected_embedding_shape = tuple(embedding_manifest["shapes"]["train"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "embedding manifest must define train order, SHA-256, and shape"
        ) from exc
    actual_embedding_sha256 = file_sha256(embedding_path)
    if expected_embedding_order != source["ordered_skill_ids_sha256"]:
        raise ValueError("embedding manifest row order differs from the train catalog")
    if expected_embedding_sha256 != actual_embedding_sha256:
        raise ValueError("train embedding SHA-256 does not match embedding manifest")
    if expected_embedding_shape != tuple(int(value) for value in embeddings.shape):
        raise ValueError("train embedding shape does not match embedding manifest")
    source["embedding_path"] = str(embedding_path)
    source["embedding_file_sha256"] = actual_embedding_sha256
    source["embedding_manifest"] = str(embedding_manifest_path)
    source["embedding_manifest_file_sha256"] = file_sha256(
        embedding_manifest_path
    )
    if args.use_graph:
        validate_npz_skill_hash(graph_path, source["ordered_skill_ids_sha256"])
        actual_graph_sha256 = file_sha256(graph_path)
        processed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_graph_sha256 = processed_manifest.get("graph", {}).get("sha256")
        if expected_graph_sha256 and expected_graph_sha256 != actual_graph_sha256:
            raise ValueError("collaborative graph SHA-256 does not match processed manifest")
        graph = SparseCollaborationGraph.from_npz(graph_path)
        source["graph_path"] = str(graph_path)
        source["graph_file_sha256"] = actual_graph_sha256
    else:
        graph = None

    num_levels = args.num_levels or len(args.branching_factors)
    sk_epsilons = args.sk_epsilons or ([0.0] * (num_levels - 1) + [0.01])
    model_config = ToolWeaverModelConfig(
        in_dim=int(embeddings.shape[1]),
        num_levels=num_levels,
        num_emb_list=tuple(args.branching_factors),
        e_dim=args.e_dim,
        layers=tuple(args.layers),
        dropout_prob=args.dropout,
        bn=args.batch_norm,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=tuple(sk_epsilons),
        sk_iters=args.sk_iters,
        graph_lambda=args.graph_lambda if args.use_graph else 0.0,
        token_format=args.token_format,
        codebook_version=args.codebook_version,
    )
    training_config = Stage1TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        gradient_clip=args.gradient_clip,
        amp_dtype=args.amp_dtype,
        seed=args.seed,
        eval_every=args.eval_every,
        max_graph_edges_per_batch=args.max_graph_edges_per_batch,
        edge_aware_batches=args.edge_aware_batches,
        normalize_embeddings=args.normalize_embeddings,
        selection_metric=args.selection_metric,
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = ToolWeaverStage1Trainer(
        model_config,
        training_config,
        embeddings,
        graph,
        output_dir,
        device=device,
        data_provenance={
            **source,
            "manifest_file_sha256": file_sha256(manifest_path),
        },
    )
    result = trainer.fit(args.resume)
    best_model, _ = load_toolweaver_rqvae(result["best_checkpoint"], device=device)
    codes = encode_embeddings(
        best_model,
        embeddings,
        device=device,
        batch_size=args.export_batch_size,
        num_levels=model_config.num_levels,
        normalize=training_config.normalize_embeddings,
    )
    registry = export_assignments(output_dir, skill_ids, codes, model_config, source)
    summary = {
        **result,
        "data": source,
        "code_metrics": code_assignment_metrics(codes, model_config.num_emb_list),
        "num_buckets": len(registry["buckets"]),
        "code_assignment": {
            "mode": "sinkhorn",
            "scope": "full_catalog",
            "sk_epsilons": list(model_config.sk_epsilons),
            "sk_iters": model_config.sk_iters,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
