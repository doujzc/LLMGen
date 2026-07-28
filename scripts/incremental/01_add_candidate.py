#!/usr/bin/env python3
"""Compute one frozen-codebook skill code and activate it in a candidate overlay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from llmgen.embeddings import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
)
from llmgen.incremental import (
    add_candidate,
    compute_frozen_skill_code,
    load_candidate_state,
    write_candidate_state,
)
from llmgen.router import RouterDataError, skill_document_text
from llmgen.skillret import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-state-dir",
        type=Path,
        required=True,
        help="Router model bundle or a previous incremental candidate-state directory.",
    )
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True, help="One JSON skill object.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--update-mode",
        choices=("index_only", "lora_train"),
        default="index_only",
        help="Record whether the state will be used directly or followed by tiny LoRA SFT.",
    )
    parser.add_argument(
        "--assignment-mode",
        choices=("nearest", "nearest_available"),
        default="nearest_available",
        help=(
            "nearest is ordinary frozen RQ inference and may collide; "
            "nearest_available keeps old codes fixed and selects the nearest unused path."
        ),
    )
    parser.add_argument("--assignment-beam-size", type=int, default=512)
    parser.add_argument("--embedding-npy", type=Path)
    parser.add_argument("--embedding-model", default=DEFAULT_OPENAI_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--embedding-timeout", type=float, default=600.0)
    parser.add_argument("--embedding-max-retries", type=int, default=5)
    parser.add_argument("--max-skill-chars", type=int, default=1400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_skill(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise RouterDataError(f"skill JSON does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RouterDataError(f"invalid skill JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise RouterDataError("--skill must contain one JSON object")
    return payload


def _embedding(
    args: argparse.Namespace,
    skill: dict[str, Any],
    document_text: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.embedding_npy is not None:
        source = args.embedding_npy.expanduser().resolve()
        if not source.is_file():
            raise RouterDataError(f"embedding NPY does not exist: {source}")
        return np.asarray(np.load(source, allow_pickle=False), dtype=np.float32), {
            "provider": "npy",
            "path": str(source),
            "sha256": sha256_file(source),
        }
    raw_embedding = skill.get("embedding")
    if isinstance(raw_embedding, list):
        return np.asarray(raw_embedding, dtype=np.float32), {
            "provider": "skill_json",
        }
    config = OpenAIEmbeddingConfig(
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimensions=args.embedding_dimensions,
        timeout=args.embedding_timeout,
        max_retries=args.embedding_max_retries,
    )
    model = OpenAIEmbeddingModel(config)
    try:
        values = model.encode([document_text], normalize_embeddings=True)[0]
    finally:
        model.close()
    return values, {
        "provider": "openai",
        "model": config.model,
        "base_url": config.base_url,
        "dimensions": int(values.shape[0]),
    }


def main() -> None:
    args = parse_args()
    if args.assignment_beam_size < 1:
        raise RouterDataError("--assignment-beam-size must be positive")
    if args.max_skill_chars < 1:
        raise RouterDataError("--max-skill-chars must be positive")
    source, decode_path, _ = load_candidate_state(args.source_state_dir)
    skill = _load_skill(args.skill)
    document_text = skill_document_text(skill).strip()[: args.max_skill_chars]
    if not document_text:
        raise RouterDataError("new skill has no text to embed")
    embedding, embedding_provenance = _embedding(
        args,
        skill,
        document_text,
    )
    code = compute_frozen_skill_code(
        skill=skill,
        stage1_checkpoint=args.stage1_checkpoint,
        source_decode_map=source,
        embedding=embedding,
        assignment_mode=args.assignment_mode,
        assignment_beam_size=args.assignment_beam_size,
        device=args.device,
    )
    updated, operation = add_candidate(
        source,
        skill=skill,
        indices=code["indices"],
        tokens=code["tokens"],
        source_sha256=sha256_file(decode_path),
        assignment_mode=args.assignment_mode,
        update_mode=args.update_mode,
        document_text=document_text,
    )
    operation.update(
        {
            "distance": code["distance"],
            "stage1_checkpoint_sha256": code["stage1_checkpoint_sha256"],
            "embedding": embedding_provenance,
        }
    )
    result = write_candidate_state(
        args.output_dir,
        updated,
        operation,
        overwrite=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
