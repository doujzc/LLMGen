#!/usr/bin/env python3
"""Attach decoder artifacts or materialize a deployable router checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from llmgen.router import (
    RouterDataError,
    code_token_id_map,
    load_virtual_tokens,
    mix_replay_sources,
    read_jsonl,
)
from llmgen.router_bundle import (
    BUNDLED_VIRTUAL_TOKENS_FILENAME,
    dump_router_decoder_artifacts,
)
from llmgen.skillret import sha256_file


MEMORIZATION_SYSTEM_PROMPT = (
    "Map the Agent Skill document to its fixed-length hierarchical skill code. "
    "Answer with code tokens only."
)
RETRIEVAL_SYSTEM_PROMPT = (
    "Select every Agent Skill needed for the user request in execution order. "
    "Output one hierarchical skill code per line, with no other text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Complete router model directory, or a completed Hugging Face "
            "checkpoint-N directory when --output-dir is supplied."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Materialized deployment directory for an intermediate checkpoint.",
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--virtual-tokens", required=True)
    parser.add_argument(
        "--training-data",
        help="Router phase training JSONL used to annotate target supervision.",
    )
    parser.add_argument("--validation-data")
    parser.add_argument(
        "--memorization-replay-data",
        "--replay-data",
        dest="replay_data",
    )
    parser.add_argument(
        "--memorization-replay-fraction",
        "--replay-fraction",
        dest="replay_fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument("--alignment-replay-data")
    parser.add_argument(
        "--alignment-replay-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument("--phase", choices=("memorization", "retrieval"))
    parser.add_argument(
        "--tokenizer-source",
        help=(
            "Tokenizer directory from the completed preceding phase. Required "
            "when materializing an intermediate checkpoint."
        ),
    )
    parser.add_argument(
        "--template-manifest",
        help="Preceding phase router_manifest.json used for shared training metadata.",
    )
    parser.add_argument("--num-levels", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model-name-or-path")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RouterDataError(f"invalid JSON object: {source}") from exc
    if not isinstance(payload, dict):
        raise RouterDataError(f"expected a JSON object: {source}")
    return payload


def _checkpoint_step(checkpoint_dir: Path) -> tuple[int, dict[str, Any]]:
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_dir.name)
    if match is None:
        raise RouterDataError(
            "intermediate model directory must be named checkpoint-<global_step>"
        )
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        raise RouterDataError(
            "checkpoint has no trainer_state.json and may still be saving; "
            f"wait for a completed checkpoint: {checkpoint_dir}"
        )
    state = _load_json_object(state_path)
    directory_step = int(match.group(1))
    try:
        recorded_step = int(state["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RouterDataError("checkpoint trainer_state.json has no global_step") from exc
    if recorded_step != directory_step:
        raise RouterDataError(
            "checkpoint directory and trainer_state.json disagree on global_step"
        )
    return directory_step, state


def _safe_index_weight_files(checkpoint_dir: Path, index_name: str) -> list[Path]:
    index_path = checkpoint_dir / index_name
    if not index_path.is_file():
        return []
    payload = _load_json_object(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RouterDataError(f"model weight index has no weight_map: {index_path}")
    checkpoint_root = checkpoint_dir.resolve()
    files = [index_path]
    for raw_name in sorted(set(weight_map.values())):
        if not isinstance(raw_name, str) or not raw_name:
            raise RouterDataError(f"invalid model shard in weight index: {index_path}")
        shard = (checkpoint_dir / raw_name).resolve()
        if shard.parent != checkpoint_root or not shard.is_file():
            raise RouterDataError(
                f"model weight index references a missing/unsafe shard: {raw_name}"
            )
        files.append(shard)
    return files


def _checkpoint_inference_files(checkpoint_dir: Path) -> tuple[list[Path], str]:
    """Return only files needed for full-model or PEFT inference."""

    is_adapter = (checkpoint_dir / "adapter_config.json").is_file()
    if is_adapter:
        indexed = (
            "adapter_model.safetensors.index.json",
            "adapter_model.bin.index.json",
        )
        single = ("adapter_model.safetensors", "adapter_model.bin")
        required_metadata = ("adapter_config.json",)
        mode = "adapter"
    else:
        indexed = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        single = ("model.safetensors", "pytorch_model.bin")
        required_metadata = ("config.json",)
        mode = "full"

    files: list[Path] = []
    for name in indexed:
        indexed_files = _safe_index_weight_files(checkpoint_dir, name)
        if indexed_files:
            files.extend(indexed_files)
            break
    if not files:
        for name in single:
            candidate = checkpoint_dir / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                files.append(candidate)
                break
    if not files:
        raise RouterDataError(
            "checkpoint has no consolidated inference weights. For ZeRO-3, enable "
            "stage3_gather_16bit_weights_on_model_save before training."
        )

    for name in required_metadata:
        candidate = checkpoint_dir / name
        if not candidate.is_file():
            raise RouterDataError(f"checkpoint is missing inference metadata: {candidate}")
        files.append(candidate)
    for name in ("config.json", "generation_config.json"):
        candidate = checkpoint_dir / name
        if candidate.is_file() and candidate not in files:
            files.append(candidate)
    files.extend(sorted(checkpoint_dir.glob("*.py")))
    return sorted(set(files)), mode


def _link_or_copy(source: Path, destination: Path) -> str:
    """Protect a checkpoint from rotation without rewriting multi-GB weights."""

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _save_checkpoint_tokenizer(
    *,
    tokenizer_source: Path,
    output_dir: Path,
    virtual_tokens_path: Path,
    full_model_config_path: Path | None,
    trust_remote_code: bool,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - real export environment
        raise RouterDataError(
            "checkpoint export requires transformers; install the training extras"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=trust_remote_code,
    )
    virtual_tokens = load_virtual_tokens(virtual_tokens_path)
    existing = list(getattr(tokenizer, "additional_special_tokens", ()))
    missing = [token for token in virtual_tokens if token not in existing]
    if missing:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [*existing, *missing]}
        )
    token_ids = code_token_id_map(tokenizer, virtual_tokens)
    tokenizer.save_pretrained(str(output_dir))

    vocab_size = len(tokenizer)
    if full_model_config_path is not None:
        config = _load_json_object(full_model_config_path)
        recorded_vocab_size = config.get("vocab_size")
        if (
            isinstance(recorded_vocab_size, bool)
            or not isinstance(recorded_vocab_size, int)
            or recorded_vocab_size != vocab_size
        ):
            raise RouterDataError(
                "checkpoint model/tokenizer vocabulary mismatch: "
                f"model={recorded_vocab_size!r}, tokenizer={vocab_size}"
            )
    return {
        "source": str(tokenizer_source.resolve()),
        "vocab_size": vocab_size,
        "num_virtual_tokens": len(token_ids),
    }


def _checkpoint_manifest(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    checkpoint_step: int,
    trainer_state: dict[str, Any],
    inference_files: list[Path],
    inference_mode: str,
    copy_modes: dict[str, str],
    tokenizer_metadata: dict[str, Any],
    template_manifest_path: Path | None,
    phase: str,
    num_levels: int,
    max_length: int,
    base_model_name_or_path: str | None,
    training_data_path: Path,
    validation_data_path: Path | None,
    replay_data_path: Path | None,
    replay_fraction: float,
    alignment_replay_data_path: Path | None,
    alignment_replay_fraction: float,
    seed: int,
    decoder_artifacts: dict[str, Any],
) -> dict[str, Any]:
    template = (
        _load_json_object(template_manifest_path)
        if template_manifest_path is not None
        else {}
    )
    primary_rows = read_jsonl(training_data_path)
    if not primary_rows:
        raise RouterDataError("checkpoint export training data is empty")
    alignment_replay_rows = (
        read_jsonl(alignment_replay_data_path)
        if alignment_replay_data_path is not None
        else []
    )
    memorization_replay_rows = (
        read_jsonl(replay_data_path)
        if replay_data_path is not None
        else []
    )
    mixed_rows, replay_counts = mix_replay_sources(
        primary_rows,
        (
            (
                "alignment",
                alignment_replay_rows,
                alignment_replay_fraction,
            ),
            ("memorization", memorization_replay_rows, replay_fraction),
        ),
        seed=seed,
    )
    replay_examples = sum(replay_counts.values())
    replay_sources = {}
    for name, path, rows, fraction in (
        (
            "alignment",
            alignment_replay_data_path,
            alignment_replay_rows,
            alignment_replay_fraction,
        ),
        (
            "memorization",
            replay_data_path,
            memorization_replay_rows,
            replay_fraction,
        ),
    ):
        if path is None:
            continue
        examples = replay_counts[name]
        replay_sources[name] = {
            "data": str(path),
            "data_sha256": sha256_file(path),
            "source_rows": len(rows),
            "examples": examples,
            "fraction_requested": fraction,
            "fraction_actual": examples / len(mixed_rows),
            "repeat_factor": examples / max(len(rows), 1),
        }
    validation_rows = (
        read_jsonl(validation_data_path)
        if validation_data_path is not None
        else []
    )
    max_target_paths = max(
        len(row.get("target_paths") or [row.get("target_tokens")])
        for row in [*mixed_rows, *validation_rows]
    )

    router_data_manifest_path = training_data_path.parent / "manifest.json"
    provenance = decoder_artifacts["provenance"]
    manifest = dict(template)
    manifest.pop("decoder_artifacts", None)
    manifest.update(
        {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "curriculum_stage": phase,
            "num_levels": num_levels,
            "virtual_tokens": BUNDLED_VIRTUAL_TOKENS_FILENAME,
            "virtual_tokens_sha256": decoder_artifacts[
                "virtual_tokens_sha256"
            ],
            "train_data": str(training_data_path),
            "train_data_sha256": sha256_file(training_data_path),
            "replay_data": (
                str(replay_data_path) if replay_data_path is not None else None
            ),
            "replay_data_sha256": (
                sha256_file(replay_data_path)
                if replay_data_path is not None
                else None
            ),
            "replay_fraction_requested": (
                replay_fraction + alignment_replay_fraction
            ),
            "replay_fraction_actual": replay_examples / len(mixed_rows),
            "replay_sources": replay_sources,
            "validation_data": (
                str(validation_data_path)
                if validation_data_path is not None
                else None
            ),
            "validation_data_sha256": (
                sha256_file(validation_data_path)
                if validation_data_path is not None
                else None
            ),
            "router_data_manifest_sha256": (
                sha256_file(router_data_manifest_path)
                if router_data_manifest_path.is_file()
                else None
            ),
            "index_manifest_sha256": provenance.get("index_manifest_sha256"),
            "stage1_checkpoint_sha256": provenance.get(
                "stage1_checkpoint_sha256"
            ),
            "base_model": (
                template.get("base_model") or base_model_name_or_path
            ),
            "finetune_mode": (
                template.get("finetune_mode")
                or ("continued_adapter" if inference_mode == "adapter" else "full")
            ),
            "system_prompt": (
                template.get("system_prompt")
                or (
                    RETRIEVAL_SYSTEM_PROMPT
                    if phase == "retrieval"
                    else MEMORIZATION_SYSTEM_PROMPT
                )
            ),
            "replay_system_prompt": (
                MEMORIZATION_SYSTEM_PROMPT
                if replay_counts.get("memorization", 0)
                else None
            ),
            "max_length": max_length,
            "generation_contract": {
                "mode": (
                    "autoregressive_multi_path"
                    if phase == "retrieval"
                    else "single_path"
                ),
                "path_separator": "\n" if phase == "retrieval" else None,
                "max_target_paths": max_target_paths,
            },
            "examples": {
                "train": len(mixed_rows),
                "primary_train": len(primary_rows),
                "replay": replay_examples,
                "replay_by_source": replay_counts,
                "validation": len(validation_rows),
            },
            "checkpoint_export": {
                "source": str(checkpoint_dir),
                "output": str(output_dir),
                "global_step": checkpoint_step,
                "epoch": trainer_state.get("epoch"),
                "trainer_state": "trainer_state.json",
                "trainer_state_sha256": sha256_file(
                    checkpoint_dir / "trainer_state.json"
                ),
                "inference_mode": inference_mode,
                "inference_files": [
                    path.name for path in inference_files
                ],
                "file_materialization": copy_modes,
                "tokenizer": tokenizer_metadata,
            },
            "decoder_artifacts": decoder_artifacts,
        }
    )
    return manifest


def materialize_checkpoint_bundle(
    *,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    tokenizer_source: str | Path,
    catalog_path: str | Path,
    codes_path: str | Path,
    registry_path: str | Path,
    virtual_tokens_path: str | Path,
    training_data_path: str | Path,
    validation_data_path: str | Path | None,
    replay_data_path: str | Path | None,
    replay_fraction: float,
    phase: str,
    num_levels: int,
    max_length: int,
    seed: int,
    template_manifest_path: str | Path | None,
    base_model_name_or_path: str | None,
    trust_remote_code: bool,
    alignment_replay_data_path: str | Path | None = None,
    alignment_replay_fraction: float = 0.0,
) -> dict[str, Any]:
    source = Path(checkpoint_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    tokenizer_source = Path(tokenizer_source).expanduser().resolve()
    catalog_path = Path(catalog_path).expanduser().resolve()
    codes_path = Path(codes_path).expanduser().resolve()
    registry_path = Path(registry_path).expanduser().resolve()
    virtual_tokens_path = Path(virtual_tokens_path).expanduser().resolve()
    training_data_path = Path(training_data_path).expanduser().resolve()
    validation_data_path = (
        Path(validation_data_path).expanduser().resolve()
        if validation_data_path is not None
        else None
    )
    replay_data_path = (
        Path(replay_data_path).expanduser().resolve()
        if replay_data_path is not None
        else None
    )
    alignment_replay_data_path = (
        Path(alignment_replay_data_path).expanduser().resolve()
        if alignment_replay_data_path is not None
        else None
    )
    template_manifest_path = (
        Path(template_manifest_path).expanduser().resolve()
        if template_manifest_path is not None
        else None
    )

    if not source.is_dir():
        raise RouterDataError(f"checkpoint directory does not exist: {source}")
    if not tokenizer_source.is_dir():
        raise RouterDataError(f"tokenizer source does not exist: {tokenizer_source}")
    if destination.exists():
        raise RouterDataError(
            f"checkpoint export destination already exists: {destination}"
        )
    if destination == source or source in destination.parents:
        raise RouterDataError("checkpoint export must be outside the source checkpoint")
    if phase not in {"memorization", "retrieval"}:
        raise RouterDataError(f"unsupported checkpoint phase: {phase!r}")
    if num_levels < 1 or max_length <= num_levels + 1:
        raise RouterDataError("invalid num_levels/max_length for checkpoint export")
    for name, path, fraction in (
        ("memorization", replay_data_path, replay_fraction),
        (
            "alignment",
            alignment_replay_data_path,
            alignment_replay_fraction,
        ),
    ):
        if not 0.0 <= fraction < 1.0:
            raise RouterDataError(f"{name} replay fraction must be in [0, 1)")
        if bool(path) != (fraction > 0.0):
            raise RouterDataError(
                f"set both {name} replay data and a positive fraction, or neither"
            )
    if replay_fraction + alignment_replay_fraction >= 1.0:
        raise RouterDataError("total replay fraction must be less than 1")
    required_files = [
        catalog_path,
        codes_path,
        registry_path,
        virtual_tokens_path,
        training_data_path,
    ]
    if validation_data_path is not None:
        required_files.append(validation_data_path)
    if replay_data_path is not None:
        required_files.append(replay_data_path)
    if alignment_replay_data_path is not None:
        required_files.append(alignment_replay_data_path)
    if template_manifest_path is not None:
        required_files.append(template_manifest_path)
    for path in required_files:
        if not path.is_file():
            raise RouterDataError(f"checkpoint export input does not exist: {path}")

    checkpoint_step, trainer_state = _checkpoint_step(source)
    inference_files, inference_mode = _checkpoint_inference_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=str(destination.parent),
        )
    )
    try:
        copy_modes: dict[str, str] = {}
        for source_file in [*inference_files, source / "trainer_state.json"]:
            destination_file = temporary / source_file.name
            copy_modes[source_file.name] = _link_or_copy(
                source_file, destination_file
            )
        tokenizer_metadata = _save_checkpoint_tokenizer(
            tokenizer_source=tokenizer_source,
            output_dir=temporary,
            virtual_tokens_path=virtual_tokens_path,
            full_model_config_path=(
                temporary / "config.json"
                if inference_mode == "full"
                else None
            ),
            trust_remote_code=trust_remote_code,
        )
        primary_supervision_rows = read_jsonl(training_data_path)
        memorization_replay_supervision_rows = (
            read_jsonl(replay_data_path)
            if replay_data_path is not None
            else []
        )
        alignment_replay_supervision_rows = (
            read_jsonl(alignment_replay_data_path)
            if alignment_replay_data_path is not None
            else []
        )
        effective_supervision_rows, _ = mix_replay_sources(
            primary_supervision_rows,
            (
                (
                    "alignment",
                    alignment_replay_supervision_rows,
                    alignment_replay_fraction,
                ),
                (
                    "memorization",
                    memorization_replay_supervision_rows,
                    replay_fraction,
                ),
            ),
            seed=seed,
        )
        decoder_artifacts = dump_router_decoder_artifacts(
            output_dir=temporary,
            catalog_path=catalog_path,
            codes_path=codes_path,
            registry_path=registry_path,
            virtual_tokens_path=virtual_tokens_path,
            training_data_path=training_data_path,
            supervision_phase=phase,
            supervision_rows=effective_supervision_rows,
        )
        manifest = _checkpoint_manifest(
            checkpoint_dir=source,
            output_dir=destination,
            checkpoint_step=checkpoint_step,
            trainer_state=trainer_state,
            inference_files=inference_files,
            inference_mode=inference_mode,
            copy_modes=copy_modes,
            tokenizer_metadata=tokenizer_metadata,
            template_manifest_path=template_manifest_path,
            phase=phase,
            num_levels=num_levels,
            max_length=max_length,
            base_model_name_or_path=base_model_name_or_path,
            training_data_path=training_data_path,
            validation_data_path=validation_data_path,
            replay_data_path=replay_data_path,
            replay_fraction=replay_fraction,
            alignment_replay_data_path=alignment_replay_data_path,
            alignment_replay_fraction=alignment_replay_fraction,
            seed=seed,
            decoder_artifacts=decoder_artifacts,
        )
        manifest_path = temporary / "router_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "mode": "checkpoint_materialization",
        "checkpoint": str(source),
        "global_step": checkpoint_step,
        "output_dir": str(destination),
        "inference_mode": inference_mode,
        "model_files": [path.name for path in inference_files],
        "decoder_artifacts": decoder_artifacts,
    }


def attach_decoder_artifacts(
    *,
    model_dir: str | Path,
    catalog_path: str | Path,
    codes_path: str | Path,
    registry_path: str | Path,
    virtual_tokens_path: str | Path,
    training_data_path: str | Path | None,
    phase: str | None,
    replay_data_path: str | Path | None = None,
    replay_fraction: float = 0.0,
    seed: int = 42,
    alignment_replay_data_path: str | Path | None = None,
    alignment_replay_fraction: float = 0.0,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise RouterDataError(f"model directory does not exist: {model_dir}")
    manifest_path = model_dir / "router_manifest.json"
    if not manifest_path.is_file():
        raise RouterDataError(
            f"router manifest does not exist; not a complete router dump: {model_dir}"
        )
    manifest = _load_json_object(manifest_path)
    training_data = training_data_path or manifest.get("train_data")
    if training_data and not Path(training_data).is_file():
        raise RouterDataError(
            f"router training data does not exist: {training_data}; "
            "pass --training-data with its current location"
        )
    manifest_replay_sources = manifest.get("replay_sources")
    structured_replay = (
        manifest_replay_sources
        if isinstance(manifest_replay_sources, dict)
        else None
    )
    if replay_data_path is None:
        memorization_source = (
            structured_replay.get("memorization")
            if structured_replay is not None
            else None
        )
        if isinstance(memorization_source, dict):
            replay_data_path = memorization_source.get("data")
            replay_fraction = float(
                memorization_source.get("fraction_requested", 0.0)
            )
        elif structured_replay is None:
            replay_data_path = manifest.get("replay_data")
            replay_fraction = float(
                manifest.get("replay_fraction_requested", replay_fraction)
            )
    if alignment_replay_data_path is None and structured_replay is not None:
        alignment_source = structured_replay.get("alignment")
        if isinstance(alignment_source, dict):
            alignment_replay_data_path = alignment_source.get("data")
            alignment_replay_fraction = float(
                alignment_source.get("fraction_requested", 0.0)
            )
    for name, path, fraction in (
        ("memorization", replay_data_path, replay_fraction),
        (
            "alignment",
            alignment_replay_data_path,
            alignment_replay_fraction,
        ),
    ):
        if path and not Path(path).is_file():
            raise RouterDataError(
                f"router {name} replay data does not exist: {path}; "
                f"pass --{name}-replay-data with its current location"
            )
        if not 0.0 <= fraction < 1.0:
            raise RouterDataError(f"{name} replay fraction must be in [0, 1)")
        if bool(path) != (fraction > 0.0):
            raise RouterDataError(
                f"set both {name} replay data and a positive fraction, or neither"
            )
    if replay_fraction + alignment_replay_fraction >= 1.0:
        raise RouterDataError("total replay fraction must be less than 1")
    effective_supervision_rows = None
    if training_data:
        effective_supervision_rows, _ = mix_replay_sources(
            read_jsonl(training_data),
            (
                (
                    "alignment",
                    (
                        read_jsonl(alignment_replay_data_path)
                        if alignment_replay_data_path
                        else []
                    ),
                    alignment_replay_fraction,
                ),
                (
                    "memorization",
                    read_jsonl(replay_data_path) if replay_data_path else [],
                    replay_fraction,
                ),
            ),
            seed=seed,
        )
    supervision_phase = phase or manifest.get("phase")
    artifacts = dump_router_decoder_artifacts(
        output_dir=model_dir,
        catalog_path=catalog_path,
        codes_path=codes_path,
        registry_path=registry_path,
        virtual_tokens_path=virtual_tokens_path,
        training_data_path=training_data,
        supervision_phase=(
            str(supervision_phase) if supervision_phase else None
        ),
        supervision_rows=effective_supervision_rows,
    )
    manifest["decoder_artifacts"] = artifacts
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return {
        "mode": "attach_decoder_artifacts",
        "model_dir": str(model_dir.resolve()),
        "decoder_artifacts": artifacts,
    }


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if args.output_dir:
        required = {
            "--tokenizer-source": args.tokenizer_source,
            "--training-data": args.training_data,
            "--phase": args.phase,
            "--num-levels": args.num_levels,
            "--max-length": args.max_length,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RouterDataError(
                "checkpoint materialization is missing: " + ", ".join(missing)
            )
        result = materialize_checkpoint_bundle(
            checkpoint_dir=model_dir,
            output_dir=args.output_dir,
            tokenizer_source=args.tokenizer_source,
            catalog_path=args.catalog,
            codes_path=args.codes,
            registry_path=args.registry,
            virtual_tokens_path=args.virtual_tokens,
            training_data_path=args.training_data,
            validation_data_path=args.validation_data,
            replay_data_path=args.replay_data,
            replay_fraction=args.replay_fraction,
            phase=args.phase,
            num_levels=args.num_levels,
            max_length=args.max_length,
            seed=args.seed,
            template_manifest_path=args.template_manifest,
            base_model_name_or_path=args.base_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            alignment_replay_data_path=args.alignment_replay_data,
            alignment_replay_fraction=args.alignment_replay_fraction,
        )
    else:
        result = attach_decoder_artifacts(
            model_dir=model_dir,
            catalog_path=args.catalog,
            codes_path=args.codes,
            registry_path=args.registry,
            virtual_tokens_path=args.virtual_tokens,
            training_data_path=args.training_data,
            phase=args.phase,
            replay_data_path=args.replay_data,
            replay_fraction=args.replay_fraction,
            seed=args.seed,
            alignment_replay_data_path=args.alignment_replay_data,
            alignment_replay_fraction=args.alignment_replay_fraction,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
