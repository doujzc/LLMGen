#!/usr/bin/env python3
"""Train one direct candidate-name Top1 router."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from llmgen.diagnostics import build_curve_summary, build_data_profile
from llmgen.experiment import (
    TRAINING_RUN_SCHEMA_VERSION,
    RunStore,
    canonical_sha256,
    directory_file_manifest,
    git_snapshot,
    json_safe,
    make_training_log_callback,
    system_snapshot,
    utc_now,
    write_model_artifact_manifest,
    write_trainer_history,
)
from llmgen.top1 import (
    CONVERSATION_TEMPLATE,
    MAX_ASSISTANT_HISTORY_CHARACTERS,
    MAX_HISTORY_CHARACTERS,
    MAX_HISTORY_MESSAGES,
    INFERENCE_DECISION_RULE,
    ROUTING_MODE,
    TARGET_CONTRACT,
    Top1DataError,
    candidate_registry_payload,
    candidate_token_sequences,
    load_candidate_names,
    prepare_example,
    prompt_implementation_sha256,
    read_jsonl,
    sha256_file,
    tokenizer_prompt_contract,
    validate_memorization_rows,
    validate_training_rows,
    write_json,
    write_jsonl,
)


SUPPORTED_DEEPSPEED_VERSION = "0.16.4"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a direct candidate-name Top1 causal-LM router."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--memorization-data")
    parser.add_argument(
        "--memorization-steps",
        type=int,
        default=0,
        help="Optimizer steps spent on description-to-label memorization before main training.",
    )
    parser.add_argument("--validation-data")
    parser.add_argument("--candidate-registry", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="top1")
    parser.add_argument("--run-id")

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--deepspeed")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing-mode",
        choices=("auto", "reentrant", "non-reentrant"),
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume-from-checkpoint")

    parser.add_argument("--finetune-mode", choices=("full", "lora"), default="full")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module suffixes.",
    )
    return parser.parse_args(argv)


def _csv(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise Top1DataError("LoRA target module list cannot be empty")
    return values


def _resume_value(value: str | None) -> str | bool | None:
    if value is None:
        return None
    return True if value.casefold() in {"latest", "true"} else value


def _read_deepspeed_config(value: str | Path) -> tuple[Path, dict[str, Any], int]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise Top1DataError(f"DeepSpeed config does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Top1DataError(f"invalid DeepSpeed JSON config: {path}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError("DeepSpeed config must be a JSON object")
    zero = payload.get("zero_optimization")
    stage = zero.get("stage") if isinstance(zero, dict) else None
    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 3:
        raise Top1DataError(
            "DeepSpeed config must define zero_optimization.stage from 0 to 3"
        )
    return path, payload, stage


def _require_supported_deepspeed_version() -> str:
    try:
        installed = importlib.metadata.version("deepspeed")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Top1DataError(
            "DeepSpeed is not installed; install the project's training dependencies"
        ) from exc
    if installed != SUPPORTED_DEEPSPEED_VERSION:
        raise Top1DataError(
            f"expected deepspeed=={SUPPORTED_DEEPSPEED_VERSION}, found {installed}"
        )
    return installed


def _gradient_checkpointing_kwargs(
    args: argparse.Namespace,
) -> dict[str, bool] | None:
    if not args.gradient_checkpointing:
        return None
    if args.gradient_checkpointing_mode == "auto":
        return {"use_reentrant": bool(args.deepspeed)}
    return {"use_reentrant": args.gradient_checkpointing_mode == "reentrant"}


def _validate_args(args: argparse.Namespace) -> None:
    required_files = {
        "train data": args.train_data,
        "candidate registry": args.candidate_registry,
        "system prompt": args.system_prompt_file,
    }
    if args.validation_data:
        required_files["validation data"] = args.validation_data
    if args.memorization_data:
        required_files["memorization data"] = args.memorization_data
    for label, value in required_files.items():
        if not Path(value).expanduser().is_file():
            raise Top1DataError(f"{label} does not exist: {value}")

    positive_values = {
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise Top1DataError("values must be positive: " + ", ".join(invalid))
    if args.dataloader_num_workers < 0:
        raise Top1DataError("dataloader_num_workers cannot be negative")
    if args.memorization_data and args.memorization_steps <= 0:
        raise Top1DataError(
            "memorization_steps must be positive when memorization_data is set"
        )
    if not args.memorization_data and args.memorization_steps:
        raise Top1DataError(
            "memorization_data is required when memorization_steps is non-zero"
        )
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise Top1DataError("warmup_ratio must be in [0, 1]")
    if args.weight_decay < 0:
        raise Top1DataError("weight_decay cannot be negative")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise Top1DataError("lora_dropout must be in [0, 1)")
    if args.validation_data and args.save_steps % args.eval_steps:
        raise Top1DataError(
            "save_steps must be a multiple of eval_steps when validation is enabled"
        )


def _import_training_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised in training environments
        raise SystemExit(
            "Top1 training requires torch and transformers. "
            "Install with: python -m pip install -e '.[train]'"
        ) from exc
    return torch, transformers


def _build_training_arguments(
    args: argparse.Namespace,
    transformers: Any,
    *,
    has_validation: bool,
    resume_from_checkpoint: str | bool | None,
    staged_training: bool = False,
) -> Any:
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    training_kwargs = {
        "output_dir": str(checkpoint_dir),
        "overwrite_output_dir": resume_from_checkpoint is None,
        "num_train_epochs": 1.0 if staged_training else args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "load_best_model_at_end": has_validation,
        "metric_for_best_model": "eval_loss" if has_validation else None,
        "greater_is_better": False if has_validation else None,
        "save_total_limit": args.save_total_limit,
        "bf16": args.precision == "bf16",
        "fp16": args.precision == "fp16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": _gradient_checkpointing_kwargs(args),
        "dataloader_num_workers": args.dataloader_num_workers,
        "deepspeed": args.deepspeed,
        "local_rank": args.local_rank,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "report_to": [],
        "logging_nan_inf_filter": False,
        "include_num_input_tokens_seen": "non_padding",
        "seed": args.seed,
        "data_seed": args.seed,
    }
    parameters = inspect.signature(transformers.TrainingArguments.__init__).parameters
    if staged_training:
        if "train_sampling_strategy" not in parameters:
            raise Top1DataError(
                "memorization curriculum requires Transformers with "
                "TrainingArguments.train_sampling_strategy"
            )
        training_kwargs["train_sampling_strategy"] = "sequential"
    evaluation_key = (
        "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    )
    training_kwargs[evaluation_key] = "steps" if has_validation else "no"
    return transformers.TrainingArguments(
        **{
            key: value
            for key, value in training_kwargs.items()
            if key in parameters
        }
    )


def _load_model_and_tokenizer(
    args: argparse.Namespace,
    torch: Any,
    transformers: Any,
) -> tuple[Any, Any]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise Top1DataError("base tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.precision == "bf16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.precision == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    args.base_model_revision = getattr(model.config, "_commit_hash", None)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    if args.finetune_mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - training environment
            raise SystemExit("LoRA training requires peft.") from exc
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=_csv(args.lora_target_modules),
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    return tokenizer, model


def _prepare_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str | Path,
    tokenizer: Any,
    candidate_tokens: Mapping[str, Sequence[int]],
    max_length: int,
    system_prompt: str,
) -> tuple[
    list[dict[str, list[int]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    encoded: list[dict[str, list[int]]] = []
    sft_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        try:
            prepared = prepare_example(
                tokenizer,
                row,
                candidate_tokens=candidate_tokens,
                max_length=max_length,
                system_prompt=system_prompt,
            )
        except Top1DataError as exc:
            raise Top1DataError(f"{source}:{row_number}: {exc}") from exc
        encoded.append(prepared.encoded)
        sft_rows.append(prepared.sft_row)
        diagnostics.append(prepared.diagnostics)
    return encoded, sft_rows, diagnostics


def _shuffled_indices(length: int, *, seed: int) -> list[int]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices


def _repeated_shuffled_indices(
    length: int,
    *,
    count: int,
    seed: int,
) -> list[int]:
    if length <= 0:
        raise Top1DataError("cannot schedule an empty training stage")
    result: list[int] = []
    cycle = 0
    while len(result) < count:
        result.extend(_shuffled_indices(length, seed=seed + cycle))
        cycle += 1
    return result[:count]


def _main_training_indices(
    row_count: int,
    *,
    epochs: float,
    seed: int,
) -> list[int]:
    if row_count <= 0:
        raise Top1DataError("main training data cannot be empty")
    full_epochs = int(math.floor(epochs))
    fractional_epoch = epochs - full_epochs
    result: list[int] = []
    for epoch_index in range(full_epochs):
        result.extend(
            _shuffled_indices(row_count, seed=seed + 100_000 + epoch_index)
        )
    if fractional_epoch:
        final_epoch = _shuffled_indices(
            row_count,
            seed=seed + 100_000 + full_epochs,
        )
        result.extend(final_epoch[: max(1, math.ceil(row_count * fractional_epoch))])
    return result


def _build_training_schedule(
    memorization_examples: Sequence[Mapping[str, Sequence[int]]],
    train_examples: Sequence[Mapping[str, Sequence[int]]],
    *,
    memorization_steps: int,
    main_epochs: float,
    effective_global_batch_size: int,
    seed: int,
) -> tuple[list[Mapping[str, Sequence[int]]], dict[str, Any]]:
    """Build a deterministic, optimizer-step-aligned memorization prefix."""

    if memorization_steps <= 0:
        raise Top1DataError("memorization_steps must be positive")
    if effective_global_batch_size <= 0:
        raise Top1DataError("effective_global_batch_size must be positive")
    memorization_sample_count = memorization_steps * effective_global_batch_size
    memorization_indices = _repeated_shuffled_indices(
        len(memorization_examples),
        count=memorization_sample_count,
        seed=seed,
    )
    main_indices = _main_training_indices(
        len(train_examples),
        epochs=main_epochs,
        seed=seed,
    )
    examples = [memorization_examples[index] for index in memorization_indices]
    examples.extend(train_examples[index] for index in main_indices)
    main_steps = math.ceil(len(main_indices) / effective_global_batch_size)
    metadata = {
        "schema_version": 1,
        "algorithm": "memorization_prefix_v1",
        "sampling": "deterministic_epoch_shuffle_then_sequential",
        "seed": seed,
        "effective_global_batch_size": effective_global_batch_size,
        "memorization": {
            "input_rows": len(memorization_examples),
            "scheduled_samples": len(memorization_indices),
            "optimizer_steps": memorization_steps,
            "ends_after_global_step": memorization_steps,
        },
        "main": {
            "input_rows": len(train_examples),
            "requested_epochs": main_epochs,
            "scheduled_samples": len(main_indices),
            "expected_optimizer_steps": main_steps,
        },
        "total": {
            "scheduled_samples": len(examples),
            "expected_optimizer_steps": memorization_steps + main_steps,
        },
        "order_sha256": canonical_sha256(
            {
                "memorization_indices": memorization_indices,
                "main_indices": main_indices,
            }
        ),
    }
    return examples, metadata


def _dataset_class(torch: Any):
    class Top1Dataset(torch.utils.data.Dataset):
        def __init__(self, examples: Sequence[Mapping[str, Sequence[int]]]) -> None:
            self.examples = list(examples)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> Mapping[str, Sequence[int]]:
            return self.examples[index]

    return Top1Dataset


def _collator(torch: Any, pad_token_id: int):
    def collate(features: list[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        max_length = max(len(row["input_ids"]) for row in features)
        input_ids = []
        attention_mask = []
        labels = []
        for row in features:
            padding = max_length - len(row["input_ids"])
            input_ids.append([*row["input_ids"], *([pad_token_id] * padding)])
            attention_mask.append([*row["attention_mask"], *([0] * padding)])
            labels.append([*row["labels"], *([-100] * padding)])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _write_bundle(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    tokenizer: Any,
    model: Any,
    candidate_names: tuple[str, ...],
    candidate_tokens: Mapping[str, Sequence[int]],
    system_prompt: str,
    train_report: Mapping[str, Any],
    memorization_report: Mapping[str, Any] | None,
    validation_report: Mapping[str, Any] | None,
    world_size: int,
    deepspeed_metadata: Mapping[str, Any] | None,
    training_run_id: str,
    prepared_train_path: Path,
    prepared_memorization_path: Path | None,
    training_schedule: Mapping[str, Any] | None,
    transformers_version: str,
) -> None:
    tokenizer.save_pretrained(str(output_dir))
    bundled_registry = output_dir / "candidate_registry.json"
    write_json(bundled_registry, candidate_registry_payload(candidate_names))
    bundled_prompt = output_dir / "router_system_prompt.md"
    bundled_prompt.write_text(system_prompt.rstrip() + "\n", encoding="utf-8")
    base_model_dependency = _base_model_dependency(args, model)
    base_model_revision = getattr(
        args,
        "base_model_revision",
        getattr(model.config, "_commit_hash", None),
    )
    manifest = {
        "schema_version": 2,
        "routing_mode": ROUTING_MODE,
        "base_model": args.model_name_or_path,
        "base_model_revision": base_model_revision,
        "base_model_dependency": base_model_dependency,
        "finetune_mode": args.finetune_mode,
        "train_data": {
            "path": str(Path(args.train_data).expanduser().resolve()),
            "sha256": sha256_file(args.train_data),
            **train_report,
        },
        "memorization_data": (
            {
                "path": str(Path(args.memorization_data).expanduser().resolve()),
                "sha256": sha256_file(args.memorization_data),
                **memorization_report,
            }
            if args.memorization_data and memorization_report is not None
            else None
        ),
        "validation_data": (
            {
                "path": str(Path(args.validation_data).expanduser().resolve()),
                "sha256": sha256_file(args.validation_data),
                **validation_report,
            }
            if args.validation_data and validation_report is not None
            else None
        ),
        "candidate_registry": {
            "path": bundled_registry.name,
            "sha256": sha256_file(bundled_registry),
            "candidate_names": list(candidate_names),
            "token_sequences": {
                name: list(candidate_tokens[name]) for name in candidate_names
            },
        },
        "system_prompt": {
            "path": bundled_prompt.name,
            "sha256": sha256_file(bundled_prompt),
        },
        "conversation": {
            "template": CONVERSATION_TEMPLATE,
            "max_history_messages": MAX_HISTORY_MESSAGES,
            "max_history_characters": MAX_HISTORY_CHARACTERS,
            "max_assistant_history_characters": MAX_ASSISTANT_HISTORY_CHARACTERS,
            "implementation_sha256": prompt_implementation_sha256(),
        },
        "tokenizer": tokenizer_prompt_contract(
            tokenizer,
            transformers_version=transformers_version,
        ),
        "target": TARGET_CONTRACT,
        "inference": {
            "decision_rule": INFERENCE_DECISION_RULE,
            "include_eos": True,
        },
        "max_length": args.max_length,
        "training": {
            "training_run_id": training_run_id,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "precision": args.precision,
            "trust_remote_code": args.trust_remote_code,
            "seed": args.seed,
            "world_size": world_size,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch_size": (
                world_size
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
            ),
            "curriculum": training_schedule,
            "deepspeed": deepspeed_metadata,
        },
        "sft_input": {
            "path": "../../prepared/train.sft.jsonl",
            "sha256": sha256_file(prepared_train_path),
            "rows": train_report["rows"],
        },
        "memorization_sft_input": (
            {
                "path": "../../prepared/memorization.sft.jsonl",
                "sha256": sha256_file(prepared_memorization_path),
                "rows": memorization_report["rows"],
            }
            if prepared_memorization_path is not None
            and memorization_report is not None
            else None
        ),
    }
    write_json(output_dir / "router_manifest.json", manifest)


def _base_model_dependency(
    args: argparse.Namespace,
    model: Any,
) -> dict[str, Any]:
    revision = getattr(
        args,
        "base_model_revision",
        getattr(model.config, "_commit_hash", None),
    )
    if args.finetune_mode == "full":
        return {"kind": "self_contained"}
    base_path = Path(args.model_name_or_path).expanduser()
    if base_path.is_dir():
        files = directory_file_manifest(base_path)
        identity = {"schema_version": 1, "files": files}
        return {
            "kind": "local_directory",
            "reference": str(base_path.resolve()),
            "content_id": canonical_sha256(identity),
        }
    if not isinstance(revision, str) or not revision:
        raise Top1DataError(
            "LoRA training requires a resolved base model revision or a local directory"
        )
    return {
        "kind": "hub_revision",
        "reference": args.model_name_or_path,
        "revision": revision,
    }


def _safe_component(value: str, *, label: str) -> str:
    cleaned = value.strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise Top1DataError(f"{label} must be one non-empty path component")
    return cleaned


def _build_run_manifest(
    *,
    args: argparse.Namespace,
    run_id: str,
    candidate_names: Sequence[str],
    system_prompt: str,
    train_report: Mapping[str, Any],
    memorization_report: Mapping[str, Any] | None,
    validation_report: Mapping[str, Any] | None,
    deepspeed_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        launcher_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise Top1DataError("WORLD_SIZE must be an integer") from exc
    repository = Path(__file__).resolve().parents[1]
    code_files = (
        Path(__file__).resolve(),
        repository / "src" / "llmgen" / "top1.py",
        repository / "src" / "llmgen" / "experiment.py",
        repository / "src" / "llmgen" / "diagnostics.py",
    )
    code = {
        "git": git_snapshot(repository),
        "files": {
            path.relative_to(repository).as_posix(): sha256_file(path)
            for path in code_files
        },
    }
    semantic_deepspeed = (
        {
            key: value
            for key, value in deepspeed_metadata.items()
            if key != "path"
        }
        if deepspeed_metadata is not None
        else None
    )
    identity = {
        "task": ROUTING_MODE,
        "base_model": args.model_name_or_path,
        "finetune_mode": args.finetune_mode,
        "inputs": {
            "train_sha256": sha256_file(args.train_data),
            "memorization_sha256": (
                sha256_file(args.memorization_data)
                if args.memorization_data
                else None
            ),
            "validation_sha256": (
                sha256_file(args.validation_data) if args.validation_data else None
            ),
            "candidate_registry_sha256": sha256_file(args.candidate_registry),
            "system_prompt_sha256": sha256_file(args.system_prompt_file),
        },
        "configuration": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "memorization_steps": args.memorization_steps,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "eval_accumulation_steps": args.eval_accumulation_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "world_size": launcher_world_size,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "save_total_limit": args.save_total_limit,
            "dataloader_num_workers": args.dataloader_num_workers,
            "seed": args.seed,
            "precision": args.precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_mode": args.gradient_checkpointing_mode,
            "trust_remote_code": args.trust_remote_code,
            "lora": (
                {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": _csv(args.lora_target_modules),
                }
                if args.finetune_mode == "lora"
                else None
            ),
            "deepspeed": semantic_deepspeed,
        },
        "code_sha256": canonical_sha256(code),
    }
    return {
        "schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "run_signature": canonical_sha256(identity),
        "run_id": run_id,
        "experiment_name": _safe_component(
            args.experiment_name,
            label="experiment_name",
        ),
        "created_at": utc_now(),
        **identity,
        "data": {
            "train": {
                "path": str(Path(args.train_data).expanduser().resolve()),
                **train_report,
            },
            "memorization": (
                {
                    "path": str(Path(args.memorization_data).expanduser().resolve()),
                    **memorization_report,
                }
                if args.memorization_data and memorization_report is not None
                else None
            ),
            "validation": (
                {
                    "path": str(Path(args.validation_data).expanduser().resolve()),
                    **validation_report,
                }
                if args.validation_data and validation_report is not None
                else None
            ),
            "candidate_names": list(candidate_names),
            "system_prompt_characters": len(system_prompt),
        },
        "code": code,
    }


def _trainer(
    *,
    transformers: Any,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    data_collator: Any,
    tokenizer: Any,
    callback: Any,
) -> Any:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "callbacks": [callback],
    }
    parameters = inspect.signature(transformers.Trainer.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    return transformers.Trainer(**kwargs)


def _annotate_history_stages(
    history: Sequence[Mapping[str, Any]],
    *,
    memorization_steps: int,
) -> list[dict[str, Any]]:
    annotated = []
    for row in history:
        item = dict(row)
        step = item.get("step")
        item["stage"] = (
            "memorization"
            if memorization_steps
            and isinstance(step, (int, float))
            and int(step) <= memorization_steps
            else "main"
        )
        annotated.append(item)
    return annotated


def main(argv: Sequence[str] | None = None) -> None:
    python_bin = str(Path(sys.executable).absolute().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join((python_bin, *path_entries))

    args = parse_args(argv)
    environment_local_rank = os.environ.get("LOCAL_RANK")
    if args.local_rank < 0 and environment_local_rank is not None:
        try:
            args.local_rank = int(environment_local_rank)
        except ValueError as exc:
            raise Top1DataError("LOCAL_RANK must be an integer") from exc
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise Top1DataError("RANK must be an integer") from exc
    is_primary = rank == 0
    try:
        launcher_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise Top1DataError("WORLD_SIZE must be an integer") from exc
    if launcher_world_size <= 0:
        raise Top1DataError("WORLD_SIZE must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    store = RunStore.training(output_dir)
    store_initialized = False

    try:
        _validate_args(args)
        run_id = _safe_component(args.run_id or output_dir.name, label="run_id")
        prompt_path = Path(args.system_prompt_file).expanduser()
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise Top1DataError("system prompt file is empty")
        candidate_names = load_candidate_names(args.candidate_registry)
        train_rows = read_jsonl(args.train_data)
        memorization_rows = (
            read_jsonl(args.memorization_data) if args.memorization_data else []
        )
        validation_rows = (
            read_jsonl(args.validation_data) if args.validation_data else []
        )
        train_report = validate_training_rows(
            train_rows,
            candidate_names,
            source=args.train_data,
        )
        memorization_report = (
            validate_memorization_rows(
                memorization_rows,
                candidate_names,
                source=args.memorization_data,
            )
            if args.memorization_data
            else None
        )
        validation_report = (
            validate_training_rows(
                validation_rows,
                candidate_names,
                source=args.validation_data,
            )
            if args.validation_data
            else None
        )

        deepspeed_metadata = None
        if args.deepspeed:
            deepspeed_path, _, stage = _read_deepspeed_config(args.deepspeed)
            version = _require_supported_deepspeed_version()
            args.deepspeed = str(deepspeed_path)
            deepspeed_metadata = {
                "path": str(deepspeed_path),
                "sha256": sha256_file(deepspeed_path),
                "zero_stage": stage,
                "version": version,
            }

        resume_from_checkpoint = _resume_value(args.resume_from_checkpoint)
        run_manifest = _build_run_manifest(
            args=args,
            run_id=run_id,
            candidate_names=candidate_names,
            system_prompt=system_prompt,
            train_report=train_report,
            memorization_report=memorization_report,
            validation_report=validation_report,
            deepspeed_metadata=deepspeed_metadata,
        )
        if is_primary:
            store.initialize(
                run_manifest,
                resume=resume_from_checkpoint is not None,
            )
            store_initialized = True
        else:
            store.ensure_layout()

        torch, transformers = _import_training_dependencies()
        if is_primary:
            write_json(output_dir / "logs" / "system.json", system_snapshot(torch))
            store.update_status("PREPARING")
        training_args = _build_training_arguments(
            args,
            transformers,
            has_validation=bool(validation_rows),
            resume_from_checkpoint=resume_from_checkpoint,
            staged_training=bool(memorization_rows),
        )
        tokenizer, model = _load_model_and_tokenizer(args, torch, transformers)
        candidate_tokens = candidate_token_sequences(tokenizer, candidate_names)
        train_examples, train_sft_rows, train_diagnostics = _prepare_rows(
            train_rows,
            source=args.train_data,
            tokenizer=tokenizer,
            candidate_tokens=candidate_tokens,
            max_length=args.max_length,
            system_prompt=system_prompt,
        )
        (
            memorization_examples,
            memorization_sft_rows,
            memorization_diagnostics,
        ) = (
            _prepare_rows(
                memorization_rows,
                source=args.memorization_data,
                tokenizer=tokenizer,
                candidate_tokens=candidate_tokens,
                max_length=args.max_length,
                system_prompt=system_prompt,
            )
            if memorization_rows
            else ([], [], [])
        )
        validation_examples, validation_sft_rows, validation_diagnostics = (
            _prepare_rows(
                validation_rows,
                source=args.validation_data,
                tokenizer=tokenizer,
                candidate_tokens=candidate_tokens,
                max_length=args.max_length,
                system_prompt=system_prompt,
            )
            if validation_rows
            else ([], [], [])
        )
        prepared_dir = output_dir / "prepared"
        prepared_train_path = prepared_dir / "train.sft.jsonl"
        prepared_memorization_path = (
            prepared_dir / "memorization.sft.jsonl"
            if memorization_rows
            else None
        )
        training_examples: Sequence[Mapping[str, Sequence[int]]] = train_examples
        training_schedule = None
        if memorization_examples:
            effective_global_batch_size = (
                launcher_world_size
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
            )
            training_examples, training_schedule = _build_training_schedule(
                memorization_examples,
                train_examples,
                memorization_steps=args.memorization_steps,
                main_epochs=args.epochs,
                effective_global_batch_size=effective_global_batch_size,
                seed=args.seed,
            )
        if is_primary:
            write_jsonl(prepared_train_path, train_sft_rows)
            write_json(
                prepared_dir / "train_profile.json",
                build_data_profile(
                    train_diagnostics,
                    candidate_names=candidate_names,
                    candidate_tokens=candidate_tokens,
                    max_length=args.max_length,
                ),
            )
            if prepared_memorization_path is not None:
                write_jsonl(prepared_memorization_path, memorization_sft_rows)
                write_json(
                    prepared_dir / "memorization_profile.json",
                    build_data_profile(
                        memorization_diagnostics,
                        candidate_names=candidate_names,
                        candidate_tokens=candidate_tokens,
                        max_length=args.max_length,
                    ),
                )
                write_json(
                    prepared_dir / "training_schedule.json",
                    training_schedule,
                )
            if validation_rows:
                write_jsonl(prepared_dir / "validation.sft.jsonl", validation_sft_rows)
                write_json(
                    prepared_dir / "validation_profile.json",
                    build_data_profile(
                        validation_diagnostics,
                        candidate_names=candidate_names,
                        candidate_tokens=candidate_tokens,
                        max_length=args.max_length,
                    ),
                )
            tokenizer.save_pretrained(str(prepared_dir / "tokenizer"))
            store.event(
                "data_prepared",
                train_rows=len(train_examples),
                memorization_rows=len(memorization_examples),
                scheduled_rows=len(training_examples),
                memorization_steps=args.memorization_steps,
                validation_rows=len(validation_examples),
            )
            print(
                "[top1] candidate supervision: "
                + ", ".join(
                    f"{name}={train_report['candidate_counts'].get(name, 0)}"
                    for name in candidate_names
                ),
                flush=True,
            )
            if training_schedule is not None:
                print(
                    "[top1] memorization curriculum: "
                    f"{training_schedule['memorization']['input_rows']} rows -> "
                    f"{training_schedule['memorization']['scheduled_samples']} "
                    "samples, "
                    f"{training_schedule['memorization']['optimizer_steps']} "
                    "optimizer steps; main training follows",
                    flush=True,
                )

        Dataset = _dataset_class(torch)
        callback = make_training_log_callback(
            store,
            transformers.TrainerCallback,
            torch,
            memorization_steps=args.memorization_steps,
        )
        trainer = _trainer(
            transformers=transformers,
            model=model,
            training_args=training_args,
            train_dataset=Dataset(training_examples),
            eval_dataset=(
                Dataset(validation_examples) if validation_examples else None
            ),
            data_collator=_collator(torch, int(tokenizer.pad_token_id)),
            tokenizer=tokenizer,
            callback=callback,
        )
        wait_for_everyone = getattr(
            getattr(trainer, "accelerator", None),
            "wait_for_everyone",
            None,
        )
        if callable(wait_for_everyone):
            wait_for_everyone()
        world_size = int(trainer.args.world_size)
        if launcher_world_size > 1 and world_size != launcher_world_size:
            raise Top1DataError(
                f"launcher requested {launcher_world_size} processes, but Trainer "
                f"initialized world_size={world_size}"
            )

        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        final_eval_metrics = (
            trainer.evaluate(metric_key_prefix="final") if validation_examples else {}
        )
        trainer.save_state()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        final_model_dir = output_dir / "final" / "model"
        trainer.save_model(str(final_model_dir))
        if callable(wait_for_everyone):
            wait_for_everyone()
        if trainer.is_world_process_zero():
            store.update_status("FINALIZING", step=int(trainer.state.global_step))
            _write_bundle(
                args=args,
                output_dir=final_model_dir,
                tokenizer=tokenizer,
                model=model,
                candidate_names=candidate_names,
                candidate_tokens=candidate_tokens,
                system_prompt=system_prompt,
                train_report=train_report,
                memorization_report=memorization_report,
                validation_report=validation_report,
                world_size=world_size,
                deepspeed_metadata=deepspeed_metadata,
                training_run_id=run_id,
                prepared_train_path=prepared_train_path,
                prepared_memorization_path=prepared_memorization_path,
                training_schedule=training_schedule,
                transformers_version=str(transformers.__version__),
            )
            artifact = write_model_artifact_manifest(
                final_model_dir,
                training_run_id=run_id,
            )
            history = _annotate_history_stages(
                trainer.state.log_history,
                memorization_steps=args.memorization_steps,
            )
            write_trainer_history(output_dir / "logs" / "trainer_history.jsonl", history)
            curves = build_curve_summary(history)
            write_json(output_dir / "final" / "curves.json", curves)
            best_checkpoint = trainer.state.best_model_checkpoint
            if best_checkpoint:
                write_json(
                    output_dir / "checkpoints" / "best_checkpoint.json",
                    {
                        "schema_version": 1,
                        "path": str(Path(best_checkpoint).resolve()),
                        "metric": trainer.state.best_metric,
                        "updated_at": utc_now(),
                    },
                )
            summary = {
                "schema_version": 1,
                "run_id": run_id,
                "run_signature": run_manifest["run_signature"],
                "model_id": artifact["model_id"],
                "global_step": int(trainer.state.global_step),
                "training_schedule": training_schedule,
                "best_checkpoint": best_checkpoint,
                "best_eval_loss": curves["best_eval_loss"],
                "train_metrics": dict(getattr(train_result, "metrics", {})),
                "final_validation_metrics": dict(final_eval_metrics),
                "completed_at": utc_now(),
            }
            write_json(output_dir / "final" / "summary.json", json_safe(summary))
            store.event(
                "run_completed",
                step=int(trainer.state.global_step),
                model_id=artifact["model_id"],
            )
            store.update_status(
                "COMPLETED",
                step=int(trainer.state.global_step),
                model_id=artifact["model_id"],
            )
        if callable(wait_for_everyone):
            wait_for_everyone()
    except BaseException as exc:
        if is_primary and store_initialized:
            store.event("run_failed", error_type=type(exc).__name__, error=str(exc))
            store.update_status(
                "FAILED",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
