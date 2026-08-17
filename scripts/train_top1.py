#!/usr/bin/env python3
"""Train one direct candidate-name Top1 router."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from llmgen.top1 import (
    CONVERSATION_TEMPLATE,
    MAX_ASSISTANT_HISTORY_CHARACTERS,
    MAX_HISTORY_CHARACTERS,
    MAX_HISTORY_MESSAGES,
    ROUTING_MODE,
    Top1DataError,
    candidate_registry_payload,
    candidate_token_sequences,
    load_candidate_names,
    prepare_example,
    read_jsonl,
    sha256_file,
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
    parser.add_argument("--validation-data")
    parser.add_argument("--candidate-registry", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)

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
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise Top1DataError("values must be positive: " + ", ".join(invalid))
    if args.dataloader_num_workers < 0:
        raise Top1DataError("dataloader_num_workers cannot be negative")
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
) -> Any:
    training_kwargs = {
        "output_dir": str(Path(args.output_dir)),
        "overwrite_output_dir": resume_from_checkpoint is None,
        "num_train_epochs": args.epochs,
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
        "seed": args.seed,
        "data_seed": args.seed,
    }
    parameters = inspect.signature(transformers.TrainingArguments.__init__).parameters
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
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]]]:
    encoded: list[dict[str, list[int]]] = []
    sft_rows: list[dict[str, Any]] = []
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
    return encoded, sft_rows


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
    system_prompt: str,
    train_report: Mapping[str, Any],
    validation_report: Mapping[str, Any] | None,
    world_size: int,
    deepspeed_metadata: Mapping[str, Any] | None,
) -> None:
    tokenizer.save_pretrained(str(output_dir))
    bundled_registry = output_dir / "candidate_registry.json"
    write_json(bundled_registry, candidate_registry_payload(candidate_names))
    bundled_prompt = output_dir / "router_system_prompt.md"
    bundled_prompt.write_text(system_prompt.rstrip() + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "routing_mode": ROUTING_MODE,
        "base_model": args.model_name_or_path,
        "base_model_revision": getattr(model.config, "_commit_hash", None),
        "finetune_mode": args.finetune_mode,
        "train_data": {
            "path": str(Path(args.train_data).expanduser().resolve()),
            "sha256": sha256_file(args.train_data),
            **train_report,
        },
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
        },
        "target": "candidate_name_tokens_plus_eos",
        "max_length": args.max_length,
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "precision": args.precision,
            "seed": args.seed,
            "world_size": world_size,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch_size": (
                world_size
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
            ),
            "deepspeed": deepspeed_metadata,
        },
        "sft_input": {
            "path": "sft_input.jsonl",
            "sha256": sha256_file(output_dir / "sft_input.jsonl"),
            "rows": train_report["rows"],
        },
    }
    write_json(output_dir / "router_manifest.json", manifest)


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
    _validate_args(args)

    prompt_path = Path(args.system_prompt_file).expanduser()
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise Top1DataError("system prompt file is empty")
    candidate_names = load_candidate_names(args.candidate_registry)
    train_rows = read_jsonl(args.train_data)
    validation_rows = read_jsonl(args.validation_data) if args.validation_data else []
    train_report = validate_training_rows(
        train_rows,
        candidate_names,
        source=args.train_data,
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

    torch, transformers = _import_training_dependencies()
    resume_from_checkpoint = _resume_value(args.resume_from_checkpoint)
    training_args = _build_training_arguments(
        args,
        transformers,
        has_validation=bool(validation_rows),
        resume_from_checkpoint=resume_from_checkpoint,
    )
    tokenizer, model = _load_model_and_tokenizer(args, torch, transformers)
    candidate_tokens = candidate_token_sequences(tokenizer, candidate_names)
    train_examples, sft_rows = _prepare_rows(
        train_rows,
        source=args.train_data,
        tokenizer=tokenizer,
        candidate_tokens=candidate_tokens,
        max_length=args.max_length,
        system_prompt=system_prompt,
    )
    validation_examples, _ = (
        _prepare_rows(
            validation_rows,
            source=args.validation_data,
            tokenizer=tokenizer,
            candidate_tokens=candidate_tokens,
            max_length=args.max_length,
            system_prompt=system_prompt,
        )
        if validation_rows
        else ([], [])
    )

    Dataset = _dataset_class(torch)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset(train_examples),
        eval_dataset=Dataset(validation_examples) if validation_examples else None,
        data_collator=_collator(torch, int(tokenizer.pad_token_id)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if trainer.is_world_process_zero():
        write_jsonl(output_dir / "sft_input.jsonl", sft_rows)
        print(
            "[top1] candidate supervision: "
            + ", ".join(
                f"{name}={train_report['candidate_counts'].get(name, 0)}"
                for name in candidate_names
            ),
            flush=True,
        )

    wait_for_everyone = getattr(
        getattr(trainer, "accelerator", None), "wait_for_everyone", None
    )
    if callable(wait_for_everyone):
        wait_for_everyone()
    try:
        launcher_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise Top1DataError("WORLD_SIZE must be an integer") from exc
    world_size = int(trainer.args.world_size)
    if launcher_world_size > 1 and world_size != launcher_world_size:
        raise Top1DataError(
            f"launcher requested {launcher_world_size} processes, but Trainer "
            f"initialized world_size={world_size}"
        )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        _write_bundle(
            args=args,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            candidate_names=candidate_names,
            system_prompt=system_prompt,
            train_report=train_report,
            validation_report=validation_report,
            world_size=world_size,
            deepspeed_metadata=deepspeed_metadata,
        )
    if callable(wait_for_everyone):
        wait_for_everyone()


if __name__ == "__main__":
    main()
