#!/usr/bin/env python3
"""Train a causal-LM skill router in memorization and/or retrieval phases.

Heavy dependencies are imported only inside ``main`` so the repository's core
data and metric tests remain runnable without a GPU training environment.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from llmgen.direct_router import (
    DIRECT_ROUTING_MODE,
    candidate_registry_payload,
    candidate_token_sequences,
    encode_candidate_name_example,
    load_candidate_registry,
    target_candidate_name,
)
from llmgen.router import (
    RouterDataError,
    code_token_id_map,
    encode_target_only_example,
    load_virtual_tokens,
    mix_replay_sources,
    read_jsonl,
)
from llmgen.router_bundle import dump_router_decoder_artifacts
from llmgen.skillret import sha256_file


MEMORIZATION_SYSTEM_PROMPT = (
    "Map the Agent Skill document to its fixed-length hierarchical skill code. "
    "Answer with code tokens only."
)
RETRIEVAL_SYSTEM_PROMPT = (
    "Select every Agent Skill needed for the user request in execution order. "
    "Output one hierarchical skill code per line, with no other text."
)
DIRECT_RETRIEVAL_SYSTEM_PROMPT = (
    "Select exactly one candidate name for the user's current intent. "
    "Output the candidate name only, with no explanation or punctuation."
)
HIERARCHICAL_ROUTING_MODE = "hierarchical_code"
SUPPORTED_DEEPSPEED_VERSION = "0.16.4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the generative skill router.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--routing-mode",
        choices=(HIERARCHICAL_ROUTING_MODE, DIRECT_ROUTING_MODE),
        default=HIERARCHICAL_ROUTING_MODE,
    )
    parser.add_argument("--virtual-tokens")
    parser.add_argument(
        "--candidate-registry",
        help="Direct-mode JSON registry containing every legal generated name.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--phase-output-subdir",
        help="Override the phase checkpoint subdirectory (for retrieval curriculum).",
    )
    parser.add_argument(
        "--skill-catalog",
        help="Catalog JSONL bundled beside each final model for human decoding.",
    )
    parser.add_argument(
        "--skill-codes",
        help="Code JSONL bundled beside each final model for human decoding.",
    )
    parser.add_argument(
        "--skill-registry",
        help="Active registry JSON bundled beside each final model for decoding.",
    )
    parser.add_argument(
        "--stage",
        choices=("memorization", "retrieval", "both"),
        default="both",
    )
    parser.add_argument("--memorization-train")
    parser.add_argument("--memorization-validation")
    parser.add_argument("--retrieval-train")
    parser.add_argument("--retrieval-validation")
    parser.add_argument(
        "--retrieval-memorization-replay-data",
        "--retrieval-replay-data",
        dest="retrieval_memorization_replay_data",
        help="Optional memorization_train.jsonl replayed during retrieval SFT.",
    )
    parser.add_argument(
        "--retrieval-memorization-replay-fraction",
        "--retrieval-replay-fraction",
        dest="retrieval_memorization_replay_fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--retrieval-alignment-replay-data",
        help="Optional retrieval_alignment_train.jsonl replayed during retrieval SFT.",
    )
    parser.add_argument(
        "--retrieval-alignment-replay-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument("--num-levels", type=int)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--memorization-system-prompt", default=MEMORIZATION_SYSTEM_PROMPT)
    parser.add_argument("--retrieval-system-prompt", default=RETRIEVAL_SYSTEM_PROMPT)
    parser.add_argument(
        "--retrieval-system-prompt-file",
        help="UTF-8 prompt file; overrides --retrieval-system-prompt.",
    )

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--memorization-epochs", type=float, default=1.0)
    parser.add_argument("--retrieval-epochs", type=float, default=3.0)
    parser.add_argument("--memorization-learning-rate", type=float, default=2e-5)
    parser.add_argument("--retrieval-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--deepspeed", help="Optional DeepSpeed JSON config.")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing-mode",
        choices=("auto", "reentrant", "non-reentrant"),
        default="auto",
        help=(
            "Activation-checkpoint implementation. 'auto' uses reentrant with "
            "DeepSpeed ZeRO-3 and the Transformers default otherwise."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume-memorization-from-checkpoint")
    parser.add_argument("--resume-retrieval-from-checkpoint")

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--adapter-name-or-path")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module suffixes.",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default="auto",
        help=(
            "Comma-separated full modules saved with the adapter. 'auto' keeps "
            "the resized input/output embeddings trainable and checkpointed; "
            "'none' creates a pure LoRA delta when the source router already "
            "contains the complete virtual-token vocabulary."
        ),
    )
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _resume_value(value: str | None) -> str | bool | None:
    if value is None:
        return None
    return True if value.lower() in {"latest", "true"} else value


def _read_deepspeed_config(value: str | Path) -> tuple[Path, dict[str, Any], int]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise RouterDataError(f"DeepSpeed config does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RouterDataError(f"invalid DeepSpeed JSON config: {path}") from exc
    if not isinstance(payload, dict):
        raise RouterDataError("DeepSpeed config must be a JSON object")
    zero = payload.get("zero_optimization")
    stage = zero.get("stage") if isinstance(zero, dict) else None
    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 3:
        raise RouterDataError(
            "DeepSpeed config must define zero_optimization.stage from 0 to 3"
        )
    return path, payload, stage


def _require_supported_deepspeed_version() -> str:
    try:
        installed = importlib.metadata.version("deepspeed")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RouterDataError(
            "DeepSpeed is not installed; install the project's training dependencies"
        ) from exc
    if installed != SUPPORTED_DEEPSPEED_VERSION:
        raise RouterDataError(
            f"DeepSpeed {installed} is incompatible with the router's ZeRO-3 + "
            "PEFT modules_to_save path; install the pinned version with: "
            f"python -m pip install --no-build-isolation --force-reinstall "
            f"--no-deps deepspeed=={SUPPORTED_DEEPSPEED_VERSION}"
        )
    return installed


def _gradient_checkpointing_kwargs(args: argparse.Namespace) -> dict[str, bool] | None:
    if not args.gradient_checkpointing:
        return None
    mode = args.gradient_checkpointing_mode
    if mode == "auto":
        # Non-reentrant checkpointing can stop recomputation early. With ZeRO-3
        # that may bypass the module hooks which gather a partitioned parameter,
        # so recomputation observes its zero-sized placeholder instead.
        return {"use_reentrant": bool(args.deepspeed)}
    return {"use_reentrant": mode == "reentrant"}


def _module_name_for(model: Any, target: Any) -> str | None:
    for name, module in model.named_modules():
        if module is target:
            return name.rsplit(".", 1)[-1]
    return None


def _load_training_stack(
    args: argparse.Namespace,
    virtual_tokens: tuple[str, ...] = (),
):
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in real training env
        raise SystemExit(
            "Training requires torch and transformers. Install the project's "
            "training dependencies first."
        ) from exc

    tokenizer_source = args.adapter_name_or_path or args.model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=args.trust_remote_code,
        )
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    if tokenizer.eos_token_id is None:
        raise RouterDataError("base tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids: dict[str, int] = {}
    if virtual_tokens:
        existing_special = list(getattr(tokenizer, "additional_special_tokens", ()))
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": existing_special
                + [token for token in virtual_tokens if token not in existing_special]
            }
        )
        token_ids = code_token_id_map(tokenizer, virtual_tokens)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    if virtual_tokens:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.config.use_cache = False

    if args.adapter_name_or_path:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Loading an adapter requires the peft package.") from exc
        model = PeftModel.from_pretrained(
            model,
            args.adapter_name_or_path,
            is_trainable=True,
        )
    elif args.lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("--lora requires the peft package.") from exc

        if args.lora_modules_to_save == "auto":
            # Hierarchical routing adds code tokens, so LoRA must preserve the
            # resized embeddings. Direct-name routing uses the untouched base
            # vocabulary and therefore needs only the LoRA delta.
            modules_to_save = [] if virtual_tokens else None
            if virtual_tokens:
                for target in (
                    model.get_input_embeddings(),
                    model.get_output_embeddings(),
                ):
                    name = _module_name_for(model, target)
                    if name and name not in modules_to_save:
                        modules_to_save.append(name)
        elif args.lora_modules_to_save.strip().casefold() == "none":
            modules_to_save = None
        else:
            modules_to_save = _csv(args.lora_modules_to_save)
        if virtual_tokens and modules_to_save == []:
            raise RouterDataError(
                "LoRA must checkpoint the resized input/output embeddings; "
                "set --lora-modules-to-save explicitly for this architecture, "
                "or use 'none' only when continuing from a trained router"
            )
        lora_kwargs: dict[str, Any] = {
            "task_type": TaskType.CAUSAL_LM,
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": _csv(args.lora_target_modules),
            "bias": "none",
        }
        if modules_to_save is not None:
            lora_kwargs["modules_to_save"] = modules_to_save
        # Some Qwen3 models tie input and output embeddings. Newer PEFT
        # versions can preserve that contract explicitly when both resized
        # modules are stored in the adapter; retain compatibility with older
        # supported PEFT releases that do not expose this argument.
        if (
            modules_to_save is not None
            and "ensure_weight_tying" in inspect.signature(LoraConfig).parameters
        ):
            lora_kwargs["ensure_weight_tying"] = bool(
                getattr(model.config, "tie_word_embeddings", False)
            )
        config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return torch, transformers, tokenizer, model, token_ids


def _dataset_class(torch: Any):
    class RouterDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            rows: list[dict[str, Any]],
            tokenizer: Any,
            token_ids: dict[str, int],
            *,
            routing_mode: str,
            candidate_names: tuple[str, ...],
            num_levels: int | None,
            max_length: int,
            system_prompt: str,
            phase_system_prompts: dict[str, str] | None = None,
        ) -> None:
            self.rows = rows
            self.tokenizer = tokenizer
            self.token_ids = token_ids
            self.routing_mode = routing_mode
            self.candidate_names = candidate_names
            self.candidate_name_tokens = (
                candidate_token_sequences(tokenizer, candidate_names)
                if routing_mode == DIRECT_ROUTING_MODE
                else None
            )
            self.num_levels = num_levels
            self.max_length = max_length
            self.system_prompt = system_prompt
            self.phase_system_prompts = dict(phase_system_prompts or {})

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = self.rows[index]
            row_system_prompt = self.phase_system_prompts.get(
                str(row.get("phase", "")), self.system_prompt
            )
            if self.routing_mode == DIRECT_ROUTING_MODE:
                return encode_candidate_name_example(
                    self.tokenizer,
                    row,
                    candidate_names=self.candidate_names,
                    candidate_name_tokens=self.candidate_name_tokens,
                    max_length=self.max_length,
                    system_prompt=row_system_prompt,
                )
            if self.num_levels is None:
                raise RouterDataError("hierarchical routing requires num_levels")
            return encode_target_only_example(
                self.tokenizer,
                row,
                code_token_ids=self.token_ids,
                num_levels=self.num_levels,
                max_length=self.max_length,
                system_prompt=row_system_prompt,
            )

    return RouterDataset


def _collator(torch: Any, pad_token_id: int):
    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(row["input_ids"]) for row in features)
        input_ids = []
        attention_mask = []
        labels = []
        for row in features:
            padding = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_token_id] * padding)
            attention_mask.append(row["attention_mask"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _require_phase_path(path: str | None, phase: str) -> str:
    if not path:
        raise RouterDataError(f"--{phase}-train is required for stage {phase!r}")
    return path


def _build_training_arguments(
    *,
    phase: str,
    has_validation: bool,
    epochs: float,
    learning_rate: float,
    resume_from_checkpoint: str | bool | None,
    args: argparse.Namespace,
    transformers: Any,
) -> Any:
    phase_dir = Path(args.output_dir) / phase
    training_kwargs = {
        "output_dir": str(phase_dir),
        "overwrite_output_dir": resume_from_checkpoint is None,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": learning_rate,
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
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": _gradient_checkpointing_kwargs(args),
        "dataloader_num_workers": args.dataloader_num_workers,
        "deepspeed": args.deepspeed,
        "local_rank": args.local_rank,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
    }
    # Transformers 4.x and 5.x differ on two TrainingArguments names. Keep the
    # complete training path runnable across both supported major versions.
    parameters = inspect.signature(transformers.TrainingArguments.__init__).parameters
    evaluation_key = (
        "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    )
    training_kwargs[evaluation_key] = "steps" if has_validation else "no"
    training_kwargs = {
        key: value for key, value in training_kwargs.items() if key in parameters
    }
    return transformers.TrainingArguments(**training_kwargs)


def _run_phase(
    *,
    phase: str,
    train_path: str,
    validation_path: str | None,
    system_prompt: str,
    epochs: float,
    learning_rate: float,
    resume_from_checkpoint: str | bool | None,
    args: argparse.Namespace,
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    model: Any,
    token_ids: dict[str, int],
    candidate_names: tuple[str, ...] = (),
    training_args: Any | None = None,
    replay_inputs: Sequence[tuple[str, str | None, float]] = (),
    replay_system_prompt: str | None = None,
    output_subdir: str | None = None,
) -> None:
    primary_train_rows = read_jsonl(train_path)
    if not primary_train_rows:
        raise RouterDataError(f"{phase} training data is empty")
    loaded_replay_sources = [
        (name, read_jsonl(path) if path else [], fraction)
        for name, path, fraction in replay_inputs
    ]
    train_rows, replay_counts = mix_replay_sources(
        primary_train_rows,
        loaded_replay_sources,
        seed=args.seed,
    )
    replay_examples = sum(replay_counts.values())
    replay_fraction = sum(fraction for _, _, fraction in replay_inputs)
    replay_sources: dict[str, dict[str, Any]] = {}
    for (name, path, fraction), (_, rows, _) in zip(
        replay_inputs,
        loaded_replay_sources,
        strict=True,
    ):
        if not path:
            continue
        examples = replay_counts[name]
        replay_sources[name] = {
            "data": str(Path(path).resolve()),
            "data_sha256": sha256_file(path),
            "source_rows": len(rows),
            "examples": examples,
            "fraction_requested": fraction,
            "fraction_actual": examples / max(len(train_rows), 1),
            "repeat_factor": examples / max(len(rows), 1),
        }
    if args.local_rank in (-1, 0):
        mixture = [
            "primary="
            f"{len(primary_train_rows)} "
            f"({len(primary_train_rows) / len(train_rows):.2%})"
        ]
        mixture.extend(
            f"{name}={metadata['examples']} "
            f"({metadata['fraction_actual']:.2%})"
            for name, metadata in replay_sources.items()
        )
        mixture.append(f"total={len(train_rows)}")
        print(f"[{phase}] training mixture: " + ", ".join(mixture), flush=True)
    validation_rows = read_jsonl(validation_path) if validation_path else []
    if args.routing_mode == DIRECT_ROUTING_MODE:
        legal_names = set(candidate_names)
        train_name_counts = Counter(
            target_candidate_name(row) for row in train_rows
        )
        unknown_names = set(train_name_counts).difference(legal_names)
        if unknown_names:
            raise RouterDataError(
                "direct training data contains candidates outside the registry: "
                + ", ".join(sorted(unknown_names))
            )
        missing_names = legal_names.difference(train_name_counts)
        if missing_names:
            raise RouterDataError(
                "direct training data has no supervision for candidates: "
                + ", ".join(sorted(missing_names))
            )
        for row in validation_rows:
            name = target_candidate_name(row)
            if name not in legal_names:
                raise RouterDataError(
                    f"direct validation target {name!r} is outside the registry"
                )
        if args.local_rank in (-1, 0):
            print(
                "[retrieval] candidate supervision: "
                + ", ".join(
                    f"{name}={train_name_counts[name]}" for name in candidate_names
                ),
                flush=True,
            )
    Dataset = _dataset_class(torch)
    train_dataset = Dataset(
        train_rows,
        tokenizer,
        token_ids,
        routing_mode=args.routing_mode,
        candidate_names=candidate_names,
        num_levels=args.num_levels,
        max_length=args.max_length,
        system_prompt=system_prompt,
        phase_system_prompts=(
            {"memorization": replay_system_prompt}
            if replay_system_prompt is not None
            else None
        ),
    )
    validation_dataset = (
        Dataset(
            validation_rows,
            tokenizer,
            token_ids,
            routing_mode=args.routing_mode,
            candidate_names=candidate_names,
            num_levels=args.num_levels,
            max_length=args.max_length,
            system_prompt=system_prompt,
        )
        if validation_rows
        else None
    )

    phase_output_name = output_subdir or phase
    if Path(phase_output_name).name != phase_output_name:
        raise RouterDataError("phase output subdirectory must be one path component")
    phase_dir = Path(args.output_dir) / phase_output_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    if training_args is None:
        training_args = _build_training_arguments(
            phase=phase_output_name,
            has_validation=validation_dataset is not None,
            epochs=epochs,
            learning_rate=learning_rate,
            resume_from_checkpoint=resume_from_checkpoint,
            args=args,
            transformers=transformers,
        )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=_collator(torch, int(tokenizer.pad_token_id)),
    )
    try:
        launcher_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise RouterDataError("WORLD_SIZE must be an integer") from exc
    actual_world_size = int(trainer.args.world_size)
    if launcher_world_size > 1 and actual_world_size != launcher_world_size:
        raise RouterDataError(
            "distributed launcher requested "
            f"{launcher_world_size} processes, but Trainer initialized "
            f"world_size={actual_world_size}; check CUDA visibility and Accelerate setup"
        )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(phase_dir))
    if trainer.is_world_process_zero():
        deepspeed_metadata = None
        if args.deepspeed:
            deepspeed_path, _, zero_stage = _read_deepspeed_config(args.deepspeed)
            deepspeed_metadata = {
                "config": str(deepspeed_path),
                "config_sha256": sha256_file(deepspeed_path),
                "zero_stage": zero_stage,
            }
        tokenizer.save_pretrained(str(phase_dir))
        router_data_manifest_path = Path(train_path).resolve().parent / "manifest.json"
        router_data_manifest_sha256 = None
        stage1_checkpoint_sha256 = None
        index_manifest_sha256 = None
        if router_data_manifest_path.is_file():
            router_data_manifest_sha256 = sha256_file(router_data_manifest_path)
            router_data_manifest = json.loads(
                router_data_manifest_path.read_text(encoding="utf-8")
            )
            index_source = router_data_manifest.get("sources", {}).get(
                "index_manifest"
            )
            if isinstance(index_source, dict):
                stage1_checkpoint_sha256 = index_source.get("checkpoint_sha256")
                index_manifest_sha256 = index_source.get("sha256")
        all_rows = [*train_rows, *validation_rows]
        is_direct = args.routing_mode == DIRECT_ROUTING_MODE
        max_target_paths = 1 if is_direct else max(
            (
                len(row["target_paths"])
                if isinstance(row.get("target_paths"), list)
                else 1
            )
            for row in all_rows
        )
        state = {
            "schema_version": 3,
            "routing_mode": args.routing_mode,
            "phase": phase,
            "curriculum_stage": phase_output_name,
            "num_levels": args.num_levels,
            "virtual_tokens": (
                str(Path(args.virtual_tokens).resolve())
                if args.virtual_tokens
                else None
            ),
            "virtual_tokens_sha256": (
                sha256_file(args.virtual_tokens) if args.virtual_tokens else None
            ),
            "train_data": str(Path(train_path).resolve()),
            "train_data_sha256": sha256_file(train_path),
            "replay_data": replay_sources.get("memorization", {}).get("data"),
            "replay_data_sha256": replay_sources.get("memorization", {}).get(
                "data_sha256"
            ),
            "replay_fraction_requested": replay_fraction,
            "replay_fraction_actual": replay_examples / max(len(train_rows), 1),
            "replay_sources": replay_sources,
            "validation_data": (
                str(Path(validation_path).resolve()) if validation_path else None
            ),
            "validation_data_sha256": (
                sha256_file(validation_path) if validation_path else None
            ),
            "router_data_manifest_sha256": router_data_manifest_sha256,
            "index_manifest_sha256": index_manifest_sha256,
            "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
            "base_model": args.model_name_or_path,
            "base_model_revision": getattr(model.config, "_commit_hash", None),
            "seed": args.seed,
            "finetune_mode": (
                "continued_adapter"
                if args.adapter_name_or_path
                else "lora"
                if args.lora
                else "full"
            ),
            "distributed": {
                "backend": (
                    "deepspeed"
                    if args.deepspeed
                    else "ddp"
                    if actual_world_size > 1
                    else "single"
                ),
                "world_size": actual_world_size,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_global_batch_size": (
                    actual_world_size
                    * args.per_device_train_batch_size
                    * args.gradient_accumulation_steps
                ),
                "deepspeed": deepspeed_metadata,
            },
            "system_prompt": system_prompt,
            "replay_system_prompt": (
                replay_system_prompt
                if replay_counts.get("memorization", 0)
                else None
            ),
            "max_length": args.max_length,
            "generation_contract": {
                "mode": (
                    DIRECT_ROUTING_MODE
                    if is_direct
                    else "autoregressive_multi_path"
                    if phase == "retrieval"
                    else "single_path"
                ),
                "path_separator": (
                    None if is_direct else "\n" if phase == "retrieval" else None
                ),
                "max_target_paths": max_target_paths,
                "candidate_names": list(candidate_names) if is_direct else None,
                "target_suffix": "eos" if is_direct else None,
            },
            "examples": {
                "train": len(train_rows),
                "primary_train": len(primary_train_rows),
                "replay": replay_examples,
                "replay_by_source": replay_counts,
                "validation": len(validation_rows),
            },
        }
        decoder_inputs = (
            args.skill_catalog,
            args.skill_codes,
            args.skill_registry,
        )
        if all(decoder_inputs):
            state["decoder_artifacts"] = dump_router_decoder_artifacts(
                output_dir=phase_dir,
                catalog_path=args.skill_catalog,
                codes_path=args.skill_codes,
                registry_path=args.skill_registry,
                virtual_tokens_path=args.virtual_tokens,
                training_data_path=train_path,
                supervision_phase=phase,
                supervision_rows=train_rows,
            )
        if is_direct:
            routes = load_candidate_registry(args.candidate_registry)
            bundled_registry = phase_dir / "candidate_registry.json"
            bundled_registry.write_text(
                json.dumps(
                    candidate_registry_payload(routes),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            bundled_prompt = phase_dir / "router_system_prompt.md"
            bundled_prompt.write_text(system_prompt.rstrip() + "\n", encoding="utf-8")
            state["candidate_registry"] = {
                "path": bundled_registry.name,
                "sha256": sha256_file(bundled_registry),
                "count": len(routes),
            }
            state["system_prompt_artifact"] = {
                "path": bundled_prompt.name,
                "sha256": sha256_file(bundled_prompt),
            }
        with (phase_dir / "router_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    wait_for_everyone = getattr(
        getattr(trainer, "accelerator", None), "wait_for_everyone", None
    )
    if callable(wait_for_everyone):
        wait_for_everyone()


def main() -> None:
    # Do not resolve the venv/Conda Python symlink: its sibling `ninja`
    # executable lives in the environment's bin directory, not /usr/bin.
    python_bin = str(Path(sys.executable).absolute().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join((python_bin, *path_entries))

    args = parse_args()
    if args.retrieval_system_prompt_file:
        prompt_path = Path(args.retrieval_system_prompt_file).expanduser()
        if not prompt_path.is_file():
            raise RouterDataError(f"retrieval system prompt does not exist: {prompt_path}")
        args.retrieval_system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not args.retrieval_system_prompt:
            raise RouterDataError("retrieval system prompt file is empty")
    elif (
        args.routing_mode == DIRECT_ROUTING_MODE
        and args.retrieval_system_prompt == RETRIEVAL_SYSTEM_PROMPT
    ):
        args.retrieval_system_prompt = DIRECT_RETRIEVAL_SYSTEM_PROMPT
    environment_local_rank = os.environ.get("LOCAL_RANK")
    if args.local_rank < 0 and environment_local_rank is not None:
        try:
            args.local_rank = int(environment_local_rank)
        except ValueError as exc:
            raise RouterDataError("LOCAL_RANK must be an integer") from exc
    if args.bf16 and args.fp16:
        raise RouterDataError("--bf16 and --fp16 are mutually exclusive")
    decoder_inputs = (
        args.skill_catalog,
        args.skill_codes,
        args.skill_registry,
    )
    if any(decoder_inputs) and not all(decoder_inputs):
        raise RouterDataError(
            "--skill-catalog, --skill-codes, and --skill-registry must be set together"
        )
    candidate_names: tuple[str, ...] = ()
    if args.routing_mode == HIERARCHICAL_ROUTING_MODE:
        if not args.virtual_tokens or args.num_levels is None:
            raise RouterDataError(
                "hierarchical routing requires --virtual-tokens and --num-levels"
            )
        if args.candidate_registry:
            raise RouterDataError(
                "--candidate-registry is only valid for candidate_name_top1 routing"
            )
        if args.num_levels < 1 or args.max_length <= args.num_levels + 1:
            raise RouterDataError("invalid num_levels/max_length combination")
    else:
        if args.stage != "retrieval":
            raise RouterDataError(
                "candidate_name_top1 routing has one retrieval phase; use --stage retrieval"
            )
        if not args.candidate_registry:
            raise RouterDataError(
                "candidate_name_top1 routing requires --candidate-registry"
            )
        if args.virtual_tokens or args.num_levels is not None or any(decoder_inputs):
            raise RouterDataError(
                "direct candidate-name routing does not use virtual tokens, num_levels, "
                "skill codes, or a Stage-1 registry"
            )
        routes = load_candidate_registry(args.candidate_registry)
        candidate_names = tuple(route.name for route in routes)
        if args.max_length < 4:
            raise RouterDataError("max_length is too small for direct routing")
    if args.eval_steps < 1 or args.save_steps < 1:
        raise RouterDataError("save_steps and eval_steps must be positive")
    if args.save_steps % args.eval_steps:
        raise RouterDataError(
            "save_steps must be a multiple of eval_steps when validation is enabled"
        )
    replay_arguments = (
        (
            "alignment",
            args.retrieval_alignment_replay_data,
            args.retrieval_alignment_replay_fraction,
        ),
        (
            "memorization",
            args.retrieval_memorization_replay_data,
            args.retrieval_memorization_replay_fraction,
        ),
    )
    for name, path, fraction in replay_arguments:
        if not 0.0 <= fraction < 1.0:
            raise RouterDataError(
                f"--retrieval-{name}-replay-fraction must be in [0, 1)"
            )
        if bool(path) != (fraction > 0.0):
            raise RouterDataError(
                f"set both --retrieval-{name}-replay-data and a positive "
                f"--retrieval-{name}-replay-fraction, or neither"
            )
    if sum(fraction for _, _, fraction in replay_arguments) >= 1.0:
        raise RouterDataError("total retrieval replay fraction must be less than 1")
    if args.routing_mode == DIRECT_ROUTING_MODE and any(
        path or fraction for _, path, fraction in replay_arguments
    ):
        raise RouterDataError(
            "candidate_name_top1 does not accept hierarchical replay datasets"
        )
    if args.adapter_name_or_path and args.lora:
        raise RouterDataError("use either --adapter-name-or-path or --lora, not both")
    if args.deepspeed and args.stage == "both":
        raise RouterDataError(
            "DeepSpeed runs memorization and retrieval as separate launches; "
            "use --stage memorization and then --stage retrieval"
        )

    deepspeed_training_args = None
    if args.deepspeed:
        deepspeed_path, _, _ = _read_deepspeed_config(args.deepspeed)
        _require_supported_deepspeed_version()
        args.deepspeed = str(deepspeed_path)
        try:
            import transformers as transformers_for_args
        except ImportError as exc:  # pragma: no cover - real training environment
            raise SystemExit(
                "DeepSpeed training requires transformers and deepspeed; reinstall "
                "the project's training dependencies."
            ) from exc
        if args.stage == "memorization":
            phase = "memorization"
            validation_path = args.memorization_validation
            epochs = args.memorization_epochs
            learning_rate = args.memorization_learning_rate
            resume_from_checkpoint = _resume_value(
                args.resume_memorization_from_checkpoint
            )
        else:
            phase = "retrieval"
            validation_path = args.retrieval_validation
            epochs = args.retrieval_epochs
            learning_rate = args.retrieval_learning_rate
            resume_from_checkpoint = _resume_value(
                args.resume_retrieval_from_checkpoint
            )
        try:
            # Trainer's DeepSpeed config must exist before from_pretrained so
            # ZeRO-3 partitions parameters during model construction as well.
            deepspeed_training_args = _build_training_arguments(
                phase=args.phase_output_subdir or phase,
                has_validation=(
                    bool(read_jsonl(validation_path)) if validation_path else False
                ),
                epochs=epochs,
                learning_rate=learning_rate,
                resume_from_checkpoint=resume_from_checkpoint,
                args=args,
                transformers=transformers_for_args,
            )
        except (ImportError, RuntimeError) as exc:  # pragma: no cover
            raise RouterDataError(
                "failed to initialize DeepSpeed TrainingArguments; verify the "
                "deepspeed installation and JSON config"
            ) from exc

    virtual_tokens = (
        load_virtual_tokens(args.virtual_tokens)
        if args.routing_mode == HIERARCHICAL_ROUTING_MODE
        else ()
    )
    torch, transformers, tokenizer, model, token_ids = _load_training_stack(
        args, virtual_tokens
    )

    if args.stage in {"memorization", "both"}:
        _run_phase(
            phase="memorization",
            train_path=_require_phase_path(
                args.memorization_train, "memorization"
            ),
            validation_path=args.memorization_validation,
            system_prompt=args.memorization_system_prompt,
            epochs=args.memorization_epochs,
            learning_rate=args.memorization_learning_rate,
            resume_from_checkpoint=_resume_value(
                args.resume_memorization_from_checkpoint
            ),
            args=args,
            torch=torch,
            transformers=transformers,
            tokenizer=tokenizer,
            model=model,
            token_ids=token_ids,
            candidate_names=candidate_names,
            training_args=deepspeed_training_args,
            replay_inputs=(),
            replay_system_prompt=None,
            output_subdir=args.phase_output_subdir,
        )

    if args.stage in {"retrieval", "both"}:
        _run_phase(
            phase="retrieval",
            train_path=_require_phase_path(args.retrieval_train, "retrieval"),
            validation_path=args.retrieval_validation,
            system_prompt=args.retrieval_system_prompt,
            epochs=args.retrieval_epochs,
            learning_rate=args.retrieval_learning_rate,
            resume_from_checkpoint=_resume_value(args.resume_retrieval_from_checkpoint),
            args=args,
            torch=torch,
            transformers=transformers,
            tokenizer=tokenizer,
            model=model,
            token_ids=token_ids,
            candidate_names=candidate_names,
            training_args=deepspeed_training_args,
            replay_inputs=replay_arguments,
            replay_system_prompt=args.memorization_system_prompt,
            output_subdir=args.phase_output_subdir,
        )


if __name__ == "__main__":
    main()
