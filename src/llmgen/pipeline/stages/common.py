"""Shared, algorithm-neutral helpers for generic pipeline stage adapters."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from ..io import read_json, sha256_file
from ..providers import provider_config
from ..resources import (
    ResourceResolutionError,
    verify_run_provenance,
    visible_devices_environment,
)
from .base import StageContext


def python(context: StageContext) -> str:
    configured = str(context.config.get("runtime.python") or "").strip()
    if configured:
        return configured
    candidate = context.repo_root / ".venv" / "bin" / "python"
    return str(candidate if candidate.is_file() else os.sys.executable)


def checkpoint_environment(context: StageContext) -> dict[str, str]:
    """Expose only the private lineage-file path to training save hooks."""

    if context.checkpoint_lineage_path is None:
        raise ValueError("training stage has no checkpoint lineage path")
    return {
        "LLMGEN_PIPELINE_CHECKPOINT_LINEAGE": str(context.checkpoint_lineage_path)
    }


def verify_training_provenance(context: StageContext) -> None:
    """Refuse to train with code, packages, devices, or base bytes that drifted."""

    provenance_path = context.run_dir / "config" / "provenance.json"
    if not provenance_path.is_file():
        raise ValueError(f"Run training provenance is missing: {provenance_path}")
    frozen = read_json(provenance_path)
    if not isinstance(frozen, dict):
        raise ValueError(f"Run training provenance is invalid: {provenance_path}")
    try:
        verify_run_provenance(
            frozen,
            repo_root=context.repo_root,
            runtime=context.config.require("runtime"),
            base_model=str(context.config.require("router.base_model")),
        )
    except ResourceResolutionError as error:
        raise ValueError(str(error)) from error
    context.logger.event(
        "training.provenance_verified",
        provenance=str(provenance_path.relative_to(context.run_dir)),
        sha256=sha256_file(provenance_path),
    )


def configured(values: Mapping[str, Any], key: str, default: Any) -> Any:
    """Return a configured value without treating numeric zero as missing."""

    value = values.get(key)
    return default if value is None or value == "" else value


def paths(context: StageContext) -> dict[str, Path]:
    export_model = context.run_dir / str(context.config.require("export.output_dir"))

    def output(stage: str) -> Path:
        current_stage = getattr(getattr(context, "spec", None), "name", None)
        return (
            context.output_dir
            if current_stage == stage
            else context.state.stage_dir(stage) / "output"
        )

    return {
        "catalog": context.run_dir / "source" / "catalog.jsonl",
        "profiles": output("enrich") / "skill_profiles.jsonl",
        "workflows": output("plan-queries") / "workflows.jsonl",
        "generated_queries": output("generate-queries") / "queries.generated.jsonl",
        "alignment_queries": output("generate-queries") / "queries.alignment.generated.jsonl",
        "review_workflows": output("review-queries") / "workflows.jsonl",
        "review_queries": output("review-queries") / "queries.generated.jsonl",
        "reviews": output("review-queries") / "query_reviews.jsonl",
        "review_alignment_queries": output("review-queries") / "queries.alignment.generated.jsonl",
        "alignment_reviews": output("review-queries") / "query_alignment_reviews.jsonl",
        "dataset": output("finalize-dataset") / "dataset",
        "processed": output("finalize-dataset") / "processed",
        "embeddings": output("finalize-dataset") / "embeddings",
        "stage1": output("train-codebook") / "stage1",
        "code_plan": output("train-codebook") / "code_plan.json",
        "index": output("assign-codes") / "index",
        "router_data": output("build-sft") / "router_data",
        "router": context.run_dir / "models",
        "evaluation": output("evaluate") / "evaluation",
        "export_model": export_model,
        "export_report": export_model.parent / "report",
    }


def device_count(context: StageContext) -> int:
    provenance_path = context.run_dir / "config" / "provenance.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        resolved = (
            provenance.get("resources", {}).get("resolved", {})
            if isinstance(provenance, dict)
            else {}
        )
        frozen_count = resolved.get("num_devices")
        if isinstance(frozen_count, int) and not isinstance(frozen_count, bool):
            if frozen_count < 1:
                raise ValueError(
                    "runtime.num_devices=auto found no requested accelerator when "
                    "the Run was created; create or fork the Run on the training host"
                )
            return frozen_count
    explicit = context.config.get("runtime.num_devices")
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        return max(1, explicit)
    devices = context.config.get("runtime.devices")
    if isinstance(devices, list):
        return max(1, len(devices))
    if isinstance(devices, str) and devices not in {"", "auto"}:
        return max(1, len([value for value in devices.split(",") if value.strip()]))
    if str(context.config.get("runtime.device") or "").startswith("cuda"):
        try:
            import torch

            return max(1, int(torch.cuda.device_count()))
        except Exception:
            return 1
    return 1


def alignment_only(context: StageContext) -> bool:
    if context.config.require("input.single_candidate_policy") != "alignment_only":
        return False
    manifest_path = context.run_dir / "source" / "candidate_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    return isinstance(manifest, dict) and int(manifest.get("candidate_count", 0)) == 1


def legacy_environment(context: StageContext) -> dict[str, str]:
    stage_paths = paths(context)
    code_config = context.config.require("code")
    router = context.config.require("router")
    checkpointing = context.config.require("checkpointing")
    evaluation = context.config.require("evaluation")
    runtime = context.config.require("runtime")
    num_devices = device_count(context)
    deepspeed = runtime.get("deepspeed")
    if deepspeed == "auto":
        deepspeed = "configs/deepspeed_zero3.json" if num_devices > 1 else "none"
    plan = read_json(stage_paths["code_plan"]) if stage_paths["code_plan"].is_file() else None
    factors = (
        list(plan["branching_factors"])
        if isinstance(plan, dict)
        else list(code_config.get("branching_factors") or [64, 64])
    )
    levels = int(plan["num_levels"]) if isinstance(plan, dict) else len(factors)
    raw_epsilons = code_config.get("sk_epsilons")
    if isinstance(raw_epsilons, list) and len(raw_epsilons) == levels:
        epsilons = [float(value) for value in raw_epsilons]
    else:
        epsilons = [0.0] * max(0, levels - 1) + [0.01]
    raw_utilization = code_config.get("min_raw_level_utilization")
    if isinstance(raw_utilization, list) and len(raw_utilization) == levels:
        raw_level_utilization = raw_utilization
    else:
        raw_level_utilization = [0.0] * levels
    configured_batch = int(code_config.get("batch_size") or 512)
    embedding = provider_config(context, "embedding")
    environment = {
        "PYTHON": python(context),
        "SKILLRET_CONFIG": str(context.repo_root / "configs" / "generic.env"),
        "SKIP_DOWNLOAD": "1",
        "PREPARE_SCRIPT": "scripts/prepare_closedset.py",
        "DATASET_NAME": str(context.config.require("run.name")),
        "RUN_DIR": str(context.run_dir),
        "DATASET_DIR": str(stage_paths["dataset"]),
        "PROCESSED_DIR": str(stage_paths["processed"]),
        "EMBEDDING_DIR": str(stage_paths["embeddings"]),
        "STAGE1_DIR": str(stage_paths["stage1"]),
        "INDEX_DIR": str(stage_paths["index"]),
        "ROUTER_DATA_DIR": str(stage_paths["router_data"]),
        "ROUTER_OUTPUT_DIR": str(stage_paths["router"]),
        "ROUTER_EXPORT_MODEL_DIR": str(stage_paths["router"] / "retrieval"),
        "EVAL_DIR": str(stage_paths["evaluation"]),
        "DEVICE": str(runtime.get("device") or "cuda"),
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": str(embedding["model"]),
        "EMBEDDING_BASE_URL": str(embedding.get("base_url") or ""),
        "EMBEDDING_BATCH_SIZE": str(int(embedding.get("batch_size") or 8)),
        "EMBEDDING_DIMENSIONS": str(embedding.get("dimensions") or ""),
        "EMBEDDING_TIMEOUT": str(float(embedding.get("timeout_seconds") or 600)),
        "EMBEDDING_MAX_RETRIES": str(int(embedding.get("max_retries") or 5)),
        "EMBEDDING_MAX_BATCH_CHARS": str(int(embedding.get("max_batch_chars") or 12000)),
        "EMBEDDING_MAX_SKILL_CHARS": str(embedding.get("max_skill_chars") or ""),
        "LLMGEN_EMBEDDING_LEDGER_ROOT": str(context.stage_dir / "ledger" / "embedding" / "candidate-catalog"),
        "LLMGEN_EMBEDDING_LEDGER_BATCH_RECORDS": str(int(checkpointing.get("embedding_batch_records") or 100)),
        "NUM_LEVELS": str(levels),
        "BRANCHING_FACTORS": " ".join(map(str, factors)),
        "SK_EPSILONS": " ".join(map(str, epsilons)),
        "RQ_LAYERS": " ".join(map(str, code_config.get("rq_layers") or [512, 256, 128])),
        "TOKENIZER_E_DIM": str(int(code_config.get("embedding_dim") or 64)),
        "TOKENIZER_BETA": str(float(code_config.get("beta") or 0.25)),
        "TOKENIZER_EPOCHS": str(int(code_config.get("epochs") or 100)),
        "TOKENIZER_BATCH_SIZE": str(max(configured_batch, max(factors))),
        "TOKENIZER_LR": str(float(code_config.get("learning_rate") or 1e-4)),
        "TOKENIZER_SCHEDULER": str(code_config.get("scheduler") or "cosine"),
        "TOKENIZER_WARMUP_RATIO": str(float(code_config.get("warmup_ratio") or 0.05)),
        "TOKENIZER_EVAL_EVERY": str(int(code_config.get("eval_every") or 1)),
        "TOKENIZER_GRAPH_LAMBDA": str(float(code_config.get("graph_lambda") or 0.001)),
        "TOKENIZER_AMP_DTYPE": str(code_config.get("amp_dtype") or "bf16"),
        "CODEBOOK_VERSION": str(code_config.get("version") or context.config.require("run.name")),
        "CODE_EXPORT_BATCH_SIZE": str(int(code_config.get("export_batch_size") or 512)),
        "CODE_ASSIGNMENT_MODE": str(code_config.get("assignment") or "balanced_hierarchical"),
        "CODE_ASSIGNMENT_EXACT_GROUP_SIZE": str(int(code_config.get("assignment_exact_group_size") or 2048)),
        "CODE_QUALITY_GATE_SPLIT": "train",
        "CODE_MAX_COLLISION_RATE": str(float(configured(code_config, "max_collision_rate", 0.01))),
        "CODE_MAX_RAW_COLLISION_RATE": str(float(configured(code_config, "max_raw_collision_rate", 1.0))),
        "CODE_MAX_BUCKET_SIZE": str(int(code_config.get("max_bucket_size") or 2)),
        "CODE_MIN_LEVEL_UTILIZATION": str(float(configured(code_config, "min_level_utilization", 0.0))),
        "CODE_MIN_NORMALIZED_ENTROPY": str(float(configured(code_config, "min_normalized_entropy", 0.0))),
        "CODE_MIN_RAW_LEVEL_UTILIZATION": " ".join(map(str, raw_level_utilization)),
        "CODE_MIN_RAW_NORMALIZED_ENTROPY": str(float(configured(code_config, "min_raw_normalized_entropy", 0.0))),
        "MEMORIZATION_VALIDATION_FRACTION": "0",
        "ROUTER_VALIDATION_FRACTION": str(float(configured(router, "validation_fraction", 0.02))),
        "ROUTER_DATA_SEED": str(int(configured(router, "data_seed", context.config.require("run.seed")))),
        "ROUTER_MODEL": str(router["base_model"]),
        "ROUTER_FINETUNE_MODE": str(router["finetune_mode"]),
        "ROUTER_ALIGNMENT_ONLY": "1" if alignment_only(context) else "0",
        "ROUTER_NUM_GPUS": str(num_devices),
        "ROUTER_DEEPSPEED_CONFIG": str(deepspeed or "none"),
        "ROUTER_PER_DEVICE_TRAIN_BATCH_SIZE": str(int(router.get("per_device_train_batch_size") or 1)),
        "ROUTER_PER_DEVICE_EVAL_BATCH_SIZE": str(int(router.get("per_device_eval_batch_size") or 4)),
        "ROUTER_GRADIENT_ACCUMULATION_STEPS": str(int(router.get("gradient_accumulation_steps") or 8)),
        "ROUTER_MAX_LENGTH": str(int(router.get("max_length") or 1024)),
        "ROUTER_MEMORIZATION_EPOCHS": str(int(router["memorization"]["epochs"])),
        "ROUTER_ALIGNMENT_EPOCHS": str(int(router["alignment"]["epochs"]) if router["alignment"].get("enabled", True) else 0),
        "ROUTER_RETRIEVAL_EPOCHS": str(int(router["retrieval"]["epochs"])),
        "ROUTER_MEMORIZATION_LR": str(float(router["memorization"]["learning_rate"])),
        "ROUTER_ALIGNMENT_LR": str(float(router["alignment"]["learning_rate"])),
        "ROUTER_RETRIEVAL_LR": str(float(router["retrieval"]["learning_rate"])),
        "ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION": str(float(configured(router["retrieval"], "alignment_replay_fraction", 0))),
        "ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION": str(float(configured(router["retrieval"], "memorization_replay_fraction", 0))),
        "ROUTER_WEIGHT_DECAY": str(float(configured(router, "weight_decay", 0))),
        "ROUTER_WARMUP_RATIO": str(float(configured(router, "warmup_ratio", 0.03))),
        "ROUTER_LOGGING_STEPS": str(int(router.get("logging_steps") or 10)),
        "ROUTER_SAVE_STEPS": str(int(checkpointing.get("training_save_steps") or 100)),
        "ROUTER_EVAL_STEPS": str(int(checkpointing.get("training_eval_steps") or checkpointing.get("training_save_steps") or 100)),
        "ROUTER_SAVE_TOTAL_LIMIT": str(int(checkpointing.get("keep_last") or 3)),
        "ROUTER_DATALOADER_NUM_WORKERS": str(int(runtime.get("dataloader_workers") or 0)),
        "ROUTER_SEED": str(int(context.config.require("run.seed"))),
        "ROUTER_PRECISION": str(router.get("precision") or "bf16"),
        "ROUTER_GRADIENT_CHECKPOINTING": "1" if router.get("gradient_checkpointing", True) else "0",
        "ROUTER_GRADIENT_CHECKPOINTING_MODE": str(router.get("gradient_checkpointing_mode") or "auto"),
        "ROUTER_TRUST_REMOTE_CODE": "1" if router.get("trust_remote_code", False) else "0",
        "ROUTER_LORA_R": str(int(router["lora"].get("r") or 16)),
        "ROUTER_LORA_ALPHA": str(int(router["lora"].get("alpha") or 32)),
        "ROUTER_LORA_DROPOUT": str(float(configured(router["lora"], "dropout", 0.05))),
        "ROUTER_LORA_TARGET_MODULES": str(router["lora"].get("target_modules") or "q_proj,k_proj,v_proj,o_proj"),
        "ROUTER_LORA_MODULES_TO_SAVE": str(router["lora"].get("modules_to_save") or "auto"),
        "EVAL_PROTOCOL": str(evaluation.get("protocol") or "closedset"),
        "QUERY_SET": str(evaluation.get("query_split") or "test"),
        "EVAL_DTYPE": str(evaluation.get("dtype") or "bfloat16"),
        "EVAL_BATCH_SIZE": str(int(evaluation.get("batch_size") or 1)),
        "EVAL_MAX_CODE_PATHS": str(int(evaluation.get("max_code_paths") or 8)),
        "EVAL_TOP_K": str(int(evaluation.get("top_k") or 20)),
        "EVAL_CUTOFFS": " ".join(map(str, evaluation.get("cutoffs") or [1, 5, 10])),
    }
    if alignment_only(context):
        environment["QUERY_SET"] = "alignment"
    visibility_environment = visible_devices_environment(runtime)
    custom_environment = runtime.get("environment")
    if isinstance(custom_environment, Mapping):
        collisions = sorted(
            set(map(str, custom_environment)).intersection(
                {*environment, *visibility_environment}
            )
        )
        if collisions:
            raise ValueError(
                "runtime.environment cannot override pipeline-managed variable: "
                f"{collisions[0]}"
            )
        environment.update({str(key): str(value) for key, value in custom_environment.items()})
    environment.update(visibility_environment)
    api_key_env = str(embedding.get("api_key_env") or "")
    if api_key_env and os.environ.get(api_key_env):
        environment["OPENAI_API_KEY"] = os.environ[api_key_env]
    return environment


def router_pipeline(
    context: StageContext,
    command: str,
    *,
    environment_overrides: Mapping[str, str] | None = None,
) -> None:
    scripts = {
        "prepare": "scripts/skillret/01_prepare.sh",
        "train-tokenizer": "scripts/skillret/02_train_tokenizer.sh",
        "export-codes": "scripts/skillret/03_export_codes.sh",
        "build-router-data": "scripts/skillret/04_build_router_data.sh",
        "train-memorization": "scripts/skillret/05_train_memorization.sh",
        "evaluate": "scripts/skillret/07_evaluate.sh",
        "export-web": "scripts/skillret/10_export_web_bundle.sh",
    }
    try:
        script = scripts[command]
    except KeyError as error:
        raise ValueError(f"unsupported generic router command: {command}") from error
    environment = legacy_environment(context)
    if environment_overrides:
        environment.update(environment_overrides)
    context.run_command(
        ["bash", script],
        environment=environment,
        label=f"generic-router-{command}",
    )


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    for suffix in (".manifest.json", ".errors.jsonl"):
        companion_source = source.with_name(source.stem + suffix)
        if companion_source.is_file():
            shutil.copy2(
                companion_source,
                destination.with_name(destination.stem + suffix),
            )
