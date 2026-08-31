"""Stage adapters around the repository's existing algorithm entry points."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from ..code_plan import plan_codes
from ..io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)
from ..schema import (
    catalog_rows,
    ensure_ordered_qrels,
    read_candidate_file,
    validate_ordered_qrels,
)
from ...router_bundle import validate_skill_decode_map
from .base import ArtifactOutput, StageContext, StageResult, StageSpec


def _python(context: StageContext) -> str:
    configured = str(context.config.get("runtime.python") or "").strip()
    if configured:
        return configured
    candidate = context.repo_root / ".venv" / "bin" / "python"
    return str(candidate if candidate.is_file() else os.sys.executable)


def _provider(context: StageContext, name: str) -> dict[str, Any]:
    value = context.config.require(f"providers.{name}")
    assert isinstance(value, dict)
    return value


def _provider_api_config(context: StageContext, name: str) -> str:
    provider = _provider(context, name)
    path = str(provider.get("api_config") or "").strip()
    if path:
        return str(Path(path).expanduser())

    # Existing data builders accept an API config *path*. Keep that algorithm
    # boundary intact while sourcing credentials exclusively from the named
    # environment variable: load_api_config() falls back to API_BASE_URL and
    # OPENAI_API_KEY when this deliberately secret-free file is empty.
    if not str(provider.get("base_url") or "").strip():
        raise ValueError(
            f"providers.{name} requires base_url when api_config is not set"
        )
    key_environment = str(provider.get("api_key_env") or "").strip()
    if not key_environment:
        raise ValueError(
            f"providers.{name} requires api_key_env when api_config is not set"
        )
    if not os.environ.get(key_environment):
        raise ValueError(
            f"providers.{name}.api_key_env references unset environment "
            f"variable {key_environment}"
        )
    marker = context.attempt_dir / "provider" / f"{name}.conf"
    if not marker.exists():
        atomic_write_text(
            marker,
            "# endpoint and credential are supplied through the child environment\n",
            mode=0o600,
        )
    return str(marker)


def _provider_environment(context: StageContext, name: str) -> dict[str, str]:
    provider = _provider(context, name)
    environment: dict[str, str] = {}
    base_url = str(provider.get("base_url") or "").strip()
    if base_url:
        environment["API_BASE_URL"] = base_url
    key_environment = str(provider.get("api_key_env") or "").strip()
    if key_environment:
        secret = os.environ.get(key_environment)
        if secret:
            environment["OPENAI_API_KEY"] = secret
    timeout = provider.get("timeout_seconds")
    if timeout is not None:
        environment["LLMGEN_CHAT_TIMEOUT_SECONDS"] = str(float(timeout))
    retries = provider.get("max_retries")
    if retries is not None:
        environment["LLMGEN_CHAT_MAX_RETRIES"] = str(int(retries))
    return environment


def _workers(provider: Mapping[str, Any]) -> str:
    return str(int(provider.get("concurrency") or provider.get("workers") or 1))


def _configured(
    values: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    """Return a configured value without treating numeric zero as missing."""

    value = values.get(key)
    return default if value is None or value == "" else value


def _paths(context: StageContext) -> dict[str, Path]:
    export_model = context.run_dir / str(context.config.require("export.output_dir"))
    return {
        "catalog": context.run_dir / "source" / "catalog.jsonl",
        "profiles": context.state.stage_dir("enrich") / "output" / "skill_profiles.jsonl",
        "workflows": context.state.stage_dir("plan-queries") / "output" / "workflows.jsonl",
        "generated_queries": context.state.stage_dir("generate-queries") / "output" / "queries.generated.jsonl",
        "alignment_queries": context.state.stage_dir("generate-queries") / "output" / "queries.alignment.generated.jsonl",
        "review_workflows": context.state.stage_dir("review-queries") / "output" / "workflows.jsonl",
        "review_queries": context.state.stage_dir("review-queries") / "output" / "queries.generated.jsonl",
        "reviews": context.state.stage_dir("review-queries") / "output" / "query_reviews.jsonl",
        "review_alignment_queries": context.state.stage_dir("review-queries") / "output" / "queries.alignment.generated.jsonl",
        "alignment_reviews": context.state.stage_dir("review-queries") / "output" / "query_alignment_reviews.jsonl",
        "dataset": context.state.stage_dir("finalize-dataset") / "output" / "dataset",
        "processed": context.state.stage_dir("finalize-dataset") / "output" / "processed",
        "embeddings": context.state.stage_dir("finalize-dataset") / "output" / "embeddings",
        "stage1": context.state.stage_dir("train-codebook") / "output" / "stage1",
        "code_plan": context.state.stage_dir("train-codebook") / "output" / "code_plan.json",
        "index": context.state.stage_dir("assign-codes") / "output" / "index",
        "router_data": context.state.stage_dir("build-sft") / "output" / "router_data",
        "router": context.run_dir / "models",
        "evaluation": context.state.stage_dir("evaluate") / "output" / "evaluation",
        "export_model": export_model,
        "export_report": export_model.parent / "report",
    }


def _device_count(context: StageContext) -> int:
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


def _legacy_environment(context: StageContext) -> dict[str, str]:
    paths = _paths(context)
    code_config = context.config.require("code")
    router = context.config.require("router")
    checkpointing = context.config.require("checkpointing")
    evaluation = context.config.require("evaluation")
    runtime = context.config.require("runtime")
    num_devices = _device_count(context)
    deepspeed = runtime.get("deepspeed")
    if deepspeed == "auto":
        deepspeed = "configs/deepspeed_zero3.json" if num_devices > 1 else "none"
    plan = read_json(paths["code_plan"]) if paths["code_plan"].is_file() else None
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
    environment = {
        "PYTHON": _python(context),
        "SKILLRET_CONFIG": str(context.repo_root / "configs" / "light.env"),
        "SKIP_DOWNLOAD": "1",
        "PREPARE_SCRIPT": "scripts/prepare_closedset.py",
        "DATASET_NAME": str(context.config.require("run.name")),
        "RUN_DIR": str(context.run_dir),
        "DATASET_DIR": str(paths["dataset"]),
        "PROCESSED_DIR": str(paths["processed"]),
        "EMBEDDING_DIR": str(paths["embeddings"]),
        "STAGE1_DIR": str(paths["stage1"]),
        "INDEX_DIR": str(paths["index"]),
        "ROUTER_DATA_DIR": str(paths["router_data"]),
        "ROUTER_OUTPUT_DIR": str(paths["router"]),
        "ROUTER_EXPORT_MODEL_DIR": str(paths["router"] / "retrieval"),
        "EVAL_DIR": str(paths["evaluation"]),
        "DEVICE": str(runtime.get("device") or "cuda"),
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": str(_provider(context, "embedding")["model"]),
        "EMBEDDING_BASE_URL": str(_provider(context, "embedding").get("base_url") or ""),
        "EMBEDDING_BATCH_SIZE": str(int(_provider(context, "embedding").get("batch_size") or 8)),
        "EMBEDDING_DIMENSIONS": str(_provider(context, "embedding").get("dimensions") or ""),
        "EMBEDDING_TIMEOUT": str(float(_provider(context, "embedding").get("timeout_seconds") or 600)),
        "EMBEDDING_MAX_RETRIES": str(int(_provider(context, "embedding").get("max_retries") or 5)),
        "EMBEDDING_MAX_BATCH_CHARS": str(int(_provider(context, "embedding").get("max_batch_chars") or 12000)),
        "EMBEDDING_MAX_SKILL_CHARS": str(_provider(context, "embedding").get("max_skill_chars") or ""),
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
        "CODE_MAX_COLLISION_RATE": str(float(_configured(code_config, "max_collision_rate", 0.01))),
        "CODE_MAX_RAW_COLLISION_RATE": str(float(_configured(code_config, "max_raw_collision_rate", 1.0))),
        "CODE_MAX_BUCKET_SIZE": str(int(code_config.get("max_bucket_size") or 2)),
        "CODE_MIN_LEVEL_UTILIZATION": str(float(_configured(code_config, "min_level_utilization", 0.0))),
        "CODE_MIN_NORMALIZED_ENTROPY": str(float(_configured(code_config, "min_normalized_entropy", 0.0))),
        "CODE_MIN_RAW_LEVEL_UTILIZATION": " ".join(map(str, raw_level_utilization)),
        "CODE_MIN_RAW_NORMALIZED_ENTROPY": str(float(_configured(code_config, "min_raw_normalized_entropy", 0.0))),
        "MEMORIZATION_VALIDATION_FRACTION": "0",
        "ROUTER_VALIDATION_FRACTION": str(float(_configured(router, "validation_fraction", 0.02))),
        "ROUTER_DATA_SEED": str(int(_configured(router, "data_seed", context.config.require("run.seed")))),
        "ROUTER_MODEL": str(router["base_model"]),
        "ROUTER_FINETUNE_MODE": str(router["finetune_mode"]),
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
        "ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION": str(float(_configured(router["retrieval"], "alignment_replay_fraction", 0))),
        "ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION": str(float(_configured(router["retrieval"], "memorization_replay_fraction", 0))),
        "ROUTER_WEIGHT_DECAY": str(float(_configured(router, "weight_decay", 0))),
        "ROUTER_WARMUP_RATIO": str(float(_configured(router, "warmup_ratio", 0.03))),
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
        "ROUTER_LORA_DROPOUT": str(float(_configured(router["lora"], "dropout", 0.05))),
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
    devices = runtime.get("devices")
    if isinstance(devices, list):
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    elif isinstance(devices, str) and devices not in {"", "auto"}:
        environment["CUDA_VISIBLE_DEVICES"] = devices
    custom_environment = runtime.get("environment")
    if isinstance(custom_environment, Mapping):
        environment.update({str(key): str(value) for key, value in custom_environment.items()})
    api_key_env = str(_provider(context, "embedding").get("api_key_env") or "")
    if api_key_env and os.environ.get(api_key_env):
        environment["OPENAI_API_KEY"] = os.environ[api_key_env]
    return environment


def _router_pipeline(
    context: StageContext,
    command: str,
    *,
    environment_overrides: Mapping[str, str] | None = None,
) -> None:
    environment = _legacy_environment(context)
    if environment_overrides:
        environment.update(environment_overrides)
    context.run_command(
        ["bash", "scripts/router_pipeline.sh", "light", command],
        environment=environment,
        label=f"legacy-router-{command}",
    )


def _ingest(context: StageContext) -> StageResult:
    configured_source = Path(str(context.config.require("input.candidates"))).expanduser()
    configured_source = (
        configured_source
        if configured_source.is_absolute()
        else context.repo_root / configured_source
    ).resolve()
    frozen_fingerprint = read_json(context.run_dir / "config" / "candidate_input.json")
    frozen_path = context.run_dir / str(
        frozen_fingerprint.get("frozen_path") or "source/candidates.input.jsonl"
    )
    source = frozen_path if frozen_path.is_file() else configured_source
    expected_hash = str(frozen_fingerprint.get("sha256") or "")
    actual_hash = sha256_file(source)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            "candidate input changed after Run creation; create a new Run or fork "
            "with the intended candidate file"
        )
    candidates = read_candidate_file(
        source,
        id_policy=str(context.config.require("input.id_policy")),
        preserve_metadata=bool(context.config.require("input.preserve_metadata")),
    )
    if len(candidates) == 1 and context.config.require("input.single_candidate_policy") == "error":
        raise ValueError(
            "a single-candidate run cannot construct multi-Skill retrieval data; "
            "the current generic adapter requires at least two candidates"
        )
    source_dir = context.run_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    input_copy = source_dir / "candidates.input.jsonl"
    normalized_path = source_dir / "candidates.normalized.jsonl"
    catalog_path = source_dir / "catalog.jsonl"
    manifest_path = source_dir / "candidate_manifest.json"
    atomic_write_bytes(input_copy, source.read_bytes())
    atomic_write_jsonl(normalized_path, candidates)
    atomic_write_jsonl(
        catalog_path,
        catalog_rows(candidates, source="source/candidates.input.jsonl"),
    )
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source": str(configured_source),
        "candidate_count": len(candidates),
        "unique_id_count": len({row["skill_id"] for row in candidates}),
        "unique_name_count": len({row["name"] for row in candidates}),
        "ordered_skill_ids": [row["skill_id"] for row in candidates],
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (input_copy, normalized_path, catalog_path)
        },
    }
    atomic_write_json(manifest_path, manifest)
    context.update_progress(completed=len(candidates), total=len(candidates))
    return StageResult(
        artifacts=(
            ArtifactOutput("candidates.input", input_copy, "candidate_input/v1"),
            ArtifactOutput("candidates.normalized", normalized_path, "candidate/v1"),
            ArtifactOutput("candidates.catalog", catalog_path, "candidate_catalog/v1"),
            ArtifactOutput("candidates.manifest", manifest_path, "candidate_manifest/v1"),
        ),
        progress={"candidate_count": len(candidates)},
    )


def _enrich(context: StageContext) -> StageResult:
    paths = _paths(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    provider = _provider(context, "generation")
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/00_profile_skills.py",
            "--catalog",
            str(paths["catalog"]),
            "--output",
            str(paths["profiles"]),
            "--api-config",
            _provider_api_config(context, "generation"),
            "--model",
            str(provider["model"]),
            "--workers",
            _workers(provider),
            "--batch-size",
            str(int(context.config.get("data_generation.profile_batch_size") or 10)),
        ],
        environment=_provider_environment(context, "generation"),
        label="profile-candidates",
    )
    return StageResult(
        artifacts=(ArtifactOutput("data.profiles", paths["profiles"], "skill_profile/v1"),)
    )


def _plan_queries(context: StageContext) -> StageResult:
    paths = _paths(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/01_build_workflows.py",
            "--profiles",
            str(paths["profiles"]),
            "--output",
            str(paths["workflows"]),
            "--workflows-per-skill",
            str(int(context.config.require("data_generation.workflows_per_skill"))),
            "--seed",
            str(int(context.config.require("run.seed"))),
        ],
        label="plan-workflows",
    )
    return StageResult(
        artifacts=(ArtifactOutput("data.workflows", paths["workflows"], "workflow_plan/v1"),)
    )


def _generate_queries(context: StageContext) -> StageResult:
    paths = _paths(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    provider = _provider(context, "generation")
    common = [
        "--api-config",
        _provider_api_config(context, "generation"),
        "--model",
        str(provider["model"]),
        "--workers",
        _workers(provider),
    ]
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/02a_generate_alignment_queries.py",
            "--profiles",
            str(paths["profiles"]),
            "--output",
            str(paths["alignment_queries"]),
            "--variants",
            str(int(context.config.require("data_generation.alignment_queries_per_skill"))),
            "--batch-size",
            str(int(context.config.require("data_generation.alignment_batch_size"))),
            *common,
        ],
        environment=_provider_environment(context, "generation"),
        label="generate-alignment-queries",
    )
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/02_generate_queries.py",
            "--workflows",
            str(paths["workflows"]),
            "--output",
            str(paths["generated_queries"]),
            "--variants",
            str(int(context.config.require("data_generation.explicit_variants"))),
            "--implicit-variants",
            str(int(context.config.require("data_generation.implicit_variants"))),
            "--batch-size",
            str(int(context.config.require("data_generation.query_batch_size"))),
            "--validation-retry-rounds",
            str(int(context.config.require("data_generation.validation_retry_rounds"))),
            "--min-completion-rate",
            str(float(context.config.require("data_generation.min_completion_rate"))),
            *common,
        ],
        environment=_provider_environment(context, "generation"),
        label="generate-multiskill-queries",
    )
    return StageResult(
        artifacts=(
            ArtifactOutput("data.queries.generated", paths["generated_queries"], "query_draft/v1"),
            ArtifactOutput("data.queries.alignment.generated", paths["alignment_queries"], "alignment_query_draft/v1"),
        )
    )


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    for suffix in (".manifest.json", ".errors.jsonl"):
        companion_source = source.with_name(source.stem + suffix)
        if companion_source.is_file():
            shutil.copy2(
                companion_source,
                destination.with_name(destination.stem + suffix),
            )


def _review_alignment(context: StageContext, paths: Mapping[str, Path]) -> None:
    provider = _provider(context, "review")
    common = [
        "--api-config",
        _provider_api_config(context, "review"),
        "--model",
        str(provider["model"]),
        "--workers",
        _workers(provider),
        "--batch-size",
        str(int(context.config.require("data_generation.review_batch_size"))),
    ]
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/03a_review_alignment_queries.py",
            "--queries",
            str(paths["review_alignment_queries"]),
            "--profiles",
            str(paths["profiles"]),
            "--output",
            str(paths["alignment_reviews"]),
            *common,
        ],
        environment=_provider_environment(context, "review"),
        label="review-alignment-queries",
    )


def _review_multiskill(context: StageContext, paths: Mapping[str, Path]) -> None:
    provider = _provider(context, "review")
    common = [
        "--api-config",
        _provider_api_config(context, "review"),
        "--model",
        str(provider["model"]),
        "--workers",
        _workers(provider),
        "--batch-size",
        str(int(context.config.require("data_generation.review_batch_size"))),
    ]
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/03_review_queries.py",
            "--queries",
            str(paths["review_queries"]),
            "--workflows",
            str(paths["review_workflows"]),
            "--output",
            str(paths["reviews"]),
            *common,
        ],
        environment=_provider_environment(context, "review"),
        label="review-multiskill-queries",
    )


def _review_queries(context: StageContext) -> StageResult:
    paths = _paths(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _copy(paths["workflows"], paths["review_workflows"])
    _copy(paths["generated_queries"], paths["review_queries"])
    _copy(paths["alignment_queries"], paths["review_alignment_queries"])
    _review_alignment(context, paths)
    generation = _provider(context, "generation")
    generation_common = [
        "--api-config",
        _provider_api_config(context, "generation"),
        "--model",
        str(generation["model"]),
        "--workers",
        _workers(generation),
    ]
    alignment_rounds = int(context.config.require("data_generation.alignment_backfill_rounds"))
    min_alignment = int(context.config.require("data_generation.alignment_queries_per_skill"))
    for round_index in range(1, alignment_rounds + 1):
        context.run_command(
            [
                _python(context),
                "scripts/clawhub_data/03a2_backfill_alignment.py",
                "--profiles",
                str(paths["profiles"]),
                "--queries",
                str(paths["review_alignment_queries"]),
                "--reviews",
                str(paths["alignment_reviews"]),
                "--round",
                str(round_index),
                "--variants",
                str(int(context.config.require("data_generation.alignment_queries_per_skill"))),
                "--batch-size",
                str(int(context.config.require("data_generation.alignment_batch_size"))),
                "--min-passed-per-skill",
                str(min_alignment),
                *generation_common,
            ],
            environment=_provider_environment(context, "generation"),
            label=f"alignment-backfill-{round_index}",
        )
        _review_alignment(context, paths)

    # Match the established light/ClawHub flow: curated alignment examples
    # participate in the combined coverage calculation that follows.
    manual = str(context.config.get("data_generation.manual_alignment_path") or "").strip()
    if manual:
        context.run_command(
            [
                _python(context),
                "scripts/light_data/02b_apply_manual_alignment.py",
                "--profiles",
                str(paths["profiles"]),
                "--queries",
                str(paths["review_alignment_queries"]),
                "--reviews",
                str(paths["alignment_reviews"]),
                "--curated",
                str(Path(manual).expanduser()),
            ],
            label="apply-manual-alignment",
        )

    _review_multiskill(context, paths)

    coverage_rounds = int(context.config.require("data_generation.max_backfill_rounds"))
    for round_index in range(1, coverage_rounds + 1):
        context.run_command(
            [
                _python(context),
                "scripts/clawhub_data/03b_build_coverage_workflows.py",
                "--profiles",
                str(paths["profiles"]),
                "--workflows",
                str(paths["review_workflows"]),
                "--queries",
                str(paths["review_queries"]),
                "--reviews",
                str(paths["reviews"]),
                "--alignment-queries",
                str(paths["review_alignment_queries"]),
                "--alignment-reviews",
                str(paths["alignment_reviews"]),
                "--round",
                str(round_index),
                "--min-train-positives-per-skill",
                str(int(context.config.require("data_generation.retrieval_positives_per_skill"))),
                "--variants-per-workflow",
                str(int(context.config.require("data_generation.explicit_variants"))),
                "--oversample-factor",
                str(float(context.config.require("data_generation.coverage_oversample_factor"))),
                "--seed",
                str(int(context.config.require("run.seed"))),
            ],
            label=f"plan-coverage-backfill-{round_index}",
        )
        context.run_command(
            [
                _python(context),
                "scripts/clawhub_data/02_generate_queries.py",
                "--workflows",
                str(paths["review_workflows"]),
                "--output",
                str(paths["review_queries"]),
                "--variants",
                str(int(context.config.require("data_generation.explicit_variants"))),
                "--implicit-variants",
                str(int(context.config.require("data_generation.implicit_variants"))),
                "--batch-size",
                str(int(context.config.require("data_generation.query_batch_size"))),
                "--validation-retry-rounds",
                str(int(context.config.require("data_generation.validation_retry_rounds"))),
                "--min-completion-rate",
                str(float(context.config.require("data_generation.min_completion_rate"))),
                *generation_common,
            ],
            environment=_provider_environment(context, "generation"),
            label=f"generate-coverage-backfill-{round_index}",
        )
        _review_multiskill(context, paths)

    final_rounds = int(context.config.require("data_generation.final_alignment_backfill_rounds"))
    for offset in range(1, final_rounds + 1):
        round_index = alignment_rounds + offset
        context.run_command(
            [
                _python(context),
                "scripts/clawhub_data/03a2_backfill_alignment.py",
                "--profiles",
                str(paths["profiles"]),
                "--queries",
                str(paths["review_alignment_queries"]),
                "--reviews",
                str(paths["alignment_reviews"]),
                "--round",
                str(round_index),
                "--variants",
                str(int(context.config.require("data_generation.alignment_queries_per_skill"))),
                "--batch-size",
                str(int(context.config.require("data_generation.alignment_batch_size"))),
                "--min-passed-per-skill",
                str(min_alignment),
                "--multiskill-queries",
                str(paths["review_queries"]),
                "--multiskill-reviews",
                str(paths["reviews"]),
                "--workflows",
                str(paths["review_workflows"]),
                "--min-combined-per-skill",
                str(int(context.config.require("data_generation.retrieval_positives_per_skill"))),
                *generation_common,
            ],
            environment=_provider_environment(context, "generation"),
            label=f"final-alignment-backfill-{offset}",
        )
        _review_alignment(context, paths)
    return StageResult(
        artifacts=(
            ArtifactOutput("data.workflows.reviewed", paths["review_workflows"], "workflow_plan/v1"),
            ArtifactOutput("data.queries.reviewed", paths["review_queries"], "query_draft/v1"),
            ArtifactOutput("data.reviews", paths["reviews"], "query_review/v1"),
            ArtifactOutput("data.queries.alignment.reviewed", paths["review_alignment_queries"], "alignment_query_draft/v1"),
            ArtifactOutput("data.reviews.alignment", paths["alignment_reviews"], "alignment_query_review/v1"),
        )
    )


def _finalize_dataset(context: StageContext) -> StageResult:
    paths = _paths(context)
    paths["dataset"].mkdir(parents=True, exist_ok=True)
    minimum = int(context.config.require("data_generation.retrieval_positives_per_skill"))
    common = [
        "--catalog",
        str(paths["catalog"]),
        "--profiles",
        str(paths["profiles"]),
        "--workflows",
        str(paths["review_workflows"]),
        "--queries",
        str(paths["review_queries"]),
        "--reviews",
        str(paths["reviews"]),
        "--alignment-queries",
        str(paths["review_alignment_queries"]),
        "--alignment-reviews",
        str(paths["alignment_reviews"]),
    ]
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/04_export_dataset.py",
            *common,
            "--output-dir",
            str(paths["dataset"]),
            "--seed",
            str(int(context.config.require("run.seed"))),
            "--min-train-positives-per-skill",
            str(minimum),
            "--min-augmented-train-queries",
            str(int(context.config.require("data_generation.min_augmented_train_queries"))),
            "--target-order-variants",
            str(int(context.config.require("data_generation.order_variants"))),
        ],
        label="export-training-dataset",
    )
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/04a_export_alignment.py",
            "--catalog",
            str(paths["catalog"]),
            "--queries",
            str(paths["review_alignment_queries"]),
            "--reviews",
            str(paths["alignment_reviews"]),
            "--output-dir",
            str(paths["dataset"]),
            "--min-queries-per-skill",
            str(int(context.config.require("data_generation.alignment_queries_per_skill"))),
        ],
        label="export-alignment-dataset",
    )
    ensure_ordered_qrels(paths["dataset"])
    validate_ordered_qrels(paths["dataset"])
    count = len(read_jsonl(paths["dataset"] / "skills.jsonl"))
    context.run_command(
        [
            _python(context),
            "scripts/clawhub_data/05_validate_dataset.py",
            "--dataset-dir",
            str(paths["dataset"]),
            "--expected-candidates",
            str(count),
        ],
        label="audit-final-dataset",
    )
    _router_pipeline(context, "prepare")
    artifacts = [
        ArtifactOutput("dataset.directory", paths["dataset"], "closedset_dataset/v3"),
        ArtifactOutput("dataset.manifest", paths["dataset"] / "manifest.json", "closedset_manifest/v3"),
        ArtifactOutput("processed.directory", paths["processed"], "processed_closedset/v1"),
        ArtifactOutput("processed.manifest", paths["processed"] / "manifest.json", "processed_manifest/v1"),
        ArtifactOutput("embeddings.directory", paths["embeddings"], "embedding_bundle/v1"),
        ArtifactOutput("embeddings.manifest", paths["embeddings"] / "manifest.json", "embedding_manifest/v1"),
    ]
    for split in ("train", "validation", "test", "alignment"):
        for kind in ("queries", "qrels"):
            path = paths["dataset"] / f"{kind}_{split}.jsonl"
            if path.is_file():
                artifacts.append(
                    ArtifactOutput(
                        f"dataset.{kind}.{split}",
                        path,
                        f"router_{kind.rstrip('s')}/1",
                    )
                )
    return StageResult(artifacts=tuple(artifacts), progress={"candidate_count": count})


def _train_codebook(context: StageContext) -> StageResult:
    paths = _paths(context)
    candidate_manifest = read_json(context.artifact("candidates.manifest"))
    count = int(candidate_manifest["candidate_count"])
    code_plan = plan_codes(count, context.config.require("code"))
    paths["code_plan"].parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["code_plan"], code_plan.to_dict())
    resume = (
        {"TOKENIZER_RESUME": context.resume_checkpoint}
        if context.resume_checkpoint
        else None
    )
    _router_pipeline(context, "train-tokenizer", environment_overrides=resume)
    return StageResult(
        artifacts=(
            ArtifactOutput("code.plan", paths["code_plan"], "code_plan/v1"),
            ArtifactOutput("codebook.directory", paths["stage1"], "toolweaver_codebook/v1"),
            ArtifactOutput("codebook.best", paths["stage1"] / "best.pt", "toolweaver_checkpoint/v1"),
        ),
        progress={"candidate_count": count, "code_plan": code_plan.to_dict()},
    )


def _assign_codes(context: StageContext) -> StageResult:
    paths = _paths(context)
    _router_pipeline(context, "export-codes")
    return StageResult(
        artifacts=(
            ArtifactOutput("codes.directory", paths["index"], "skill_code_index/v1"),
            ArtifactOutput("codes.train", paths["index"] / "train_codes.jsonl", "skill_code/v1"),
            ArtifactOutput("codes.registry", paths["index"] / "train_registry.json", "skill_registry/v1"),
            ArtifactOutput("codes.virtual_tokens", paths["index"] / "virtual_tokens.txt", "virtual_tokens/v1"),
            ArtifactOutput("codes.manifest", paths["index"] / "manifest.json", "code_index_manifest/v1"),
        )
    )


def _build_sft(context: StageContext) -> StageResult:
    paths = _paths(context)
    _router_pipeline(context, "build-router-data")
    artifacts = [
        ArtifactOutput("sft.directory", paths["router_data"], "router_sft_bundle/v1"),
        ArtifactOutput("sft.manifest", paths["router_data"] / "manifest.json", "router_sft_manifest/v1"),
    ]
    for name in (
        "memorization_train",
        "memorization_validation",
        "retrieval_alignment_train",
        "retrieval_train",
        "retrieval_validation",
    ):
        path = paths["router_data"] / f"{name}.jsonl"
        if path.is_file():
            artifacts.append(
                ArtifactOutput(f"sft.{name.replace('_', '.')}", path, "router_sft/v1")
            )
    return StageResult(artifacts=tuple(artifacts))


def _train_memorization(context: StageContext) -> StageResult:
    paths = _paths(context)
    resume = (
        {"ROUTER_RESUME_MEMORIZATION": context.resume_checkpoint}
        if context.resume_checkpoint
        else None
    )
    _router_pipeline(context, "train-memorization", environment_overrides=resume)
    model = paths["router"] / "memorization"
    return StageResult(
        artifacts=(ArtifactOutput("model.memorization", model, "router_model/v1"),)
    )


def _train_alignment(context: StageContext) -> StageResult:
    paths = _paths(context)
    enabled = bool(context.config.require("router.alignment.enabled"))
    if enabled and int(context.config.require("router.alignment.epochs")) > 0:
        environment = _legacy_environment(context)
        if context.resume_checkpoint:
            environment["ROUTER_RESUME_ALIGNMENT"] = context.resume_checkpoint
        context.run_command(
            ["bash", "scripts/skillret/06a_train_alignment.sh"],
            environment=environment,
            label="legacy-router-train-alignment",
        )
        model = paths["router"] / "retrieval_alignment"
        metadata = {"passthrough": False}
    else:
        model = paths["router"] / "memorization"
        metadata = {"passthrough": True, "reason": "alignment disabled"}
        context.logger.event("stage.passthrough", source=str(model), reason="alignment disabled")
    return StageResult(
        artifacts=(ArtifactOutput("model.alignment", model, "router_model/v1", metadata=metadata),)
    )


def _train_retrieval(context: StageContext) -> StageResult:
    paths = _paths(context)
    environment = _legacy_environment(context)
    if context.resume_checkpoint:
        environment["ROUTER_RESUME_RETRIEVAL"] = context.resume_checkpoint
    context.run_command(
        ["bash", "scripts/skillret/06b_train_retrieval.sh"],
        environment=environment,
        label="legacy-router-train-retrieval",
    )
    model = paths["router"] / "retrieval"
    return StageResult(
        artifacts=(ArtifactOutput("model.retrieval", model, "router_model/v1"),)
    )


def _evaluate(context: StageContext) -> StageResult:
    paths = _paths(context)
    _router_pipeline(context, "evaluate")
    metrics = paths["evaluation"] / "metrics.json"
    artifacts = [
        ArtifactOutput("evaluation.directory", paths["evaluation"], "router_evaluation/v1")
    ]
    if metrics.is_file():
        artifacts.append(
            ArtifactOutput("evaluation.metrics", metrics, "router_metrics/v1")
        )
    predictions = paths["evaluation"] / "predictions.jsonl"
    if predictions.is_file():
        artifacts.append(
            ArtifactOutput("evaluation.predictions", predictions, "router_predictions/v1")
        )
    return StageResult(artifacts=tuple(artifacts))


def _copy_model_tree(source: Path, destination: Path) -> None:
    def link_weights_or_copy(src: str, dst: str) -> str:
        # export-web rewrites JSON/tokenizer/decoder files in place. Only
        # immutable weight blobs may be hard-linked; metadata must be copied so
        # exporting can never mutate the completed training artifact.
        source_path = Path(src)
        immutable_weight = (
            source_path.suffix == ".safetensors"
            or source_path.name.startswith("pytorch_model-")
            or source_path.name in {"pytorch_model.bin", "adapter_model.bin"}
        )
        if immutable_weight:
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        shutil.copy2(src, dst)
        return dst

    shutil.copytree(
        source,
        destination,
        copy_function=link_weights_or_copy,
        ignore=shutil.ignore_patterns("checkpoint-*"),
    )


def _root_weight_files(model_dir: Path) -> list[Path]:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    files = [model_dir / name for name in names if (model_dir / name).is_file()]
    files.extend(sorted(model_dir.glob("model-*.safetensors")))
    files.extend(sorted(model_dir.glob("pytorch_model-*.bin")))
    return list(dict.fromkeys(files))


def _full_weights_are_present(model_dir: Path, weight_files: Sequence[Path]) -> bool:
    """Check that a root weight file or every shard named by its index exists."""

    if any(
        path.is_file() and path.stat().st_size > 0
        for path in (model_dir / "model.safetensors", model_dir / "pytorch_model.bin")
    ):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / name
        if not index_path.is_file():
            continue
        try:
            payload = read_json(index_path)
            weight_map = payload.get("weight_map")
            shard_names = (
                {str(value) for value in weight_map.values()}
                if isinstance(weight_map, dict)
                else set()
            )
        except (OSError, TypeError, ValueError):
            shard_names = set()
        safe_shards = [
            model_dir / shard
            for shard in shard_names
            if Path(shard).name == shard
        ]
        if (
            len(safe_shards) == len(shard_names)
            and safe_shards
            and all(path.is_file() and path.stat().st_size > 0 for path in safe_shards)
        ):
            return True
    return False


def _export_quality(
    context: StageContext,
    model_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required_names = (
        "config.json",
        "tokenizer_config.json",
        "router_manifest.json",
        "skill_decode_map.json",
        "virtual_tokens.txt",
    )
    required_files = {
        name: (model_dir / name).is_file()
        for name in required_names
    }
    weight_files = _root_weight_files(model_dir)
    tokenizer_files = [
        name
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
        if (model_dir / name).is_file()
    ]

    decode_map: dict[str, Any] = {}
    decode_error: str | None = None
    try:
        raw_decode = read_json(model_dir / "skill_decode_map.json")
        if not isinstance(raw_decode, dict):
            raise ValueError("decode map must be an object")
        validate_skill_decode_map(raw_decode)
        decode_map = raw_decode
    except Exception as error:
        decode_error = f"{type(error).__name__}: {error}"
    candidate_count = int(
        read_json(context.artifact("candidates.manifest"))["candidate_count"]
    )
    decoded_skills = decode_map.get("skills")
    decoded_count = len(decoded_skills) if isinstance(decoded_skills, dict) else 0
    candidate_coverage = decoded_count / candidate_count if candidate_count else 0.0

    token_consistency = False
    if decode_map:
        try:
            file_tokens = [
                line.strip()
                for line in (model_dir / "virtual_tokens.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            map_tokens = [str(value) for value in decode_map["virtual_tokens"]]
            token_consistency = file_tokens == map_tokens and len(file_tokens) == len(
                set(file_tokens)
            )
        except (KeyError, OSError, TypeError, ValueError):
            token_consistency = False

    predictions_path = _paths(context)["evaluation"] / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.is_file() else []
    invalid_predictions = [
        row
        for row in predictions
        if not isinstance(row.get("paths"), list) or not row.get("paths")
    ]
    format_valid_rate = (
        (len(predictions) - len(invalid_predictions)) / len(predictions)
        if predictions
        else 0.0
    )
    required_format_rate = float(
        context.config.require("evaluation.require_format_valid_rate")
    )
    required_candidate_coverage = float(
        context.config.require("evaluation.require_candidate_coverage")
    )
    metrics_path = _paths(context)["evaluation"] / "metrics.json"
    metrics_payload: dict[str, Any] = {}
    metrics_error: str | None = None
    try:
        raw_metrics = read_json(metrics_path)
        if not isinstance(raw_metrics, dict) or not isinstance(
            raw_metrics.get("metrics"), dict
        ):
            raise ValueError("metrics.json must contain a metrics object")
        metrics_payload = raw_metrics
    except Exception as error:
        metrics_error = f"{type(error).__name__}: {error}"
    metric_values = metrics_payload.get("metrics", {})
    metric_thresholds = context.config.require("evaluation.metric_thresholds")
    gates = {
        "required_files": all(required_files.values()),
        "full_model_weights": _full_weights_are_present(model_dir, weight_files),
        "tokenizer_assets": bool(tokenizer_files),
        "decoder_token_consistency": token_consistency,
        "evaluation_metrics_present": metrics_error is None,
        "format_valid_rate": format_valid_rate >= required_format_rate,
        "candidate_coverage": candidate_coverage >= required_candidate_coverage,
    }
    for name, threshold in metric_thresholds.items():
        value = metric_values.get(name)
        gates[f"metric:{name}"] = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= float(threshold)
        )
    return (
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "required_files": required_files,
            "weight_files": [path.name for path in weight_files],
            "tokenizer_files": tokenizer_files,
            "decode_error": decode_error,
            "candidate_count": candidate_count,
            "decoded_candidate_count": decoded_count,
            "candidate_coverage": candidate_coverage,
            "required_candidate_coverage": required_candidate_coverage,
            "prediction_count": len(predictions),
            "invalid_prediction_count": len(invalid_predictions),
            "format_valid_rate": format_valid_rate,
            "required_format_valid_rate": required_format_rate,
            "metrics_error": metrics_error,
            "metrics": metric_values,
            "metric_thresholds": metric_thresholds,
            "gates": gates,
            "passed": all(gates.values()),
            "require_all_gates": bool(
                context.config.require("export.require_all_gates")
            ),
            "allow_failed_gates": bool(
                context.config.require("export.allow_failed_gates")
            ),
            "model_load_smoke_test": "not_requested",
        },
        invalid_predictions,
    )


def _run_export_model_smoke(context: StageContext, model_dir: Path) -> None:
    if not context.config.require("export.smoke_test"):
        return
    script = (
        "from transformers import AutoModelForCausalLM, AutoTokenizer; "
        "import sys; p=sys.argv[1]; trust=sys.argv[2]=='1'; "
        "AutoTokenizer.from_pretrained(p, local_files_only=True, "
        "trust_remote_code=trust); "
        "AutoModelForCausalLM.from_pretrained(p, local_files_only=True, "
        "trust_remote_code=trust)"
    )
    context.run_command(
        [
            _python(context),
            "-c",
            script,
            str(model_dir),
            "1" if context.config.require("router.trust_remote_code") else "0",
        ],
        label="export-model-load-smoke-test",
    )


def _export(context: StageContext) -> StageResult:
    paths = _paths(context)
    source = context.artifact("model.retrieval")
    temporary = context.attempt_dir / "model"
    if temporary.exists():
        shutil.rmtree(temporary)
    _copy_model_tree(source, temporary)
    environment = _legacy_environment(context)
    environment["ROUTER_EXPORT_MODEL_DIR"] = str(temporary)
    context.run_command(
        ["bash", "scripts/router_pipeline.sh", "light", "export-web"],
        environment=environment,
        label="materialize-router-bundle",
    )
    run_manifest = context.state.read_run()
    router_manifest_path = temporary / "router_manifest.json"
    if router_manifest_path.is_file():
        router_manifest = read_json(router_manifest_path)
        if not isinstance(router_manifest, dict):
            raise ValueError("exported router_manifest.json must be an object")
        router_manifest["pipeline_lineage"] = {
            "schema_version": 1,
            "run_id": run_manifest["run_id"],
            "config_sha256": context.config.hash,
            "git_commit": run_manifest.get("git_commit"),
            "candidate_input_sha256": read_json(
                context.run_dir / "config" / "candidate_input.json"
            )["sha256"],
            "source_artifact": "model.retrieval",
        }
        atomic_write_json(router_manifest_path, router_manifest)

    smoke_status = "not_requested"
    smoke_error: str | None = None
    if context.config.require("export.smoke_test"):
        try:
            _run_export_model_smoke(context, temporary)
            smoke_status = "passed"
        except Exception as error:
            smoke_status = "failed"
            smoke_error = f"{type(error).__name__}: {error}"
            context.logger.event(
                "export.smoke_test_failed",
                level="ERROR",
                error_type=type(error).__name__,
                error=str(error),
            )

    quality, failed_examples = _export_quality(context, temporary)
    quality["model_load_smoke_test"] = smoke_status
    if smoke_error is not None:
        quality["model_load_smoke_error"] = smoke_error
    if smoke_status != "not_requested":
        quality["gates"]["model_load_smoke_test"] = smoke_status == "passed"
    quality["passed"] = all(quality["gates"].values())
    quality["deployment_qualified"] = (
        quality["passed"] and smoke_status == "passed"
    )

    if router_manifest_path.is_file():
        router_manifest = read_json(router_manifest_path)
        router_manifest["pipeline_quality"] = {
            "schema_version": 1,
            "passed": quality["passed"],
            "deployment_qualified": quality["deployment_qualified"],
            "failed_gates": [
                name for name, passed in quality["gates"].items() if not passed
            ],
            "allow_failed_gates": bool(
                context.config.require("export.allow_failed_gates")
            ),
            "model_load_smoke_test": smoke_status,
        }
        atomic_write_json(router_manifest_path, router_manifest)

    attempt_report = context.attempt_dir / "report"
    if attempt_report.exists():
        shutil.rmtree(attempt_report)
    attempt_report.mkdir(parents=True)
    registry_snapshot = {
        name: record.to_dict()
        for name, record in context.registry.all().items()
    }
    atomic_write_json(attempt_report / "quality_gates.json", quality)
    atomic_write_json(attempt_report / "artifact_lineage.json", registry_snapshot)
    atomic_write_jsonl(attempt_report / "failed_examples.jsonl", failed_examples)

    require_all = bool(context.config.require("export.require_all_gates"))
    allow_failed = bool(context.config.require("export.allow_failed_gates"))
    if not quality["passed"] and require_all and not allow_failed:
        failed = ", ".join(
            name for name, passed in quality["gates"].items() if not passed
        )
        raise ValueError(
            "export quality gates failed: " + failed
            + f"; diagnostics remain in {attempt_report}"
        )

    destination = paths["export_model"]
    report_dir = paths["export_report"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": run_manifest["run_id"],
        "model": str(destination),
        "candidate_count": read_json(context.artifact("candidates.manifest"))["candidate_count"],
        "config_hash": context.config.hash,
        "quality_gates": quality,
    }
    atomic_write_json(attempt_report / "run_summary.json", summary)
    summary_text = (
        f"# Pipeline Run {summary['run_id']}\n\n"
        f"- Model: `{destination}`\n"
        f"- Candidates: {summary['candidate_count']}\n"
        f"- Config SHA-256: `{context.config.hash}`\n"
        f"- Quality gates: {'passed' if quality['passed'] else 'failed'}\n"
        f"- Deployment qualified: {str(quality['deployment_qualified']).lower()}\n"
    )
    atomic_write_text(attempt_report / "run_summary.md", summary_text)

    previous_model = context.attempt_dir / "previous-model"
    previous_report = context.attempt_dir / "previous-report"
    for previous in (previous_model, previous_report):
        if previous.exists():
            shutil.rmtree(previous)
    if destination.exists():
        shutil.move(destination, previous_model)
    if report_dir.exists():
        shutil.move(report_dir, previous_report)
    try:
        os.replace(temporary, destination)
        os.replace(attempt_report, report_dir)
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        if report_dir.exists():
            shutil.rmtree(report_dir)
        if previous_model.exists():
            shutil.move(previous_model, destination)
        if previous_report.exists():
            shutil.move(previous_report, report_dir)
        raise
    return StageResult(
        artifacts=(
            ArtifactOutput("export.model", destination, "deployable_router/v1"),
            ArtifactOutput("export.report", report_dir, "pipeline_run_report/v1"),
            ArtifactOutput("export.summary", report_dir / "run_summary.json", "pipeline_run_summary/v1"),
            ArtifactOutput("export.quality_gates", report_dir / "quality_gates.json", "quality_gates/v1"),
            ArtifactOutput("export.lineage", report_dir / "artifact_lineage.json", "artifact_lineage/v1"),
        )
    )


def default_stage_specs() -> tuple[StageSpec, ...]:
    """Return the stable generic-candidate pipeline DAG."""

    return (
        StageSpec("ingest", "00_ingest", (), (), _ingest, "validate and freeze candidates"),
        StageSpec("enrich", "01_enrich", ("ingest",), ("candidates.catalog",), _enrich, "profile candidate capabilities"),
        StageSpec("plan-queries", "02_plan_queries", ("enrich",), ("data.profiles",), _plan_queries, "plan multi-Skill workflows"),
        StageSpec("generate-queries", "03_generate_queries", ("plan-queries",), ("data.profiles", "data.workflows"), _generate_queries, "generate alignment and retrieval queries"),
        StageSpec("review-queries", "04_review_queries", ("generate-queries",), ("data.profiles", "data.workflows", "data.queries.generated", "data.queries.alignment.generated"), _review_queries, "review queries and backfill coverage"),
        StageSpec("finalize-dataset", "05_finalize_dataset", ("review-queries",), ("candidates.catalog", "data.profiles", "data.workflows.reviewed", "data.queries.reviewed", "data.reviews", "data.queries.alignment.reviewed", "data.reviews.alignment"), _finalize_dataset, "export ordered qrels and prepare embeddings"),
        StageSpec("train-codebook", "06_train_codebook", ("finalize-dataset",), ("candidates.manifest", "processed.manifest", "embeddings.manifest"), _train_codebook, "plan and train the hierarchical codebook"),
        StageSpec("assign-codes", "07_assign_codes", ("train-codebook",), ("code.plan", "codebook.best"), _assign_codes, "assign codes and run quality gates"),
        StageSpec("build-sft", "08_build_sft", ("assign-codes",), ("dataset.directory", "codes.train", "codes.registry", "codes.virtual_tokens"), _build_sft, "build target-only Router SFT data"),
        StageSpec("train-memorization", "09_train_memorization", ("build-sft",), ("sft.directory", "codes.virtual_tokens"), _train_memorization, "train Skill document memorization"),
        StageSpec("train-alignment", "10_train_alignment", ("train-memorization",), ("model.memorization", "sft.directory"), _train_alignment, "train single-Skill alignment"),
        StageSpec("train-retrieval", "11_train_retrieval", ("train-alignment",), ("model.memorization", "model.alignment", "sft.directory"), _train_retrieval, "train multi-Skill retrieval"),
        StageSpec("evaluate", "12_evaluate", ("train-retrieval",), ("model.retrieval", "codes.registry", "codes.virtual_tokens"), _evaluate, "run constrained closed-set evaluation"),
        StageSpec("export", "13_export", ("evaluate",), ("model.retrieval", "evaluation.directory", "codes.registry", "codes.virtual_tokens"), _export, "export a self-contained deployment model"),
    )
