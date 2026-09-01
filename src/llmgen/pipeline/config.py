"""Versioned, strict configuration for generic candidate pipeline runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .io import canonical_json, sha256_json
from .resources import ResourceResolutionError, validate_runtime_device_request


class PipelineConfigError(ValueError):
    """Raised when a pipeline configuration is ambiguous or invalid."""


_TOP_LEVEL_KEYS = {
    "schema_version",
    "run",
    "input",
    "providers",
    "data_generation",
    "code",
    "router",
    "runtime",
    "checkpointing",
    "evaluation",
    "export",
    "logging",
}

_SECTION_KEYS: dict[str, set[str]] = {
    "run": {"name", "output_dir", "seed"},
    "input": {
        "candidates",
        "id_policy",
        "preserve_metadata",
        "single_candidate_policy",
    },
    "data_generation": {
        "alignment_queries_per_skill",
        "retrieval_positives_per_skill",
        "skills_per_query",
        "explicit_variants",
        "implicit_variants",
        "order_variants",
        "max_backfill_rounds",
        "alignment_backfill_rounds",
        "final_alignment_backfill_rounds",
        "coverage_oversample_factor",
        "workflows_per_skill",
        "profile_batch_size",
        "query_batch_size",
        "review_batch_size",
        "alignment_batch_size",
        "validation_retry_rounds",
        "min_completion_rate",
        "min_augmented_train_queries",
        "manual_alignment_path",
        "split",
    },
    "code": {
        "mode",
        "latency_priority",
        "num_levels",
        "branching_factors",
        "spare_capacity_ratio",
        "max_virtual_tokens",
        "max_branching_factor",
        "assignment",
        "assignment_exact_group_size",
        "max_collision_rate",
        "max_raw_collision_rate",
        "max_bucket_size",
        "min_level_utilization",
        "min_normalized_entropy",
        "min_raw_level_utilization",
        "min_raw_normalized_entropy",
        "rq_layers",
        "embedding_dim",
        "sk_epsilons",
        "beta",
        "epochs",
        "batch_size",
        "learning_rate",
        "scheduler",
        "warmup_ratio",
        "eval_every",
        "graph_lambda",
        "amp_dtype",
        "version",
        "export_batch_size",
    },
    "runtime": {
        "python",
        "device",
        "devices",
        "num_devices",
        "distributed",
        "deepspeed",
        "dataloader_workers",
        "environment",
    },
    "checkpointing": {
        "llm_batch_records",
        "embedding_batch_records",
        "training_save_steps",
        "training_eval_steps",
        "keep_last",
    },
    "evaluation": {
        "protocol",
        "query_split",
        "cutoffs",
        "top_k",
        "max_code_paths",
        "batch_size",
        "dtype",
        "require_format_valid_rate",
        "require_candidate_coverage",
        "metric_thresholds",
    },
    "export": {
        "output_dir",
        "require_all_gates",
        "smoke_test",
        "allow_failed_gates",
    },
    "logging": {
        "console_level",
        "file_level",
        "marker",
        "progress_interval_seconds",
        "capture_subprocess",
        "save_llm_requests",
        "save_llm_responses",
        "console_text_preview",
        "file_text_preview_chars",
    },
}

_PROVIDER_KEYS = {
    "type",
    "base_url",
    "api_key_env",
    "api_config",
    "model",
    "concurrency",
    "workers",
    "timeout_seconds",
    "max_retries",
    "batch_size",
    "dimensions",
    "max_batch_chars",
    "max_skill_chars",
}

_ROUTER_KEYS = {
    "base_model",
    "finetune_mode",
    "precision",
    "max_length",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "weight_decay",
    "warmup_ratio",
    "logging_steps",
    "gradient_checkpointing",
    "gradient_checkpointing_mode",
    "trust_remote_code",
    "validation_fraction",
    "data_seed",
    "lora",
    "memorization",
    "alignment",
    "retrieval",
}

_ROUTER_PHASE_KEYS = {
    "enabled",
    "epochs",
    "learning_rate",
    "alignment_replay_fraction",
    "memorization_replay_fraction",
}

_ROUTER_LORA_KEYS = {
    "r",
    "alpha",
    "dropout",
    "target_modules",
    "modules_to_save",
}

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SCIENTIFIC_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$"
)

_MANAGED_RUNTIME_ENVIRONMENT_NAMES = {
    "PYTHON",
    "SKIP_DOWNLOAD",
    "PREPARE_SCRIPT",
    "DATASET_NAME",
    "RUN_DIR",
    "DATASET_DIR",
    "PROCESSED_DIR",
    "STAGE1_DIR",
    "INDEX_DIR",
    "DEVICE",
    "NUM_LEVELS",
    "BRANCHING_FACTORS",
    "SK_EPSILONS",
    "RQ_LAYERS",
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
}
_MANAGED_RUNTIME_ENVIRONMENT_PREFIXES = (
    "EMBEDDING_",
    "LLMGEN_",
    "TOKENIZER_",
    "CODE_",
    "ROUTER_",
    "EVAL_",
)


_ROUTER_SHARED_PATHS = (
    "router.base_model",
    "router.finetune_mode",
    "router.precision",
    "router.max_length",
    "router.per_device_train_batch_size",
    "router.per_device_eval_batch_size",
    "router.gradient_accumulation_steps",
    "router.weight_decay",
    "router.warmup_ratio",
    "router.logging_steps",
    "router.gradient_checkpointing",
    "router.gradient_checkpointing_mode",
    "router.trust_remote_code",
    "router.lora",
)


STAGE_CONFIG_PATHS: dict[str, tuple[str, ...]] = {
    # Candidate bytes are tracked separately as an external input hash. The
    # filesystem path itself must not invalidate an otherwise identical fork.
    "ingest": (
        "schema_version",
        "run.seed",
        "input.id_policy",
        "input.preserve_metadata",
        "input.single_candidate_policy",
    ),
    "enrich": (
        "schema_version",
        "run.seed",
        "runtime.python",
        "providers.generation",
        "data_generation.profile_batch_size",
        "checkpointing.llm_batch_records",
    ),
    "plan-queries": (
        "schema_version",
        "run.seed",
        "runtime.python",
        "data_generation.workflows_per_skill",
        "data_generation.skills_per_query",
    ),
    "generate-queries": (
        "schema_version",
        "run.seed",
        "runtime.python",
        "providers.generation",
        "data_generation.alignment_queries_per_skill",
        "data_generation.explicit_variants",
        "data_generation.implicit_variants",
        "data_generation.query_batch_size",
        "data_generation.alignment_batch_size",
        "data_generation.validation_retry_rounds",
        "data_generation.min_completion_rate",
        "checkpointing.llm_batch_records",
    ),
    "review-queries": (
        "schema_version",
        "run.seed",
        "runtime.python",
        "providers.generation",
        "providers.review",
        "data_generation.alignment_queries_per_skill",
        "data_generation.retrieval_positives_per_skill",
        "data_generation.skills_per_query",
        "data_generation.explicit_variants",
        "data_generation.implicit_variants",
        "data_generation.max_backfill_rounds",
        "data_generation.alignment_backfill_rounds",
        "data_generation.final_alignment_backfill_rounds",
        "data_generation.coverage_oversample_factor",
        "data_generation.query_batch_size",
        "data_generation.review_batch_size",
        "data_generation.alignment_batch_size",
        "data_generation.validation_retry_rounds",
        "data_generation.min_completion_rate",
        "data_generation.manual_alignment_path",
        "data_generation.split",
        "checkpointing.llm_batch_records",
    ),
    "finalize-dataset": (
        "schema_version",
        "run.seed",
        "runtime.python",
        "runtime.environment",
        "providers.embedding",
        "data_generation.alignment_queries_per_skill",
        "data_generation.retrieval_positives_per_skill",
        "data_generation.order_variants",
        "data_generation.min_augmented_train_queries",
        "data_generation.split",
        "checkpointing.embedding_batch_records",
    ),
    "train-codebook": (
        "schema_version",
        "run.seed",
        "code",
        "runtime.python",
        "runtime.device",
        "runtime.devices",
        "runtime.num_devices",
        "runtime.environment",
        "checkpointing",
    ),
    "assign-codes": (
        "schema_version",
        "run.seed",
        "code",
        "runtime.python",
        "runtime.device",
        "runtime.devices",
        "runtime.num_devices",
        "runtime.environment",
    ),
    "build-sft": (
        "schema_version",
        "run.seed",
        "runtime.python",
        "runtime.environment",
        "router.validation_fraction",
        "router.data_seed",
        "router.alignment.enabled",
        "router.alignment.epochs",
        "router.retrieval.alignment_replay_fraction",
    ),
    "train-memorization": (
        "schema_version",
        "run.seed",
        *_ROUTER_SHARED_PATHS,
        "router.memorization",
        "runtime",
        "checkpointing",
    ),
    "train-alignment": (
        "schema_version",
        "run.seed",
        *_ROUTER_SHARED_PATHS,
        "router.alignment",
        "runtime",
        "checkpointing",
    ),
    "train-retrieval": (
        "schema_version",
        "run.seed",
        *_ROUTER_SHARED_PATHS,
        "router.alignment.enabled",
        "router.alignment.epochs",
        "router.retrieval",
        "runtime",
        "checkpointing",
    ),
    "evaluate": (
        "schema_version",
        "run.seed",
        "evaluation.protocol",
        "evaluation.query_split",
        "evaluation.cutoffs",
        "evaluation.top_k",
        "evaluation.max_code_paths",
        "evaluation.batch_size",
        "evaluation.dtype",
        "evaluation.require_format_valid_rate",
        "evaluation.metric_thresholds",
        "runtime.python",
        "runtime.device",
        "runtime.devices",
        "runtime.num_devices",
        "runtime.environment",
        "router.base_model",
        "router.finetune_mode",
        "router.trust_remote_code",
    ),
    "export": (
        "schema_version",
        "export",
        "evaluation",
        "runtime.python",
        "runtime.device",
        "runtime.devices",
        "runtime.num_devices",
        "runtime.environment",
        "router.base_model",
        "router.finetune_mode",
        "router.retrieval.alignment_replay_fraction",
        "router.retrieval.memorization_replay_fraction",
        "router.trust_remote_code",
    ),
}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{path} must be a mapping")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(map(str, value)).difference(allowed))
    if unknown:
        raise PipelineConfigError(
            f"unknown configuration key at {path}: {unknown[0]}"
        )


def _expand_environment(value: Any, environment: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, fallback = match.groups()
            resolved = environment.get(name)
            if resolved is None or resolved == "":
                if fallback is None:
                    raise PipelineConfigError(
                        f"configuration references unset environment variable {name}"
                    )
                return fallback
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item, environment) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _expand_environment(item, environment)
            for key, item in value.items()
        }
    return value


def _reject_embedded_provider_secrets(
    data: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    """Prevent referenced provider credentials from entering persisted config."""

    providers = data.get("providers")
    if not isinstance(providers, Mapping):
        return
    secrets = {
        environment[name]
        for provider in providers.values()
        if isinstance(provider, Mapping)
        for name in [str(provider.get("api_key_env") or "")]
        if name and environment.get(name)
    }
    if not secrets:
        return

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(value, str):
            return
        if any(
            value == secret or (len(secret) >= 8 and secret in value)
            for secret in secrets
        ):
            raise PipelineConfigError(
                f"configuration value at {path} contains a provider credential; "
                "reference it only through providers.*.api_key_env"
            )

    visit(data, "")


def _deep_get(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise PipelineConfigError(
                f"configuration does not define {dotted_path}"
            )
        current = current[component]
    return current


def _deep_set(value: dict[str, Any], dotted_path: str, item: Any) -> None:
    components = dotted_path.split(".")
    current = value
    for component in components[:-1]:
        child = current.get(component)
        if child is None:
            child = {}
            current[component] = child
        if not isinstance(child, dict):
            raise PipelineConfigError(
                f"cannot set {dotted_path}: {component} is not a mapping"
            )
        current = child
    current[components[-1]] = item


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise PipelineConfigError(f"override must use key=value: {raw!r}")
    key, text = raw.split("=", 1)
    key = key.strip()
    if not key or any(not part for part in key.split(".")):
        raise PipelineConfigError(f"invalid override key: {key!r}")
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - packaging regression
        raise PipelineConfigError("PyYAML is required to parse pipeline config") from error
    value = yaml.safe_load(text)
    # PyYAML follows YAML 1.1 and treats values such as ``1e-5`` as strings,
    # while scientific notation is the natural CLI spelling for learning
    # rates. Normalize this one unambiguous scalar form explicitly.
    if isinstance(value, str) and _SCIENTIFIC_NUMBER.fullmatch(text.strip()):
        value = float(text)
    return key, value


def _positive_int(value: Any, path: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineConfigError(f"{path} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise PipelineConfigError(f"{path} must be >= {minimum}")
    return value


def _probability(value: Any, path: str, *, include_one: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineConfigError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise PipelineConfigError(f"{path} must be finite")
    if number < 0 or number > 1 or (not include_one and number == 1):
        bound = "[0, 1]" if include_one else "[0, 1)"
        raise PipelineConfigError(f"{path} must be in {bound}")
    return number


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineConfigError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise PipelineConfigError(f"{path} must be finite")
    if strictly_positive and number <= 0:
        raise PipelineConfigError(f"{path} must be > 0")
    if minimum is not None and number < minimum:
        raise PipelineConfigError(f"{path} must be >= {minimum}")
    return number


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise PipelineConfigError(f"{path} must be a boolean")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigError(f"{path} must be a non-empty string")
    return value


def _validate(data: dict[str, Any]) -> None:
    _reject_unknown(data, _TOP_LEVEL_KEYS, "root")
    if data.get("schema_version") != 1:
        raise PipelineConfigError("schema_version must be 1")
    for section in _TOP_LEVEL_KEYS.difference({"schema_version", "providers", "router"}):
        mapping = _mapping(data.get(section), section)
        _reject_unknown(mapping, _SECTION_KEYS[section], section)
        data[section] = mapping

    providers = _mapping(data.get("providers"), "providers")
    _reject_unknown(providers, {"generation", "review", "embedding"}, "providers")
    for name in ("generation", "review", "embedding"):
        provider = _mapping(providers.get(name), f"providers.{name}")
        _reject_unknown(provider, _PROVIDER_KEYS, f"providers.{name}")
        if not str(provider.get("model") or "").strip():
            raise PipelineConfigError(f"providers.{name}.model must be non-empty")
        if provider.get("type") != "openai_compatible":
            raise PipelineConfigError(
                f"providers.{name}.type must be openai_compatible"
            )
        api_config = provider.get("api_config")
        if api_config is not None and not isinstance(api_config, str):
            raise PipelineConfigError(
                f"providers.{name}.api_config must be a string"
            )
        if str(api_config or "").strip():
            raise PipelineConfigError(
                f"providers.{name}.api_config is not accepted by the generic "
                "pipeline; configure base_url and api_key_env so credentials "
                "never enter a Run snapshot"
            )
        base_url = provider.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise PipelineConfigError(
                f"providers.{name}.base_url must be a string"
            )
        if not str(base_url or "").strip():
            raise PipelineConfigError(f"providers.{name}.base_url is required")
        api_key_env = provider.get("api_key_env")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise PipelineConfigError(
                f"providers.{name}.api_key_env must be a string"
            )
        if not str(api_key_env or "").strip():
            raise PipelineConfigError(
                f"providers.{name}.api_key_env is required"
            )
        for key in ("concurrency", "workers", "batch_size"):
            if provider.get(key) is not None:
                _positive_int(provider[key], f"providers.{name}.{key}")
        if (
            provider.get("concurrency") is not None
            and provider.get("workers") is not None
            and provider["concurrency"] != provider["workers"]
        ):
            raise PipelineConfigError(
                f"providers.{name}.concurrency and workers disagree"
            )
        if provider.get("timeout_seconds") is not None:
            _number(
                provider["timeout_seconds"],
                f"providers.{name}.timeout_seconds",
                strictly_positive=True,
            )
        if provider.get("max_retries") is not None:
            _positive_int(
                provider["max_retries"],
                f"providers.{name}.max_retries",
                allow_zero=True,
            )
        for key in ("dimensions", "max_batch_chars", "max_skill_chars"):
            if provider.get(key) is not None:
                _positive_int(provider[key], f"providers.{name}.{key}")
        if name in {"generation", "review"} and any(
            provider.get(key) is not None
            for key in ("batch_size", "dimensions", "max_batch_chars", "max_skill_chars")
        ):
            raise PipelineConfigError(
                f"providers.{name} batching belongs under data_generation"
            )
        if name == "embedding" and any(
            provider.get(key) is not None
            for key in ("concurrency", "workers")
        ):
            raise PipelineConfigError(
                "the embedding adapter is batch-sequential; configure only batch_size"
            )
        providers[name] = provider
    data["providers"] = providers

    router = _mapping(data.get("router"), "router")
    _reject_unknown(router, _ROUTER_KEYS, "router")
    if not str(router.get("base_model") or "").strip():
        raise PipelineConfigError("router.base_model must be non-empty")
    if router.get("finetune_mode") not in {"full", "lora"}:
        raise PipelineConfigError(
            "router.finetune_mode must be full or lora"
        )
    for name in ("memorization", "alignment", "retrieval"):
        phase = _mapping(router.get(name), f"router.{name}")
        _reject_unknown(phase, _ROUTER_PHASE_KEYS, f"router.{name}")
        _positive_int(phase.get("epochs"), f"router.{name}.epochs", allow_zero=True)
        _boolean(phase.get("enabled"), f"router.{name}.enabled")
        _number(
            phase.get("learning_rate"),
            f"router.{name}.learning_rate",
            strictly_positive=True,
        )
        for fraction in (
            "alignment_replay_fraction",
            "memorization_replay_fraction",
        ):
            _probability(
                phase.get(fraction),
                f"router.{name}.{fraction}",
                include_one=False,
            )
        router[name] = phase
    lora = _mapping(router.get("lora"), "router.lora")
    _reject_unknown(lora, _ROUTER_LORA_KEYS, "router.lora")
    router["lora"] = lora
    if not router["memorization"]["enabled"] or not router["retrieval"]["enabled"]:
        raise PipelineConfigError(
            "the current training adapter requires memorization and retrieval"
        )
    if router.get("precision") not in {"bf16", "fp16", "fp32"}:
        raise PipelineConfigError("router.precision must be bf16, fp16, or fp32")
    for key in (
        "max_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
    ):
        _positive_int(router.get(key), f"router.{key}")
    for key in ("weight_decay", "warmup_ratio", "validation_fraction"):
        _probability(router.get(key), f"router.{key}")
    _positive_int(router.get("data_seed"), "router.data_seed", allow_zero=True)
    _boolean(router.get("gradient_checkpointing"), "router.gradient_checkpointing")
    _boolean(router.get("trust_remote_code"), "router.trust_remote_code")
    if router.get("gradient_checkpointing_mode") not in {"auto", "reentrant", "non_reentrant"}:
        raise PipelineConfigError(
            "router.gradient_checkpointing_mode must be auto, reentrant, or non_reentrant"
        )
    for key in ("r", "alpha"):
        _positive_int(lora.get(key), f"router.lora.{key}")
    _probability(lora.get("dropout"), "router.lora.dropout")
    for key in ("target_modules", "modules_to_save"):
        _nonempty_string(lora.get(key), f"router.lora.{key}")
    data["router"] = router

    run = data["run"]
    if not str(run.get("name") or "").strip():
        raise PipelineConfigError("run.name must be non-empty")
    if not str(run.get("output_dir") or "").strip():
        raise PipelineConfigError("run.output_dir must be non-empty")
    _positive_int(run.get("seed"), "run.seed", allow_zero=True)

    input_config = data["input"]
    if not str(input_config.get("candidates") or "").strip():
        raise PipelineConfigError("input.candidates must be non-empty")
    if input_config.get("id_policy") not in {"explicit", "explicit_or_name"}:
        raise PipelineConfigError(
            "input.id_policy must be explicit or explicit_or_name"
        )
    if input_config.get("single_candidate_policy") not in {"error", "alignment_only"}:
        raise PipelineConfigError(
            "input.single_candidate_policy must be error or alignment_only"
        )
    _boolean(input_config.get("preserve_metadata"), "input.preserve_metadata")

    data_generation = data["data_generation"]
    for key in (
        "alignment_queries_per_skill",
        "retrieval_positives_per_skill",
        "explicit_variants",
        "implicit_variants",
        "order_variants",
        "max_backfill_rounds",
        "alignment_backfill_rounds",
        "final_alignment_backfill_rounds",
        "workflows_per_skill",
        "profile_batch_size",
        "query_batch_size",
        "review_batch_size",
        "alignment_batch_size",
        "validation_retry_rounds",
        "min_augmented_train_queries",
    ):
        _positive_int(
            data_generation.get(key),
            f"data_generation.{key}",
            allow_zero=key in {
                "implicit_variants",
                "max_backfill_rounds",
                "alignment_backfill_rounds",
                "final_alignment_backfill_rounds",
                "min_augmented_train_queries",
            },
        )
    if data_generation["explicit_variants"] < 2:
        raise PipelineConfigError("data_generation.explicit_variants must be >= 2")
    if data_generation["implicit_variants"] >= data_generation["explicit_variants"]:
        raise PipelineConfigError(
            "data_generation.implicit_variants must be below explicit_variants"
        )
    _number(
        data_generation.get("coverage_oversample_factor"),
        "data_generation.coverage_oversample_factor",
        strictly_positive=True,
    )
    _probability(
        data_generation.get("min_completion_rate"),
        "data_generation.min_completion_rate",
    )
    manual_alignment = data_generation.get("manual_alignment_path")
    if manual_alignment is not None and not isinstance(manual_alignment, str):
        raise PipelineConfigError(
            "data_generation.manual_alignment_path must be a string"
        )
    skills_per_query = _mapping(
        data_generation.get("skills_per_query"),
        "data_generation.skills_per_query",
    )
    _reject_unknown(skills_per_query, {"min", "max"}, "data_generation.skills_per_query")
    minimum = _positive_int(skills_per_query.get("min"), "skills_per_query.min")
    maximum = _positive_int(skills_per_query.get("max"), "skills_per_query.max")
    if minimum < 2 or minimum > maximum or maximum > 4:
        raise PipelineConfigError("skills_per_query must satisfy 2 <= min <= max <= 4")
    data_generation["skills_per_query"] = skills_per_query
    split = _mapping(data_generation.get("split"), "data_generation.split")
    _reject_unknown(split, {"train", "validation", "test"}, "data_generation.split")
    if abs(sum(float(split.get(name, 0)) for name in split) - 1.0) > 1e-9:
        raise PipelineConfigError("data_generation.split values must sum to 1")
    for name in ("train", "validation", "test"):
        _probability(split.get(name), f"data_generation.split.{name}")
    if float(split["train"]) <= 0:
        raise PipelineConfigError("data_generation.split.train must be positive")
    data_generation["split"] = split

    code = data["code"]
    if code.get("mode") not in {"auto", "manual"}:
        raise PipelineConfigError("code.mode must be auto or manual")
    _positive_int(code.get("max_virtual_tokens"), "code.max_virtual_tokens")
    _positive_int(code.get("max_branching_factor"), "code.max_branching_factor")
    if float(code.get("spare_capacity_ratio", 0)) < 1:
        raise PipelineConfigError("code.spare_capacity_ratio must be >= 1")
    if code.get("latency_priority") not in {"latency", "balanced", "vocabulary"}:
        raise PipelineConfigError(
            "code.latency_priority must be latency, balanced, or vocabulary"
        )
    if code.get("num_levels") is not None:
        _positive_int(code["num_levels"], "code.num_levels")
    branching_factors = code.get("branching_factors")
    if not isinstance(branching_factors, list):
        raise PipelineConfigError("code.branching_factors must be a list")
    if code.get("assignment") not in {"balanced_hierarchical", "nearest"}:
        raise PipelineConfigError(
            "code.assignment must be balanced_hierarchical or nearest"
        )
    for key in (
        "assignment_exact_group_size",
        "max_bucket_size",
        "embedding_dim",
        "epochs",
        "batch_size",
        "eval_every",
        "export_batch_size",
    ):
        _positive_int(code.get(key), f"code.{key}")
    for key in (
        "max_collision_rate",
        "max_raw_collision_rate",
        "min_level_utilization",
        "min_normalized_entropy",
        "min_raw_normalized_entropy",
        "warmup_ratio",
    ):
        _probability(code.get(key), f"code.{key}")
    for key in ("beta", "graph_lambda"):
        _number(code.get(key), f"code.{key}", minimum=0)
    _number(code.get("learning_rate"), "code.learning_rate", strictly_positive=True)
    for key in ("rq_layers", "sk_epsilons", "min_raw_level_utilization"):
        if not isinstance(code.get(key), list):
            raise PipelineConfigError(f"code.{key} must be a list")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in code["rq_layers"]
    ):
        raise PipelineConfigError("code.rq_layers must contain positive integers")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in code["sk_epsilons"]
    ):
        raise PipelineConfigError("code.sk_epsilons must contain numbers")
    if any(
        not 0 <= float(value) <= 1
        for value in code["min_raw_level_utilization"]
    ):
        raise PipelineConfigError(
            "code.min_raw_level_utilization values must be in [0, 1]"
        )
    if code.get("mode") == "manual":
        factors = branching_factors
        if not isinstance(factors, list) or not factors:
            raise PipelineConfigError(
                "manual code mode requires branching_factors"
            )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in factors):
            raise PipelineConfigError("branching_factors must be positive integers")
        if code.get("num_levels") not in {None, len(factors)}:
            raise PipelineConfigError(
                "code.num_levels must equal len(branching_factors)"
            )

    replay_alignment = _probability(
        router["retrieval"].get("alignment_replay_fraction"),
        "router.retrieval.alignment_replay_fraction",
        include_one=False,
    )
    replay_memorization = _probability(
        router["retrieval"].get("memorization_replay_fraction"),
        "router.retrieval.memorization_replay_fraction",
        include_one=False,
    )
    if replay_alignment + replay_memorization >= 1:
        raise PipelineConfigError("retrieval replay fractions must sum to less than 1")

    runtime = data["runtime"]
    if not isinstance(runtime.get("python"), str):
        raise PipelineConfigError("runtime.python must be a string")
    _nonempty_string(runtime.get("device"), "runtime.device")
    devices = runtime.get("devices")
    if not (
        devices == "auto"
        or isinstance(devices, list)
        or (isinstance(devices, str) and bool(devices.strip()))
    ):
        raise PipelineConfigError(
            "runtime.devices must be auto, a device list, or a comma-separated string"
        )
    if isinstance(devices, list) and not devices:
        raise PipelineConfigError("runtime.devices list must not be empty")
    num_devices = runtime.get("num_devices")
    if num_devices != "auto":
        _positive_int(num_devices, "runtime.num_devices")
    try:
        validate_runtime_device_request(runtime)
    except ResourceResolutionError as error:
        raise PipelineConfigError(str(error)) from error
    if runtime.get("distributed") != "auto":
        raise PipelineConfigError(
            "the current adapter supports only runtime.distributed=auto"
        )
    deepspeed = runtime.get("deepspeed")
    if not isinstance(deepspeed, str) or not deepspeed.strip():
        raise PipelineConfigError("runtime.deepspeed must be a non-empty string")
    _positive_int(
        runtime.get("dataloader_workers"),
        "runtime.dataloader_workers",
        allow_zero=True,
    )
    custom_environment = _mapping(runtime.get("environment"), "runtime.environment")
    secret_name = re.compile(
        r"(?i)(api.?key|token|secret|password|authorization|credential)"
    )
    unsafe = sorted(name for name in custom_environment if secret_name.search(name))
    if unsafe:
        raise PipelineConfigError(
            "runtime.environment may not contain secrets; reference provider "
            f"api_key_env instead: {unsafe[0]}"
        )
    if any(not isinstance(value, (str, int, float, bool)) for value in custom_environment.values()):
        raise PipelineConfigError(
            "runtime.environment values must be scalar"
        )
    managed = sorted(
        name
        for name in custom_environment
        if name in _MANAGED_RUNTIME_ENVIRONMENT_NAMES
        or name.startswith(_MANAGED_RUNTIME_ENVIRONMENT_PREFIXES)
    )
    if managed:
        raise PipelineConfigError(
            "runtime.environment cannot override pipeline-managed variable: "
            f"{managed[0]}"
        )
    runtime["environment"] = custom_environment

    checkpointing = data["checkpointing"]
    for key in (
        "llm_batch_records",
        "embedding_batch_records",
        "training_save_steps",
        "training_eval_steps",
        "keep_last",
    ):
        _positive_int(checkpointing.get(key), f"checkpointing.{key}")

    evaluation = data["evaluation"]
    if evaluation.get("protocol") != "closedset":
        raise PipelineConfigError(
            "the generic candidate adapter supports evaluation.protocol=closedset"
        )
    if evaluation.get("query_split") not in {
        "validation",
        "dataset-validation",
        "test",
        "train",
    }:
        raise PipelineConfigError("evaluation.query_split is invalid")
    cutoffs = evaluation.get("cutoffs")
    if not isinstance(cutoffs, list) or not cutoffs:
        raise PipelineConfigError("evaluation.cutoffs must be a non-empty list")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in cutoffs
    ):
        raise PipelineConfigError(
            "evaluation.cutoffs must contain positive integers"
        )
    for key in ("top_k", "max_code_paths", "batch_size"):
        _positive_int(evaluation.get(key), f"evaluation.{key}")
    _nonempty_string(evaluation.get("dtype"), "evaluation.dtype")
    _probability(
        evaluation.get("require_format_valid_rate"),
        "evaluation.require_format_valid_rate",
    )
    _probability(
        evaluation.get("require_candidate_coverage"),
        "evaluation.require_candidate_coverage",
    )
    metric_thresholds = _mapping(
        evaluation.get("metric_thresholds"),
        "evaluation.metric_thresholds",
    )
    for name, threshold in metric_thresholds.items():
        if not name.strip():
            raise PipelineConfigError(
                "evaluation.metric_thresholds keys must be non-empty"
            )
        _probability(
            threshold,
            f"evaluation.metric_thresholds.{name}",
        )
    evaluation["metric_thresholds"] = metric_thresholds

    export = data["export"]
    output_dir = Path(str(export.get("output_dir") or ""))
    if not str(output_dir) or output_dir.is_absolute() or ".." in output_dir.parts:
        raise PipelineConfigError(
            "export.output_dir must be a relative path inside the Run"
        )
    for key in ("require_all_gates", "smoke_test", "allow_failed_gates"):
        _boolean(export.get(key), f"export.{key}")

    logging = data["logging"]
    for key in ("console_level", "file_level"):
        if logging.get(key) not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise PipelineConfigError(
                f"logging.{key} must be DEBUG, INFO, WARNING, or ERROR"
            )
    _nonempty_string(logging.get("marker"), "logging.marker")
    _number(
        logging.get("progress_interval_seconds"),
        "logging.progress_interval_seconds",
        strictly_positive=True,
    )
    for key in (
        "capture_subprocess",
        "save_llm_requests",
        "save_llm_responses",
        "console_text_preview",
    ):
        _boolean(logging.get(key), f"logging.{key}")
    if not logging["capture_subprocess"]:
        raise PipelineConfigError(
            "the current adapter requires logging.capture_subprocess=true"
        )
    if not logging["save_llm_requests"] or not logging["save_llm_responses"]:
        raise PipelineConfigError(
            "the resumable generic pipeline requires private LLM request and "
            "response ledgers; logging.save_llm_requests and "
            "logging.save_llm_responses must both be true"
        )
    _positive_int(
        logging.get("file_text_preview_chars"),
        "logging.file_text_preview_chars",
        allow_zero=True,
    )


@dataclass(frozen=True)
class PipelineConfig:
    """An immutable-by-convention resolved pipeline configuration."""

    data: Mapping[str, Any]
    source_path: Path
    overrides: tuple[str, ...] = ()

    @property
    def hash(self) -> str:
        return sha256_json(self.data)

    def get(self, dotted_path: str, default: Any = None) -> Any:
        try:
            return deepcopy(_deep_get(self.data, dotted_path))
        except PipelineConfigError:
            return deepcopy(default)

    def require(self, dotted_path: str) -> Any:
        return deepcopy(_deep_get(self.data, dotted_path))

    def stage_view(self, stage: str) -> dict[str, Any]:
        paths = STAGE_CONFIG_PATHS.get(stage)
        if paths is None:
            raise PipelineConfigError(f"unknown stage: {stage}")
        return {path: self.require(path) for path in paths}

    def stage_hash(self, stage: str) -> str:
        return sha256_json(self.stage_view(stage))

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.data))

    def to_yaml(self) -> str:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover
            raise PipelineConfigError("PyYAML is required to write pipeline config") from error
        return yaml.safe_dump(
            self.to_dict(),
            allow_unicode=True,
            sort_keys=False,
        )


def load_pipeline_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    candidates: str | Path | None = None,
    output: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> PipelineConfig:
    """Load, expand, override, and strictly validate one YAML/JSON config."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise PipelineConfigError("PyYAML is required to parse pipeline config") from error
    source_text = source.read_text(encoding="utf-8")
    raw = (
        json.loads(source_text)
        if source.suffix.casefold() == ".json"
        else yaml.safe_load(source_text)
    )
    if not isinstance(raw, Mapping):
        raise PipelineConfigError("pipeline config must be a mapping")
    selected_environment = dict(os.environ if environment is None else environment)
    data = _expand_environment(
        {str(key): deepcopy(value) for key, value in raw.items()},
        selected_environment,
    )
    for override in overrides:
        key, value = _parse_override(override)
        _deep_set(data, key, value)
    if candidates is not None:
        _deep_set(data, "input.candidates", str(Path(candidates).expanduser()))
    if output is not None:
        _deep_set(data, "run.output_dir", str(Path(output).expanduser()))
    _reject_embedded_provider_secrets(data, selected_environment)
    _validate(data)
    # Ensure the value is JSON-compatible before it becomes a persisted Run
    # snapshot. This also rejects YAML-specific object constructors.
    try:
        normalized = json.loads(canonical_json(data))
    except (TypeError, ValueError) as error:
        raise PipelineConfigError("pipeline config must be JSON-compatible") from error
    return PipelineConfig(
        data=normalized,
        source_path=source,
        overrides=tuple(overrides),
    )
