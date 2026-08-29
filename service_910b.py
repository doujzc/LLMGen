#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-contained Ascend 910B deployment service for an LLMGen router.

The hosting contract intentionally matches the reference retrieval service:

* ``load()`` initializes the long-lived model runtime.
* ``calc({"data": {"query": ..., "top_k": ...}})`` returns a JSON string
  containing a list of Skill names.

The model directory must be a complete LLMGen Router bundle containing the
Hugging Face model/tokenizer files, ``skill_decode_map.json``,
``virtual_tokens.txt``, and ``router_manifest.json``.  Model startup and
generation mirror ``skillhub`` develop commit ``486ef18``: the reference 910B
flow uses the custom in-process ``AsyncLLMEngine``, explicitly loads the
model, starts its background loop, waits for engine health, and submits the
full custom asynchronous ``generate`` request.  Its model-specific defaults
are adapted for dense Qwen3.5-2B on one NPU while preserving that lifecycle
and request protocol.  Candidate decoding remains the LLMGen token-ID Trie
algorithm so this service has the same input/output semantics as
``service.py``.

For local contract tests, set ``MOCK_MODE=1``.  This skips model and candidate
artifact loading.  ``MOCK_RESPONSES_JSON`` may be either one result list used
for every query or an object mapping exact queries (plus optional ``"*"``
fallback) to result lists.

Startup and request diagnostics are emitted at ``INFO`` by default, including
request/load correlation IDs, resolved configuration sources, artifact probes,
stage timings, engine settings, token counts, constrained-decoding state,
async-loop lifecycle, and cleanup results.  Structured query and prompt fields
stay hidden unless
``SERVICE_910B_LOG_TEXT=1``; previews are bounded by
``SERVICE_910B_LOG_PREVIEW_CHARS``.  Per-token Trie traces can be toggled with
``SERVICE_910B_TRACE_TRIE=1``.  Prompt token IDs and marked traceback lines
require ``SERVICE_910B_LOG_TOKEN_IDS=1`` and
``SERVICE_910B_LOG_TRACEBACKS=1`` respectively.  Tracebacks are intended for
development because third-party exception messages can contain request text.
Service records are written directly to stdout and do not propagate through
the host/root logger.  Stdout backpressure therefore counts toward
request/startup time, especially with verbose per-token tracing enabled.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import gc
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import threading
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence
import uuid


_LOG_MARKER = "[[LLMGEN-910B]]"
_LOG_FILTER_TAG = "_llmgen_service_910b_marker_filter"
_STDOUT_HANDLER_TAG = "_llmgen_service_910b_stdout_handler"


class _ServiceLogMarker(logging.Filter):
    """Prefix every record so service logs remain grepable in shared output."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = str(record.msg)
        if not message.startswith(_LOG_MARKER):
            record.msg = f"{_LOG_MARKER} {message}"
        if isinstance(record.args, tuple):
            record.args = tuple(
                _safe_log_value(f"arg_{index}", value)
                for index, value in enumerate(record.args)
            )
        elif isinstance(record.args, Mapping):
            record.args = {
                _safe_log_string(key): _safe_log_value(key, value)
                for key, value in record.args.items()
            }
        if record.exc_info:
            if _log_flag("SERVICE_910B_LOG_TRACEBACKS", False):
                traceback_text = logging.Formatter().formatException(record.exc_info)
                record.exc_text = "\n".join(
                    f"{_LOG_MARKER} {line}"
                    for line in traceback_text.splitlines()
                )
            else:
                record.exc_text = None
                # Tracebacks are opt-in because third-party exception messages
                # may contain request text.  When disabled, prevent a downstream
                # formatter from regenerating one.  When enabled, retain
                # exc_info for structured JSON/Sentry-style handlers; standard
                # text formatters reuse the marked exc_text above.
                record.exc_info = None
        return True


logger = logging.getLogger("web_demo.service_910b")
for _existing_filter in tuple(logger.filters):
    if getattr(_existing_filter, _LOG_FILTER_TAG, False):
        logger.removeFilter(_existing_filter)
_service_log_marker = _ServiceLogMarker()
setattr(_service_log_marker, _LOG_FILTER_TAG, True)
logger.addFilter(_service_log_marker)


def _install_stdout_log_handler() -> None:
    """Route this service's records directly to stdout without propagation."""

    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
    stdout_handler = logging.StreamHandler(sys.stdout)
    setattr(stdout_handler, _STDOUT_HANDLER_TAG, True)
    stdout_handler.setLevel(logging.NOTSET)
    # Keep the grep marker at the start of every physical service-log line.
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stdout_handler)
    logger.propagate = False
    logger.disabled = False
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


_install_stdout_log_handler()

_VLLM_TENSOR_PARALLEL_SIZE = 1
_VLLM_DTYPE = "bfloat16"
_VLLM_HEALTH_CHECK_INTERVAL = 1.0
_VLLM_910B_ARCHITECTURE = "Qwen3_5ForConditionalGeneration_OnlyLLM"
_QWEN35_DENSE_BUNDLE_ARCHITECTURES = frozenset(
    {"Qwen3_5ForConditionalGeneration"}
)
_DEFAULT_QUERY = "查天气"
_DEFAULT_SYSTEM_PROMPT = (
    "Select every Agent Skill needed for the user request in execution order. "
    "Output one hierarchical skill code per line, with no other text."
)
_CODE_PATH_SEPARATOR = "\n"
_DECODE_MAP_FILENAME = "skill_decode_map.json"
_VIRTUAL_TOKENS_FILENAME = "virtual_tokens.txt"
_ROUTER_MANIFEST_FILENAME = "router_manifest.json"
_DECODE_MAP_SCHEMA_VERSION = 1
_DEFAULT_MOCK_RESPONSES = ("Mock Weather Skill", "Mock Map Skill")
_DEFAULT_LOG_PREVIEW_CHARS = 320
_DEFAULT_LOG_SEQUENCE_ITEMS = 16
_SENSITIVE_LOG_KEY_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)

# These defaults are intentionally kept in sync with the SkillHub 910B
# deployment flow.  Each value can be overridden by VLLM_<NAME> (or <NAME>)
# and VLLM_KWARGS_JSON, matching the reference service.
_VLLM_ENGINE_ARG_DEFAULTS: Dict[str, Any] = {
    "model_vision": "facebook/opt-125m",
    "architectures": _VLLM_910B_ARCHITECTURE,
    "tokenizer_mode": "auto",
    "trust_remote_code": True,
    "download_dir": None,
    "load_format": "auto",
    "seed": 0,
    "max_model_len": None,
    "rope_scaling_type": None,
    "rope_scaling_factor": 1.0,
    "pipeline_parallel_size": 1,
    "tensor_parallel_size": _VLLM_TENSOR_PARALLEL_SIZE,
    "data_parallel_size": 1,
    "context_parallel_size": 1,
    "pipeline_parallel_layer_partitions": "",
    "mla_wo_tensor_parallel_size": -1,
    "enable_expert_parallel": False,
    "decode_enable_expert_parallel": False,
    "decode_pipeline_parallel_size": 1,
    "decode_tensor_parallel_size": _VLLM_TENSOR_PARALLEL_SIZE,
    "decode_data_parallel_size": 1,
    "decode_context_parallel_size": 1,
    "block_size": 128,
    "kernel_block_size": 128,
    "prefix_sharing_chunk_size": 128,
    "scheduler_budget_len": 102400,
    "prefix_sharing_kwargs": {"gpu_usage_threshold": 0.7},
    "enable_datasystem": True,
    "multipath_devices": "",
    "swap_space": 0,
    "gpu_memory_utilization": 0.9,
    "max_num_batched_tokens": None,
    "max_num_seqs": 8,
    "disable_log_stats": False,
    "revision": None,
    "tokenizer_revision": None,
    "quantization": None,
    "block_sliding_window": None,
    "sink_block_num": 0,
    "schedule_policy": "fcfs",
    "schedule_policy_kwargs": None,
    "first_token_timeout": 300.0,
    "max_swapped_req_num": 128,
    "sys_prefix_prompts": None,
    "ops_dev_mode": None,
    "speculate_type": None,
    "speculate_kwargs": None,
    "disaggregate_prefill_decoding": False,
    "dispd_args": None,
    "ranks": None,
    "engine_name": "",
    "sparse_mode": "",
    "sparse_threshold_len": 4096,
    "sparse_minimum_len": 2048,
    "sparse_budget_len": 4096,
    "sparse_compress_ratio": 0.5,
    "cluster_window_size": 32,
    "cluster_sink_size": 64,
    "cluster_recent_size": 128,
    "cluster_kernel_size": 9,
    "cluster_block_size": 64,
    "inf_prefix_len": 64,
    "inf_query_len": 32,
    "inf_window_size": 1024,
    "inf_overlap_size": 32,
    "turbo_share_sysprefix": False,
    "turbo_sysprefix_num": 0,
    "turbo_separator_set": None,
    "speculative_config": None,
    "enable_chunked_prefill": True,
    "enable_batching_prefill": False,
    "enable_fuse_prefill_and_decode": False,
    "enable_lookahead_scheduling": False,
    "need_kv_transfer": False,
    "prefill_group_num": 1,
    "decode_group_num": 1,
    "global_group_meta": None,
    "stage_id": None,
    "head_candidate_role_set": None,
    "need_bypass_balancer": False,
    "group_name": "",
    "dllm_blockwise_type": None,
    "dllm_blockwise_kwargs": None,
    "dense_prefetch_config": None,
    "tokenizer_group_mode": "process",
    "tokenizer_group_workers": 4,
    "disable_log_requests": True,
    "max_log_len": None,
    "new_requests_que_size": 128,
    "finished_requests_que_size": 1024,
    "detokenizer_group_mode": None,
    "detokenizer_group_workers": 1,
}


class ServiceConfigurationError(RuntimeError):
    """Raised when deployment artifacts or environment settings disagree."""


def _configure_service_logging() -> None:
    """Apply the service-local log level without touching root logging."""

    raw_level = os.environ.get("SERVICE_910B_LOG_LEVEL", "").strip()
    if raw_level:
        if raw_level.isdigit():
            level: Any = int(raw_level)
        else:
            level = getattr(logging, raw_level.upper(), None)
        if not isinstance(level, int):
            logger.warning(
                "event=logging.invalid_level value=%r; retaining configured level",
                raw_level,
            )
        else:
            logger.setLevel(level)
    logger.info(
        "event=logging.configured logger=%s effective_level=%s text_enabled=%s "
        "token_ids_enabled=%s tracebacks_enabled=%s trie_trace_enabled=%s "
        "preview_chars=%s sequence_items=%s marker_enabled=true",
        logger.name,
        logging.getLevelName(logger.getEffectiveLevel()),
        _log_flag("SERVICE_910B_LOG_TEXT", False),
        _log_flag("SERVICE_910B_LOG_TOKEN_IDS", False),
        _log_flag("SERVICE_910B_LOG_TRACEBACKS", False),
        _log_flag("SERVICE_910B_TRACE_TRIE", False),
        _log_preview_chars(),
        _log_sequence_items(),
    )


def _log_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _log_positive_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        logger.warning(
            "event=logging.invalid_integer name=%s value=%r default=%s",
            name,
            value,
            default,
        )
        return default


def _log_preview_chars() -> int:
    return _log_positive_int(
        "SERVICE_910B_LOG_PREVIEW_CHARS",
        _DEFAULT_LOG_PREVIEW_CHARS,
        minimum=32,
    )


def _log_sequence_items() -> int:
    return _log_positive_int(
        "SERVICE_910B_LOG_SEQUENCE_ITEMS",
        _DEFAULT_LOG_SEQUENCE_ITEMS,
    )


def _configured_environment_source(
    names: Sequence[str], *, fallback: str
) -> str:
    """Return the first non-empty environment variable controlling a value."""

    for name in names:
        if os.environ.get(name, "").strip():
            return name
    return fallback


def _environment_log_value(name: str, raw_value: str) -> object:
    """Render a deployment environment value without exposing nested secrets."""

    if not raw_value:
        return "<unset>"
    if name.endswith("_JSON"):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return f"<invalid-json chars={len(raw_value)}>"
        return _safe_log_value(name, parsed)
    return _safe_log_value(name, raw_value)


def _model_sfs_log_details(raw_value: str | None) -> tuple[str, object]:
    """Describe MODEL_SFS parsing without logging its complete JSON payload."""

    if raw_value is None or not raw_value.strip():
        return "unset", "<unset>"
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        return "invalid_json", f"<unavailable chars={len(raw_value)}>"
    if not isinstance(payload, Mapping):
        return "not_object", "<unavailable>"
    base_path = payload.get("sfsBasePath")
    if not isinstance(base_path, str) or not base_path.strip():
        return "missing_sfsBasePath", "<unset>"
    return "valid", _safe_log_value("sfsBasePath", base_path)


def _startup_runtime_environment() -> dict[str, str]:
    """Return relevant runtime variables for verbose startup diagnostics."""

    explicit_names = {
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "HCCL_CONNECT_TIMEOUT",
        "HCCL_EXEC_TIMEOUT",
        "NPU_VISIBLE_DEVICES",
    }
    return {
        name: value
        for name, value in os.environ.items()
        if name.startswith("VLLM_") or name in explicit_names
    }


def _text_log_value(value: object) -> str:
    text = str(value)
    if not _log_flag("SERVICE_910B_LOG_TEXT", False):
        return f"<hidden chars={len(text)}>"
    escaped = text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")
    limit = _log_preview_chars()
    if len(escaped) <= limit:
        return escaped
    return f"{escaped[:limit]}...<truncated chars={len(escaped) - limit}>"


def _sequence_log_value(values: Sequence[Any] | Iterable[Any]) -> object:
    limit = _log_sequence_items()
    if isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        count = len(values)
        if count <= limit:
            return list(values)
        head_count = max(1, limit // 2)
        tail_count = max(1, limit - head_count)
        return {
            "count": count,
            "head": list(values[:head_count]),
            "tail": list(values[-tail_count:]),
        }
    sequence = list(values)
    if len(sequence) <= limit:
        return sequence
    head_count = max(1, limit // 2)
    tail_count = max(1, limit - head_count)
    return {
        "count": len(sequence),
        "head": sequence[:head_count],
        "tail": sequence[-tail_count:],
    }


def _token_ids_log_value(values: Sequence[Any] | Iterable[Any]) -> object:
    if isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        count: int | None = len(values)
    else:
        count = None
    if not _log_flag("SERVICE_910B_LOG_TOKEN_IDS", False):
        return f"<hidden count={count if count is not None else 'unknown'}>"
    return _sequence_log_value(values)


def _safe_log_string(value: Any) -> str:
    try:
        return str(value)
    except Exception as exc:
        return f"<unprintable type={_type_name(value)} error={type(exc).__name__}>"


def _safe_log_value(name: object, value: Any) -> Any:
    return _safe_log_value_inner(name, value, depth=0, seen=set())


def _safe_log_value_inner(
    name: object,
    value: Any,
    *,
    depth: int,
    seen: set[int],
) -> Any:
    try:
        normalized_name = _safe_log_string(name).lower().replace("-", "_")
        components = {part for part in normalized_name.split("_") if part}
        sensitive = (
            bool(components.intersection(_SENSITIVE_LOG_KEY_PARTS))
            or {"api", "key"} <= components
            or "authorization" in normalized_name
            or (
                not _log_flag("SERVICE_910B_LOG_TEXT", False)
                and bool(components.intersection({"prompt", "prompts", "query"}))
            )
        )
        if sensitive:
            return "<redacted>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if depth >= 4:
            return f"<max-depth type={_type_name(value)}>"

        is_container = isinstance(
            value, (Mapping, list, tuple, set, frozenset)
        )
        object_id = id(value)
        if is_container:
            if object_id in seen:
                return f"<cycle type={_type_name(value)}>"
            seen.add(object_id)
        try:
            if isinstance(value, Mapping):
                result: dict[str, Any] = {}
                limit = _log_sequence_items_quiet()
                for index, (key, nested_value) in enumerate(value.items()):
                    if index >= limit:
                        result["<truncated>"] = f"remaining={len(value) - limit}"
                        break
                    safe_key = _safe_log_string(key)
                    result[safe_key] = _safe_log_value_inner(
                        safe_key,
                        nested_value,
                        depth=depth + 1,
                        seen=seen,
                    )
                return result
            if isinstance(value, (list, tuple, set, frozenset)):
                sequence = list(value)
                limit = _log_sequence_items_quiet()
                sampled = sequence[:limit]
                result = [
                    _safe_log_value_inner(
                        name,
                        item,
                        depth=depth + 1,
                        seen=seen,
                    )
                    for item in sampled
                ]
                if len(sequence) > limit:
                    result.append(f"<truncated remaining={len(sequence) - limit}>")
                return result
            text = _safe_log_string(value)
            sanitized = (
                text.replace("\\", "\\\\")
                .replace("\r", "\\r")
                .replace("\n", "\\n")
            )
            limit = _log_preview_chars_quiet()
            if len(sanitized) > limit:
                return f"{sanitized[:limit]}...<truncated>"
            return sanitized
        finally:
            if is_container:
                seen.discard(object_id)
    except Exception as exc:
        return f"<log-sanitize-error error={type(exc).__name__}>"


def _log_preview_chars_quiet() -> int:
    value = os.environ.get("SERVICE_910B_LOG_PREVIEW_CHARS")
    if value is None or not value.strip():
        return _DEFAULT_LOG_PREVIEW_CHARS
    try:
        return max(32, int(value))
    except ValueError:
        return _DEFAULT_LOG_PREVIEW_CHARS


def _log_sequence_items_quiet() -> int:
    """Read the log sampling limit without recursively logging from a filter."""

    value = os.environ.get("SERVICE_910B_LOG_SEQUENCE_ITEMS")
    if value is None or not value.strip():
        return _DEFAULT_LOG_SEQUENCE_ITEMS
    try:
        return max(1, int(value))
    except ValueError:
        return _DEFAULT_LOG_SEQUENCE_ITEMS


def _safe_getattr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception as exc:
        return f"<attribute-error name={name} error={type(exc).__name__}>"


def _directory_entries_log_value(path: Path) -> object:
    """Return a bounded, best-effort directory sample for DEBUG diagnostics."""

    limit = _log_sequence_items_quiet()
    entries: list[str] = []
    try:
        for index, entry in enumerate(path.iterdir()):
            if index >= limit:
                entries.append("<truncated>")
                break
            entries.append(entry.name)
    except OSError as exc:
        return f"<diagnostic-error {type(exc).__name__}>"
    return sorted(entries)


def _engine_log_summary(options: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic_keys = (
        "architectures",
        "dtype",
        "load_format",
        "trust_remote_code",
        "max_model_len",
        "pipeline_parallel_size",
        "tensor_parallel_size",
        "data_parallel_size",
        "context_parallel_size",
        "decode_pipeline_parallel_size",
        "decode_tensor_parallel_size",
        "decode_data_parallel_size",
        "decode_context_parallel_size",
        "enable_expert_parallel",
        "decode_enable_expert_parallel",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "block_size",
        "scheduler_budget_len",
        "prefix_sharing_type",
        "prefix_sharing_kwargs",
        "enable_datasystem",
        "enable_chunked_prefill",
        "enable_batching_prefill",
        "disable_log_stats",
        "disable_log_requests",
        "tokenizer_group_mode",
        "tokenizer_group_workers",
        "health_check_interval",
        "health_check_timeout",
        "request_model",
    )
    return {
        key: _safe_log_value(key, options[key])
        for key in diagnostic_keys
        if key in options
    }


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


class MultiPathTokenTrie:
    """Grammar for ``path (separator path)* EOS`` constrained decoding."""

    def __init__(
        self,
        paths: Iterable[Sequence[int]],
        *,
        eos_token_id: int,
        separator_token_ids: Sequence[int],
        max_paths: int,
    ) -> None:
        normalized = {tuple(int(value) for value in path) for path in paths}
        if not normalized:
            raise ServiceConfigurationError("candidate state contains no active paths")
        path_lengths = {len(path) for path in normalized}
        if len(path_lengths) != 1 or next(iter(path_lengths)) < 1:
            raise ServiceConfigurationError(
                "all active code paths must have one fixed positive length"
            )
        if any(value < 0 for path in normalized for value in path):
            raise ServiceConfigurationError("code token IDs must be non-negative")
        separator = tuple(int(value) for value in separator_token_ids)
        if not separator or any(value < 0 for value in separator):
            raise ServiceConfigurationError(
                "the code-path separator must encode to non-negative token IDs"
            )
        if int(eos_token_id) in separator:
            raise ServiceConfigurationError("the code-path separator contains EOS")
        if max_paths < 1:
            raise ServiceConfigurationError("MAX_CODE_PATHS must be positive")

        self.paths = frozenset(normalized)
        self.num_levels = next(iter(path_lengths))
        self.eos_token_id = int(eos_token_id)
        self.separator_token_ids = separator
        self.max_paths = min(int(max_paths), len(self.paths))
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=trie.initialized paths=%s levels=%s requested_max_paths=%s "
                "effective_max_paths=%s eos_token_id=%s separator_token_ids=%s "
                "sample_paths=%s",
                len(self.paths),
                self.num_levels,
                max_paths,
                self.max_paths,
                _token_ids_log_value([self.eos_token_id]),
                _token_ids_log_value(self.separator_token_ids),
                _token_ids_log_value(sorted(self.paths)),
            )

    def _state(
        self, generated: Sequence[int]
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], str, int] | None:
        completed: list[tuple[int, ...]] = []
        prefix: list[int] = []
        mode = "path"
        separator_offset = 0
        for raw_token_id in generated:
            token_id = int(raw_token_id)
            if token_id == self.eos_token_id:
                return None
            if mode == "path":
                prefix.append(token_id)
                current_prefix = tuple(prefix)
                if not any(
                    path not in completed
                    and path[: len(current_prefix)] == current_prefix
                    for path in self.paths
                ):
                    return None
                if len(prefix) == self.num_levels:
                    completed.append(current_prefix)
                    prefix = []
                    mode = "boundary"
            elif mode == "boundary":
                if token_id != self.separator_token_ids[0]:
                    return None
                if len(self.separator_token_ids) == 1:
                    mode = "path"
                else:
                    mode = "separator"
                    separator_offset = 1
            else:
                if token_id != self.separator_token_ids[separator_offset]:
                    return None
                separator_offset += 1
                if separator_offset == len(self.separator_token_ids):
                    mode = "path"
        return tuple(completed), tuple(prefix), mode, separator_offset

    def allowed_next(self, generated: Sequence[int]) -> tuple[int, ...]:
        state = self._state(generated)
        if state is None:
            return ()
        completed, prefix, mode, separator_offset = state
        if mode == "separator":
            return (self.separator_token_ids[separator_offset],)
        if mode == "boundary":
            if len(completed) >= self.max_paths or len(completed) >= len(self.paths):
                return (self.eos_token_id,)
            return tuple(sorted({self.eos_token_id, self.separator_token_ids[0]}))

        candidates = [
            path
            for path in self.paths
            if path not in completed and path[: len(prefix)] == prefix
        ]
        if not candidates or len(prefix) >= self.num_levels:
            return ()
        return tuple(sorted({path[len(prefix)] for path in candidates}))

    def parse_complete(
        self,
        generated: Sequence[int],
        *,
        request_id: str | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=trie.parse.begin request_id=%s generated_count=%s generated=%s",
                request_id,
                len(generated),
                _token_ids_log_value(generated),
            )
        state = self._state(generated)
        if state is None:
            logger.error(
                "event=trie.parse.invalid request_id=%s generated_count=%s generated=%s",
                request_id,
                len(generated),
                _token_ids_log_value(generated),
            )
            raise RuntimeError("vLLM generated an invalid code sequence")
        completed, prefix, mode, _ = state
        if (
            not completed
            or len(completed) > self.max_paths
            or prefix
            or mode != "boundary"
        ):
            logger.error(
                "event=trie.parse.incomplete request_id=%s completed=%s prefix=%s "
                "mode=%s max_paths=%s",
                request_id,
                _token_ids_log_value(completed),
                _token_ids_log_value(prefix),
                mode,
                self.max_paths,
            )
            raise RuntimeError("vLLM generation did not end at a code-path boundary")
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=trie.parse.complete request_id=%s path_count=%s paths=%s",
                request_id,
                len(completed),
                _token_ids_log_value(completed),
            )
        return completed


class TrieLogitsProcessor:
    """Request-level logits processor for active LLMGen Skill paths."""

    def __init__(self, trie: MultiPathTokenTrie) -> None:
        self.trie = trie

    def clone(self) -> "TrieLogitsProcessor":
        # The processor is stateless; generated IDs are supplied by vLLM.
        return self

    def __call__(self, output_token_ids: list[int], scores: Any) -> Any:
        allowed = self.trie.allowed_next(output_token_ids)
        if logger.isEnabledFor(logging.INFO) and _log_flag(
            "SERVICE_910B_TRACE_TRIE", False
        ):
            logger.info(
                "event=trie.mask.step generated_count=%s generated=%s allowed_count=%s "
                "allowed=%s score_shape=%s",
                len(output_token_ids),
                _token_ids_log_value(output_token_ids),
                len(allowed),
                _token_ids_log_value(allowed),
                getattr(scores, "shape", None),
            )
        if not allowed:
            logger.error(
                "event=trie.mask.invalid_prefix generated_count=%s generated=%s",
                len(output_token_ids),
                _token_ids_log_value(output_token_ids),
            )
            raise RuntimeError(
                f"generation reached an invalid code prefix: {output_token_ids!r}"
            )
        vocabulary_size = int(scores.shape[-1])
        if any(token_id >= vocabulary_size for token_id in allowed):
            logger.error(
                "event=trie.mask.out_of_vocabulary vocabulary_size=%s allowed=%s",
                vocabulary_size,
                _token_ids_log_value(allowed),
            )
            raise RuntimeError("candidate path contains a token outside model vocabulary")
        masked = scores.new_full(scores.shape, -float("inf"))
        indices = list(allowed)
        masked[indices] = scores[indices]
        return masked


@dataclass(frozen=True, slots=True)
class CandidateBundle:
    decode_map: dict[str, Any]
    virtual_tokens: tuple[str, ...]
    skills: dict[str, dict[str, Any]]
    token_paths: dict[tuple[str, ...], tuple[str, ...]]
    num_levels: int


@dataclass(frozen=True, slots=True)
class RouterManifestSettings:
    max_length: int | None
    system_prompt: str


def _load_candidate_bundle(directory: Path) -> CandidateBundle:
    started = perf_counter()
    decode_path = directory / _DECODE_MAP_FILENAME
    token_path = directory / _VIRTUAL_TOKENS_FILENAME
    logger.info(
        "event=candidate.load.begin directory=%s decode_path=%s token_path=%s",
        directory,
        decode_path,
        token_path,
    )
    if not decode_path.is_file() or not token_path.is_file():
        raise ServiceConfigurationError(
            f"candidate state {directory} must contain {_DECODE_MAP_FILENAME} "
            f"and {_VIRTUAL_TOKENS_FILENAME}"
        )
    decode_map = _read_json_object(decode_path)
    file_tokens_list: list[str] = []
    seen_file_tokens: set[str] = set()
    for line_number, line in enumerate(
        token_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        token = line.strip()
        if not token:
            continue
        if any(character.isspace() for character in token):
            raise ServiceConfigurationError(
                f"virtual token at line {line_number} contains whitespace"
            )
        if token in seen_file_tokens:
            raise ServiceConfigurationError(f"duplicate virtual token: {token!r}")
        file_tokens_list.append(token)
        seen_file_tokens.add(token)
    file_tokens = tuple(file_tokens_list)
    if not file_tokens:
        raise ServiceConfigurationError("virtual_tokens.txt must be non-empty")

    if decode_map.get("schema_version") != _DECODE_MAP_SCHEMA_VERSION:
        raise ServiceConfigurationError("unsupported skill decode map schema")
    map_tokens = decode_map.get("virtual_tokens")
    if (
        not isinstance(map_tokens, list)
        or any(not isinstance(token, str) for token in map_tokens)
        or tuple(map_tokens) != file_tokens
    ):
        raise ServiceConfigurationError(
            "virtual_tokens.txt disagrees with skill_decode_map.json"
        )
    raw_levels = decode_map.get("num_levels")
    if not isinstance(raw_levels, int) or isinstance(raw_levels, bool) or raw_levels < 1:
        raise ServiceConfigurationError("decode map has an invalid num_levels")
    skills = decode_map.get("skills")
    skill_to_code = decode_map.get("skill_to_code")
    raw_paths = decode_map.get("paths")
    if not isinstance(skills, dict) or not skills:
        raise ServiceConfigurationError("decode map contains no Skills")
    if not isinstance(skill_to_code, dict) or set(skill_to_code) != set(skills):
        raise ServiceConfigurationError("decode map Skill metadata and codes disagree")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ServiceConfigurationError("decode map contains no active paths")
    for skill_id, metadata in skills.items():
        if not skill_id or not isinstance(metadata, Mapping):
            raise ServiceConfigurationError(
                "decode map Skill metadata must contain non-empty object entries"
            )

    token_namespace = set(file_tokens)
    token_paths: dict[tuple[str, ...], tuple[str, ...]] = {}
    reconstructed: dict[str, tuple[str, ...]] = {}
    for path_number, raw_path in enumerate(raw_paths, start=1):
        if not isinstance(raw_path, Mapping):
            raise ServiceConfigurationError(
                f"decode map path {path_number} is not an object"
            )
        tokens = raw_path.get("tokens")
        members = raw_path.get("skill_ids")
        if not isinstance(tokens, list) or len(tokens) != raw_levels:
            raise ServiceConfigurationError(
                f"decode map path {path_number} has the wrong code length"
            )
        if any(not isinstance(token, str) or not token for token in tokens):
            raise ServiceConfigurationError(
                f"decode map path {path_number} contains an invalid token"
            )
        path = tuple(tokens)
        if not set(path) <= token_namespace:
            raise ServiceConfigurationError(
                f"decode map path {path_number} uses an unknown virtual token"
            )
        if not isinstance(members, list) or not members:
            raise ServiceConfigurationError(
                f"decode map path {path_number} contains no Skills"
            )
        if any(not isinstance(skill_id, str) or not skill_id for skill_id in members):
            raise ServiceConfigurationError(
                f"decode map path {path_number} contains an invalid Skill ID"
            )
        if len(set(members)) != len(members):
            raise ServiceConfigurationError(
                f"decode map path {path_number} contains duplicate Skill IDs"
            )
        member_ids = tuple(sorted(members))
        if path in token_paths:
            raise ServiceConfigurationError("decode map contains a duplicate code path")
        token_paths[path] = member_ids
        for skill_id in member_ids:
            if skill_id not in skills:
                raise ServiceConfigurationError(
                    f"decode map path references unknown Skill {skill_id!r}"
                )
            if skill_id in reconstructed:
                raise ServiceConfigurationError(
                    f"Skill {skill_id!r} appears in multiple code paths"
                )
            reconstructed[skill_id] = path

    if set(reconstructed) != set(skills):
        raise ServiceConfigurationError("decode map paths do not cover every Skill")
    for skill_id, path in reconstructed.items():
        details = skill_to_code[skill_id]
        if not isinstance(details, Mapping) or tuple(details.get("tokens", ())) != path:
            raise ServiceConfigurationError(
                f"decode map has inconsistent code metadata for {skill_id!r}"
            )
    if decode_map.get("num_skills", len(skills)) != len(skills):
        raise ServiceConfigurationError("decode map num_skills is inconsistent")
    if decode_map.get("num_paths", len(token_paths)) != len(token_paths):
        raise ServiceConfigurationError("decode map num_paths is inconsistent")

    supervision = decode_map.get("supervision")
    if isinstance(supervision, Mapping) and supervision.get("phase") == "retrieval":
        target_counts = supervision.get("target_counts")
        if not isinstance(target_counts, Mapping) or set(target_counts) != set(skills):
            raise ServiceConfigurationError(
                "retrieval decode map does not cover one complete candidate set"
            )
        try:
            uncovered = [
                skill_id
                for skill_id, count in target_counts.items()
                if int(count) < 1
            ]
        except (TypeError, ValueError) as exc:
            raise ServiceConfigurationError(
                "retrieval decode map has invalid supervision counts"
            ) from exc
        if uncovered:
            raise ServiceConfigurationError(
                "retrieval decode map contains Skills without train positives"
            )

    normalized_skills = {
        str(skill_id): dict(metadata) for skill_id, metadata in skills.items()
    }
    bundle = CandidateBundle(
        decode_map=decode_map,
        virtual_tokens=file_tokens,
        skills=normalized_skills,
        token_paths=token_paths,
        num_levels=raw_levels,
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=candidate.load.complete elapsed_ms=%.3f schema_version=%s skills=%s "
            "paths=%s levels=%s virtual_tokens=%s supervision_phase=%s "
            "sample_path_member_counts=%s",
            (perf_counter() - started) * 1000.0,
            decode_map.get("schema_version"),
            len(bundle.skills),
            len(bundle.token_paths),
            bundle.num_levels,
            len(bundle.virtual_tokens),
            supervision.get("phase") if isinstance(supervision, Mapping) else None,
            _token_ids_log_value(
                [(path, len(members)) for path, members in bundle.token_paths.items()]
            ),
        )
    return bundle


class _AsyncLoopRunner:
    """Run the custom vLLM async engine on its dedicated SkillHub-style loop."""

    def __init__(self) -> None:
        self.runner_id = uuid.uuid4().hex[:12]
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run,
            name="local-vllm-910b-async-loop",
            daemon=True,
        )
        self._closed = False
        logger.info(
            "event=async_loop.create runner_id=%s loop_id=%s thread_name=%s",
            self.runner_id,
            id(self._loop),
            self._thread.name,
        )
        self._thread.start()
        logger.info(
            "event=async_loop.thread_started runner_id=%s thread_alive=%s "
            "thread_ident=%s",
            self.runner_id,
            self._thread.is_alive(),
            self._thread.ident,
        )

    def submit(
        self,
        coroutine: Any,
        *,
        timeout: float | None = None,
        operation: str = "coroutine",
        request_id: str | None = None,
    ) -> Any:
        if self._closed:
            close_coroutine = getattr(coroutine, "close", None)
            if callable(close_coroutine):
                close_coroutine()
            logger.error(
                "event=async_loop.submit_rejected runner_id=%s operation=%s "
                "request_id=%s reason=closed",
                self.runner_id,
                operation,
                request_id,
            )
            raise RuntimeError("custom vLLM async loop is closed")
        started = perf_counter()
        logger.info(
            "event=async_loop.submit runner_id=%s operation=%s request_id=%s "
            "timeout_seconds=%s loop_running=%s thread_alive=%s",
            self.runner_id,
            operation,
            request_id,
            timeout,
            self._loop.is_running(),
            self._thread.is_alive(),
        )
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            result = future.result(timeout=timeout)
            logger.info(
                "event=async_loop.submit_complete runner_id=%s operation=%s "
                "request_id=%s elapsed_ms=%.3f result_type=%s future_done=%s",
                self.runner_id,
                operation,
                request_id,
                (perf_counter() - started) * 1000.0,
                _type_name(result),
                future.done(),
            )
            return result
        except FutureTimeoutError as exc:
            # On Python 3.11+, concurrent.futures.TimeoutError aliases the
            # built-in TimeoutError.  A completed coroutine can therefore
            # raise the same type itself; only classify this as a submit wait
            # timeout when a finite wait expired and the future is unfinished.
            if timeout is None or future.done():
                logger.info(
                    "event=async_loop.submit_failed runner_id=%s operation=%s "
                    "request_id=%s elapsed_ms=%.3f error_type=%s "
                    "future_done=%s timeout_seconds=%s",
                    self.runner_id,
                    operation,
                    request_id,
                    (perf_counter() - started) * 1000.0,
                    type(exc).__name__,
                    future.done(),
                    timeout,
                    exc_info=True,
                )
                raise
            cancel_requested = future.cancel()
            logger.warning(
                "event=async_loop.submit_timeout runner_id=%s operation=%s "
                "request_id=%s elapsed_ms=%.3f timeout_seconds=%s "
                "cancel_requested=%s future_done=%s",
                self.runner_id,
                operation,
                request_id,
                (perf_counter() - started) * 1000.0,
                timeout,
                cancel_requested,
                future.done(),
            )
            raise
        except Exception as exc:
            logger.info(
                "event=async_loop.submit_failed runner_id=%s operation=%s "
                "request_id=%s elapsed_ms=%.3f error_type=%s",
                self.runner_id,
                operation,
                request_id,
                (perf_counter() - started) * 1000.0,
                type(exc).__name__,
                exc_info=True,
            )
            raise

    def close(self) -> None:
        if self._closed:
            logger.info(
                "event=async_loop.close_skipped runner_id=%s reason=already_closed",
                self.runner_id,
            )
            return
        started = perf_counter()
        logger.info(
            "event=async_loop.close_begin runner_id=%s loop_running=%s "
            "loop_closed=%s thread_alive=%s thread_ident=%s",
            self.runner_id,
            self._loop.is_running(),
            self._loop.is_closed(),
            self._thread.is_alive(),
            self._thread.ident,
        )
        self._closed = True
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.warning(
                "event=async_loop.close_timeout runner_id=%s elapsed_ms=%.3f "
                "thread_ident=%s",
                self.runner_id,
                (perf_counter() - started) * 1000.0,
                self._thread.ident,
            )
        else:
            logger.info(
                "event=async_loop.close_complete runner_id=%s elapsed_ms=%.3f "
                "loop_closed=%s",
                self.runner_id,
                (perf_counter() - started) * 1000.0,
                self._loop.is_closed(),
            )

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        logger.info(
            "event=async_loop.run_begin runner_id=%s thread_ident=%s",
            self.runner_id,
            threading.get_ident(),
        )
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        logger.info(
            "event=async_loop.run_stopping runner_id=%s pending_tasks=%s",
            self.runner_id,
            len(pending),
        )
        for task in pending:
            task.cancel()
        if pending:
            results = self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
            logger.info(
                "event=async_loop.pending_cancelled runner_id=%s pending_tasks=%s "
                "exception_results=%s",
                self.runner_id,
                len(pending),
                sum(isinstance(result, BaseException) for result in results),
            )
        self._loop.close()
        logger.info(
            "event=async_loop.run_complete runner_id=%s thread_ident=%s",
            self.runner_id,
            threading.get_ident(),
        )


class LocalVLLM910BRuntime:
    """Owned in-process custom-vLLM runtime using SkillHub's 910B API."""

    def __init__(
        self,
        *,
        engine: Any,
        tokenizer: Any,
        sampling_params_cls: Any,
        loop_runner: _AsyncLoopRunner,
        request_timeout: float | None,
        engine_kwargs: Mapping[str, Any],
    ) -> None:
        self.runtime_id = uuid.uuid4().hex[:12]
        self.engine = engine
        self.tokenizer = tokenizer
        self.sampling_params_cls = sampling_params_cls
        self.loop_runner = loop_runner
        self.request_timeout = request_timeout
        self.engine_kwargs = dict(engine_kwargs)
        self._close_lock = threading.Lock()
        self._closed = False
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=runtime.created runtime_id=%s runner_id=%s engine_type=%s "
                "tokenizer_type=%s sampling_params_type=%s request_timeout=%s "
                "engine_summary=%s",
                self.runtime_id,
                self.loop_runner.runner_id,
                _type_name(self.engine),
                _type_name(self.tokenizer),
                _safe_getattr(
                    self.sampling_params_cls,
                    "__qualname__",
                    _type_name(self.sampling_params_cls),
                ),
                self.request_timeout,
                _engine_log_summary(self.engine_kwargs),
            )

    def generate(
        self,
        *,
        prompt: str,
        prompt_token_ids: Sequence[int],
        sampling_params: Any,
        request_id: str | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("custom vLLM 910B runtime is closed")
        resolved_request_id = request_id or str(uuid.uuid4())
        started = perf_counter()
        if logger.isEnabledFor(logging.INFO):
            processors = _safe_getattr(sampling_params, "logits_processors", ())
            try:
                processor_count: Any = len(processors or ())
            except Exception as exc:
                processor_count = f"<length-error {type(exc).__name__}>"
            logger.info(
                "event=runtime.generate_begin runtime_id=%s request_id=%s "
                "prompt_chars=%s prompt_tokens=%s timeout_seconds=%s "
                "sampling_max_tokens=%s sampling_min_tokens=%s processor_count=%s",
                self.runtime_id,
                resolved_request_id,
                len(prompt),
                len(prompt_token_ids),
                self.request_timeout,
                _safe_getattr(sampling_params, "max_tokens"),
                _safe_getattr(sampling_params, "min_tokens"),
                processor_count,
            )
        try:
            result = self.loop_runner.submit(
                _generate_on_910b_loop(
                    engine=self.engine,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    sampling_params=sampling_params,
                    request_id=resolved_request_id,
                ),
                timeout=self.request_timeout,
                operation="generate",
                request_id=resolved_request_id,
            )
        except Exception as exc:
            logger.info(
                "event=runtime.generate_failed runtime_id=%s request_id=%s "
                "elapsed_ms=%.3f error_type=%s",
                self.runtime_id,
                resolved_request_id,
                (perf_counter() - started) * 1000.0,
                type(exc).__name__,
                exc_info=True,
            )
            raise
        logger.info(
            "event=runtime.generate_complete runtime_id=%s request_id=%s "
            "elapsed_ms=%.3f output_type=%s",
            self.runtime_id,
            resolved_request_id,
            (perf_counter() - started) * 1000.0,
            _type_name(result),
        )
        return result

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                logger.info(
                    "event=runtime.close_skipped runtime_id=%s reason=already_closed",
                    self.runtime_id,
                )
                return
            started = perf_counter()
            logger.info(
                "event=runtime.close_begin runtime_id=%s runner_id=%s "
                "engine_type=%s",
                self.runtime_id,
                self.loop_runner.runner_id,
                _type_name(self.engine),
            )
            try:
                operations = (
                    (
                        "shutdown_background_loop",
                        getattr(self.engine, "shutdown_background_loop", None),
                    ),
                    (
                        "llm_engine.shutdown",
                        getattr(
                            getattr(self.engine, "llm_engine", None),
                            "shutdown",
                            None,
                        ),
                    ),
                )
                for label, operation in operations:
                    if not callable(operation):
                        logger.info(
                            "event=runtime.close_operation_skipped runtime_id=%s "
                            "operation=%s reason=not_callable",
                            self.runtime_id,
                            label,
                        )
                        continue
                    operation_started = perf_counter()
                    try:
                        operation()
                    except Exception:
                        logger.exception(
                            "event=runtime.close_operation_failed runtime_id=%s "
                            "operation=%s",
                            self.runtime_id,
                            label,
                        )
                    else:
                        logger.info(
                            "event=runtime.close_operation_complete runtime_id=%s "
                            "operation=%s elapsed_ms=%.3f",
                            self.runtime_id,
                            label,
                            (perf_counter() - operation_started) * 1000.0,
                        )
            finally:
                self.loop_runner.close()
                self._closed = True
                logger.info(
                    "event=runtime.close_complete runtime_id=%s elapsed_ms=%.3f "
                    "loop_closed=%s",
                    self.runtime_id,
                    (perf_counter() - started) * 1000.0,
                    self.loop_runner._loop.is_closed(),
                )


class RetriverTest:
    """Long-lived LLMGen retrieval service backed by custom 910B vLLM."""

    def __init__(self) -> None:
        _configure_service_logging()
        self.instance_id = uuid.uuid4().hex[:12]
        self.llm: Any | None = None
        self.tokenizer: Any | None = None
        self.sampling_params: Any | None = None
        self.bundle: CandidateBundle | None = None
        self.trie: MultiPathTokenTrie | None = None
        self.path_skill_ids: dict[tuple[int, ...], tuple[str, ...]] = {}
        self.model_path = ""
        self.tokenizer_path = ""
        self.skill_index_path = ""
        # Retain the misspelled attribute used by the reference service.
        self.skill_indes_path = ""
        self.served_model_name = ""
        self.backend = "vllm_910b"
        self.mock_responses: dict[str, tuple[str, ...]] = {}
        self.default_top_k = 2
        self.max_code_paths = 2
        self.max_input_length = 1
        self.system_prompt = _DEFAULT_SYSTEM_PROMPT
        self._load_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._loaded = False
        logger.info(
            "event=service.created instance_id=%s backend=%s thread_name=%s "
            "thread_ident=%s",
            self.instance_id,
            self.backend,
            threading.current_thread().name,
            threading.get_ident(),
        )

    def load(self) -> None:
        _configure_service_logging()
        load_id = uuid.uuid4().hex[:12]
        lock_wait_started = perf_counter()
        logger.info(
            "event=service.load_lock_wait instance_id=%s load_id=%s loaded=%s "
            "backend=%s thread_name=%s thread_ident=%s",
            self.instance_id,
            load_id,
            self._loaded,
            self.backend,
            threading.current_thread().name,
            threading.get_ident(),
        )
        with self._load_lock:
            logger.info(
                "event=service.load_lock_acquired instance_id=%s load_id=%s "
                "wait_ms=%.3f",
                self.instance_id,
                load_id,
                (perf_counter() - lock_wait_started) * 1000.0,
            )
            if self._loaded:
                logger.info(
                    "event=service.load_skipped instance_id=%s load_id=%s "
                    "reason=already_loaded backend=%s",
                    self.instance_id,
                    load_id,
                    self.backend,
                )
                return

            started = perf_counter()
            stage = "configuration"
            logger.info(
                "event=service.load_begin instance_id=%s load_id=%s backend=%s",
                self.instance_id,
                load_id,
                self.backend,
            )
            current_dir = Path(__file__).resolve().parent
            try:
                working_directory: object = Path.cwd()
            except OSError as exc:
                working_directory = f"<unavailable error={type(exc).__name__}>"
            logger.info(
                "event=service.process_context instance_id=%s load_id=%s pid=%s "
                "ppid=%s python_version=%s python_executable=%s platform=%s "
                "service_file=%s service_dir=%s working_directory=%s",
                self.instance_id,
                load_id,
                os.getpid(),
                os.getppid(),
                sys.version.split()[0],
                sys.executable,
                sys.platform,
                Path(__file__).resolve(),
                current_dir,
                working_directory,
            )

            model_object_id = os.environ.get("MODEL_OBJECT_ID", "")
            model_sfs = os.environ.get("MODEL_SFS")
            model_sfs_status, model_sfs_base_path = _model_sfs_log_details(
                model_sfs
            )
            if model_object_id.strip() or (model_sfs and model_sfs.strip()):
                model_source = (
                    "MODEL_SFS+MODEL_OBJECT_ID"
                    if model_object_id.strip() and model_sfs and model_sfs.strip()
                    else "incomplete_MODEL_SFS_configuration"
                )
            else:
                model_source = _configured_environment_source(
                    ("MODEL_PATH", "MODEL_DIR"), fallback="service_parent/model"
                )
            tokenizer_source = _configured_environment_source(
                ("TOKENIZER_PATH",), fallback="model"
            )
            candidate_source = _configured_environment_source(
                ("CANDIDATE_STATE_PATH", "SKILL_INDEX_PATH"), fallback="model"
            )
            logger.info(
                "event=service.deployment_environment instance_id=%s load_id=%s "
                "model_source=%s tokenizer_source=%s candidate_source=%s "
                "model_object_id=%s model_sfs_status=%s model_sfs_base_path=%s "
                "model_path_override=%s model_dir_override=%s "
                "tokenizer_path_override=%s candidate_state_path_override=%s "
                "skill_index_path_override=%s",
                self.instance_id,
                load_id,
                model_source,
                tokenizer_source,
                candidate_source,
                model_object_id or "<unset>",
                model_sfs_status,
                model_sfs_base_path,
                os.environ.get("MODEL_PATH", "") or "<unset>",
                os.environ.get("MODEL_DIR", "") or "<unset>",
                os.environ.get("TOKENIZER_PATH", "") or "<unset>",
                os.environ.get("CANDIDATE_STATE_PATH", "") or "<unset>",
                os.environ.get("SKILL_INDEX_PATH", "") or "<unset>",
            )
            runtime_environment = _startup_runtime_environment()
            if not runtime_environment:
                logger.info(
                    "event=service.runtime_environment instance_id=%s load_id=%s "
                    "configured_count=0",
                    self.instance_id,
                    load_id,
                )
            else:
                logger.info(
                    "event=service.runtime_environment instance_id=%s load_id=%s "
                    "configured_count=%s names=%s",
                    self.instance_id,
                    load_id,
                    len(runtime_environment),
                    sorted(runtime_environment),
                )
                for environment_name in sorted(runtime_environment):
                    logger.info(
                        "event=service.runtime_environment_value instance_id=%s "
                        "load_id=%s name=%s value=%s",
                        self.instance_id,
                        load_id,
                        environment_name,
                        _environment_log_value(
                            environment_name,
                            runtime_environment[environment_name],
                        ),
                    )

            self.default_top_k = _env_int("TOP_K", 2)
            self.max_code_paths = _env_int(
                "MAX_CODE_PATHS", max(1, self.default_top_k)
            )
            if self.default_top_k < 1:
                raise ServiceConfigurationError("TOP_K must be positive")
            if self.max_code_paths < 1:
                raise ServiceConfigurationError("MAX_CODE_PATHS must be positive")
            mock_mode = _env_bool("MOCK_MODE", False)
            logger.info(
                "event=service.load_configuration instance_id=%s load_id=%s "
                "current_dir=%s top_k=%s max_code_paths=%s mock_mode=%s "
                "served_model_override=%s system_prompt_override=%s "
                "candidate_override=%s tokenizer_override=%s",
                self.instance_id,
                load_id,
                current_dir,
                self.default_top_k,
                self.max_code_paths,
                mock_mode,
                bool(os.environ.get("SERVED_MODEL_NAME", "").strip()),
                bool(os.environ.get("SYSTEM_PROMPT", "").strip()),
                bool(
                    os.environ.get("CANDIDATE_STATE_PATH", "").strip()
                    or os.environ.get("SKILL_INDEX_PATH", "").strip()
                ),
                bool(os.environ.get("TOKENIZER_PATH", "").strip()),
            )

            if mock_mode:
                stage = "mock_configuration"
                self.backend = "mock"
                self.mock_responses = _load_mock_responses()
                self.model_path = _env_first_text(
                    ("MODEL_PATH", "MODEL_DIR"), "<mock>"
                )
                self.tokenizer_path = _env_text("TOKENIZER_PATH", "<mock>")
                self.skill_index_path = _env_first_text(
                    ("CANDIDATE_STATE_PATH", "SKILL_INDEX_PATH"), "<mock>"
                )
                self.skill_indes_path = self.skill_index_path
                self.served_model_name = _env_text(
                    "SERVED_MODEL_NAME", "llmgen-router-mock"
                )
                self.system_prompt = _env_text(
                    "SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT
                )
                self._loaded = True
                logger.info(
                    "event=service.load_complete instance_id=%s load_id=%s "
                    "backend=mock elapsed_ms=%.3f queries=%s top_k=%s",
                    self.instance_id,
                    load_id,
                    (perf_counter() - started) * 1000.0,
                    len(self.mock_responses),
                    self.default_top_k,
                )
                logger.info(
                    "event=service.mock_configuration instance_id=%s load_id=%s "
                    "served_model=%s response_queries=%s has_fallback=%s "
                    "system_prompt=%s",
                    self.instance_id,
                    load_id,
                    self.served_model_name,
                    len(self.mock_responses),
                    "*" in self.mock_responses,
                    _text_log_value(self.system_prompt),
                )
                return

            stage = "resolve_paths"
            resolve_started = perf_counter()
            model_dir = _resolve_model_directory(current_dir)
            tokenizer_dir = Path(
                _env_first_text(("TOKENIZER_PATH",), str(model_dir))
            ).expanduser().resolve()
            candidate_dir = Path(
                _env_first_text(
                    ("CANDIDATE_STATE_PATH", "SKILL_INDEX_PATH"), str(model_dir)
                )
            ).expanduser().resolve()

            self.model_path = str(model_dir)
            self.tokenizer_path = str(tokenizer_dir)
            self.skill_index_path = str(candidate_dir)
            self.skill_indes_path = self.skill_index_path
            self.served_model_name = _env_text(
                "SERVED_MODEL_NAME", model_dir.name or str(model_dir)
            )
            logger.info(
                "event=service.paths_resolved instance_id=%s load_id=%s "
                "elapsed_ms=%.3f model_source=%s tokenizer_source=%s "
                "candidate_source=%s model=%s tokenizer=%s candidate=%s "
                "served_model=%s",
                self.instance_id,
                load_id,
                (perf_counter() - resolve_started) * 1000.0,
                model_source,
                tokenizer_source,
                candidate_source,
                model_dir,
                tokenizer_dir,
                candidate_dir,
                self.served_model_name,
            )
            path_checks = (
                ("model", model_dir),
                ("tokenizer", tokenizer_dir),
                ("candidate state", candidate_dir),
            )
            invalid_paths: list[tuple[str, Path]] = []
            for label, path in path_checks:
                exists = path.exists()
                is_directory = path.is_dir()
                logger.info(
                    "event=service.path_check instance_id=%s load_id=%s label=%s "
                    "path=%s exists=%s is_directory=%s",
                    self.instance_id,
                    load_id,
                    label,
                    path,
                    exists,
                    is_directory,
                )
                if not is_directory:
                    invalid_paths.append((label, path))
            _log_startup_artifact_probes(
                instance_id=self.instance_id,
                load_id=load_id,
                model_dir=model_dir,
                tokenizer_dir=tokenizer_dir,
                candidate_dir=candidate_dir,
            )
            if invalid_paths:
                for label, path in invalid_paths:
                    logger.error(
                        "event=service.path_invalid instance_id=%s load_id=%s "
                        "label=%s path=%s",
                        self.instance_id,
                        load_id,
                        label,
                        path,
                    )
                first_label, first_path = invalid_paths[0]
                raise ServiceConfigurationError(
                    f"{first_label} directory does not exist: {first_path}"
                )
            stage = "validate_model_bundle"
            validation_started = perf_counter()
            _validate_full_model_bundle(model_dir)
            logger.info(
                "event=service.model_bundle_validated instance_id=%s load_id=%s "
                "elapsed_ms=%.3f",
                self.instance_id,
                load_id,
                (perf_counter() - validation_started) * 1000.0,
            )

            try:
                stage = "load_candidate_bundle"
                self.bundle = _load_candidate_bundle(candidate_dir)
                logger.info(
                    "event=service.candidate_ready instance_id=%s load_id=%s "
                    "skills=%s paths=%s levels=%s virtual_tokens=%s",
                    self.instance_id,
                    load_id,
                    len(self.bundle.skills),
                    len(self.bundle.token_paths),
                    self.bundle.num_levels,
                    len(self.bundle.virtual_tokens),
                )
                stage = "load_router_manifest"
                manifest_settings = _load_router_settings(model_dir)
                self.system_prompt = _env_text(
                    "SYSTEM_PROMPT", manifest_settings.system_prompt
                )
                logger.info(
                    "event=service.manifest_ready instance_id=%s load_id=%s "
                    "trained_max_length=%s system_prompt_chars=%s "
                    "system_prompt=%s override=%s",
                    self.instance_id,
                    load_id,
                    manifest_settings.max_length,
                    len(self.system_prompt),
                    _text_log_value(self.system_prompt),
                    bool(os.environ.get("SYSTEM_PROMPT", "").strip()),
                )

                stage = "build_engine_options"
                engine_options = _build_vllm_kwargs(model_path=model_dir)
                # Match SkillHub's VLLMClientConfig adapter: the service-level
                # dtype and request model are applied after generic JSON args.
                engine_options["request_model"] = self.served_model_name
                engine_options["dtype"] = _vllm_dtype()
                logger.info(
                    "event=service.engine_load_begin instance_id=%s load_id=%s "
                    "served_model=%s architecture=%s dtype=%s tp=%s pp=%s dp=%s",
                    self.instance_id,
                    load_id,
                    self.served_model_name,
                    engine_options.get("architectures"),
                    engine_options.get("dtype"),
                    engine_options.get("tensor_parallel_size"),
                    engine_options.get("pipeline_parallel_size"),
                    engine_options.get("data_parallel_size"),
                )
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "event=service.engine_options instance_id=%s load_id=%s "
                        "option_count=%s option_keys=%s summary=%s",
                        self.instance_id,
                        load_id,
                        len(engine_options),
                        sorted(engine_options),
                        _engine_log_summary(engine_options),
                    )
                stage = "load_vllm_runtime"
                engine_load_started = perf_counter()
                self.llm = load_vllm_model(
                    model_path=model_dir,
                    tokenizer_path=tokenizer_dir,
                    vllm_kwargs=engine_options,
                )
                self.tokenizer = self.llm.tokenizer
                logger.info(
                    "event=service.engine_ready instance_id=%s load_id=%s "
                    "elapsed_ms=%.3f runtime_id=%s engine_type=%s tokenizer_type=%s",
                    self.instance_id,
                    load_id,
                    (perf_counter() - engine_load_started) * 1000.0,
                    self.llm.runtime_id,
                    _type_name(self.llm.engine),
                    _type_name(self.tokenizer),
                )

                stage = "build_token_trie"
                trie_started = perf_counter()
                token_ids = _code_token_id_map(
                    self.tokenizer, self.bundle.virtual_tokens
                )
                eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_token_id is None:
                    raise ServiceConfigurationError("Router tokenizer has no EOS token")
                if int(eos_token_id) in token_ids.values():
                    raise ServiceConfigurationError(
                        "a hierarchical virtual token shares the tokenizer EOS ID"
                    )
                separator_ids = tuple(
                    int(value)
                    for value in self.tokenizer.encode(
                        _CODE_PATH_SEPARATOR, add_special_tokens=False
                    )
                )
                if not separator_ids:
                    raise ServiceConfigurationError(
                        "Router tokenizer encodes the path separator as empty"
                    )
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "event=service.tokenizer_contract instance_id=%s load_id=%s "
                        "tokenizer_type=%s eos_token_id=%s virtual_token_count=%s "
                        "virtual_token_ids=%s separator_token_ids=%s "
                        "has_chat_template=%s",
                        self.instance_id,
                        load_id,
                        _type_name(self.tokenizer),
                        _token_ids_log_value([eos_token_id]),
                        len(token_ids),
                        _token_ids_log_value(sorted(token_ids.values())),
                        _token_ids_log_value(separator_ids),
                        _safe_getattr(
                            self.tokenizer, "chat_template", None
                        )
                        is not None,
                    )

                self.path_skill_ids = {
                    tuple(token_ids[token] for token in path): members
                    for path, members in self.bundle.token_paths.items()
                }
                self.trie = MultiPathTokenTrie(
                    self.path_skill_ids,
                    eos_token_id=int(eos_token_id),
                    separator_token_ids=separator_ids,
                    max_paths=self.max_code_paths,
                )
                logger.info(
                    "event=service.trie_ready instance_id=%s load_id=%s "
                    "elapsed_ms=%.3f token_paths=%s effective_max_paths=%s",
                    self.instance_id,
                    load_id,
                    (perf_counter() - trie_started) * 1000.0,
                    len(self.path_skill_ids),
                    self.trie.max_paths,
                )
                output_budget = (
                    self.trie.max_paths * self.bundle.num_levels
                    + (self.trie.max_paths - 1) * len(separator_ids)
                )
                engine_max_length = _custom_engine_max_model_len(self.llm)
                stage = "resolve_context_limits"
                self.max_input_length = _resolve_max_input_length(
                    trained_max_length=manifest_settings.max_length,
                    engine_max_length=engine_max_length,
                    output_budget=output_budget,
                )
                logger.info(
                    "event=service.context_limits instance_id=%s load_id=%s "
                    "trained_max_length=%s engine_max_length=%s output_budget=%s "
                    "max_input_length=%s explicit_max_input=%s",
                    self.instance_id,
                    load_id,
                    manifest_settings.max_length,
                    engine_max_length,
                    output_budget,
                    self.max_input_length,
                    os.environ.get("MAX_INPUT_LENGTH"),
                )
                stage = "build_sampling_params"
                self.sampling_params = _build_910b_sampling_params(
                    self.llm.sampling_params_cls,
                    output_budget=output_budget,
                    num_levels=self.bundle.num_levels,
                    trie=self.trie,
                )
                self.backend = "vllm_910b"
                self._loaded = True
                logger.info(
                    "event=service.load_complete instance_id=%s load_id=%s "
                    "backend=vllm_910b elapsed_ms=%.3f skills=%s paths=%s "
                    "levels=%s max_input_length=%s output_budget=%s",
                    self.instance_id,
                    load_id,
                    (perf_counter() - started) * 1000.0,
                    len(self.bundle.skills),
                    len(self.path_skill_ids),
                    self.bundle.num_levels,
                    self.max_input_length,
                    output_budget,
                )
            except BaseException as exc:
                cleanup_started = perf_counter()
                cleanup_error: BaseException | None = None
                try:
                    self._cleanup_after_failed_load(
                        load_id=load_id,
                        failed_stage=stage,
                    )
                except BaseException as cleanup_exc:
                    cleanup_error = cleanup_exc
                logger.exception(
                    "event=service.load_failed instance_id=%s load_id=%s stage=%s "
                    "elapsed_ms=%.3f cleanup_ms=%.3f error_type=%s",
                    self.instance_id,
                    load_id,
                    stage,
                    (perf_counter() - started) * 1000.0,
                    (perf_counter() - cleanup_started) * 1000.0,
                    type(exc).__name__,
                )
                if cleanup_error is not None:
                    logger.error(
                        "event=service.load_cleanup_failed instance_id=%s "
                        "load_id=%s failed_stage=%s cleanup_error_type=%s",
                        self.instance_id,
                        load_id,
                        stage,
                        type(cleanup_error).__name__,
                    )
                raise

    def calc(self, req_data: Mapping[str, Any] | None) -> str:
        _configure_service_logging()
        request_id = str(uuid.uuid4())
        total_started = perf_counter()
        container = req_data if isinstance(req_data, Mapping) else {}
        data = container.get("data", {})
        request = dict(data) if isinstance(data, Mapping) else {}
        query = str(request.get("query", _DEFAULT_QUERY)).strip()
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=service.calc_begin instance_id=%s request_id=%s loaded=%s "
                "backend=%s request_type=%s data_type=%s request_keys=%s "
                "query_chars=%s query=%s top_k_raw=%s topk_raw=%s",
                self.instance_id,
                request_id,
                self._loaded,
                self.backend,
                _type_name(req_data),
                _type_name(data),
                sorted(_safe_log_string(key) for key in request),
                len(query),
                _text_log_value(query),
                _safe_log_value("top_k", request.get("top_k")),
                _safe_log_value("topk", request.get("topk")),
            )
        if not query:
            logger.info(
                "event=service.calc_complete instance_id=%s request_id=%s "
                "status=empty_query elapsed_ms=%.3f results=0",
                self.instance_id,
                request_id,
                (perf_counter() - total_started) * 1000.0,
            )
            return json.dumps([], ensure_ascii=False)
        phase = "load_lock"
        load_lock_started = perf_counter()
        try:
            logger.info(
                "event=service.calc_load_lock_wait instance_id=%s request_id=%s",
                self.instance_id,
                request_id,
            )
            with self._load_lock:
                load_lock_wait_ms = (perf_counter() - load_lock_started) * 1000.0
                logger.info(
                    "event=service.calc_load_lock_acquired instance_id=%s "
                    "request_id=%s wait_ms=%.3f loaded=%s",
                    self.instance_id,
                    request_id,
                    load_lock_wait_ms,
                    self._loaded,
                )
                if not self._loaded:
                    phase = "automatic_load"
                    logger.info(
                        "event=service.calc_auto_load instance_id=%s request_id=%s",
                        self.instance_id,
                        request_id,
                    )
                    self.load()

                phase = "resolve_top_k"
                requested_top_k = _coerce_optional_int(request.get("top_k"))
                top_k_source = "top_k"
                if requested_top_k is None:
                    requested_top_k = _coerce_optional_int(request.get("topk"))
                    top_k_source = "topk" if requested_top_k is not None else "default"
                resolved_top_k = (
                    self.default_top_k
                    if requested_top_k is None
                    else max(1, requested_top_k)
                )
                clamped = resolved_top_k > self.default_top_k
                if clamped:
                    logger.warning(
                        "event=service.calc_top_k_clamped instance_id=%s request_id=%s "
                        "requested=%s initialized=%s",
                        self.instance_id,
                        request_id,
                        resolved_top_k,
                        self.default_top_k,
                    )
                    resolved_top_k = self.default_top_k
                logger.info(
                    "event=service.calc_route instance_id=%s request_id=%s "
                    "top_k_source=%s requested_top_k=%s resolved_top_k=%s "
                    "default_top_k=%s clamped=%s backend=%s",
                    self.instance_id,
                    request_id,
                    top_k_source,
                    requested_top_k,
                    resolved_top_k,
                    self.default_top_k,
                    clamped,
                    self.backend,
                )

                phase = "inference_lock"
                inference_wait_started = perf_counter()
                logger.info(
                    "event=service.calc_inference_lock_wait instance_id=%s "
                    "request_id=%s",
                    self.instance_id,
                    request_id,
                )
                with self._inference_lock:
                    inference_wait_ms = (
                        perf_counter() - inference_wait_started
                    ) * 1000.0
                    logger.info(
                        "event=service.calc_inference_lock_acquired instance_id=%s "
                        "request_id=%s wait_ms=%.3f",
                        self.instance_id,
                        request_id,
                        inference_wait_ms,
                    )
                    phase = "inference"
                    inference_started = perf_counter()
                    names = self._search_names(query, request_id=request_id)
                    inference_ms = (perf_counter() - inference_started) * 1000.0
            selected_names = names[:resolved_top_k]
            serialization_started = perf_counter()
            response_json = json.dumps(selected_names, ensure_ascii=False)
            serialization_ms = (perf_counter() - serialization_started) * 1000.0
            total_ms = (perf_counter() - total_started) * 1000.0
            queue_wait_ms = load_lock_wait_ms + inference_wait_ms
            calc_overhead_ms = max(
                0.0,
                total_ms - queue_wait_ms - inference_ms - serialization_ms,
            )
            logger.info(
                "event=service.calc_complete instance_id=%s request_id=%s "
                "status=ok elapsed_ms=%.3f inference_ms=%.3f load_lock_wait_ms=%.3f "
                "inference_lock_wait_ms=%.3f queue_wait_ms=%.3f "
                "json_serialize_ms=%.3f calc_overhead_ms=%.3f query_chars=%s "
                "decoded_results=%s returned_results=%s response_chars=%s",
                self.instance_id,
                request_id,
                total_ms,
                inference_ms,
                load_lock_wait_ms,
                inference_wait_ms,
                queue_wait_ms,
                serialization_ms,
                calc_overhead_ms,
                len(query),
                len(names),
                len(selected_names),
                len(response_json),
            )
            return response_json
        except Exception as exc:
            logger.exception(
                "event=service.calc_failed instance_id=%s request_id=%s phase=%s "
                "elapsed_ms=%.3f error_type=%s",
                self.instance_id,
                request_id,
                phase,
                (perf_counter() - total_started) * 1000.0,
                type(exc).__name__,
            )
            raise

    def close(self) -> None:
        close_id = uuid.uuid4().hex[:12]
        started = perf_counter()
        logger.info(
            "event=service.close_begin instance_id=%s close_id=%s loaded=%s "
            "backend=%s has_runtime=%s",
            self.instance_id,
            close_id,
            self._loaded,
            self.backend,
            self.llm is not None,
        )
        with self._load_lock:
            logger.info(
                "event=service.close_load_lock_acquired instance_id=%s close_id=%s "
                "elapsed_ms=%.3f",
                self.instance_id,
                close_id,
                (perf_counter() - started) * 1000.0,
            )
            with self._inference_lock:
                logger.info(
                    "event=service.close_inference_lock_acquired instance_id=%s "
                    "close_id=%s elapsed_ms=%.3f",
                    self.instance_id,
                    close_id,
                    (perf_counter() - started) * 1000.0,
                )
                _shutdown_vllm(self.llm)
                self.llm = None
                self.tokenizer = None
                self.sampling_params = None
                self.bundle = None
                self.trie = None
                self.path_skill_ids = {}
                self.backend = "vllm_910b"
                self.mock_responses = {}
                self.system_prompt = _DEFAULT_SYSTEM_PROMPT
                self._loaded = False
            gc.collect()
        logger.info(
            "event=service.close_complete instance_id=%s close_id=%s "
            "elapsed_ms=%.3f loaded=%s runtime_present=%s",
            self.instance_id,
            close_id,
            (perf_counter() - started) * 1000.0,
            self._loaded,
            self.llm is not None,
        )

    def _search_names(self, query: str, *, request_id: str | None = None) -> list[str]:
        resolved_request_id = request_id or str(uuid.uuid4())
        started = perf_counter()
        logger.info(
            "event=search.begin instance_id=%s request_id=%s backend=%s "
            "query_chars=%s query=%s",
            self.instance_id,
            resolved_request_id,
            self.backend,
            len(query),
            _text_log_value(query),
        )
        if self.backend == "mock":
            if not self._loaded:
                raise RuntimeError("LLMGen mock retrieval service is not loaded")
            exact_match = query in self.mock_responses
            results = list(
                self.mock_responses.get(query, self.mock_responses.get("*", ()))
            )
            logger.info(
                "event=search.mock_complete instance_id=%s request_id=%s "
                "elapsed_ms=%.3f exact_match=%s result_count=%s",
                self.instance_id,
                resolved_request_id,
                (perf_counter() - started) * 1000.0,
                exact_match,
                len(results),
            )
            return results
        if (
            self.llm is None
            or self.tokenizer is None
            or self.sampling_params is None
            or self.bundle is None
            or self.trie is None
        ):
            logger.error(
                "event=search.runtime_missing instance_id=%s request_id=%s "
                "has_runtime=%s has_tokenizer=%s has_sampling_params=%s "
                "has_bundle=%s has_trie=%s loaded=%s",
                self.instance_id,
                resolved_request_id,
                self.llm is not None,
                self.tokenizer is not None,
                self.sampling_params is not None,
                self.bundle is not None,
                self.trie is not None,
                self._loaded,
            )
            raise RuntimeError("LLMGen retrieval service is not loaded")
        render_started = perf_counter()
        prompt = _render_router_prompt(
            self.tokenizer,
            query,
            self.system_prompt,
        )
        render_ms = (perf_counter() - render_started) * 1000.0
        logger.info(
            "event=search.prompt_rendered instance_id=%s request_id=%s "
            "elapsed_ms=%.3f prompt_chars=%s prompt=%s system_prompt_chars=%s",
            self.instance_id,
            resolved_request_id,
            render_ms,
            len(prompt),
            _text_log_value(prompt),
            len(self.system_prompt),
        )
        tokenize_started = perf_counter()
        prompt_ids = [
            int(value)
            for value in self.tokenizer.encode(prompt, add_special_tokens=False)
        ]
        if not prompt_ids:
            raise RuntimeError("Router tokenizer encoded the prompt as empty")
        original_prompt_tokens = len(prompt_ids)
        truncated = False
        if len(prompt_ids) > self.max_input_length:
            truncated = True
            # Match the repository inference path: keep the prompt prefix.
            prompt_ids = prompt_ids[: self.max_input_length]
            decode = getattr(self.tokenizer, "decode", None)
            if callable(decode):
                decode_started = perf_counter()
                try:
                    prompt = str(
                        decode(prompt_ids, skip_special_tokens=False)
                    )
                except TypeError:
                    logger.info(
                        "event=search.prompt_decode_retry instance_id=%s "
                        "request_id=%s reason=skip_special_tokens_not_supported",
                        self.instance_id,
                        resolved_request_id,
                        exc_info=True,
                    )
                    prompt = str(decode(prompt_ids))
                logger.info(
                    "event=search.prompt_decoded_after_truncation instance_id=%s "
                    "request_id=%s elapsed_ms=%.3f prompt_chars=%s prompt=%s",
                    self.instance_id,
                    resolved_request_id,
                    (perf_counter() - decode_started) * 1000.0,
                    len(prompt),
                    _text_log_value(prompt),
                )
            else:
                logger.warning(
                    "event=search.prompt_truncated_without_decode instance_id=%s "
                    "request_id=%s original_prompt_chars=%s original_tokens=%s "
                    "kept_tokens=%s",
                    self.instance_id,
                    resolved_request_id,
                    len(prompt),
                    original_prompt_tokens,
                    len(prompt_ids),
                )
        tokenize_ms = (perf_counter() - tokenize_started) * 1000.0
        logger.info(
            "event=search.prompt_tokenized instance_id=%s request_id=%s "
            "elapsed_ms=%.3f original_tokens=%s submitted_tokens=%s "
            "max_input_length=%s truncated=%s token_ids=%s",
            self.instance_id,
            resolved_request_id,
            tokenize_ms,
            original_prompt_tokens,
            len(prompt_ids),
            self.max_input_length,
            truncated,
            _token_ids_log_value(prompt_ids),
        )

        generation_started = perf_counter()
        request_output = self.llm.generate(
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            sampling_params=self.sampling_params,
            request_id=resolved_request_id,
        )
        generation_ms = (perf_counter() - generation_started) * 1000.0
        logger.info(
            "event=search.generation_returned instance_id=%s request_id=%s "
            "elapsed_ms=%.3f output_type=%s",
            self.instance_id,
            resolved_request_id,
            generation_ms,
            _type_name(request_output),
        )
        decode_started = perf_counter()
        completion = _first_completion_output(
            request_output, request_id=resolved_request_id
        )
        finish_reason = _output_field(completion, "finish_reason")
        raw_token_ids = _output_field(completion, "token_ids")
        if raw_token_ids is None:
            raise RuntimeError("custom vLLM 910B completion has no token IDs")
        generated = [int(value) for value in raw_token_ids]
        logger.info(
            "event=search.decode_begin instance_id=%s request_id=%s "
            "finish_reason=%s generated_count=%s generated=%s",
            self.instance_id,
            resolved_request_id,
            finish_reason,
            len(generated),
            _token_ids_log_value(generated),
        )
        try:
            eos_position = generated.index(self.trie.eos_token_id)
        except ValueError as exc:
            if finish_reason != "length":
                raise RuntimeError(
                    "constrained vLLM generation did not emit EOS"
                ) from exc
            eos_position = None
            logger.info(
                "event=search.length_truncation_accepted instance_id=%s "
                "request_id=%s generated_count=%s",
                self.instance_id,
                resolved_request_id,
                len(generated),
            )
        else:
            trailing_count = len(generated) - eos_position - 1
            if trailing_count:
                logger.warning(
                    "event=search.tokens_after_eos instance_id=%s request_id=%s "
                    "eos_position=%s trailing_count=%s trailing_tokens=%s",
                    self.instance_id,
                    resolved_request_id,
                    eos_position,
                    trailing_count,
                    _token_ids_log_value(generated[eos_position + 1 :]),
                )
            generated = generated[:eos_position]
        paths = self.trie.parse_complete(
            generated, request_id=resolved_request_id
        )
        decode_ms = (perf_counter() - decode_started) * 1000.0
        logger.info(
            "event=search.paths_decoded instance_id=%s request_id=%s "
            "eos_position=%s path_count=%s paths=%s",
            self.instance_id,
            resolved_request_id,
            eos_position,
            len(paths),
            _token_ids_log_value(paths),
        )

        mapping_started = perf_counter()
        skill_ids: list[str] = []
        seen: set[str] = set()
        for path in paths:
            members = self.path_skill_ids.get(path)
            if members is None:
                logger.error(
                    "event=search.path_missing instance_id=%s request_id=%s path=%s",
                    self.instance_id,
                    resolved_request_id,
                    _token_ids_log_value(path),
                )
                raise RuntimeError("generated path is absent from the candidate state")
            for skill_id in members:
                if skill_id not in seen:
                    skill_ids.append(skill_id)
                    seen.add(skill_id)
        names = [
            str(self.bundle.skills[skill_id].get("name") or skill_id)
            for skill_id in skill_ids
        ]
        mapping_ms = (perf_counter() - mapping_started) * 1000.0
        search_total_ms = (perf_counter() - started) * 1000.0
        service_non_generation_ms = max(0.0, search_total_ms - generation_ms)
        measured_stage_ms = (
            render_ms + tokenize_ms + generation_ms + decode_ms + mapping_ms
        )
        unattributed_ms = max(0.0, search_total_ms - measured_stage_ms)
        logger.info(
            "event=search.complete instance_id=%s request_id=%s elapsed_ms=%.3f "
            "render_ms=%.3f tokenize_ms=%.3f generation_ms=%.3f "
            "decode_ms=%.3f mapping_ms=%.3f service_non_generation_ms=%.3f "
            "unattributed_ms=%.3f prompt_tokens=%s completion_tokens=%s "
            "path_count=%s skill_id_count=%s result_count=%s",
            self.instance_id,
            resolved_request_id,
            search_total_ms,
            render_ms,
            tokenize_ms,
            generation_ms,
            decode_ms,
            mapping_ms,
            service_non_generation_ms,
            unattributed_ms,
            len(prompt_ids),
            len(generated),
            len(paths),
            len(skill_ids),
            len(names),
        )
        return names

    def _cleanup_after_failed_load(
        self,
        *,
        load_id: str | None = None,
        failed_stage: str | None = None,
    ) -> None:
        started = perf_counter()
        logger.info(
            "event=service.load_cleanup_begin instance_id=%s load_id=%s "
            "failed_stage=%s has_runtime=%s has_tokenizer=%s has_bundle=%s",
            self.instance_id,
            load_id,
            failed_stage,
            self.llm is not None,
            self.tokenizer is not None,
            self.bundle is not None,
        )
        _shutdown_vllm(self.llm)
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self.bundle = None
        self.trie = None
        self.path_skill_ids = {}
        self.backend = "vllm_910b"
        self.mock_responses = {}
        self.system_prompt = _DEFAULT_SYSTEM_PROMPT
        self._loaded = False
        gc.collect()
        logger.info(
            "event=service.load_cleanup_complete instance_id=%s load_id=%s "
            "failed_stage=%s elapsed_ms=%.3f loaded=%s runtime_present=%s",
            self.instance_id,
            load_id,
            failed_stage,
            (perf_counter() - started) * 1000.0,
            self._loaded,
            self.llm is not None,
        )


# A correctly-spelled alias for integrations that do not depend on the
# historical reference class name.
SkillRouterService = RetriverTest


def _deployment_parent(current_dir: Path) -> Path:
    model_object_id = os.environ.get("MODEL_OBJECT_ID")
    model_sfs = os.environ.get("MODEL_SFS")
    logger.info(
        "event=deployment.resolve_parent current_dir=%s model_object_id_present=%s "
        "model_sfs_present=%s",
        current_dir,
        bool(model_object_id),
        bool(model_sfs),
    )
    if model_object_id or model_sfs:
        if not model_object_id or not model_sfs:
            raise ServiceConfigurationError(
                "MODEL_OBJECT_ID and MODEL_SFS must be set together"
            )
        try:
            payload = json.loads(model_sfs)
        except json.JSONDecodeError as exc:
            raise ServiceConfigurationError("MODEL_SFS is not valid JSON") from exc
        base_path = payload.get("sfsBasePath") if isinstance(payload, Mapping) else None
        if not isinstance(base_path, str) or not base_path.strip():
            raise ServiceConfigurationError("MODEL_SFS has no non-empty sfsBasePath")
        resolved = (Path(base_path) / model_object_id).expanduser().resolve()
        logger.info(
            "event=deployment.parent_resolved source=sfs parent=%s",
            resolved,
        )
        return resolved
    resolved = current_dir.parent.resolve()
    logger.info(
        "event=deployment.parent_resolved source=service_parent parent=%s",
        resolved,
    )
    return resolved


def _resolve_model_directory(current_dir: Path) -> Path:
    deployment_parent = _deployment_parent(current_dir)
    platform_model_dir = deployment_parent / "model"
    if os.environ.get("MODEL_OBJECT_ID") or os.environ.get("MODEL_SFS"):
        # Preserve the reference service contract: the hosting system's SFS
        # location is authoritative when it is supplied.
        resolved = platform_model_dir.expanduser().resolve()
        logger.info(
            "event=deployment.model_resolved source=sfs model=%s",
            resolved,
        )
        return resolved
    resolved = Path(
        _env_first_text(("MODEL_PATH", "MODEL_DIR"), str(platform_model_dir))
    ).expanduser().resolve()
    logger.info(
        "event=deployment.model_resolved source=%s model=%s",
        "environment"
        if os.environ.get("MODEL_PATH") or os.environ.get("MODEL_DIR")
        else "default",
        resolved,
    )
    return resolved


def _log_startup_artifact_probes(
    *,
    instance_id: str,
    load_id: str,
    model_dir: Path,
    tokenizer_dir: Path,
    candidate_dir: Path,
) -> None:
    """Log every deployment artifact relevant to startup before validation."""

    artifacts = (
        ("model_config", model_dir / "config.json", "required"),
        ("model_safetensors", model_dir / "model.safetensors", "weights_one_of"),
        (
            "model_safetensors_index",
            model_dir / "model.safetensors.index.json",
            "weights_one_of",
        ),
        ("pytorch_model", model_dir / "pytorch_model.bin", "weights_one_of"),
        (
            "pytorch_model_index",
            model_dir / "pytorch_model.bin.index.json",
            "weights_one_of",
        ),
        (
            "tokenizer_config",
            tokenizer_dir / "tokenizer_config.json",
            "tokenizer_expected",
        ),
        ("tokenizer_json", tokenizer_dir / "tokenizer.json", "tokenizer_expected"),
        (
            "special_tokens_map",
            tokenizer_dir / "special_tokens_map.json",
            "tokenizer_optional",
        ),
        (
            "added_tokens",
            tokenizer_dir / "added_tokens.json",
            "tokenizer_optional",
        ),
        (
            "chat_template",
            tokenizer_dir / "chat_template.jinja",
            "tokenizer_optional",
        ),
        (
            "generation_config",
            model_dir / "generation_config.json",
            "optional",
        ),
        ("training_args", model_dir / "training_args.bin", "optional"),
        (
            "router_manifest",
            model_dir / _ROUTER_MANIFEST_FILENAME,
            "required",
        ),
        (
            "candidate_decode_map",
            candidate_dir / _DECODE_MAP_FILENAME,
            "required",
        ),
        (
            "candidate_virtual_tokens",
            candidate_dir / _VIRTUAL_TOKENS_FILENAME,
            "required",
        ),
    )
    for role, path, requirement in artifacts:
        try:
            exists = path.exists()
            is_file = path.is_file()
            size_bytes: object = path.stat().st_size if is_file else None
            diagnostic_error: object = None
        except OSError as exc:
            exists = False
            is_file = False
            size_bytes = None
            diagnostic_error = type(exc).__name__
        logger.info(
            "event=service.artifact_probe instance_id=%s load_id=%s role=%s "
            "requirement=%s path=%s exists=%s is_file=%s size_bytes=%s "
            "diagnostic_error=%s",
            instance_id,
            load_id,
            role,
            requirement,
            path,
            exists,
            is_file,
            size_bytes,
            diagnostic_error,
        )


def _validate_full_model_bundle(model_dir: Path) -> None:
    """Require the consolidated Hugging Face files that vLLM can load."""

    started = perf_counter()
    config_path = model_dir / "config.json"
    indexed_weights = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    single_weights = ("model.safetensors", "pytorch_model.bin")
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=model_bundle.validate_begin model=%s config_exists=%s "
            "directory_entries=%s",
            model_dir,
            config_path.is_file(),
            _directory_entries_log_value(model_dir),
        )

    has_full_weights = False
    selected_weights: list[tuple[str, int]] = []
    for index_name in indexed_weights:
        index_path = model_dir / index_name
        if index_path.is_file():
            selected_weights = _validate_model_weight_index(model_dir, index_path)
            has_full_weights = True
            break
    if not has_full_weights:
        for name in single_weights:
            weight_path = model_dir / name
            if not weight_path.is_file():
                continue
            weight_size = weight_path.stat().st_size
            if weight_size > 0:
                selected_weights = [(name, weight_size)]
                has_full_weights = True
                break

    if not has_full_weights:
        if (model_dir / "adapter_config.json").is_file():
            raise ServiceConfigurationError(
                "vLLM service deployment requires a merged/full Router model; "
                "the selected bundle contains only a PEFT adapter"
            )
        raise ServiceConfigurationError(
            "model directory has no consolidated Hugging Face inference weights; "
            "expected model.safetensors, pytorch_model.bin, or a valid shard index"
        )
    if not config_path.is_file():
        raise ServiceConfigurationError(
            f"full Router model is missing Hugging Face config: {config_path}"
        )
    config = _read_json_object(config_path)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=model_bundle.validate_complete model=%s elapsed_ms=%.3f "
            "weight_files=%s total_weight_bytes=%s config_keys=%s model_type=%s "
            "architectures=%s vocab_size=%s max_position_embeddings=%s",
            model_dir,
            (perf_counter() - started) * 1000.0,
            [name for name, _ in selected_weights],
            sum(size for _, size in selected_weights),
            sorted(config),
            config.get("model_type"),
            config.get("architectures"),
            config.get("vocab_size"),
            config.get("max_position_embeddings"),
        )


def _validate_model_weight_index(
    model_dir: Path, index_path: Path
) -> list[tuple[str, int]]:
    started = perf_counter()
    payload = _read_json_object(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ServiceConfigurationError(
            f"model weight index has no weight_map: {index_path}"
        )
    model_root = model_dir.resolve()
    shard_names: set[str] = set()
    for raw_name in weight_map.values():
        if not isinstance(raw_name, str) or not raw_name:
            raise ServiceConfigurationError(
                f"model weight index contains an invalid shard: {index_path}"
            )
        shard_names.add(raw_name)
    validated_shards: list[tuple[str, int]] = []
    for raw_name in sorted(shard_names):
        shard = (model_dir / raw_name).resolve()
        if shard.parent != model_root or not shard.is_file():
            raise ServiceConfigurationError(
                "model weight index references a missing, empty, or unsafe shard: "
                f"{raw_name}"
            )
        shard_size = shard.stat().st_size
        if shard_size < 1:
            raise ServiceConfigurationError(
                "model weight index references a missing, empty, or unsafe shard: "
                f"{raw_name}"
            )
        validated_shards.append((raw_name, shard_size))
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=model_bundle.index_validated index=%s elapsed_ms=%.3f "
            "weight_entries=%s shards=%s total_bytes=%s shard_names=%s",
            index_path,
            (perf_counter() - started) * 1000.0,
            len(weight_map),
            len(validated_shards),
            sum(size for _, size in validated_shards),
            _sequence_log_value([name for name, _ in validated_shards]),
        )
    return validated_shards


def _build_vllm_kwargs(*, model_path: Path) -> Dict[str, Any]:
    """Build SkillHub-style custom-engine options for the owned Router bundle."""

    started = perf_counter()
    logger.info("event=engine_options.build_begin model=%s", model_path)
    kwargs: Dict[str, Any] = {}
    for key, default in _vllm_engine_arg_defaults(model_path=model_path).items():
        kwargs[key] = _env_engine_arg(key, default)
    kwargs["health_check_timeout"] = _env_engine_arg(
        "health_check_timeout", None
    )
    kwargs["health_check_interval"] = _env_engine_arg(
        "health_check_interval", _VLLM_HEALTH_CHECK_INTERVAL
    )
    extra_json = os.environ.get("VLLM_KWARGS_JSON")
    if extra_json:
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError as exc:
            raise ServiceConfigurationError(
                "VLLM_KWARGS_JSON must be valid JSON"
            ) from exc
        if not isinstance(extra, dict):
            raise ServiceConfigurationError(
                "VLLM_KWARGS_JSON must decode to an object"
            )
        reserved = {"model", "tokenizer", "engine_role"}.intersection(extra)
        if reserved:
            raise ServiceConfigurationError(
                "VLLM_KWARGS_JSON cannot override: "
                + ", ".join(sorted(reserved))
            )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=engine_options.json_overrides keys=%s safe_values=%s",
                sorted(extra),
                {
                    key: _safe_log_value(key, value)
                    for key, value in extra.items()
                },
            )
        kwargs.update(extra)
    if not bool(kwargs.get("disable_log_requests", True)):
        logger.warning(
            "event=engine_options.request_logging_enabled max_log_len=%s "
            "message=custom_vllm_may_log_user_prompts",
            kwargs.get("max_log_len"),
        )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=engine_options.build_complete elapsed_ms=%.3f option_count=%s "
            "option_keys=%s summary=%s",
            (perf_counter() - started) * 1000.0,
            len(kwargs),
            sorted(kwargs),
            _engine_log_summary(kwargs),
        )
    return kwargs


def _vllm_engine_arg_defaults(*, model_path: Path) -> Dict[str, Any]:
    defaults = dict(_VLLM_ENGINE_ARG_DEFAULTS)
    # SkillHub's original profile hard-codes a Qwen3.5-MoE-only architecture.
    # This service targets dense Qwen3.5-2B instead: translate the official
    # bundle class to the custom runtime's text-only class, preserve explicit
    # alternative classes, and use the dense text-only class as the fallback.
    defaults["architectures"] = _model_architecture(
        model_path, fallback=_VLLM_910B_ARCHITECTURE
    )
    tensor_parallel_size = int(
        _env_engine_arg(
            "tensor_parallel_size", _VLLM_TENSOR_PARALLEL_SIZE
        )
    )
    defaults["tensor_parallel_size"] = tensor_parallel_size
    defaults["decode_tensor_parallel_size"] = tensor_parallel_size
    defaults["prefix_sharing_type"] = "auto"
    logger.info(
        "event=engine_options.defaults architecture=%s tensor_parallel_size=%s "
        "decode_tensor_parallel_size=%s defaults=%s",
        defaults["architectures"],
        defaults["tensor_parallel_size"],
        defaults["decode_tensor_parallel_size"],
        len(defaults),
    )
    return defaults


def _model_architecture(model_path: Path, *, fallback: str) -> str:
    config = _read_json_object(model_path / "config.json")
    raw_architectures = config.get("architectures")
    if isinstance(raw_architectures, str) and raw_architectures.strip():
        selected = raw_architectures.strip()
        logger.info(
            "event=model_architecture.resolved source=config_string architecture=%s",
            selected,
        )
        return _custom_910b_architecture(selected)
    if isinstance(raw_architectures, list):
        for value in raw_architectures:
            if isinstance(value, str) and value.strip():
                selected = value.strip()
                logger.info(
                    "event=model_architecture.resolved source=config_list "
                    "architecture=%s",
                    selected,
                )
                return _custom_910b_architecture(selected)
    logger.warning(
        "event=model_architecture.fallback architecture=%s config_value_type=%s",
        fallback,
        _type_name(raw_architectures),
    )
    return fallback


def _custom_910b_architecture(exported_architecture: str) -> str:
    """Translate the official dense bundle class to SkillHub's text-only class."""

    if exported_architecture in _QWEN35_DENSE_BUNDLE_ARCHITECTURES:
        logger.info(
            "event=model_architecture.adapted exported=%s runtime=%s",
            exported_architecture,
            _VLLM_910B_ARCHITECTURE,
        )
        return _VLLM_910B_ARCHITECTURE
    return exported_architecture


def _env_engine_arg(
    name: str, default: Any, *, aliases: tuple[str, ...] = ()
) -> Any:
    for env_name in _engine_arg_env_names(name, aliases=aliases):
        value = os.environ.get(env_name)
        if value is not None and str(value).strip():
            parsed = _parse_env_value(value, default)
            logger.info(
                "event=engine_options.environment_override option=%s env_name=%s "
                "default_type=%s parsed_type=%s value=%s",
                name,
                env_name,
                _type_name(default),
                _type_name(parsed),
                _safe_log_value(name, parsed),
            )
            return parsed
    return default


def _engine_arg_env_names(
    name: str, *, aliases: tuple[str, ...] = ()
) -> tuple[str, ...]:
    upper_name = name.upper()
    names = (f"VLLM_{upper_name}", upper_name, *aliases)
    deduped: list[str] = []
    for item in names:
        if item and item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _parse_env_value(value: object, default: Any) -> Any:
    text = str(value).strip()
    if isinstance(default, bool):
        return text.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(text)
    if isinstance(default, float):
        return float(text)
    if isinstance(default, str):
        return text
    if isinstance(default, dict):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ServiceConfigurationError(
                "environment value must decode to a JSON object"
            )
        return parsed
    if text.lower() in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_vllm_model(
    *,
    model_path: str | Path,
    tokenizer_path: str | Path,
    vllm_kwargs: Mapping[str, Any] | None = None,
) -> LocalVLLM910BRuntime:
    """Start and health-check SkillHub's in-process custom vLLM runtime.

    This is an independent entry point for debugging model startup on a 910B
    host.  The returned runtime owns its async loop and must be closed by the
    caller.  ``RetriverTest.load()`` calls this same function.
    """

    _configure_service_logging()
    runtime_load_id = uuid.uuid4().hex[:12]
    started = perf_counter()
    phase = "resolve_arguments"
    resolved_model_path = str(Path(model_path).expanduser().resolve())
    resolved_tokenizer_path = str(Path(tokenizer_path).expanduser().resolve())
    options = dict(vllm_kwargs or {})
    logger.info(
        "event=runtime.load_begin runtime_load_id=%s model_name=%s "
        "tokenizer_name=%s option_count=%s",
        runtime_load_id,
        Path(resolved_model_path).name,
        Path(resolved_tokenizer_path).name,
        len(options),
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=runtime.load_arguments runtime_load_id=%s model=%s tokenizer=%s "
            "option_keys=%s summary=%s",
            runtime_load_id,
            resolved_model_path,
            resolved_tokenizer_path,
            sorted(options),
            _engine_log_summary(options),
        )
    reserved = {"model", "tokenizer", "engine_role"}.intersection(options)
    if reserved:
        raise ServiceConfigurationError(
            "vllm_kwargs cannot override: " + ", ".join(sorted(reserved))
        )
    removed_service_options = {
        key: options.pop(key)
        for key in ("request_model", "model_name")
        if key in options
    }
    # SkillHub accepts device in its public config but uses it only for logs;
    # it is not part of the custom AsyncEngineArgs payload.
    if "device" in options:
        removed_service_options["device"] = options.pop("device")
    trust_remote_code = bool(options.pop("trust_remote_code", True))
    health_check_timeout = _pop_float_optional(
        options, "health_check_timeout"
    )
    health_check_interval = max(
        0.1, float(options.pop("health_check_interval", 1.0))
    )
    dtype = str(options.pop("dtype", _vllm_dtype()) or _VLLM_DTYPE)
    request_timeout = _env_float_optional("PROGRESSIVE_REQUEST_TIMEOUT")
    logger.info(
        "event=runtime.load_normalized_options runtime_load_id=%s dtype=%s "
        "trust_remote_code=%s health_check_timeout=%s "
        "health_check_interval=%s request_timeout=%s removed_service_options=%s "
        "engine_option_keys=%s",
        runtime_load_id,
        dtype,
        trust_remote_code,
        health_check_timeout,
        health_check_interval,
        request_timeout,
        {
            key: _safe_log_value(key, value)
            for key, value in removed_service_options.items()
        },
        sorted(options),
    )
    phase = "import_runtime"
    import_started = perf_counter()
    try:
        import vllm
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        from vllm.global_consts import EngineRole
    except ImportError as exc:
        raise RuntimeError(
            "service_910b requires the custom vLLM package used by SkillHub"
        ) from exc
    logger.info(
        "event=runtime.dependencies_loaded runtime_load_id=%s elapsed_ms=%.3f "
        "vllm_version=%s async_engine_args=%s async_engine=%s sampling_params=%s "
        "engine_role_type=%s",
        runtime_load_id,
        (perf_counter() - import_started) * 1000.0,
        _safe_getattr(vllm, "__version__", "<unknown>"),
        _safe_getattr(AsyncEngineArgs, "__qualname__", _type_name(AsyncEngineArgs)),
        _safe_getattr(AsyncLLMEngine, "__qualname__", _type_name(AsyncLLMEngine)),
        _safe_getattr(SamplingParams, "__qualname__", _type_name(SamplingParams)),
        _type_name(EngineRole.M),
    )
    phase = "load_tokenizer"
    tokenizer_started = perf_counter()
    tokenizer = _load_chat_template_tokenizer(
        tokenizer_path=resolved_tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    logger.info(
        "event=runtime.tokenizer_ready runtime_load_id=%s elapsed_ms=%.3f "
        "tokenizer_type=%s eos_token_id=%s vocab_size=%s "
        "has_chat_template=%s",
        runtime_load_id,
        (perf_counter() - tokenizer_started) * 1000.0,
        _type_name(tokenizer),
        _token_ids_log_value([_safe_getattr(tokenizer, "eos_token_id")]),
        _safe_getattr(tokenizer, "vocab_size"),
        _safe_getattr(tokenizer, "chat_template", None) is not None,
    )
    phase = "build_engine_kwargs"
    engine_kwargs = _build_custom_engine_kwargs(
        model_path=resolved_model_path,
        tokenizer_path=resolved_tokenizer_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        engine_role=EngineRole.M,
        options=options,
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=runtime.engine_kwargs runtime_load_id=%s dtype=%s "
            "option_count=%s option_keys=%s summary=%s",
            runtime_load_id,
            dtype,
            len(engine_kwargs),
            sorted(engine_kwargs),
            _engine_log_summary(engine_kwargs),
        )
    phase = "build_engine_args"
    engine_args_started = perf_counter()
    accepted_engine_kwargs = _filter_callable_kwargs(
        AsyncEngineArgs, engine_kwargs
    )
    engine_args = build_engine_args(AsyncEngineArgs, **engine_kwargs)
    logger.info(
        "event=runtime.engine_args_ready runtime_load_id=%s elapsed_ms=%.3f "
        "accepted=%s skipped=%s engine_args_type=%s",
        runtime_load_id,
        (perf_counter() - engine_args_started) * 1000.0,
        len(accepted_engine_kwargs),
        sorted(set(engine_kwargs).difference(accepted_engine_kwargs)),
        _type_name(engine_args),
    )
    phase = "construct_engine"
    engine_construct_started = perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info(
        "event=runtime.engine_constructed runtime_load_id=%s elapsed_ms=%.3f "
        "engine_type=%s architecture=%s dtype=%s tp=%s pp=%s dp=%s",
        runtime_load_id,
        (perf_counter() - engine_construct_started) * 1000.0,
        _type_name(engine),
        engine_kwargs.get("architectures"),
        dtype,
        engine_kwargs.get("tensor_parallel_size"),
        engine_kwargs.get("pipeline_parallel_size"),
        engine_kwargs.get("data_parallel_size"),
    )
    phase = "create_async_loop"
    loop_runner = _AsyncLoopRunner()
    runtime = LocalVLLM910BRuntime(
        engine=engine,
        tokenizer=tokenizer,
        sampling_params_cls=SamplingParams,
        loop_runner=loop_runner,
        request_timeout=request_timeout,
        engine_kwargs=accepted_engine_kwargs,
    )
    try:
        phase = "start_engine"
        startup_started = perf_counter()
        loop_runner.submit(
            _start_custom_engine(
                engine=engine,
                health_check_interval=health_check_interval,
                health_check_timeout=health_check_timeout,
            ),
            # The coroutine checks the total startup deadline itself.  An
            # outer Future timeout cannot safely interrupt synchronous
            # engine.load_model() and could otherwise let the engine start
            # after failure cleanup has already run.
            timeout=None,
            operation="engine_startup",
            request_id=runtime_load_id,
        )
        logger.info(
            "event=runtime.load_complete runtime_load_id=%s runtime_id=%s "
            "elapsed_ms=%.3f startup_ms=%.3f resolved_max_model_len=%s",
            runtime_load_id,
            runtime.runtime_id,
            (perf_counter() - started) * 1000.0,
            (perf_counter() - startup_started) * 1000.0,
            _custom_engine_max_model_len(runtime),
        )
    except BaseException as exc:
        cleanup_started = perf_counter()
        cleanup_error: BaseException | None = None
        try:
            runtime.close()
        except BaseException as cleanup_exc:
            cleanup_error = cleanup_exc
        logger.exception(
            "event=runtime.load_failed runtime_load_id=%s runtime_id=%s phase=%s "
            "elapsed_ms=%.3f cleanup_ms=%.3f error_type=%s",
            runtime_load_id,
            runtime.runtime_id,
            phase,
            (perf_counter() - started) * 1000.0,
            (perf_counter() - cleanup_started) * 1000.0,
            type(exc).__name__,
        )
        if cleanup_error is not None:
            logger.error(
                "event=runtime.load_cleanup_failed runtime_load_id=%s "
                "runtime_id=%s phase=%s cleanup_error_type=%s",
                runtime_load_id,
                runtime.runtime_id,
                phase,
                type(cleanup_error).__name__,
            )
        raise
    return runtime


def _build_custom_engine_kwargs(
    *,
    model_path: str,
    tokenizer_path: str,
    dtype: str,
    trust_remote_code: bool,
    engine_role: object,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the custom AsyncEngineArgs payload used by SkillHub."""

    resolved_dtype = str(dtype or "").strip()
    if not resolved_dtype or resolved_dtype.lower() == "auto":
        resolved_dtype = _VLLM_DTYPE
    kwargs: dict[str, Any] = {
        "model": model_path,
        "model_vision": "facebook/opt-125m",
        "architectures": _VLLM_910B_ARCHITECTURE,
        "tokenizer": tokenizer_path,
        "tokenizer_mode": "auto",
        "trust_remote_code": bool(trust_remote_code),
        "download_dir": None,
        "load_format": "auto",
        "dtype": resolved_dtype,
        "seed": 0,
        "max_model_len": None,
        "rope_scaling_type": None,
        "rope_scaling_factor": 1.0,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": _VLLM_TENSOR_PARALLEL_SIZE,
        "data_parallel_size": 1,
        "context_parallel_size": 1,
        "pipeline_parallel_layer_partitions": "",
        "mla_wo_tensor_parallel_size": -1,
        "enable_expert_parallel": False,
        "decode_enable_expert_parallel": False,
        "decode_pipeline_parallel_size": 1,
        "decode_tensor_parallel_size": _VLLM_TENSOR_PARALLEL_SIZE,
        "decode_data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "block_size": 128,
        "kernel_block_size": 128,
        "prefix_sharing_chunk_size": 128,
        "scheduler_budget_len": 102400,
        "prefix_sharing_type": "auto",
        "prefix_sharing_kwargs": {"gpu_usage_threshold": 0.7},
        "enable_datasystem": True,
        "multipath_devices": "",
        "swap_space": 0,
        "gpu_memory_utilization": 0.9,
        "max_num_batched_tokens": None,
        "max_num_seqs": 8,
        "disable_log_stats": False,
        "revision": None,
        "tokenizer_revision": None,
        "quantization": None,
        "block_sliding_window": None,
        "sink_block_num": 0,
        "schedule_policy": "fcfs",
        "schedule_policy_kwargs": None,
        "first_token_timeout": 300.0,
        "max_swapped_req_num": 128,
        "sys_prefix_prompts": None,
        "ops_dev_mode": None,
        "speculate_type": None,
        "speculate_kwargs": None,
        "disaggregate_prefill_decoding": False,
        "dispd_args": None,
        "ranks": None,
        "engine_name": "",
        "sparse_mode": "",
        "sparse_threshold_len": 4096,
        "sparse_minimum_len": 2048,
        "sparse_budget_len": 4096,
        "sparse_compress_ratio": 0.5,
        "cluster_window_size": 32,
        "cluster_sink_size": 64,
        "cluster_recent_size": 128,
        "cluster_kernel_size": 9,
        "cluster_block_size": 64,
        "inf_prefix_len": 64,
        "inf_query_len": 32,
        "inf_window_size": 1024,
        "inf_overlap_size": 32,
        "turbo_share_sysprefix": False,
        "turbo_sysprefix_num": 0,
        "turbo_separator_set": None,
        "speculative_config": None,
        "enable_chunked_prefill": True,
        "enable_batching_prefill": False,
        "enable_fuse_prefill_and_decode": False,
        "enable_lookahead_scheduling": False,
        "need_kv_transfer": False,
        "prefill_group_num": 1,
        "decode_group_num": 1,
        "global_group_meta": None,
        "stage_id": None,
        "engine_role": engine_role,
        "head_candidate_role_set": None,
        "need_bypass_balancer": False,
        "group_name": "",
        "dllm_blockwise_type": None,
        "dllm_blockwise_kwargs": None,
        "dense_prefetch_config": None,
        "tokenizer_group_mode": "process",
        "tokenizer_group_workers": 4,
        "disable_log_requests": True,
        "max_log_len": None,
        "new_requests_que_size": 128,
        "finished_requests_que_size": 1024,
        "detokenizer_group_mode": None,
        "detokenizer_group_workers": 1,
    }
    kwargs.update(dict(options or {}))
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=engine_kwargs.custom_built model_name=%s tokenizer_name=%s "
            "dtype=%s trust_remote_code=%s role_type=%s option_count=%s "
            "override_keys=%s summary=%s",
            Path(model_path).name,
            Path(tokenizer_path).name,
            resolved_dtype,
            trust_remote_code,
            _type_name(engine_role),
            len(kwargs),
            sorted(options),
            _engine_log_summary(kwargs),
        )
    return kwargs


def build_engine_args(async_engine_args_cls: Any, **kwargs: Any) -> Any:
    started = perf_counter()
    filtered = _filter_callable_kwargs(async_engine_args_cls, kwargs)
    skipped = sorted(set(kwargs).difference(filtered))
    if skipped:
        logger.warning(
            "event=engine_args.options_skipped class=%s skipped=%s",
            _safe_getattr(
                async_engine_args_cls,
                "__qualname__",
                _type_name(async_engine_args_cls),
            ),
            skipped,
        )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=engine_args.construct_begin class=%s provided=%s accepted=%s "
            "skipped=%s summary=%s",
            _safe_getattr(
                async_engine_args_cls,
                "__qualname__",
                _type_name(async_engine_args_cls),
            ),
            len(kwargs),
            len(filtered),
            skipped,
            _engine_log_summary(filtered),
        )
    result = async_engine_args_cls(**filtered)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=engine_args.construct_complete class=%s elapsed_ms=%.3f "
            "result_type=%s",
            _safe_getattr(
                async_engine_args_cls,
                "__qualname__",
                _type_name(async_engine_args_cls),
            ),
            (perf_counter() - started) * 1000.0,
            _type_name(result),
        )
    return result


def _filter_callable_kwargs(
    callable_obj: object, kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=callable_kwargs.signature_unavailable callable=%s provided=%s",
                _safe_getattr(
                    callable_obj, "__qualname__", _type_name(callable_obj)
                ),
                len(kwargs),
            )
        return dict(kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=callable_kwargs.accepts_var_keyword callable=%s provided=%s",
                _safe_getattr(
                    callable_obj, "__qualname__", _type_name(callable_obj)
                ),
                len(kwargs),
            )
        return dict(kwargs)
    supported = set(signature.parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=callable_kwargs.filtered callable=%s provided=%s accepted=%s "
            "skipped=%s",
            _safe_getattr(
                callable_obj, "__qualname__", _type_name(callable_obj)
            ),
            len(kwargs),
            len(filtered),
            sorted(set(kwargs).difference(filtered)),
        )
    return filtered


def _load_chat_template_tokenizer(
    *, tokenizer_path: str, trust_remote_code: bool
) -> Any:
    started = perf_counter()
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=tokenizer.load_begin tokenizer=%s trust_remote_code=%s",
            tokenizer_path,
            trust_remote_code,
        )
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "service_910b requires transformers to load the Router tokenizer"
        ) from exc
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=tokenizer.dependency version=%s auto_tokenizer=%s",
            _safe_getattr(transformers, "__version__", "<unknown>"),
            _safe_getattr(
                AutoTokenizer,
                "__qualname__",
                _type_name(AutoTokenizer),
            ),
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=bool(trust_remote_code)
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=tokenizer.load_complete tokenizer=%s elapsed_ms=%.3f "
            "tokenizer_type=%s vocab_size=%s eos_token_id=%s bos_token_id=%s "
            "pad_token_id=%s chat_template_state=%s",
            tokenizer_path,
            (perf_counter() - started) * 1000.0,
            _type_name(tokenizer),
            _safe_getattr(tokenizer, "vocab_size"),
            _token_ids_log_value([_safe_getattr(tokenizer, "eos_token_id")]),
            _token_ids_log_value([_safe_getattr(tokenizer, "bos_token_id")]),
            _token_ids_log_value([_safe_getattr(tokenizer, "pad_token_id")]),
            _safe_getattr(tokenizer, "chat_template", None) is not None,
        )
    return tokenizer


async def _start_custom_engine(
    *,
    engine: Any,
    health_check_interval: float,
    health_check_timeout: float | None,
) -> None:
    started = perf_counter()
    attempt = 0

    def emit(log_method: Any, message: str, *args: Any) -> None:
        log_method(message, *args)

    emit(
        logger.info,
        "event=engine.startup_begin engine_type=%s health_timeout=%s "
        "health_interval=%s",
        _type_name(engine),
        health_check_timeout,
        health_check_interval,
    )

    def remaining_seconds() -> float | None:
        if health_check_timeout is None:
            return None
        return float(health_check_timeout) - (perf_counter() - started)

    def require_time_remaining(phase: str) -> float | None:
        remaining = remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(
                f"custom vLLM engine startup timed out {phase}"
            )
        return remaining

    load_model = getattr(getattr(engine, "engine", None), "load_model", None)
    if callable(load_model):
        phase_started = perf_counter()
        emit(
            logger.info,
            "event=engine.load_model_begin engine_type=%s",
            _type_name(engine),
        )
        load_model()
        emit(
            logger.info,
            "event=engine.load_model_complete elapsed_ms=%.3f total_elapsed_ms=%.3f",
            (perf_counter() - phase_started) * 1000.0,
            (perf_counter() - started) * 1000.0,
        )
    else:
        emit(
            logger.warning,
            "event=engine.load_model_skipped engine_type=%s reason=not_callable",
            _type_name(engine),
        )
    require_time_remaining("while loading model")
    start_background_loop = getattr(engine, "start_background_loop", None)
    if callable(start_background_loop):
        phase_started = perf_counter()
        emit(
            logger.info,
            "event=engine.background_loop_begin total_elapsed_ms=%.3f",
            (perf_counter() - started) * 1000.0,
        )
        start_background_loop()
        emit(
            logger.info,
            "event=engine.background_loop_started elapsed_ms=%.3f "
            "total_elapsed_ms=%.3f",
            (perf_counter() - phase_started) * 1000.0,
            (perf_counter() - started) * 1000.0,
        )
    else:
        emit(
            logger.warning,
            "event=engine.background_loop_skipped reason=not_callable"
        )
    require_time_remaining("while starting the background loop")
    is_health = getattr(engine, "is_health", None)
    if not callable(is_health):
        emit(
            logger.warning,
            "event=engine.health_check_skipped reason=not_callable "
            "startup_elapsed_ms=%.3f",
            (perf_counter() - started) * 1000.0,
        )
        return
    while True:
        attempt += 1
        remaining = require_time_remaining("during health check")
        poll_started = perf_counter()
        emit(
            logger.info,
            "event=engine.health_poll_begin attempt=%s elapsed_ms=%.3f "
            "remaining_seconds=%s",
            attempt,
            (poll_started - started) * 1000.0,
            remaining,
        )
        try:
            healthy = (
                await is_health()
                if remaining is None
                else await asyncio.wait_for(is_health(), timeout=remaining)
            )
        except asyncio.TimeoutError as exc:
            emit(
                logger.error,
                "event=engine.health_poll_timeout attempt=%s poll_ms=%.3f "
                "total_elapsed_ms=%.3f timeout_seconds=%s",
                attempt,
                (perf_counter() - poll_started) * 1000.0,
                (perf_counter() - started) * 1000.0,
                health_check_timeout,
            )
            raise TimeoutError(
                "custom vLLM engine startup timed out during health check"
            ) from exc
        require_time_remaining("during health check")
        emit(
            logger.info,
            "event=engine.health_poll_complete attempt=%s healthy=%s "
            "poll_ms=%.3f total_elapsed_ms=%.3f",
            attempt,
            bool(healthy),
            (perf_counter() - poll_started) * 1000.0,
            (perf_counter() - started) * 1000.0,
        )
        if healthy:
            emit(
                logger.info,
                "event=engine.startup_complete attempts=%s elapsed_ms=%.3f",
                attempt,
                (perf_counter() - started) * 1000.0,
            )
            return
        sleep_seconds = max(0.1, float(health_check_interval))
        remaining = require_time_remaining("during health check")
        if remaining is not None:
            sleep_seconds = min(sleep_seconds, remaining)
        emit(
            logger.info,
            "event=engine.health_poll_sleep attempt=%s sleep_seconds=%.3f "
            "remaining_seconds=%s",
            attempt,
            sleep_seconds,
            remaining,
        )
        await asyncio.sleep(sleep_seconds)


def _completion_token_count_quiet(request_output: Any) -> int | None:
    """Read the first completion token count without affecting inference."""

    try:
        outputs = _output_field(request_output, "outputs")
        if (
            not isinstance(outputs, Sequence)
            or isinstance(outputs, (str, bytes, bytearray))
            or not outputs
        ):
            return None
        token_ids = _output_field(outputs[0], "token_ids")
        if (
            not isinstance(token_ids, Sequence)
            or isinstance(token_ids, (str, bytes, bytearray))
        ):
            return None
        return len(token_ids)
    except Exception:
        return None


def _finite_float_quiet(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed != parsed or abs(parsed) == float("inf"):
        return None
    return parsed


def _request_metric_seconds_quiet(request_output: Any, name: str) -> float | None:
    """Read an optional vLLM RequestMetrics value expressed in seconds."""

    try:
        metrics = _output_field(request_output, "metrics")
        if metrics is None:
            outputs = _output_field(request_output, "outputs")
            if (
                isinstance(outputs, Sequence)
                and not isinstance(outputs, (str, bytes, bytearray))
                and outputs
            ):
                metrics = _output_field(outputs[0], "metrics")
        if metrics is None:
            return None
        return _finite_float_quiet(_output_field(metrics, name))
    except Exception:
        return None


async def _generate_on_910b_loop(
    *,
    engine: Any,
    prompt: str,
    prompt_token_ids: Sequence[int],
    sampling_params: Any,
    request_id: str,
) -> Any:
    started = perf_counter()
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=engine.generate_begin request_id=%s engine_type=%s prompt_chars=%s "
            "prompt_tokens=%s prompt=%s token_ids=%s",
            request_id,
            _type_name(engine),
            len(prompt),
            len(prompt_token_ids),
            _text_log_value(prompt),
            _token_ids_log_value(prompt_token_ids),
        )
    results_generator = engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        prompt_token_ids=[int(token_id) for token_id in prompt_token_ids],
        tag=None,
        arrival_time=None,
        multi_modal_data=None,
        scheduler_result=None,
        is_stream=False,
    )
    logger.info(
        "event=engine.generate_iterator request_id=%s iterator_type=%s "
        "create_elapsed_ms=%.3f",
        request_id,
        _type_name(results_generator),
        (perf_counter() - started) * 1000.0,
    )
    final_output = None
    frame_count = 0
    first_frame_ms: float | None = None
    first_token_frame_ms: float | None = None
    first_token_frame_count: int | None = None
    try:
        async for request_output in results_generator:
            frame_count += 1
            if first_frame_ms is None:
                first_frame_ms = (perf_counter() - started) * 1000.0
            frame_token_count = _completion_token_count_quiet(request_output)
            if (
                first_token_frame_ms is None
                and frame_token_count is not None
                and frame_token_count > 0
            ):
                first_token_frame_ms = (perf_counter() - started) * 1000.0
                first_token_frame_count = frame_token_count
            final_output = request_output
            if logger.isEnabledFor(logging.INFO):
                try:
                    outputs = _output_field(request_output, "outputs")
                    output_count = (
                        len(outputs)
                        if isinstance(outputs, Sequence)
                        and not isinstance(outputs, (str, bytes, bytearray))
                        else None
                    )
                except Exception as exc:
                    output_count = f"<diagnostic-error {type(exc).__name__}>"
                logger.info(
                    "event=engine.generate_frame request_id=%s frame=%s "
                    "elapsed_ms=%.3f output_count=%s output_type=%s",
                    request_id,
                    frame_count,
                    (perf_counter() - started) * 1000.0,
                    output_count,
                    _type_name(request_output),
                )
    except asyncio.CancelledError:
        logger.warning(
            "event=engine.generate_cancelled request_id=%s elapsed_ms=%.3f "
            "frames=%s message=python_future_cancelled_engine_abort_not_confirmed",
            request_id,
            (perf_counter() - started) * 1000.0,
            frame_count,
        )
        raise
    except Exception as exc:
        logger.info(
            "event=engine.generate_failed request_id=%s elapsed_ms=%.3f "
            "frames=%s error_type=%s",
            request_id,
            (perf_counter() - started) * 1000.0,
            frame_count,
            type(exc).__name__,
            exc_info=True,
        )
        raise
    if final_output is None:
        logger.error(
            "event=engine.generate_empty request_id=%s elapsed_ms=%.3f frames=0",
            request_id,
            (perf_counter() - started) * 1000.0,
        )
        raise RuntimeError("custom vLLM returned no request outputs")
    total_ms = (perf_counter() - started) * 1000.0
    completion_tokens = _completion_token_count_quiet(final_output)
    arrival_time = _request_metric_seconds_quiet(final_output, "arrival_time")
    first_token_time = _request_metric_seconds_quiet(
        final_output, "first_token_time"
    )
    last_token_time = _request_metric_seconds_quiet(
        final_output, "last_token_time"
    )
    if last_token_time is None:
        last_token_time = _request_metric_seconds_quiet(
            final_output, "finished_time"
        )
    first_scheduled_time = _request_metric_seconds_quiet(
        final_output, "first_scheduled_time"
    )
    engine_queue_seconds = _request_metric_seconds_quiet(
        final_output, "time_in_queue"
    )
    if (
        engine_queue_seconds is None
        and arrival_time is not None
        and first_scheduled_time is not None
        and first_scheduled_time >= arrival_time
    ):
        engine_queue_seconds = first_scheduled_time - arrival_time

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    timing_source = "unavailable"
    if (
        arrival_time is not None
        and first_token_time is not None
        and first_token_time >= arrival_time
    ):
        ttft_ms = (first_token_time - arrival_time) * 1000.0
        timing_source = "request_output_metrics"
        if (
            completion_tokens is not None
            and completion_tokens > 1
            and last_token_time is not None
            and last_token_time >= first_token_time
        ):
            tpot_ms = (
                (last_token_time - first_token_time)
                * 1000.0
                / (completion_tokens - 1)
            )
    elif first_token_frame_ms is not None:
        # The SkillHub-compatible request is deliberately non-streaming.  A
        # custom engine may still yield intermediate RequestOutputs, but this
        # fallback is only a frame-level proxy rather than device-level TTFT.
        ttft_ms = first_token_frame_ms
        timing_source = "first_nonempty_frame_proxy"
        if (
            completion_tokens is not None
            and first_token_frame_count is not None
            and completion_tokens > first_token_frame_count
        ):
            tpot_ms = max(0.0, total_ms - first_token_frame_ms) / (
                completion_tokens - first_token_frame_count
            )

    scheduler_seconds = _request_metric_seconds_quiet(
        final_output, "scheduler_time"
    )
    model_forward_seconds = _request_metric_seconds_quiet(
        final_output, "model_forward_time"
    )
    model_execute_seconds = _request_metric_seconds_quiet(
        final_output, "model_execute_time"
    )
    logger.info(
        "event=engine.generate_complete request_id=%s elapsed_ms=%.3f "
        "first_frame_ms=%s first_token_frame_ms=%s first_token_frame_count=%s "
        "ttft_ms=%s tpot_ms=%s timing_source=%s completion_tokens=%s "
        "engine_queue_ms=%s scheduler_ms=%s model_forward_ms=%s "
        "model_execute_ms=%s frames=%s is_stream=False final_type=%s",
        request_id,
        total_ms,
        first_frame_ms,
        first_token_frame_ms,
        first_token_frame_count,
        ttft_ms,
        tpot_ms,
        timing_source,
        completion_tokens,
        None if engine_queue_seconds is None else engine_queue_seconds * 1000.0,
        None if scheduler_seconds is None else scheduler_seconds * 1000.0,
        None if model_forward_seconds is None else model_forward_seconds * 1000.0,
        None if model_execute_seconds is None else model_execute_seconds * 1000.0,
        frame_count,
        _type_name(final_output),
    )
    return final_output


def _build_910b_sampling_params(
    sampling_params_cls: Any,
    *,
    output_budget: int,
    num_levels: int,
    trie: MultiPathTokenTrie,
) -> Any:
    started = perf_counter()
    kwargs = {
        "temperature": 0.0,
        "max_tokens": int(output_budget),
        "min_tokens": int(num_levels),
        "detokenize": False,
        "skip_special_tokens": False,
        "logits_processors": [TrieLogitsProcessor(trie)],
    }
    if not _callable_accepts_keyword(
        sampling_params_cls, "logits_processors"
    ):
        raise RuntimeError(
            "the custom 910B vLLM SamplingParams does not expose "
            "request-level logits_processors"
        )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=sampling_params.construct_begin class=%s output_budget=%s "
            "num_levels=%s trie_paths=%s trie_max_paths=%s kwargs=%s",
            _safe_getattr(
                sampling_params_cls,
                "__qualname__",
                _type_name(sampling_params_cls),
            ),
            output_budget,
            num_levels,
            len(trie.paths),
            trie.max_paths,
            {
                key: (
                    len(value)
                    if key == "logits_processors" and isinstance(value, list)
                    else value
                )
                for key, value in kwargs.items()
            },
        )
    try:
        result = sampling_params_cls(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "the custom 910B vLLM SamplingParams must support LLMGen's "
            "request-level logits_processors"
        ) from exc
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=sampling_params.construct_complete class=%s elapsed_ms=%.3f "
            "result_type=%s",
            _safe_getattr(
                sampling_params_cls,
                "__qualname__",
                _type_name(sampling_params_cls),
            ),
            (perf_counter() - started) * 1000.0,
            _type_name(result),
        )
    return result


def _custom_engine_max_model_len(runtime: LocalVLLM910BRuntime) -> int | None:
    """Read the custom engine's resolved context limit when it exposes one."""

    engine = runtime.engine
    for owner in (
        _safe_getattr(engine, "llm_engine", None),
        _safe_getattr(engine, "engine", None),
        engine,
    ):
        if owner is None:
            continue
        for config_name in ("model_config", "engine_config", "vllm_config"):
            config = _safe_getattr(owner, config_name, None)
            nested = _safe_getattr(config, "model_config", config)
            raw_value = _safe_getattr(nested, "max_model_len", None)
            if (
                isinstance(raw_value, (int, float))
                and not isinstance(raw_value, bool)
                and int(raw_value) > 0
            ):
                logger.info(
                    "event=context_limit.runtime_resolved runtime_id=%s "
                    "owner_type=%s config_name=%s max_model_len=%s",
                    runtime.runtime_id,
                    _type_name(owner),
                    config_name,
                    int(raw_value),
                )
                return int(raw_value)
    configured = runtime.engine_kwargs.get("max_model_len")
    if (
        isinstance(configured, (int, float))
        and not isinstance(configured, bool)
        and int(configured) > 0
    ):
        logger.info(
            "event=context_limit.engine_args_fallback runtime_id=%s "
            "max_model_len=%s",
            runtime.runtime_id,
            int(configured),
        )
        return int(configured)
    logger.info(
        "event=context_limit.unavailable runtime_id=%s",
        runtime.runtime_id,
    )
    return None


def _callable_accepts_keyword(callable_obj: object, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=callable_keyword.signature_unavailable callable=%s keyword=%s "
                "assumed_supported=true",
                _safe_getattr(
                    callable_obj, "__qualname__", _type_name(callable_obj)
                ),
                name,
            )
        return True
    if name in signature.parameters:
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=callable_keyword.supported callable=%s keyword=%s "
                "support=explicit",
                _safe_getattr(
                    callable_obj, "__qualname__", _type_name(callable_obj)
                ),
                name,
            )
        return True
    supports_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=callable_keyword.checked callable=%s keyword=%s support=%s",
            _safe_getattr(
                callable_obj, "__qualname__", _type_name(callable_obj)
            ),
            name,
            "var_keyword" if supports_kwargs else "unsupported",
        )
    return supports_kwargs


def _first_completion_output(
    request_output: Any, *, request_id: str | None = None
) -> Any:
    outputs = _output_field(request_output, "outputs")
    if (
        not isinstance(outputs, Sequence)
        or isinstance(outputs, (str, bytes, bytearray))
        or not outputs
    ):
        logger.error(
            "event=completion.outputs_invalid request_output_type=%s "
            "outputs_type=%s",
            _type_name(request_output),
            _type_name(outputs),
        )
        raise RuntimeError("custom vLLM returned no Router completion outputs")
    if len(outputs) > 1:
        logger.warning(
            "event=completion.multiple_outputs count=%s selected_index=0",
            len(outputs),
        )
    first = outputs[0]
    if logger.isEnabledFor(logging.INFO):
        try:
            raw_token_ids = _output_field(first, "token_ids")
            token_count: Any = (
                len(raw_token_ids)
                if isinstance(raw_token_ids, Sequence)
                and not isinstance(raw_token_ids, (str, bytes, bytearray))
                else None
            )
            finish_reason: Any = _output_field(first, "finish_reason")
        except Exception as exc:
            token_count = f"<diagnostic-error {type(exc).__name__}>"
            finish_reason = token_count
        logger.info(
            "event=completion.selected request_id=%s outputs=%s selected_type=%s "
            "finish_reason=%s token_count=%s",
            request_id,
            len(outputs),
            _type_name(first),
            finish_reason,
            token_count,
        )
    return first


def _output_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _pop_float_optional(options: dict[str, Any], key: str) -> float | None:
    value = options.pop(key, None)
    if value is None or not str(value).strip():
        logger.info(
            "event=options.optional_float key=%s present=%s parsed=None",
            key,
            value is not None,
        )
        return None
    parsed = float(value)
    logger.info(
        "event=options.optional_float key=%s present=true parsed=%s",
        key,
        parsed,
    )
    return parsed


def _resolve_max_input_length(
    *,
    trained_max_length: int | None,
    engine_max_length: Any,
    output_budget: int,
) -> int:
    explicit = _env_int_optional("MAX_INPUT_LENGTH")
    limits = [
        int(value)
        for value in (trained_max_length, engine_max_length)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and int(value) > 0
    ]
    total_limit = min(limits) if limits else 1024
    available = total_limit - output_budget
    logger.info(
        "event=context_limit.resolve trained_max_length=%s engine_max_length=%s "
        "valid_limits=%s selected_total_limit=%s output_budget=%s available=%s "
        "explicit=%s",
        trained_max_length,
        engine_max_length,
        limits,
        total_limit,
        output_budget,
        available,
        explicit,
    )
    if available < 1:
        raise ServiceConfigurationError(
            "model context length leaves no room for a Router prompt"
        )
    if explicit is None:
        logger.info(
            "event=context_limit.resolved source=automatic max_input_length=%s",
            available,
        )
        return available
    if explicit < 1 or explicit > available:
        raise ServiceConfigurationError(
            f"MAX_INPUT_LENGTH must be between 1 and {available}"
        )
    logger.info(
        "event=context_limit.resolved source=environment max_input_length=%s",
        explicit,
    )
    return explicit


def _load_router_settings(model_dir: Path) -> RouterManifestSettings:
    started = perf_counter()
    manifest_path = model_dir / _ROUTER_MANIFEST_FILENAME
    logger.info(
        "event=manifest.load_begin path=%s exists=%s",
        manifest_path,
        manifest_path.is_file(),
    )
    if not manifest_path.is_file():
        raise ServiceConfigurationError(
            f"Router model is missing required prompt manifest: {manifest_path}"
        )
    manifest = _read_json_object(manifest_path)

    raw_max_length = manifest.get("max_length")
    max_length = (
        raw_max_length
        if isinstance(raw_max_length, int)
        and not isinstance(raw_max_length, bool)
        and raw_max_length > 0
        else None
    )
    raw_system_prompt = manifest.get("system_prompt")
    if not isinstance(raw_system_prompt, str):
        raise ServiceConfigurationError(
            "Router manifest has no valid training system_prompt"
        )
    settings = RouterManifestSettings(
        max_length=max_length,
        system_prompt=raw_system_prompt,
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=manifest.load_complete path=%s elapsed_ms=%.3f schema_version=%s "
            "phase=%s max_length=%s system_prompt_chars=%s system_prompt=%s keys=%s",
            manifest_path,
            (perf_counter() - started) * 1000.0,
            manifest.get("schema_version"),
            manifest.get("phase"),
            settings.max_length,
            len(settings.system_prompt),
            _text_log_value(settings.system_prompt),
            sorted(manifest),
        )
    return settings


def _code_token_id_map(tokenizer: Any, tokens: Sequence[str]) -> dict[str, int]:
    started = perf_counter()
    logger.info(
        "event=tokenizer.virtual_tokens_begin tokenizer_type=%s token_count=%s",
        _type_name(tokenizer),
        len(tokens),
    )
    mapping: dict[str, int] = {}
    used_ids: dict[int, str] = {}
    for token in tokens:
        token_started = perf_counter()
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "event=tokenizer.virtual_token_encoded token=%s token_ids=%s "
                "elapsed_ms=%.3f",
                _safe_log_value("virtual_token_text", token),
                _token_ids_log_value(token_ids),
                (perf_counter() - token_started) * 1000.0,
            )
        if len(token_ids) != 1:
            raise ServiceConfigurationError(
                f"hierarchical token {token!r} is not atomic for the Router tokenizer"
            )
        token_id = int(token_ids[0])
        if token_id in used_ids and used_ids[token_id] != token:
            raise ServiceConfigurationError(
                f"virtual tokens {used_ids[token_id]!r} and {token!r} share an ID"
            )
        mapping[token] = token_id
        used_ids[token_id] = token
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=tokenizer.virtual_tokens_complete elapsed_ms=%.3f token_count=%s "
            "token_ids=%s",
            (perf_counter() - started) * 1000.0,
            len(mapping),
            _token_ids_log_value(sorted(mapping.values())),
        )
    return mapping


def _render_router_prompt(tokenizer: Any, query: str, system_prompt: str) -> str:
    started = perf_counter()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query.strip()})
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if getattr(tokenizer, "chat_template", None) and callable(apply_template):
        logger.info(
            "event=prompt.render_begin branch=chat_template tokenizer_type=%s "
            "messages=%s query_chars=%s system_prompt_chars=%s",
            _type_name(tokenizer),
            len(messages),
            len(query),
            len(system_prompt),
        )
        try:
            prompt = apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            logger.info(
                "event=prompt.render_complete branch=chat_template_thinking_disabled "
                "elapsed_ms=%.3f prompt_chars=%s prompt=%s",
                (perf_counter() - started) * 1000.0,
                len(prompt),
                _text_log_value(prompt),
            )
            return prompt
        except TypeError:
            logger.info(
                "event=prompt.render_retry branch=chat_template_legacy "
                "reason=type_error_from_enable_thinking",
                exc_info=True,
            )
            prompt = apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.info(
                "event=prompt.render_complete branch=chat_template_legacy "
                "elapsed_ms=%.3f prompt_chars=%s prompt=%s",
                (perf_counter() - started) * 1000.0,
                len(prompt),
                _text_log_value(prompt),
            )
            return prompt
    system = f"System: {system_prompt.strip()}\n" if system_prompt else ""
    prompt = f"{system}User: {query.strip()}\nAssistant:"
    logger.info(
        "event=prompt.render_complete branch=plain_fallback tokenizer_type=%s "
        "elapsed_ms=%.3f prompt_chars=%s prompt=%s query_chars=%s "
        "system_prompt_chars=%s",
        _type_name(tokenizer),
        (perf_counter() - started) * 1000.0,
        len(prompt),
        _text_log_value(prompt),
        len(query),
        len(system_prompt),
    )
    return prompt


def _shutdown_vllm(llm: Any | None) -> None:
    if llm is None:
        logger.info("event=shutdown.skipped reason=runtime_is_none")
        return
    started = perf_counter()
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=shutdown.begin runtime_type=%s runtime_id=%s",
            _type_name(llm),
            _safe_getattr(llm, "runtime_id", None),
        )
    targets = (
        llm,
        getattr(llm, "llm_engine", None),
        getattr(getattr(llm, "llm_engine", None), "model_executor", None),
    )
    for target_index, target in enumerate(targets):
        if target is None:
            logger.info(
                "event=shutdown.target_skipped target_index=%s reason=none",
                target_index,
            )
            continue
        for method_name in ("shutdown", "close"):
            method = getattr(target, method_name, None)
            if callable(method):
                operation_started = perf_counter()
                logger.info(
                    "event=shutdown.operation_begin target_index=%s "
                    "target_type=%s method=%s",
                    target_index,
                    _type_name(target),
                    method_name,
                )
                try:
                    method()
                except Exception:
                    logger.exception(
                        "event=shutdown.operation_failed target_index=%s "
                        "target_type=%s method=%s elapsed_ms=%.3f",
                        target_index,
                        _type_name(target),
                        method_name,
                        (perf_counter() - operation_started) * 1000.0,
                    )
                    continue
                logger.info(
                    "event=shutdown.complete target_index=%s target_type=%s "
                    "method=%s operation_ms=%.3f total_ms=%.3f",
                    target_index,
                    _type_name(target),
                    method_name,
                    (perf_counter() - operation_started) * 1000.0,
                    (perf_counter() - started) * 1000.0,
                )
                return
            logger.info(
                "event=shutdown.method_skipped target_index=%s target_type=%s "
                "method=%s reason=not_callable",
                target_index,
                _type_name(target),
                method_name,
            )
    logger.warning(
        "event=shutdown.no_supported_method runtime_type=%s elapsed_ms=%.3f",
        _type_name(llm),
        (perf_counter() - started) * 1000.0,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    started = perf_counter()
    if logger.isEnabledFor(logging.INFO):
        try:
            is_file = path.is_file()
            byte_count: Any = path.stat().st_size if is_file else None
        except OSError as exc:
            is_file = f"<diagnostic-error {type(exc).__name__}>"
            byte_count = is_file
        logger.info(
            "event=json.read_begin path=%s exists=%s bytes=%s",
            path,
            is_file,
            byte_count,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ServiceConfigurationError(f"expected a JSON object: {path}")
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "event=json.read_complete path=%s elapsed_ms=%.3f keys=%s",
            path,
            (perf_counter() - started) * 1000.0,
            sorted(payload),
        )
    return payload


def _load_mock_responses() -> dict[str, tuple[str, ...]]:
    started = perf_counter()
    raw_payload = os.environ.get("MOCK_RESPONSES_JSON")
    if raw_payload is None or not raw_payload.strip():
        responses = {"*": _DEFAULT_MOCK_RESPONSES}
        logger.info(
            "event=mock_responses.loaded source=default elapsed_ms=%.3f "
            "queries=%s total_results=%s",
            (perf_counter() - started) * 1000.0,
            len(responses),
            sum(len(names) for names in responses.values()),
        )
        return responses
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(
            "MOCK_RESPONSES_JSON must be valid JSON"
        ) from exc

    if isinstance(payload, list):
        responses = {"*": _normalize_mock_names(payload, "mock fallback")}
        logger.info(
            "event=mock_responses.loaded source=list elapsed_ms=%.3f queries=%s "
            "total_results=%s",
            (perf_counter() - started) * 1000.0,
            len(responses),
            sum(len(names) for names in responses.values()),
        )
        return responses
    if not isinstance(payload, dict):
        raise ServiceConfigurationError(
            "MOCK_RESPONSES_JSON must be a result list or query-to-results object"
        )

    responses: dict[str, tuple[str, ...]] = {}
    for raw_query, raw_names in payload.items():
        query = str(raw_query).strip()
        if not query:
            raise ServiceConfigurationError("mock response query must be non-empty")
        if query in responses:
            raise ServiceConfigurationError(
                f"MOCK_RESPONSES_JSON contains duplicate query {query!r}"
            )
        responses[query] = _normalize_mock_names(
            raw_names, f"mock response for {query!r}"
        )
    logger.info(
        "event=mock_responses.loaded source=mapping elapsed_ms=%.3f queries=%s "
        "total_results=%s has_fallback=%s",
        (perf_counter() - started) * 1000.0,
        len(responses),
        sum(len(names) for names in responses.values()),
        "*" in responses,
    )
    return responses


def _normalize_mock_names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ServiceConfigurationError(f"{label} must be a JSON list")
    names: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ServiceConfigurationError(
                f"{label} must contain only non-empty strings"
            )
        name = item.strip()
        if name not in seen:
            names.append(name)
            seen.add(name)
    logger.info(
        "event=mock_responses.normalized label=%s input_count=%s output_count=%s "
        "duplicates_removed=%s",
        _text_log_value(label),
        len(value),
        len(names),
        len(value) - len(names),
    )
    return tuple(names)


def _env_first_text(names: Sequence[str], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(default)


def _env_text(name: str, default: str) -> str:
    return _env_first_text((name,), default)


def _vllm_dtype() -> str:
    return _env_first_text(
        ("GENERATION_DTYPE", "VLLM_DTYPE", "DTYPE"), _VLLM_DTYPE
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default) if value is None or not value.strip() else int(value)


def _env_int_optional(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value is None or not value.strip() else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default) if value is None or not value.strip() else float(value)


def _env_float_optional(name: str) -> float | None:
    value = os.environ.get(name)
    return None if value is None or not value.strip() else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ServiceConfigurationError(f"{name} must be a boolean value")


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or not str(value).strip():
        return None
    return int(value)


if __name__ == "__main__":
    # Third-party/root records also use stdout when this file is run directly.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    service = RetriverTest()
    try:
        service.load()
        print(service.calc({"data": {"query": _DEFAULT_QUERY}}))
    finally:
        service.close()
