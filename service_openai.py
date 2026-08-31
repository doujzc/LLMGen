#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-contained HTTP service for an exported LLMGen router.

The hosting contract intentionally matches the reference retrieval service:

* ``load()`` starts ``python -m vllm.entrypoints.api_server`` on an
  automatically selected loopback port, then initializes the local tokenizer.
* ``calc({"data": {"query": ..., "top_k": ...}})`` returns a JSON string
  containing a list of Skill names.

The model directory must be a complete LLMGen Router bundle containing the
Hugging Face model/tokenizer files, ``skill_decode_map.json``,
``virtual_tokens.txt``, and ``router_manifest.json``. Both ``--model`` and
``--tokenizer`` point to this same directory, resolved through the existing
``MODEL_SFS``/``MODEL_OBJECT_ID`` or ``MODEL_PATH`` deployment contract.

``load()`` calls ``load_vllm_model()`` to start the child server, waits for
``/health``, and owns that process until ``close()``. Set
``VLLM_SERVER_PYTHON`` to launch the server from a different Python
environment. Common deployment options use ``VLLM_*`` environment variables;
additional CLI arguments can be supplied as a JSON string list in
``VLLM_SERVER_ARGS_JSON``. ``VLLM_SERVER_PORT`` is an optional preferred port;
when it is unset, zero, or occupied, the service chooses another free port.

Inference uses the server's ``POST /generate`` endpoint. The simple API-server
protocol cannot transport an in-process Python Trie callback, so generated
token IDs are validated against the same LLMGen Trie before Skill mapping. If
an invalid suffix follows one or more complete registered paths, the longest
complete prefix is retained. Deployments whose custom endpoint supports
additional constrained decoding fields can add them through
``VLLM_GENERATE_KWARGS_JSON``.

For local contract tests, set ``MOCK_MODE=1``.  This skips model and candidate
artifact loading.  ``MOCK_RESPONSES_JSON`` may be either one result list used
for every query or an object mapping exact queries (plus optional ``"*"``
fallback) to result lists.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
from time import perf_counter, sleep
from typing import Any, Dict, Iterable, Mapping, Sequence
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_LOG_MARKER = "[[LLMGEN-OPENAI]]"
_LOG_FILTER_TAG = "_llmgen_service_openai_marker_filter"
_STDOUT_HANDLER_TAG = "_llmgen_service_openai_stdout_handler"


class _ServiceLogMarker(logging.Filter):
    """Keep this service's records recognizable in shared stdout logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = str(record.msg)
        if not message.startswith(_LOG_MARKER):
            record.msg = f"{_LOG_MARKER} {message}"
        return True


logger = logging.getLogger("web_demo.service_openai")


def _configure_service_logging() -> None:
    """Write service diagnostics directly to stdout without duplication."""

    for existing_filter in tuple(logger.filters):
        if getattr(existing_filter, _LOG_FILTER_TAG, False):
            logger.removeFilter(existing_filter)
    marker_filter = _ServiceLogMarker()
    setattr(marker_filter, _LOG_FILTER_TAG, True)
    logger.addFilter(marker_filter)

    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
    stdout_handler = logging.StreamHandler(sys.stdout)
    setattr(stdout_handler, _STDOUT_HANDLER_TAG, True)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stdout_handler)
    logger.propagate = False
    logger.disabled = False
    configured_level = os.environ.get("SERVICE_OPENAI_LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, configured_level.strip().upper(), logging.INFO))


_configure_service_logging()

_VLLM_TENSOR_PARALLEL_SIZE = 1
_VLLM_DTYPE = "bfloat16"
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
_DEFAULT_VLLM_SERVER_HOST = "127.0.0.1"
_DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS = 900.0
_DEFAULT_VLLM_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_DEFAULT_VLLM_REQUEST_TIMEOUT_SECONDS = 300.0
_DEFAULT_MODEL_OUTPUT_PREVIEW_CHARS = 4000
_DEFAULT_MODEL_OUTPUT_TOKEN_ITEMS = 256
_VLLM_SERVER_SUPPORTED_KWARGS = frozenset(
    {
        "tokenizer_mode",
        "trust_remote_code",
        "dtype",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "gpu_memory_utilization",
        "max_num_seqs",
        "seed",
        "swap_space",
        "disable_log_stats",
        "max_model_len",
        "download_dir",
        "scheduler_budget_len",
        "max_num_batched_tokens",
        "first_token_timeout",
        "max_log_len",
        "block_size",
        "decode_tensor_parallel_size",
        "disable_log_requests",
    }
)
_VLLM_INTEGER_SERVER_OPTIONS = frozenset(
    {
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "max_num_seqs",
        "seed",
        "max_model_len",
        "scheduler_budget_len",
        "max_num_batched_tokens",
        "first_token_timeout",
        "max_log_len",
        "block_size",
        "decode_tensor_parallel_size",
    }
)


class ServiceConfigurationError(RuntimeError):
    """Raised when deployment artifacts or environment settings disagree."""


class ServiceRuntimeUnavailableError(RuntimeError):
    """Raised when the owned local vLLM server can no longer serve requests."""


class VllmServerHandle:
    """Owned vLLM simple API-server process with idempotent shutdown."""

    def __init__(
        self,
        process: subprocess.Popen[Any],
        *,
        origin: str,
        host: str,
        port: int,
        process_group: bool,
    ) -> None:
        self.process = process
        self.origin = origin.rstrip("/")
        self.host = host
        self.port = int(port)
        self.generate_url = self.origin + "/generate"
        self.health_url = self.origin + "/health"
        self.process_group = process_group
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            stopped = _terminate_vllm_process(
                self.process,
                process_group=self.process_group,
            )
            if stopped:
                self._closed = True


class VllmGenerateClient:
    """Small standard-library client for vLLM's simple ``/generate`` API."""

    def __init__(self, generate_url: str, *, timeout: float) -> None:
        if timeout <= 0:
            raise ServiceConfigurationError(
                "VLLM_REQUEST_TIMEOUT_SECONDS must be positive"
            )
        self.generate_url = str(generate_url)
        self.timeout = float(timeout)

    def generate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            self.generate_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read()
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM /generate returned HTTP {exc.code}: {details[:1000]}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"cannot call local vLLM /generate endpoint: {exc}"
            ) from exc
        try:
            result = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("vLLM /generate returned invalid JSON") from exc
        if not isinstance(result, Mapping):
            raise RuntimeError("vLLM /generate response is not a JSON object")
        return result

    def close(self) -> None:
        """Match the service lifecycle; urllib keeps no owned session."""


@dataclass(frozen=True)
class TrieParseResult:
    """Result of one strict-or-recoverable token grammar scan."""

    paths: tuple[tuple[int, ...], ...]
    recovered: bool
    consumed_tokens: int
    discarded_tokens: int
    reason: str | None = None


@dataclass(frozen=True)
class _TrieScanState:
    completed: tuple[tuple[int, ...], ...]
    prefix: tuple[int, ...]
    mode: str
    separator_offset: int
    invalid_reason: str | None
    last_complete_token_count: int


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

        # Build immutable prefix indexes once during load. Request-time parsing
        # then performs O(generated tokens) dictionary lookups instead of
        # scanning every candidate path for every generated token.
        paths_by_prefix: dict[
            tuple[int, ...], set[tuple[int, ...]]
        ] = {}
        next_tokens_by_prefix: dict[tuple[int, ...], set[int]] = {}
        for path in self.paths:
            for depth in range(self.num_levels + 1):
                prefix = path[:depth]
                paths_by_prefix.setdefault(prefix, set()).add(path)
                if depth < self.num_levels:
                    next_tokens_by_prefix.setdefault(prefix, set()).add(path[depth])
        self._paths_by_prefix = {
            prefix: frozenset(descendants)
            for prefix, descendants in paths_by_prefix.items()
        }
        self._next_tokens_by_prefix = {
            prefix: tuple(sorted(token_ids))
            for prefix, token_ids in next_tokens_by_prefix.items()
        }

    def _state(
        self, generated: Sequence[int]
    ) -> _TrieScanState:
        completed: list[tuple[int, ...]] = []
        completed_set: set[tuple[int, ...]] = set()
        prefix: list[int] = []
        mode = "path"
        separator_offset = 0
        invalid_reason: str | None = None
        last_complete_token_count = 0
        for token_offset, raw_token_id in enumerate(generated):
            token_id = int(raw_token_id)
            if token_id == self.eos_token_id:
                invalid_reason = "unexpected_eos"
                break
            if mode == "path":
                prefix.append(token_id)
                current_prefix = tuple(prefix)
                descendants = self._paths_by_prefix.get(current_prefix)
                if descendants is None or descendants.issubset(completed_set):
                    invalid_reason = "invalid_or_duplicate_path_prefix"
                    break
                if len(prefix) == self.num_levels:
                    completed.append(current_prefix)
                    completed_set.add(current_prefix)
                    last_complete_token_count = token_offset + 1
                    prefix = []
                    mode = "boundary"
            elif mode == "boundary":
                if len(completed) >= self.max_paths:
                    invalid_reason = "token_after_max_paths"
                    break
                if token_id != self.separator_token_ids[0]:
                    invalid_reason = "invalid_path_boundary"
                    break
                if len(self.separator_token_ids) == 1:
                    mode = "path"
                else:
                    mode = "separator"
                    separator_offset = 1
            else:
                if token_id != self.separator_token_ids[separator_offset]:
                    invalid_reason = "invalid_path_separator"
                    break
                separator_offset += 1
                if separator_offset == len(self.separator_token_ids):
                    mode = "path"
        return _TrieScanState(
            completed=tuple(completed),
            prefix=tuple(prefix),
            mode=mode,
            separator_offset=separator_offset,
            invalid_reason=invalid_reason,
            last_complete_token_count=last_complete_token_count,
        )

    @staticmethod
    def _incomplete_reason(state: _TrieScanState) -> str:
        if state.invalid_reason is not None:
            return state.invalid_reason
        if state.mode == "separator":
            return "incomplete_path_separator"
        if state.mode == "path":
            return (
                "incomplete_path"
                if state.prefix
                else "missing_path_after_separator"
            )
        return "empty_generation"

    def allowed_next(self, generated: Sequence[int]) -> tuple[int, ...]:
        state = self._state(generated)
        if state.invalid_reason is not None:
            return ()
        if state.mode == "separator":
            return (self.separator_token_ids[state.separator_offset],)
        if state.mode == "boundary":
            if (
                len(state.completed) >= self.max_paths
                or len(state.completed) >= len(self.paths)
            ):
                return (self.eos_token_id,)
            return tuple(sorted({self.eos_token_id, self.separator_token_ids[0]}))

        if len(state.prefix) >= self.num_levels:
            return ()
        completed_set = set(state.completed)
        allowed: list[int] = []
        for token_id in self._next_tokens_by_prefix.get(state.prefix, ()):
            descendants = self._paths_by_prefix[state.prefix + (token_id,)]
            if not descendants.issubset(completed_set):
                allowed.append(token_id)
        return tuple(allowed)

    def parse_with_recovery(self, generated: Sequence[int]) -> TrieParseResult:
        """Parse once, falling back to the longest complete registered prefix."""

        state = self._state(generated)
        generated_count = len(generated)
        if (
            state.invalid_reason is None
            and state.completed
            and len(state.completed) <= self.max_paths
            and not state.prefix
            and state.mode == "boundary"
        ):
            return TrieParseResult(
                paths=state.completed,
                recovered=False,
                consumed_tokens=generated_count,
                discarded_tokens=0,
            )
        reason = self._incomplete_reason(state)
        if not state.completed:
            return TrieParseResult(
                paths=(),
                recovered=False,
                consumed_tokens=0,
                discarded_tokens=generated_count,
                reason=reason,
            )
        return TrieParseResult(
            paths=state.completed,
            recovered=True,
            consumed_tokens=state.last_complete_token_count,
            discarded_tokens=max(
                0, generated_count - state.last_complete_token_count
            ),
            reason=reason,
        )

    def parse_complete(self, generated: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        state = self._state(generated)
        if state.invalid_reason is not None:
            raise RuntimeError("vLLM generated an invalid code sequence")
        if (
            not state.completed
            or len(state.completed) > self.max_paths
            or state.prefix
            or state.mode != "boundary"
        ):
            raise RuntimeError("vLLM generation did not end at a code-path boundary")
        return state.completed


class TrieLogitsProcessor:
    """vLLM 0.8 V0 request-level logits processor for active Skill paths."""

    def __init__(self, trie: MultiPathTokenTrie) -> None:
        self.trie = trie

    def clone(self) -> "TrieLogitsProcessor":
        # The processor is stateless; generated IDs are supplied by vLLM.
        return self

    def __call__(self, output_token_ids: list[int], scores: Any) -> Any:
        allowed = self.trie.allowed_next(output_token_ids)
        if not allowed:
            raise RuntimeError(
                f"generation reached an invalid code prefix: {output_token_ids!r}"
            )
        vocabulary_size = int(scores.shape[-1])
        if any(token_id >= vocabulary_size for token_id in allowed):
            raise RuntimeError("candidate path contains a token outside model vocabulary")
        masked = scores.new_full(scores.shape, -float("inf"))
        indices = list(allowed)
        masked[indices] = scores[indices]
        return masked


def create_trie_logits_processor(
    *,
    paths: Sequence[Sequence[int]],
    eos_token_id: int,
    separator_token_ids: Sequence[int],
    max_paths: int,
) -> TrieLogitsProcessor:
    """Build the JSON-constructible processor used by the vLLM API server."""

    return TrieLogitsProcessor(
        MultiPathTokenTrie(
            paths,
            eos_token_id=eos_token_id,
            separator_token_ids=separator_token_ids,
            max_paths=max_paths,
        )
    )


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
    decode_path = directory / _DECODE_MAP_FILENAME
    token_path = directory / _VIRTUAL_TOKENS_FILENAME
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
    return CandidateBundle(
        decode_map=decode_map,
        virtual_tokens=file_tokens,
        skills=normalized_skills,
        token_paths=token_paths,
        num_levels=raw_levels,
    )


class RetriverTest:
    """Skill retrieval service backed by an owned vLLM API server or a mock."""

    def __init__(self) -> None:
        _configure_service_logging()
        self.instance_id = uuid.uuid4().hex[:12]
        self.vllm_server: VllmServerHandle | None = None
        self.vllm_client: VllmGenerateClient | None = None
        self.tokenizer: Any | None = None
        self.bundle: CandidateBundle | None = None
        self.trie: MultiPathTokenTrie | None = None
        self.path_skill_ids: dict[tuple[int, ...], tuple[str, ...]] = {}
        self.model_path = ""
        self.tokenizer_path = ""
        self.skill_index_path = ""
        # Retain the misspelled attribute used by the reference service.
        self.skill_indes_path = ""
        self.served_model_name = ""
        self.backend = "vllm_http"
        self.mock_responses: dict[str, tuple[str, ...]] = {}
        self.default_top_k = 2
        self.max_code_paths = 2
        self.max_input_length = 1
        self.output_budget = 1
        self.system_prompt = _DEFAULT_SYSTEM_PROMPT
        self._load_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._loaded = False
        logger.info(
            "event=service.created instance_id=%s backend=%s",
            self.instance_id,
            self.backend,
        )

    def load(self) -> None:
        _configure_service_logging()
        load_id = uuid.uuid4().hex[:12]
        load_wait_started = perf_counter()
        logger.info(
            "event=service.load_lock_wait instance_id=%s load_id=%s loaded=%s "
            "backend=%s",
            self.instance_id,
            load_id,
            self._loaded,
            self.backend,
        )
        with self._load_lock:
            logger.info(
                "event=service.load_lock_acquired instance_id=%s load_id=%s "
                "wait_ms=%.3f",
                self.instance_id,
                load_id,
                (perf_counter() - load_wait_started) * 1000.0,
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
            if self.vllm_server is not None:
                self.vllm_server.close()
                if not self.vllm_server.closed:
                    raise RuntimeError(
                        "cannot restart while the previous vLLM server is still running"
                    )
                self.vllm_server = None

            started = perf_counter()
            logger.info(
                "event=service.load_begin instance_id=%s load_id=%s backend=%s",
                self.instance_id,
                load_id,
                self.backend,
            )
            current_dir = Path(__file__).resolve().parent
            self.default_top_k = _env_int("TOP_K", 2)
            self.max_code_paths = _env_int(
                "MAX_CODE_PATHS", max(1, self.default_top_k)
            )
            if self.default_top_k < 1:
                raise ServiceConfigurationError("TOP_K must be positive")
            if self.max_code_paths < 1:
                raise ServiceConfigurationError("MAX_CODE_PATHS must be positive")

            if _env_bool("MOCK_MODE", False):
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
                return

            model_dir = _resolve_model_directory(current_dir)
            # The simple vLLM server and local protocol tokenizer must use the
            # exact tokenizer bundled with the resolved model directory.
            tokenizer_dir = model_dir
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
            for label, path in (
                ("model", model_dir),
                ("tokenizer", tokenizer_dir),
                ("candidate state", candidate_dir),
            ):
                if not path.is_dir():
                    raise ServiceConfigurationError(
                        f"{label} directory does not exist: {path}"
                    )
            _validate_full_model_bundle(model_dir)

            try:
                self.bundle = _load_candidate_bundle(candidate_dir)
                manifest_settings = _load_router_settings(model_dir)
                self.system_prompt = _env_text(
                    "SYSTEM_PROMPT", manifest_settings.system_prompt
                )
                logger.info(
                    "event=service.paths_resolved instance_id=%s load_id=%s "
                    "model=%s tokenizer=%s candidate=%s served_model=%s",
                    self.instance_id,
                    load_id,
                    model_dir,
                    tokenizer_dir,
                    candidate_dir,
                    self.served_model_name,
                )
                self.tokenizer = _load_tokenizer(tokenizer_dir)

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
                self.output_budget = (
                    self.trie.max_paths * self.bundle.num_levels
                    + (self.trie.max_paths - 1) * len(separator_ids)
                )
                configured_vllm_kwargs = _build_vllm_kwargs(
                    model_path=model_dir,
                    tokenizer_path=tokenizer_dir,
                )
                self.max_input_length = _resolve_max_input_length(
                    trained_max_length=manifest_settings.max_length,
                    engine_max_length=configured_vllm_kwargs.get(
                        "max_model_len"
                    ),
                    output_budget=self.output_budget,
                )
                self.vllm_server = load_vllm_model(
                    model_dir,
                    tokenizer_dir,
                    served_model_name=self.served_model_name,
                )
                self.vllm_client = VllmGenerateClient(
                    self.vllm_server.generate_url,
                    timeout=_env_float(
                        "VLLM_REQUEST_TIMEOUT_SECONDS",
                        _DEFAULT_VLLM_REQUEST_TIMEOUT_SECONDS,
                    ),
                )
                logger.warning(
                    "event=service.constraint_mode instance_id=%s load_id=%s "
                    "mode=post_validation message="
                    "vllm.entrypoints.api_server uses HTTP post-validation; "
                    "request-level Trie enforcement requires a custom field in "
                    "VLLM_GENERATE_KWARGS_JSON supported by the target image",
                    self.instance_id,
                    load_id,
                )
                logger.info(
                    "event=service.model_output_logging instance_id=%s "
                    "load_id=%s raw_output_enabled=%s token_ids_enabled=%s "
                    "preview_chars=%s token_items=%s",
                    self.instance_id,
                    load_id,
                    _env_bool("SERVICE_OPENAI_LOG_MODEL_OUTPUT", False),
                    _env_bool("SERVICE_OPENAI_LOG_TOKEN_IDS", False),
                    _debug_log_positive_int(
                        "SERVICE_OPENAI_LOG_PREVIEW_CHARS",
                        _DEFAULT_MODEL_OUTPUT_PREVIEW_CHARS,
                    ),
                    _debug_log_positive_int(
                        "SERVICE_OPENAI_LOG_TOKEN_ITEMS",
                        _DEFAULT_MODEL_OUTPUT_TOKEN_ITEMS,
                    ),
                )
                self.backend = "vllm_http"
                self._loaded = True
                logger.info(
                    "event=service.load_complete instance_id=%s load_id=%s "
                    "backend=vllm_http elapsed_ms=%.3f skills=%s paths=%s "
                    "levels=%s max_input_length=%s endpoint=%s",
                    self.instance_id,
                    load_id,
                    (perf_counter() - started) * 1000.0,
                    len(self.bundle.skills),
                    len(self.path_skill_ids),
                    self.bundle.num_levels,
                    self.max_input_length,
                    self.vllm_server.generate_url,
                )
            except BaseException:
                logger.exception(
                    "event=service.load_failed instance_id=%s load_id=%s "
                    "elapsed_ms=%.3f",
                    self.instance_id,
                    load_id,
                    (perf_counter() - started) * 1000.0,
                )
                self._cleanup_after_failed_load()
                raise

    def calc(self, req_data: Mapping[str, Any] | None) -> str:
        _configure_service_logging()
        request_id = str(uuid.uuid4())
        total_started = perf_counter()
        try:
            container = req_data if isinstance(req_data, Mapping) else {}
            data = container.get("data", {})
            request = dict(data) if isinstance(data, Mapping) else {}
            query = str(request.get("query", _DEFAULT_QUERY)).strip()
        except MemoryError:
            raise
        except Exception as exc:
            logger.error(
                "event=service.calc_recovered instance_id=%s request_id=%s "
                "phase=request_parse recoverable=True elapsed_ms=%.3f "
                "error_type=%s response=[]",
                self.instance_id,
                request_id,
                (perf_counter() - total_started) * 1000.0,
                type(exc).__name__,
            )
            return json.dumps([], ensure_ascii=False)
        logger.info(
            "event=service.calc_begin instance_id=%s request_id=%s loaded=%s "
            "backend=%s query_chars=%s top_k_raw=%s topk_raw=%s",
            self.instance_id,
            request_id,
            self._loaded,
            self.backend,
            len(query),
            request.get("top_k"),
            request.get("topk"),
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
        load_wait_started = perf_counter()
        try:
            with self._load_lock:
                load_lock_wait_ms = (
                    perf_counter() - load_wait_started
                ) * 1000.0
                if not self._loaded:
                    phase = "automatic_load"
                    self.load()

                phase = "resolve_top_k"
                requested_top_k = _coerce_optional_int(request.get("top_k"))
                if requested_top_k is None:
                    requested_top_k = _coerce_optional_int(request.get("topk"))
                resolved_top_k = (
                    self.default_top_k
                    if requested_top_k is None
                    else max(1, requested_top_k)
                )
                if resolved_top_k > self.default_top_k:
                    logger.warning(
                        "event=service.calc_top_k_clamped instance_id=%s "
                        "request_id=%s requested=%s initialized=%s",
                        self.instance_id,
                        request_id,
                        resolved_top_k,
                        self.default_top_k,
                    )
                    resolved_top_k = self.default_top_k

                phase = "inference_lock"
                inference_wait_started = perf_counter()
                with self._inference_lock:
                    inference_lock_wait_ms = (
                        perf_counter() - inference_wait_started
                    ) * 1000.0
                    phase = "inference"
                    inference_started = perf_counter()
                    names = self._search_names(
                        query,
                        request_id=request_id,
                        requested_max_paths=resolved_top_k,
                    )
                    inference_ms = (
                        perf_counter() - inference_started
                    ) * 1000.0

            selected_names = names[:resolved_top_k]
            serialization_started = perf_counter()
            response_json = json.dumps(selected_names, ensure_ascii=False)
            serialization_ms = (
                perf_counter() - serialization_started
            ) * 1000.0
            total_ms = (perf_counter() - total_started) * 1000.0
            queue_wait_ms = load_lock_wait_ms + inference_lock_wait_ms
            calc_overhead_ms = max(
                0.0,
                total_ms - queue_wait_ms - inference_ms - serialization_ms,
            )
            logger.info(
                "event=service.calc_complete instance_id=%s request_id=%s "
                "status=ok elapsed_ms=%.3f inference_ms=%.3f "
                "load_lock_wait_ms=%.3f inference_lock_wait_ms=%.3f "
                "queue_wait_ms=%.3f json_serialize_ms=%.3f "
                "calc_overhead_ms=%.3f query_chars=%s decoded_results=%s "
                "returned_results=%s response_chars=%s",
                self.instance_id,
                request_id,
                total_ms,
                inference_ms,
                load_lock_wait_ms,
                inference_lock_wait_ms,
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
            if (
                self.vllm_server is not None
                and self.vllm_server.process.poll() is not None
                and not isinstance(exc, ServiceRuntimeUnavailableError)
            ):
                exc = ServiceRuntimeUnavailableError(
                    "owned local vLLM server exited during inference"
                )
            unrecoverable = (
                phase == "automatic_load"
                or not self._loaded
                or isinstance(
                    exc,
                    (
                        MemoryError,
                        ServiceConfigurationError,
                        ServiceRuntimeUnavailableError,
                    ),
                )
            )
            if unrecoverable:
                logger.exception(
                    "event=service.calc_failed instance_id=%s request_id=%s "
                    "phase=%s recoverable=False elapsed_ms=%.3f error_type=%s",
                    self.instance_id,
                    request_id,
                    phase,
                    (perf_counter() - total_started) * 1000.0,
                    type(exc).__name__,
                )
                raise exc
            logger.error(
                "event=service.calc_recovered instance_id=%s request_id=%s "
                "phase=%s recoverable=True elapsed_ms=%.3f error_type=%s "
                "error=%s response=[]",
                self.instance_id,
                request_id,
                phase,
                (perf_counter() - total_started) * 1000.0,
                type(exc).__name__,
                _debug_error_log_value(exc),
            )
            return json.dumps([], ensure_ascii=False)

    def close(self) -> None:
        with self._load_lock:
            with self._inference_lock:
                _close_vllm_client(self.vllm_client)
                self.vllm_client = None
                if self.vllm_server is not None:
                    self.vllm_server.close()
                    if self.vllm_server.closed:
                        self.vllm_server = None
                self.tokenizer = None
                self.bundle = None
                self.trie = None
                self.path_skill_ids = {}
                self.backend = "vllm_http"
                self.mock_responses = {}
                self.output_budget = 1
                self.system_prompt = _DEFAULT_SYSTEM_PROMPT
                self._loaded = False
            gc.collect()

    def _search_names(
        self,
        query: str,
        *,
        request_id: str | None = None,
        requested_max_paths: int | None = None,
    ) -> list[str]:
        resolved_request_id = request_id or str(uuid.uuid4())
        started = perf_counter()
        logger.info(
            "event=search.begin instance_id=%s request_id=%s backend=%s "
            "query_chars=%s",
            self.instance_id,
            resolved_request_id,
            self.backend,
            len(query),
        )
        if self.backend == "mock":
            if not self._loaded:
                raise ServiceRuntimeUnavailableError(
                    "LLMGen mock retrieval service is not loaded"
                )
            results = list(
                self.mock_responses.get(query, self.mock_responses.get("*", ()))
            )
            logger.info(
                "event=search.mock_complete instance_id=%s request_id=%s "
                "elapsed_ms=%.3f result_count=%s",
                self.instance_id,
                resolved_request_id,
                (perf_counter() - started) * 1000.0,
                len(results),
            )
            return results
        if (
            self.vllm_server is None
            or self.vllm_client is None
            or self.tokenizer is None
            or self.bundle is None
            or self.trie is None
        ):
            raise ServiceRuntimeUnavailableError(
                "LLMGen retrieval service is not loaded"
            )
        exit_code = self.vllm_server.process.poll()
        if exit_code is not None:
            raise ServiceRuntimeUnavailableError(
                f"local vLLM API server exited with code {exit_code}"
            )
        render_started = perf_counter()
        prompt = _render_router_prompt(
            self.tokenizer,
            query,
            self.system_prompt,
        )
        render_ms = (perf_counter() - render_started) * 1000.0
        tokenize_started = perf_counter()
        prompt_ids = [
            int(value)
            for value in self.tokenizer.encode(prompt, add_special_tokens=False)
        ]
        if not prompt_ids:
            raise RuntimeError("Router tokenizer encoded the prompt as empty")
        truncated = len(prompt_ids) > self.max_input_length
        if truncated:
            # Match the repository inference path: keep the prompt prefix.
            prompt_ids = prompt_ids[: self.max_input_length]

        if truncated:
            decode = getattr(self.tokenizer, "decode", None)
            if not callable(decode):
                raise RuntimeError(
                    "Router tokenizer must expose decode() for truncated HTTP prompts"
                )
            prompt = str(
                decode(
                    prompt_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        tokenize_ms = (perf_counter() - tokenize_started) * 1000.0
        logger.info(
            "event=search.prompt_ready instance_id=%s request_id=%s "
            "render_ms=%.3f tokenize_ms=%.3f prompt_chars=%s "
            "prompt_tokens=%s max_input_length=%s truncated=%s",
            self.instance_id,
            resolved_request_id,
            render_ms,
            tokenize_ms,
            len(prompt),
            len(prompt_ids),
            self.max_input_length,
            truncated,
        )

        effective_max_paths = (
            self.trie.max_paths
            if requested_max_paths is None
            else min(
                self.trie.max_paths,
                max(1, int(requested_max_paths)),
            )
        )
        request_output_budget = (
            effective_max_paths * self.bundle.num_levels
            + (effective_max_paths - 1)
            * len(self.trie.separator_token_ids)
        )
        logger.info(
            "event=search.generation_configuration instance_id=%s "
            "request_id=%s requested_max_paths=%s effective_max_paths=%s "
            "initialized_max_paths=%s request_output_budget=%s "
            "initialized_output_budget=%s num_levels=%s separator_tokens=%s",
            self.instance_id,
            resolved_request_id,
            requested_max_paths,
            effective_max_paths,
            self.trie.max_paths,
            request_output_budget,
            self.output_budget,
            self.bundle.num_levels,
            len(self.trie.separator_token_ids),
        )

        payload: dict[str, Any] = {
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "max_tokens": request_output_budget,
            "min_tokens": self.bundle.num_levels,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        payload.update(_load_vllm_generate_kwargs())
        generation_started = perf_counter()
        response = self.vllm_client.generate(payload)
        generation_ms = (perf_counter() - generation_started) * 1000.0
        _log_vllm_raw_response(
            response,
            instance_id=self.instance_id,
            request_id=resolved_request_id,
        )
        decode_started = perf_counter()
        try:
            generated = _vllm_generate_token_ids(
                response,
                tokenizer=self.tokenizer,
                prompt=prompt,
            )
        except Exception as exc:
            logger.error(
                "event=search.model_output_decode_failed instance_id=%s "
                "request_id=%s error_type=%s error=%s",
                self.instance_id,
                resolved_request_id,
                type(exc).__name__,
                _debug_error_log_value(exc),
            )
            raise
        _log_vllm_token_ids(
            generated,
            stage="raw_generation",
            instance_id=self.instance_id,
            request_id=resolved_request_id,
        )
        try:
            eos_position = generated.index(self.trie.eos_token_id)
        except ValueError:
            # The 910B service intentionally budgets only path/separator
            # tokens. A valid boundary at max_tokens is therefore accepted.
            eos_position = None
        else:
            generated = generated[:eos_position]
        _log_vllm_token_ids(
            generated,
            stage="trie_input",
            instance_id=self.instance_id,
            request_id=resolved_request_id,
        )
        parse_result = self.trie.parse_with_recovery(generated)
        if not parse_result.paths:
            exc = RuntimeError(
                "vLLM generation contains no complete registered code path"
            )
            logger.error(
                "event=search.trie_parse_failed instance_id=%s request_id=%s "
                "error_type=%s error=%s generated_count=%s token_ids=%s reason=%s",
                self.instance_id,
                resolved_request_id,
                type(exc).__name__,
                _debug_error_log_value(exc),
                len(generated),
                _debug_token_ids_log_value(generated),
                parse_result.reason,
            )
            raise exc
        paths = parse_result.paths
        if parse_result.recovered:
            logger.warning(
                "event=search.trie_recovered instance_id=%s request_id=%s "
                "reason=%s consumed_tokens=%s discarded_tokens=%s "
                "path_count=%s",
                self.instance_id,
                resolved_request_id,
                parse_result.reason,
                parse_result.consumed_tokens,
                parse_result.discarded_tokens,
                len(paths),
            )
        decode_ms = (perf_counter() - decode_started) * 1000.0

        mapping_started = perf_counter()
        skill_ids: list[str] = []
        seen: set[str] = set()
        for path in paths:
            members = self.path_skill_ids.get(path)
            if members is None:
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
        total_ms = (perf_counter() - started) * 1000.0
        logger.info(
            "event=search.complete instance_id=%s request_id=%s "
            "elapsed_ms=%.3f render_ms=%.3f tokenize_ms=%.3f "
            "generation_ms=%.3f decode_ms=%.3f mapping_ms=%.3f "
            "service_non_generation_ms=%.3f prompt_tokens=%s "
            "completion_tokens=%s path_count=%s result_count=%s",
            self.instance_id,
            resolved_request_id,
            total_ms,
            render_ms,
            tokenize_ms,
            generation_ms,
            decode_ms,
            mapping_ms,
            max(0.0, total_ms - generation_ms),
            len(prompt_ids),
            len(generated),
            len(paths),
            len(names),
        )
        return names

    def _cleanup_after_failed_load(self) -> None:
        _close_vllm_client(self.vllm_client)
        self.vllm_client = None
        if self.vllm_server is not None:
            self.vllm_server.close()
            if self.vllm_server.closed:
                self.vllm_server = None
        self.tokenizer = None
        self.bundle = None
        self.trie = None
        self.path_skill_ids = {}
        self.backend = "vllm_http"
        self.mock_responses = {}
        self.output_budget = 1
        self.system_prompt = _DEFAULT_SYSTEM_PROMPT
        self._loaded = False
        gc.collect()


# A correctly-spelled alias for integrations that do not depend on the
# historical reference class name.
SkillRouterService = RetriverTest


def _deployment_parent(current_dir: Path) -> Path:
    model_object_id = os.environ.get("MODEL_OBJECT_ID")
    model_sfs = os.environ.get("MODEL_SFS")
    logger.info("model_object_id: %s", model_object_id)
    logger.info("model_sfs_path: %s", model_sfs)
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
        return (Path(base_path) / model_object_id).expanduser().resolve()
    return current_dir.parent.resolve()


def _resolve_model_directory(current_dir: Path) -> Path:
    deployment_parent = _deployment_parent(current_dir)
    platform_model_dir = deployment_parent / "model"
    if os.environ.get("MODEL_OBJECT_ID") or os.environ.get("MODEL_SFS"):
        # Preserve the reference service contract: the hosting system's SFS
        # location is authoritative when it is supplied.
        return platform_model_dir.expanduser().resolve()
    return Path(
        _env_first_text(("MODEL_PATH", "MODEL_DIR"), str(platform_model_dir))
    ).expanduser().resolve()


def _validate_full_model_bundle(model_dir: Path) -> None:
    """Require the consolidated Hugging Face files that vLLM can load."""

    config_path = model_dir / "config.json"
    indexed_weights = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    single_weights = ("model.safetensors", "pytorch_model.bin")

    has_full_weights = False
    for index_name in indexed_weights:
        index_path = model_dir / index_name
        if index_path.is_file():
            _validate_model_weight_index(model_dir, index_path)
            has_full_weights = True
            break
    if not has_full_weights:
        has_full_weights = any(
            (model_dir / name).is_file() and (model_dir / name).stat().st_size > 0
            for name in single_weights
        )

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
    _read_json_object(config_path)


def _validate_model_weight_index(model_dir: Path, index_path: Path) -> None:
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
    for raw_name in shard_names:
        shard = (model_dir / raw_name).resolve()
        if (
            shard.parent != model_root
            or not shard.is_file()
            or shard.stat().st_size < 1
        ):
            raise ServiceConfigurationError(
                "model weight index references a missing, empty, or unsafe shard: "
                f"{raw_name}"
            )


def _build_vllm_kwargs(*, model_path: Path, tokenizer_path: Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_mode": _env_text("VLLM_TOKENIZER_MODE", "auto"),
        "trust_remote_code": _env_bool("VLLM_TRUST_REMOTE_CODE", False),
        "dtype": _vllm_dtype(),
        "tensor_parallel_size": int(
            _env_first_text(
                (
                    "VLLM_TENSOR_PARALLEL_SIZE",
                    "LLM_TENSOR_PARALLEL_SIZE",
                    "TENSOR_PARALLEL_SIZE",
                    "tensor_parallel_size",
                ),
                str(_VLLM_TENSOR_PARALLEL_SIZE),
            )
        ),
        "pipeline_parallel_size": _env_int("VLLM_PIPELINE_PARALLEL_SIZE", 1),
        "gpu_memory_utilization": _env_float(
            "VLLM_GPU_MEMORY_UTILIZATION", 0.9
        ),
        "max_num_seqs": _env_int("VLLM_MAX_NUM_SEQS", 8),
        "seed": _env_int("VLLM_SEED", 0),
        "swap_space": _env_int("VLLM_SWAP_SPACE", 0),
        "disable_log_stats": _env_bool("VLLM_DISABLE_LOG_STATS", False),
        "disable_log_requests": _env_bool(
            "VLLM_DISABLE_LOG_REQUESTS", True
        ),
    }
    max_model_len = _env_int_optional("VLLM_MAX_MODEL_LEN")
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    for environment_name, option_name in (
        ("VLLM_SCHEDULER_BUDGET_LEN", "scheduler_budget_len"),
        ("VLLM_MAX_NUM_BATCHED_TOKENS", "max_num_batched_tokens"),
        ("VLLM_FIRST_TOKEN_TIMEOUT", "first_token_timeout"),
        ("VLLM_MAX_LOG_LEN", "max_log_len"),
        ("VLLM_BLOCK_SIZE", "block_size"),
        (
            "VLLM_DECODE_TENSOR_PARALLEL_SIZE",
            "decode_tensor_parallel_size",
        ),
    ):
        value = _env_int_optional(environment_name)
        if value is not None:
            kwargs[option_name] = value
    download_dir = os.environ.get("VLLM_DOWNLOAD_DIR")
    if download_dir and download_dir.strip():
        kwargs["download_dir"] = download_dir.strip()

    extra_json = os.environ.get("VLLM_KWARGS_JSON")
    if extra_json:
        extra = json.loads(extra_json)
        if not isinstance(extra, dict):
            raise ServiceConfigurationError(
                "VLLM_KWARGS_JSON must decode to an object"
            )
        reserved = {"model", "tokenizer"}.intersection(extra)
        if reserved:
            raise ServiceConfigurationError(
                "VLLM_KWARGS_JSON cannot override: " + ", ".join(sorted(reserved))
            )
        kwargs.update(extra)
    return kwargs


def _build_vllm_server_command(
    *,
    model_path: Path,
    tokenizer_path: Path,
    host: str,
    port: int,
    vllm_overrides: Mapping[str, Any],
) -> list[str]:
    kwargs = _build_vllm_kwargs(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    unsupported_overrides = set(vllm_overrides) - _VLLM_SERVER_SUPPORTED_KWARGS
    if unsupported_overrides:
        raise ServiceConfigurationError(
            "unsupported vLLM server overrides: "
            + ", ".join(sorted(unsupported_overrides))
        )
    kwargs.update(vllm_overrides)
    unsupported_kwargs = (
        set(kwargs) - {"model", "tokenizer"} - _VLLM_SERVER_SUPPORTED_KWARGS
    )
    if unsupported_kwargs:
        raise ServiceConfigurationError(
            "VLLM_KWARGS_JSON contains options without a safe CLI mapping: "
            + ", ".join(sorted(unsupported_kwargs))
        )
    for option_name in _VLLM_INTEGER_SERVER_OPTIONS:
        raw_value = kwargs.get(option_name)
        if raw_value is None:
            continue
        kwargs[option_name] = _normalize_cli_integer(
            raw_value,
            option_name=option_name,
        )
    raw_swap_space = kwargs.get("swap_space", 0)
    try:
        swap_space_is_zero = (
            not isinstance(raw_swap_space, bool)
            and float(raw_swap_space) == 0.0
        )
    except (TypeError, ValueError):
        swap_space_is_zero = False
    if not swap_space_is_zero:
        raise ServiceConfigurationError(
            "VLLM_SWAP_SPACE/swap_space must be exactly 0"
        )
    # The customized server's argparse requires the integer spelling ``0``;
    # it rejects the otherwise numerically equivalent string ``0.0``.
    kwargs["swap_space"] = 0

    command = [
        _env_first_text(
            ("VLLM_SERVER_PYTHON", "VLLM_PYTHON_EXECUTABLE"),
            sys.executable,
        ),
        "-m",
        "vllm.entrypoints.api_server",
        "--model",
        str(model_path),
        "--tokenizer",
        str(tokenizer_path),
        "--host",
        host,
        "--port",
        str(port),
    ]
    value_options = (
        ("tokenizer_mode", "--tokenizer-mode"),
        ("dtype", "--dtype"),
        ("tensor_parallel_size", "--tensor-parallel-size"),
        ("pipeline_parallel_size", "--pipeline-parallel-size"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
        ("max_num_seqs", "--max-num-seqs"),
        ("seed", "--seed"),
        ("swap_space", "--swap-space"),
        ("max_model_len", "--max-model-len"),
        ("download_dir", "--download-dir"),
        ("scheduler_budget_len", "--scheduler-budget-len"),
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("first_token_timeout", "--first-token-timeout"),
        ("max_log_len", "--max-log-len"),
        ("block_size", "--block-size"),
        (
            "decode_tensor_parallel_size",
            "--decode-tensor-parallel-size",
        ),
    )
    for name, flag in value_options:
        value = kwargs.get(name)
        if value is not None:
            command.extend((flag, str(value)))
    if kwargs.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if kwargs.get("disable_log_stats"):
        command.append("--disable-log-stats")
    if kwargs.get("disable_log_requests"):
        command.append("--disable-log-requests")
    command.extend(_load_vllm_server_extra_args())
    return command


def _normalize_cli_integer(value: Any, *, option_name: str) -> int:
    """Return an integer CLI value without ever rendering a ``.0`` suffix."""

    if isinstance(value, bool):
        raise ServiceConfigurationError(
            f"vLLM option {option_name} must be an integer"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ServiceConfigurationError(
            f"vLLM option {option_name} must be an integer"
        )
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ServiceConfigurationError(
                f"vLLM option {option_name} must be an integer"
            ) from exc
    raise ServiceConfigurationError(
        f"vLLM option {option_name} must be an integer"
    )


def _load_vllm_server_extra_args() -> list[str]:
    raw_value = os.environ.get("VLLM_SERVER_ARGS_JSON")
    if raw_value is None or not raw_value.strip():
        return []
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(
            "VLLM_SERVER_ARGS_JSON must be valid JSON"
        ) from exc
    if (
        not isinstance(payload, list)
        or any(not isinstance(value, str) or not value for value in payload)
    ):
        raise ServiceConfigurationError(
            "VLLM_SERVER_ARGS_JSON must be a JSON list of non-empty strings"
        )
    protected = {
        "--model",
        "--tokenizer",
        "--host",
        "--port",
        "--config",
        "--tokenizer-mode",
        "--trust-remote-code",
        "--dtype",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--gpu-memory-utilization",
        "--max-num-seqs",
        "--seed",
        "--swap-space",
        "--disable-log-stats",
        "--max-model-len",
        "--download-dir",
        "--scheduler-budget-len",
        "--max-num-batched-tokens",
        "--first-token-timeout",
        "--max-log-len",
        "--block-size",
        "--decode-tensor-parallel-size",
        "--disable-log-requests",
    }
    short_protected = {
        "-pp": "--pipeline-parallel-size",
        "-tp": "--tensor-parallel-size",
    }
    conflicts: set[str] = set()
    for value in payload:
        option = value.partition("=")[0].replace("_", "-")
        if option in short_protected:
            conflicts.add(option)
            continue
        if not option.startswith("--"):
            continue
        if option in protected:
            conflicts.add(option)
            continue
        # vLLM's FlexibleArgumentParser accepts unambiguous long-option
        # abbreviations, so a prefix such as ``--api-k`` must be treated as
        # an attempt to override the protected ``--api-key`` argument.
        if any(protected_option.startswith(option) for protected_option in protected):
            conflicts.add(option)
    if conflicts:
        raise ServiceConfigurationError(
            "VLLM_SERVER_ARGS_JSON cannot override: "
            + ", ".join(sorted(conflicts))
        )
    return list(payload)


def load_vllm_model(
    model_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    *,
    served_model_name: str | None = None,
    host: str | None = None,
    port: int | None = None,
    **vllm_overrides: Any,
) -> VllmServerHandle:
    """Start the owned local vLLM simple API server and wait until ready.

    The function is independently callable for environment-specific startup
    debugging; standalone callers must close the returned handle.
    ``RetriverTest.load()`` calls this same function in production.
    """

    service_dir = Path(__file__).resolve().parent
    if model_path is None:
        model_dir = _resolve_model_directory(service_dir)
    else:
        model_dir = Path(model_path).expanduser().resolve()
    tokenizer_dir = model_dir
    if tokenizer_path is not None:
        requested_tokenizer_dir = Path(tokenizer_path).expanduser().resolve()
        if requested_tokenizer_dir != model_dir:
            raise ServiceConfigurationError(
                "vllm.entrypoints.api_server requires --tokenizer to use the "
                "same resolved model directory as --model"
            )
    for label, path in (("model", model_dir), ("tokenizer", tokenizer_dir)):
        if not path.is_dir():
            raise ServiceConfigurationError(
                f"{label} directory does not exist: {path}"
            )
    _validate_full_model_bundle(model_dir)

    resolved_model_name = (
        str(served_model_name).strip()
        if served_model_name is not None and str(served_model_name).strip()
        else _env_text("SERVED_MODEL_NAME", model_dir.name or str(model_dir))
    )
    resolved_host = (
        str(host).strip()
        if host is not None and str(host).strip()
        else _env_text("VLLM_SERVER_HOST", _DEFAULT_VLLM_SERVER_HOST)
    )
    if resolved_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ServiceConfigurationError(
            "VLLM_SERVER_HOST must be a loopback address"
        )
    preferred_port = (
        int(port)
        if port is not None
        else _env_int("VLLM_SERVER_PORT", 0)
    )
    resolved_port = _find_available_vllm_port(
        resolved_host,
        preferred_port=preferred_port,
    )
    origin_host = (
        f"[{resolved_host}]" if ":" in resolved_host else resolved_host
    )
    origin = f"http://{origin_host}:{resolved_port}"
    command = _build_vllm_server_command(
        model_path=model_dir,
        tokenizer_path=tokenizer_dir,
        host=resolved_host,
        port=resolved_port,
        vllm_overrides=vllm_overrides,
    )

    child_environment = os.environ.copy()
    logger.info(
        "starting local vLLM API server model=%s tokenizer=%s model_name=%s "
        "origin=%s command=%s",
        model_dir,
        tokenizer_dir,
        resolved_model_name,
        origin,
        command,
    )
    process_group = os.name == "posix"
    process = subprocess.Popen(
        command,
        cwd=str(service_dir),
        env=child_environment,
        stdin=subprocess.DEVNULL,
        start_new_session=process_group,
    )
    handle = VllmServerHandle(
        process,
        origin=origin,
        host=resolved_host,
        port=resolved_port,
        process_group=process_group,
    )
    try:
        _wait_for_vllm_ready(
            process,
            health_url=handle.health_url,
        )
    except BaseException as exc:
        handle.close()
        if not handle.closed:
            raise RuntimeError(
                "vLLM startup failed and its child process could not be cleaned up; "
                f"pid={process.pid}"
            ) from exc
        raise
    return handle


def _find_available_vllm_port(host: str, *, preferred_port: int = 0) -> int:
    """Return a free loopback port, falling back when the preferred one is busy."""

    if preferred_port < 0 or preferred_port > 65535:
        raise ServiceConfigurationError(
            "VLLM_SERVER_PORT must be between 0 and 65535"
        )
    family = socket.AF_INET6 if ":" in host else socket.AF_INET

    def try_bind(candidate: int) -> int | None:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, candidate))
                return int(probe.getsockname()[1])
        except OSError:
            return None

    if preferred_port:
        selected = try_bind(preferred_port)
        if selected is not None:
            return selected
        logger.warning(
            "preferred vLLM port %s is occupied; selecting another port",
            preferred_port,
        )
    selected = try_bind(0)
    if selected is None:
        raise ServiceConfigurationError(
            f"cannot allocate a local vLLM port on {host}"
        )
    return selected


def _wait_for_vllm_ready(
    process: subprocess.Popen[Any],
    *,
    health_url: str,
) -> None:
    timeout = _env_float(
        "VLLM_STARTUP_TIMEOUT_SECONDS",
        _DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS,
    )
    poll_interval = _env_float("VLLM_READY_POLL_SECONDS", 0.5)
    if timeout <= 0 or poll_interval <= 0:
        raise ServiceConfigurationError(
            "vLLM startup timeout and poll interval must be positive"
        )
    deadline = perf_counter() + timeout
    last_error = "server has not accepted a health request"
    while True:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"vLLM API server exited during startup with code {exit_code}"
            )
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError(
                f"vLLM API server did not become ready within {timeout:g}s; "
                f"last check: {last_error}"
            )
        request_timeout = min(2.0, max(0.1, remaining))
        try:
            health_request = Request(health_url, method="GET")
            with urlopen(health_request, timeout=request_timeout) as response:
                response.read()
            logger.info("local vLLM API server is ready pid=%s", process.pid)
            return
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        sleep(min(poll_interval, max(0.0, deadline - perf_counter())))


def _load_tokenizer(tokenizer_path: Path) -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "HTTP Router service requires transformers to load its local tokenizer"
        ) from exc

    kwargs: dict[str, Any] = {
        "trust_remote_code": _env_bool("VLLM_TRUST_REMOTE_CODE", False),
        "local_files_only": _env_bool("TOKENIZER_LOCAL_FILES_ONLY", True),
    }
    if _env_text("VLLM_TOKENIZER_MODE", "auto").lower() == "slow":
        kwargs["use_fast"] = False
    kwargs.update(
        _transformers_tokenizer_compatibility_kwargs(
            tokenizer_path,
            transformers_version=str(
                getattr(transformers, "__version__", "")
            ),
        )
    )
    return transformers.AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        **kwargs,
    )


def _transformers_tokenizer_compatibility_kwargs(
    tokenizer_path: Path,
    *,
    transformers_version: str,
) -> dict[str, Any]:
    """Adapt Transformers-v5 list tokens to the 4.57 tokenizer contract.

    Transformers 4.57 treats ``extra_special_tokens`` as a mapping of named
    model-specific tokens and unconditionally calls ``.keys()`` on it. Some
    exported Qwen3.5 tokenizers instead store the v5 list form. Override only
    the in-memory initialization arguments; never rewrite deployment files.
    """

    if not transformers_version.startswith("4."):
        return {}
    config_path = tokenizer_path / "tokenizer_config.json"
    if not config_path.is_file():
        return {}
    config = _read_json_object(config_path)
    extra_tokens = config.get("extra_special_tokens")
    if not isinstance(extra_tokens, list):
        return {}

    existing_tokens = config.get("additional_special_tokens")
    merged_tokens: list[Any] = []
    if isinstance(existing_tokens, list):
        merged_tokens.extend(existing_tokens)
    for token in extra_tokens:
        if token not in merged_tokens:
            merged_tokens.append(token)
    logger.warning(
        "event=tokenizer.transformers_4_compatibility version=%s "
        "source_field=extra_special_tokens source_count=%s "
        "target_field=additional_special_tokens merged_count=%s "
        "model_files_modified=False",
        transformers_version,
        len(extra_tokens),
        len(merged_tokens),
    )
    return {
        "extra_special_tokens": {},
        "additional_special_tokens": merged_tokens,
    }


def _close_vllm_client(client: Any | None) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.exception("failed to close local vLLM HTTP client")


def _debug_log_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "event=service.debug_log_limit_invalid name=%s value=%s "
            "fallback=%s",
            name,
            raw_value,
            default,
        )
        return default
    if value < 1:
        logger.warning(
            "event=service.debug_log_limit_invalid name=%s value=%s "
            "fallback=%s",
            name,
            raw_value,
            default,
        )
        return default
    return value


def _bounded_log_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"<truncated chars={len(value) - limit}>"


def _log_vllm_raw_response(
    response: Mapping[str, Any],
    *,
    instance_id: str,
    request_id: str,
) -> None:
    if not _env_bool("SERVICE_OPENAI_LOG_MODEL_OUTPUT", False):
        return
    serialized = json.dumps(
        response,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    limit = _debug_log_positive_int(
        "SERVICE_OPENAI_LOG_PREVIEW_CHARS",
        _DEFAULT_MODEL_OUTPUT_PREVIEW_CHARS,
    )
    logger.info(
        "event=search.model_raw_output instance_id=%s request_id=%s "
        "response_chars=%s response=%s",
        instance_id,
        request_id,
        len(serialized),
        _bounded_log_text(serialized, limit=limit),
    )


def _debug_token_ids_log_value(token_ids: Sequence[int]) -> str:
    if not _env_bool("SERVICE_OPENAI_LOG_TOKEN_IDS", False):
        return f"<hidden count={len(token_ids)}>"
    item_limit = _debug_log_positive_int(
        "SERVICE_OPENAI_LOG_TOKEN_ITEMS",
        _DEFAULT_MODEL_OUTPUT_TOKEN_ITEMS,
    )
    visible = [int(value) for value in token_ids[:item_limit]]
    if len(token_ids) <= item_limit:
        return repr(visible)
    return repr(visible) + f"<truncated items={len(token_ids) - item_limit}>"


def _log_vllm_token_ids(
    token_ids: Sequence[int],
    *,
    stage: str,
    instance_id: str,
    request_id: str,
) -> None:
    if not _env_bool("SERVICE_OPENAI_LOG_TOKEN_IDS", False):
        return
    logger.info(
        "event=search.model_token_ids instance_id=%s request_id=%s "
        "stage=%s count=%s token_ids=%s",
        instance_id,
        request_id,
        stage,
        len(token_ids),
        _debug_token_ids_log_value(token_ids),
    )


def _debug_error_log_value(exc: BaseException) -> str:
    if not (
        _env_bool("SERVICE_OPENAI_LOG_MODEL_OUTPUT", False)
        or _env_bool("SERVICE_OPENAI_LOG_TOKEN_IDS", False)
    ):
        return (
            "<hidden; enable SERVICE_OPENAI_LOG_MODEL_OUTPUT or "
            "SERVICE_OPENAI_LOG_TOKEN_IDS>"
        )
    limit = _debug_log_positive_int(
        "SERVICE_OPENAI_LOG_PREVIEW_CHARS",
        _DEFAULT_MODEL_OUTPUT_PREVIEW_CHARS,
    )
    return _bounded_log_text(str(exc), limit=limit)


def _object_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _load_vllm_generate_kwargs() -> dict[str, Any]:
    raw_value = os.environ.get("VLLM_GENERATE_KWARGS_JSON")
    if raw_value is None or not raw_value.strip():
        return {}
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(
            "VLLM_GENERATE_KWARGS_JSON must be valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ServiceConfigurationError(
            "VLLM_GENERATE_KWARGS_JSON must decode to an object"
        )
    protected = {
        "prompt",
        "stream",
        "temperature",
        "max_tokens",
        "min_tokens",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    }.intersection(payload)
    if protected:
        raise ServiceConfigurationError(
            "VLLM_GENERATE_KWARGS_JSON cannot override: "
            + ", ".join(sorted(protected))
        )
    return dict(payload)


def _vllm_generate_token_ids(
    response: Mapping[str, Any],
    *,
    tokenizer: Any,
    prompt: str,
) -> list[int]:
    """Extract completion IDs from custom responses or re-encode text."""

    raw_token_ids: Any = response.get("token_ids")
    outputs = response.get("outputs")
    if (
        raw_token_ids is None
        and isinstance(outputs, Sequence)
        and not isinstance(outputs, (str, bytes, bytearray))
        and outputs
        and isinstance(outputs[0], Mapping)
    ):
        raw_token_ids = outputs[0].get("token_ids")
    if (
        isinstance(raw_token_ids, Sequence)
        and not isinstance(raw_token_ids, (str, bytes, bytearray))
        and raw_token_ids
    ):
        try:
            token_ids = [int(value) for value in raw_token_ids]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "vLLM /generate returned invalid completion token IDs"
            ) from exc
        if any(token_id < 0 for token_id in token_ids):
            raise RuntimeError(
                "vLLM /generate returned a negative completion token ID"
            )
        return token_ids

    generated_text: Any = response.get("generated_text")
    if generated_text is None:
        generated_text = response.get("text")
    if (
        generated_text is None
        and isinstance(outputs, Sequence)
        and not isinstance(outputs, (str, bytes, bytearray))
        and outputs
        and isinstance(outputs[0], Mapping)
    ):
        generated_text = outputs[0].get("text")
    if isinstance(generated_text, Sequence) and not isinstance(
        generated_text, (str, bytes, bytearray)
    ):
        generated_text = generated_text[0] if generated_text else None
    if not isinstance(generated_text, str):
        raise RuntimeError(
            "vLLM /generate returned neither completion token IDs nor text"
        )
    completion_text = (
        generated_text[len(prompt) :]
        if generated_text.startswith(prompt)
        else generated_text
    )
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Router tokenizer does not expose encode()")
    token_ids = [
        int(value)
        for value in encode(completion_text, add_special_tokens=False)
    ]
    if not token_ids:
        raise RuntimeError("vLLM /generate returned an empty completion")
    return token_ids


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
    if available < 1:
        raise ServiceConfigurationError(
            "model context length leaves no room for a Router prompt"
        )
    if explicit is None:
        return available
    if explicit < 1 or explicit > available:
        raise ServiceConfigurationError(
            f"MAX_INPUT_LENGTH must be between 1 and {available}"
        )
    return explicit


def _load_router_settings(model_dir: Path) -> RouterManifestSettings:
    manifest_path = model_dir / _ROUTER_MANIFEST_FILENAME
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
    return RouterManifestSettings(
        max_length=max_length,
        system_prompt=raw_system_prompt,
    )


def _code_token_id_map(tokenizer: Any, tokens: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    used_ids: dict[int, str] = {}
    for token in tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
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
    return mapping


def _render_router_prompt(tokenizer: Any, query: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query.strip()})
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if getattr(tokenizer, "chat_template", None) and callable(apply_template):
        try:
            return apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    system = f"System: {system_prompt.strip()}\n" if system_prompt else ""
    return f"{system}User: {query.strip()}\nAssistant:"


def _terminate_vllm_process(
    process: subprocess.Popen[Any],
    *,
    process_group: bool,
) -> bool:
    try:
        shutdown_timeout = _env_float(
            "VLLM_SHUTDOWN_TIMEOUT_SECONDS",
            _DEFAULT_VLLM_SHUTDOWN_TIMEOUT_SECONDS,
        )
    except (TypeError, ValueError):
        logger.exception(
            "invalid VLLM_SHUTDOWN_TIMEOUT_SECONDS; using %.1fs",
            _DEFAULT_VLLM_SHUTDOWN_TIMEOUT_SECONDS,
        )
        shutdown_timeout = _DEFAULT_VLLM_SHUTDOWN_TIMEOUT_SECONDS
    if not 0 < shutdown_timeout < float("inf"):
        logger.error(
            "VLLM_SHUTDOWN_TIMEOUT_SECONDS is not positive; using %.1fs",
            _DEFAULT_VLLM_SHUTDOWN_TIMEOUT_SECONDS,
        )
        shutdown_timeout = _DEFAULT_VLLM_SHUTDOWN_TIMEOUT_SECONDS
    safe_process_group = (
        process_group
        and os.name == "posix"
        and isinstance(process.pid, int)
        and process.pid > 1
    )

    leader_exited = process.poll() is not None
    if leader_exited:
        if safe_process_group and _process_group_exists(process.pid):
            # Once the leader has already exited, its numeric PID/PGID may be
            # stale and could have been reused.  Never signal that group from
            # the stale identifier; vLLM normally reaps its workers while the
            # leader handles shutdown.
            logger.warning(
                "vLLM server pid=%s exited before shutdown; refusing to "
                "signal its stale process-group identifier",
                process.pid,
            )
        return True

    if not leader_exited:
        try:
            if safe_process_group:
                # Signal the owned session while its leader is still known to
                # be alive.  This avoids later treating a stale PID as a PGID.
                if os.getpgid(process.pid) != process.pid:
                    logger.error(
                        "vLLM pid=%s is not the leader of its expected process group",
                        process.pid,
                    )
                    return False
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=shutdown_timeout)
            leader_exited = True
        except subprocess.TimeoutExpired:
            logger.warning(
                "vLLM server pid=%s did not stop within %.1fs; killing it",
                process.pid,
                shutdown_timeout,
            )
        except ProcessLookupError:
            leader_exited = process.poll() is not None
        except Exception:
            logger.exception("failed to terminate vLLM server pid=%s", process.pid)

    if leader_exited:
        if safe_process_group:
            return _stop_vllm_process_group(
                process.pid,
                timeout=shutdown_timeout,
            )
        if process_group and os.name == "posix":
            logger.error(
                "refusing to signal unsafe vLLM process group id=%r",
                process.pid,
            )
            return False
        return True

    try:
        if safe_process_group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=max(5.0, shutdown_timeout))
        return True
    except ProcessLookupError:
        return process.poll() is not None
    except Exception:
        logger.exception("failed to kill/reap vLLM server pid=%s", process.pid)
        return process.poll() is not None


def _stop_vllm_process_group(process_group_id: int, *, timeout: float) -> bool:
    """Stop workers left in the owned POSIX session after its leader exits."""

    if not isinstance(process_group_id, int) or process_group_id <= 1:
        logger.error(
            "refusing to signal unsafe vLLM process group id=%r",
            process_group_id,
        )
        return False
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except Exception:
        logger.exception(
            "failed to terminate vLLM process group pgid=%s",
            process_group_id,
        )

    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        sleep(min(0.1, max(0.0, deadline - perf_counter())))

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except Exception:
        logger.exception(
            "failed to kill vLLM process group pgid=%s",
            process_group_id,
        )
        return False

    deadline = perf_counter() + 5.0
    while perf_counter() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        sleep(0.05)
    logger.error("vLLM process group pgid=%s is still present", process_group_id)
    return False


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ServiceConfigurationError(f"expected a JSON object: {path}")
    return payload


def _load_mock_responses() -> dict[str, tuple[str, ...]]:
    raw_payload = os.environ.get("MOCK_RESPONSES_JSON")
    if raw_payload is None or not raw_payload.strip():
        return {"*": _DEFAULT_MOCK_RESPONSES}
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(
            "MOCK_RESPONSES_JSON must be valid JSON"
        ) from exc

    if isinstance(payload, list):
        return {"*": _normalize_mock_names(payload, "mock fallback")}
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
    if value is None:
        return None
    try:
        if not str(value).strip():
            return None
        return int(value)
    except Exception:
        logger.warning("invalid optional top_k value; using service default")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = RetriverTest()
    try:
        service.load()
        print(service.calc({"data": {"query": _DEFAULT_QUERY}}))
    finally:
        service.close()
