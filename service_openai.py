#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-contained OpenAI-client service for an exported LLMGen router.

The hosting contract intentionally matches the reference retrieval service:

* ``load()`` starts vLLM, then initializes the local tokenizer and OpenAI
  client.
* ``calc({"data": {"query": ..., "top_k": ...}})`` returns a JSON string
  containing a list of Skill names.

The model directory must be a complete LLMGen Router bundle containing the
Hugging Face model/tokenizer files, ``skill_decode_map.json``,
``virtual_tokens.txt``, and ``router_manifest.json``.

``load()`` calls ``load_vllm_model()`` to start a child vLLM 0.8.5 OpenAI
server, waits for it to become healthy, and owns that process until ``close()``.
The child uses ``VLLM_USE_V1=0`` because request-level logits processors are a
vLLM V0 feature in this pinned release.  Set ``VLLM_SERVER_PYTHON`` to launch
the server from a different Python environment; optional extra CLI arguments
can be supplied as a JSON string list in ``VLLM_SERVER_ARGS_JSON``.  The
default endpoint is ``http://127.0.0.1:8000/v1`` and can be changed with
``OPENAI_BASE_URL``.

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
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
from time import perf_counter, sleep
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


logger = logging.getLogger("web_demo")

_VLLM_TENSOR_PARALLEL_SIZE = 2
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
_DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_OPENAI_API_KEY = "EMPTY"
_DEFAULT_LOGITS_PROCESSOR_QUALNAME = (
    "service_openai.create_trie_logits_processor"
)
_DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS = 900.0
_DEFAULT_VLLM_SHUTDOWN_TIMEOUT_SECONDS = 30.0
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
    }
)


class ServiceConfigurationError(RuntimeError):
    """Raised when deployment artifacts or environment settings disagree."""


class VllmServerHandle:
    """Owned vLLM API-server process with idempotent shutdown."""

    def __init__(
        self,
        process: subprocess.Popen[Any],
        *,
        base_url: str,
        api_key: str,
        process_group: bool,
    ) -> None:
        self.process = process
        self.base_url = base_url
        self.api_key = api_key
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

    def parse_complete(self, generated: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        state = self._state(generated)
        if state is None:
            raise RuntimeError("vLLM generated an invalid code sequence")
        completed, prefix, mode, _ = state
        if (
            not completed
            or len(completed) > self.max_paths
            or prefix
            or mode != "boundary"
        ):
            raise RuntimeError("vLLM generation did not end at a code-path boundary")
        return completed


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
        self.vllm_server: VllmServerHandle | None = None
        self.openai_client: Any | None = None
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
        self.backend = "openai"
        self.mock_responses: dict[str, tuple[str, ...]] = {}
        self.default_top_k = 2
        self.max_code_paths = 2
        self.max_input_length = 1
        self.output_budget = 1
        self.logits_processor_qualname = _DEFAULT_LOGITS_PROCESSOR_QUALNAME
        self.system_prompt = _DEFAULT_SYSTEM_PROMPT
        self._load_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        with self._load_lock:
            if self._loaded:
                logger.info("LLMGen retrieval service load skipped; already loaded")
                return
            if self.vllm_server is not None:
                self.vllm_server.close()
                if not self.vllm_server.closed:
                    raise RuntimeError(
                        "cannot restart while the previous vLLM server is still running"
                    )
                self.vllm_server = None

            started = perf_counter()
            logger.info("LLMGen retrieval service load started")
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
                    "LLMGen retrieval service loaded mock backend elapsed_ms=%.3f "
                    "queries=%s top_k=%s",
                    (perf_counter() - started) * 1000.0,
                    len(self.mock_responses),
                    self.default_top_k,
                )
                return

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
                self.logits_processor_qualname = _env_text(
                    "VLLM_LOGITS_PROCESSOR_QUALNAME",
                    _DEFAULT_LOGITS_PROCESSOR_QUALNAME,
                )
                logger.info(
                    "preparing OpenAI Router model=%s tokenizer=%s candidates=%s "
                    "base_url=%s served_model=%s",
                    model_dir,
                    tokenizer_dir,
                    candidate_dir,
                    _env_text("OPENAI_BASE_URL", _DEFAULT_OPENAI_BASE_URL),
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
                    + 1
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
                    base_url=_env_text(
                        "OPENAI_BASE_URL", _DEFAULT_OPENAI_BASE_URL
                    ),
                    logits_processor_qualname=(
                        self.logits_processor_qualname
                    ),
                )
                self.openai_client = _create_openai_client(
                    base_url=self.vllm_server.base_url,
                    api_key=self.vllm_server.api_key,
                )
                self.backend = "openai"
                self._loaded = True
                logger.info(
                    "LLMGen OpenAI retrieval service loaded elapsed_ms=%.3f "
                    "skills=%s paths=%s levels=%s max_input_length=%s",
                    (perf_counter() - started) * 1000.0,
                    len(self.bundle.skills),
                    len(self.path_skill_ids),
                    self.bundle.num_levels,
                    self.max_input_length,
                )
            except BaseException:
                logger.exception(
                    "LLMGen retrieval service load failed elapsed_ms=%.3f",
                    (perf_counter() - started) * 1000.0,
                )
                self._cleanup_after_failed_load()
                raise

    def calc(self, req_data: Mapping[str, Any] | None) -> str:
        container = req_data if isinstance(req_data, Mapping) else {}
        data = container.get("data", {})
        request = dict(data) if isinstance(data, Mapping) else {}
        query = str(request.get("query", _DEFAULT_QUERY)).strip()
        if not query:
            logger.info("service calc skipped because query is empty")
            return json.dumps([], ensure_ascii=False)
        with self._load_lock:
            if not self._loaded:
                self.load()

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
                    "requested top_k=%s exceeds initialized top_k=%s; "
                    "returning at most initialized results",
                    resolved_top_k,
                    self.default_top_k,
                )
                resolved_top_k = self.default_top_k

            started = perf_counter()
            with self._inference_lock:
                names = self._search_names(query)
        logger.info(
            "service calc complete elapsed_ms=%.3f query_chars=%s results=%s",
            (perf_counter() - started) * 1000.0,
            len(query),
            min(len(names), resolved_top_k),
        )
        return json.dumps(names[:resolved_top_k], ensure_ascii=False)

    def close(self) -> None:
        with self._load_lock:
            with self._inference_lock:
                _close_openai_client(self.openai_client)
                self.openai_client = None
                if self.vllm_server is not None:
                    self.vllm_server.close()
                    if self.vllm_server.closed:
                        self.vllm_server = None
                self.tokenizer = None
                self.bundle = None
                self.trie = None
                self.path_skill_ids = {}
                self.backend = "openai"
                self.mock_responses = {}
                self.output_budget = 1
                self.logits_processor_qualname = (
                    _DEFAULT_LOGITS_PROCESSOR_QUALNAME
                )
                self.system_prompt = _DEFAULT_SYSTEM_PROMPT
                self._loaded = False
            gc.collect()

    def _search_names(self, query: str) -> list[str]:
        if self.backend == "mock":
            if not self._loaded:
                raise RuntimeError("LLMGen mock retrieval service is not loaded")
            return list(
                self.mock_responses.get(query, self.mock_responses.get("*", ()))
            )
        if (
            self.vllm_server is None
            or self.openai_client is None
            or self.tokenizer is None
            or self.bundle is None
            or self.trie is None
        ):
            raise RuntimeError("LLMGen retrieval service is not loaded")
        exit_code = self.vllm_server.process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"local vLLM OpenAI server exited with code {exit_code}"
            )
        prompt = _render_router_prompt(
            self.tokenizer,
            query,
            self.system_prompt,
        )
        prompt_ids = [
            int(value)
            for value in self.tokenizer.encode(prompt, add_special_tokens=False)
        ]
        if not prompt_ids:
            raise RuntimeError("Router tokenizer encoded the prompt as empty")
        if len(prompt_ids) > self.max_input_length:
            # Match the repository inference path: keep the prompt prefix.
            prompt_ids = prompt_ids[: self.max_input_length]

        response = self.openai_client.completions.create(
            model=self.served_model_name,
            prompt=prompt_ids,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=self.output_budget,
            logprobs=0,
            extra_body={
                "add_special_tokens": False,
                "min_tokens": self.bundle.num_levels,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "return_tokens_as_token_ids": True,
                "logits_processors": [
                    {
                        "qualname": self.logits_processor_qualname,
                        "kwargs": {
                            "paths": [
                                list(path) for path in sorted(self.path_skill_ids)
                            ],
                            "eos_token_id": self.trie.eos_token_id,
                            "separator_token_ids": list(
                                self.trie.separator_token_ids
                            ),
                            "max_paths": self.trie.max_paths,
                        },
                    }
                ],
            },
        )
        choices = _object_field(response, "choices")
        if (
            not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
            or len(choices) != 1
        ):
            raise RuntimeError("OpenAI endpoint returned no Router generation")
        completion = choices[0]
        if _object_field(completion, "finish_reason") == "length":
            raise RuntimeError("constrained vLLM generation exhausted its token budget")
        generated = _openai_completion_token_ids(completion)
        try:
            eos_position = generated.index(self.trie.eos_token_id)
        except ValueError as exc:
            raise RuntimeError("constrained vLLM generation did not emit EOS") from exc
        generated = generated[:eos_position]
        paths = self.trie.parse_complete(generated)

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
        return [
            str(self.bundle.skills[skill_id].get("name") or skill_id)
            for skill_id in skill_ids
        ]

    def _cleanup_after_failed_load(self) -> None:
        _close_openai_client(self.openai_client)
        self.openai_client = None
        if self.vllm_server is not None:
            self.vllm_server.close()
            if self.vllm_server.closed:
                self.vllm_server = None
        self.tokenizer = None
        self.bundle = None
        self.trie = None
        self.path_skill_ids = {}
        self.backend = "openai"
        self.mock_responses = {}
        self.output_budget = 1
        self.logits_processor_qualname = _DEFAULT_LOGITS_PROCESSOR_QUALNAME
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
        "tensor_parallel_size": _env_int(
            "VLLM_TENSOR_PARALLEL_SIZE", _VLLM_TENSOR_PARALLEL_SIZE
        ),
        "pipeline_parallel_size": _env_int("VLLM_PIPELINE_PARALLEL_SIZE", 1),
        "gpu_memory_utilization": _env_float(
            "VLLM_GPU_MEMORY_UTILIZATION", 0.9
        ),
        "max_num_seqs": _env_int("VLLM_MAX_NUM_SEQS", 8),
        "seed": _env_int("VLLM_SEED", 0),
        "swap_space": _env_float("VLLM_SWAP_SPACE", 0.0),
        "disable_log_stats": _env_bool("VLLM_DISABLE_LOG_STATS", False),
    }
    max_model_len = _env_int_optional("VLLM_MAX_MODEL_LEN")
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
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
    served_model_name: str,
    host: str,
    port: int,
    logits_processor_qualname: str,
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

    command = [
        _env_first_text(
            ("VLLM_SERVER_PYTHON", "VLLM_PYTHON_EXECUTABLE"),
            sys.executable,
        ),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--tokenizer",
        str(tokenizer_path),
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--generation-config",
        "vllm",
        "--logits-processor-pattern",
        f"^{re.escape(logits_processor_qualname)}$",
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
    )
    for name, flag in value_options:
        value = kwargs.get(name)
        if value is not None:
            command.extend((flag, str(value)))
    if kwargs.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if kwargs.get("disable_log_stats"):
        command.append("--disable-log-stats")
    command.extend(_load_vllm_server_extra_args())
    return command


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
        "--served-model-name",
        "--host",
        "--port",
        "--generation-config",
        "--logits-processor-pattern",
        "--api-key",
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
    base_url: str | None = None,
    logits_processor_qualname: str | None = None,
    **vllm_overrides: Any,
) -> VllmServerHandle:
    """Start the owned local vLLM OpenAI server and wait until it is ready.

    The function is independently callable for environment-specific startup
    debugging; standalone callers must close the returned handle.
    ``RetriverTest.load()`` calls this same function in production.
    """

    service_dir = Path(__file__).resolve().parent
    if model_path is None:
        model_dir = _resolve_model_directory(service_dir)
    else:
        model_dir = Path(model_path).expanduser().resolve()
    tokenizer_dir = Path(
        _env_text("TOKENIZER_PATH", str(model_dir))
        if tokenizer_path is None
        else tokenizer_path
    ).expanduser().resolve()
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
    resolved_base_url, host, port, origin = _parse_local_openai_base_url(
        base_url
        if base_url is not None
        else _env_text("OPENAI_BASE_URL", _DEFAULT_OPENAI_BASE_URL)
    )
    resolved_qualname = (
        str(logits_processor_qualname).strip()
        if logits_processor_qualname is not None
        and str(logits_processor_qualname).strip()
        else _env_text(
            "VLLM_LOGITS_PROCESSOR_QUALNAME",
            _DEFAULT_LOGITS_PROCESSOR_QUALNAME,
        )
    )
    command = _build_vllm_server_command(
        model_path=model_dir,
        tokenizer_path=tokenizer_dir,
        served_model_name=resolved_model_name,
        host=host,
        port=port,
        logits_processor_qualname=resolved_qualname,
        vllm_overrides=vllm_overrides,
    )
    _ensure_vllm_port_available(host, port)

    child_environment = os.environ.copy()
    child_environment["VLLM_USE_V1"] = "0"
    configured_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    api_key = configured_api_key or secrets.token_urlsafe(32)
    child_environment["VLLM_API_KEY"] = api_key
    existing_python_path = child_environment.get("PYTHONPATH", "").strip()
    child_environment["PYTHONPATH"] = (
        str(service_dir)
        if not existing_python_path
        else str(service_dir) + os.pathsep + existing_python_path
    )
    logger.info(
        "starting local vLLM OpenAI server model=%s tokenizer=%s "
        "base_url=%s command=%s",
        model_dir,
        tokenizer_dir,
        resolved_base_url,
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
        base_url=resolved_base_url,
        api_key=api_key,
        process_group=process_group,
    )
    try:
        _wait_for_vllm_ready(
            process,
            health_url=origin + "/health",
            models_url=resolved_base_url + "/models",
            served_model_name=resolved_model_name,
            api_key=api_key,
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


def _parse_local_openai_base_url(
    base_url: str,
) -> tuple[str, str, int, str]:
    value = str(base_url).strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "http":
        raise ServiceConfigurationError(
            "OPENAI_BASE_URL must use http for the owned local vLLM server"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ServiceConfigurationError("OPENAI_BASE_URL cannot contain credentials")
    host = parsed.hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ServiceConfigurationError(
            "OPENAI_BASE_URL must point to a loopback host"
        )
    if parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment:
        raise ServiceConfigurationError(
            "OPENAI_BASE_URL must end with /v1 and contain no query or fragment"
        )
    try:
        port = parsed.port or 80
    except ValueError as exc:
        raise ServiceConfigurationError("OPENAI_BASE_URL has an invalid port") from exc
    origin = urlunsplit(("http", parsed.netloc, "", "", ""))
    normalized = urlunsplit(("http", parsed.netloc, "/v1", "", ""))
    return normalized, host, port, origin


def _ensure_vllm_port_available(host: str, port: int) -> None:
    try:
        connection = socket.create_connection((host, port), timeout=0.25)
    except OSError:
        return
    connection.close()
    raise ServiceConfigurationError(
        f"local OpenAI endpoint is already in use at {host}:{port}"
    )


def _wait_for_vllm_ready(
    process: subprocess.Popen[Any],
    *,
    health_url: str,
    models_url: str,
    served_model_name: str,
    api_key: str,
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
                f"vLLM OpenAI server exited during startup with code {exit_code}"
            )
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError(
                f"vLLM OpenAI server did not become ready within {timeout:g}s; "
                f"last check: {last_error}"
            )
        request_timeout = min(2.0, max(0.1, remaining))
        try:
            health_request = Request(health_url, method="GET")
            with urlopen(health_request, timeout=request_timeout) as response:
                response.read()

            models_request = Request(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"},
                method="GET",
            )
            with urlopen(models_request, timeout=request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw_models = payload.get("data") if isinstance(payload, Mapping) else None
            model_names = {
                str(item.get("id"))
                for item in raw_models or ()
                if isinstance(item, Mapping) and item.get("id") is not None
            }
            if served_model_name in model_names:
                logger.info(
                    "local vLLM OpenAI server is ready model=%s pid=%s",
                    served_model_name,
                    process.pid,
                )
                return
            last_error = (
                f"/v1/models returned {sorted(model_names)!r}, expected "
                f"{served_model_name!r}"
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        sleep(min(poll_interval, max(0.0, deadline - perf_counter())))


def _load_tokenizer(tokenizer_path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Router service requires transformers to load its local tokenizer"
        ) from exc

    kwargs: dict[str, Any] = {
        "trust_remote_code": _env_bool("VLLM_TRUST_REMOTE_CODE", False),
        "local_files_only": _env_bool("TOKENIZER_LOCAL_FILES_ONLY", True),
    }
    if _env_text("VLLM_TOKENIZER_MODE", "auto").lower() == "slow":
        kwargs["use_fast"] = False
    return AutoTokenizer.from_pretrained(str(tokenizer_path), **kwargs)


def _create_openai_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Router service requires the openai Python package"
        ) from exc

    timeout = _env_float("OPENAI_TIMEOUT_SECONDS", 300.0)
    max_retries = _env_int("OPENAI_MAX_RETRIES", 0)
    if timeout <= 0:
        raise ServiceConfigurationError("OPENAI_TIMEOUT_SECONDS must be positive")
    if max_retries < 0:
        raise ServiceConfigurationError("OPENAI_MAX_RETRIES cannot be negative")
    return OpenAI(
        base_url=(
            str(base_url)
            if base_url is not None
            else _env_text("OPENAI_BASE_URL", _DEFAULT_OPENAI_BASE_URL)
        ),
        api_key=(
            str(api_key)
            if api_key is not None
            else _env_text("OPENAI_API_KEY", _DEFAULT_OPENAI_API_KEY)
        ),
        timeout=timeout,
        max_retries=max_retries,
    )


def _close_openai_client(client: Any | None) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.exception("failed to close OpenAI client")


def _object_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _openai_completion_token_ids(completion: Any) -> list[int]:
    logprobs = _object_field(completion, "logprobs")
    raw_tokens = _object_field(logprobs, "tokens")
    if (
        not isinstance(raw_tokens, Sequence)
        or isinstance(raw_tokens, (str, bytes))
        or not raw_tokens
    ):
        raise RuntimeError(
            "OpenAI endpoint did not return completion token IDs; use vLLM "
            "with logprobs and return_tokens_as_token_ids support"
        )

    token_ids: list[int] = []
    prefix = "token_id:"
    for raw_token in raw_tokens:
        if not isinstance(raw_token, str) or not raw_token.startswith(prefix):
            raise RuntimeError(
                "OpenAI endpoint returned a non-token-ID completion logprob"
            )
        try:
            token_id = int(raw_token[len(prefix) :])
        except ValueError as exc:
            raise RuntimeError(
                f"OpenAI endpoint returned an invalid token ID: {raw_token!r}"
            ) from exc
        if token_id < 0:
            raise RuntimeError(
                f"OpenAI endpoint returned a negative token ID: {token_id}"
            )
        token_ids.append(token_id)
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
    if value is None or not str(value).strip():
        return None
    return int(value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = RetriverTest()
    try:
        service.load()
        print(service.calc({"data": {"query": _DEFAULT_QUERY}}))
    finally:
        service.close()
