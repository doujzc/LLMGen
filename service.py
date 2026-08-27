#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-contained vLLM deployment service for an exported LLMGen router.

The hosting contract intentionally matches the reference retrieval service:

* ``load()`` initializes the long-lived model runtime.
* ``calc({"data": {"query": ..., "top_k": ...}})`` returns a JSON string
  containing a list of Skill names.

The model directory must be a complete LLMGen Router bundle containing the
Hugging Face model/tokenizer files, ``skill_decode_map.json``,
``virtual_tokens.txt``, and ``router_manifest.json``.

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
import threading
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence


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


class ServiceConfigurationError(RuntimeError):
    """Raised when deployment artifacts or environment settings disagree."""


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
    """Long-lived LLMGen Skill retrieval service backed by vLLM or a mock."""

    def __init__(self) -> None:
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
        self.backend = "vllm"
        self.mock_responses: dict[str, tuple[str, ...]] = {}
        self.default_top_k = 2
        self.max_code_paths = 2
        self.max_input_length = 1
        self.system_prompt = _DEFAULT_SYSTEM_PROMPT
        self._load_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        with self._load_lock:
            if self._loaded:
                logger.info("LLMGen retrieval service load skipped; already loaded")
                return

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

                # vLLM 0.8.5 request-level logits processors are a V0 API.
                os.environ["VLLM_USE_V1"] = "0"
                try:
                    from vllm import LLM, SamplingParams
                except ImportError as exc:
                    raise RuntimeError(
                        "LLMGen service requires vllm==0.8.5.post1"
                    ) from exc

                engine_kwargs = _build_vllm_kwargs(
                    model_path=model_dir,
                    tokenizer_path=tokenizer_dir,
                )
                logger.info(
                    "loading vLLM model=%s tokenizer=%s candidates=%s kwargs=%s",
                    model_dir,
                    tokenizer_dir,
                    candidate_dir,
                    {key: value for key, value in engine_kwargs.items() if key != "model"},
                )
                self.llm = LLM(**engine_kwargs)
                self.tokenizer = self.llm.get_tokenizer()

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
                output_budget = (
                    self.trie.max_paths * self.bundle.num_levels
                    + (self.trie.max_paths - 1) * len(separator_ids)
                    + 1
                )
                engine_max_length = getattr(
                    getattr(getattr(self.llm, "llm_engine", None), "model_config", None),
                    "max_model_len",
                    None,
                )
                self.max_input_length = _resolve_max_input_length(
                    trained_max_length=manifest_settings.max_length,
                    engine_max_length=engine_max_length,
                    output_budget=output_budget,
                )
                self.sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=output_budget,
                    min_tokens=self.bundle.num_levels,
                    detokenize=False,
                    skip_special_tokens=False,
                    logits_processors=[TrieLogitsProcessor(self.trie)],
                )
                self.backend = "vllm"
                self._loaded = True
                logger.info(
                    "LLMGen retrieval service loaded elapsed_ms=%.3f "
                    "skills=%s paths=%s levels=%s max_input_length=%s",
                    (perf_counter() - started) * 1000.0,
                    len(self.bundle.skills),
                    len(self.path_skill_ids),
                    self.bundle.num_levels,
                    self.max_input_length,
                )
            except Exception:
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
                _shutdown_vllm(self.llm)
                self.llm = None
                self.tokenizer = None
                self.sampling_params = None
                self.bundle = None
                self.trie = None
                self.path_skill_ids = {}
                self.backend = "vllm"
                self.mock_responses = {}
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
            self.llm is None
            or self.tokenizer is None
            or self.sampling_params is None
            or self.bundle is None
            or self.trie is None
        ):
            raise RuntimeError("LLMGen retrieval service is not loaded")
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

        outputs = self.llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != 1 or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no Router generation")
        completion = outputs[0].outputs[0]
        if getattr(completion, "finish_reason", None) == "length":
            raise RuntimeError("constrained vLLM generation exhausted its token budget")
        generated = [int(value) for value in completion.token_ids]
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
        _shutdown_vllm(self.llm)
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self.bundle = None
        self.trie = None
        self.path_skill_ids = {}
        self.backend = "vllm"
        self.mock_responses = {}
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


def _shutdown_vllm(llm: Any | None) -> None:
    if llm is None:
        return
    targets = (
        llm,
        getattr(llm, "llm_engine", None),
        getattr(getattr(llm, "llm_engine", None), "model_executor", None),
    )
    for target in targets:
        if target is None:
            continue
        for method_name in ("shutdown", "close"):
            method = getattr(target, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    logger.exception("failed to %s vLLM runtime", method_name)
                    continue
                return


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
