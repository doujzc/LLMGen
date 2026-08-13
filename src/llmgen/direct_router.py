"""Direct candidate-name routing for multi-turn intent selection.

This module deliberately has no dependency on torch or transformers.  Training
and inference share the same conversation normalization, prompt rendering,
candidate registry, and variable-length token trie through these utilities.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from llmgen.router import RouterDataError


DIRECT_ROUTING_MODE = "candidate_name_top1"
LATEST_TRUNCATION_MARKER = "\n...[当前用户消息中间内容已截断]...\n"
HISTORY_TRUNCATION_MARKER = "\n...[历史消息中间内容已截断]...\n"
ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True)
class CandidateRoute:
    """One legal generated name and its downstream route metadata."""

    name: str
    candidate_id: str
    intent_label: str | None
    virtual: bool


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RouterDataError(f"{field} must be a non-empty string")
    return value.strip()


def load_candidate_registry(path: str | Path) -> tuple[CandidateRoute, ...]:
    """Load and validate the direct-routing candidate registry."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RouterDataError(f"invalid candidate registry JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise RouterDataError("candidate registry must be a JSON object")
    mode = payload.get("routing_mode")
    if mode != DIRECT_ROUTING_MODE:
        raise RouterDataError(
            f"candidate registry routing_mode must be {DIRECT_ROUTING_MODE!r}"
        )
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise RouterDataError("candidate registry must contain a non-empty candidates list")

    routes: list[CandidateRoute] = []
    names: set[str] = set()
    ids: set[str] = set()
    for index, row in enumerate(raw_candidates):
        if not isinstance(row, dict):
            raise RouterDataError(f"candidate registry row {index} must be an object")
        name = _nonempty_string(row.get("name"), field=f"candidates[{index}].name")
        candidate_id = _nonempty_string(
            row.get("candidate_id"), field=f"candidates[{index}].candidate_id"
        )
        intent_label = row.get("intent_label")
        if intent_label is not None:
            intent_label = _nonempty_string(
                intent_label, field=f"candidates[{index}].intent_label"
            )
        virtual = row.get("virtual")
        if not isinstance(virtual, bool):
            raise RouterDataError(f"candidates[{index}].virtual must be boolean")
        if name in names:
            raise RouterDataError(f"duplicate candidate name: {name!r}")
        if candidate_id in ids:
            raise RouterDataError(f"duplicate candidate id: {candidate_id!r}")
        if virtual == (intent_label is not None):
            raise RouterDataError(
                f"candidate {name!r} must have an intent_label exactly when non-virtual"
            )
        names.add(name)
        ids.add(candidate_id)
        routes.append(CandidateRoute(name, candidate_id, intent_label, virtual))
    return tuple(routes)


def candidate_registry_payload(
    routes: Iterable[CandidateRoute],
) -> dict[str, Any]:
    """Serialize validated routes using the portable registry schema."""

    normalized = tuple(routes)
    if not normalized:
        raise RouterDataError("cannot serialize an empty candidate registry")
    return {
        "schema_version": 1,
        "routing_mode": DIRECT_ROUTING_MODE,
        "candidates": [asdict(route) for route in normalized],
    }


def _truncate_middle(content: str, limit: int, marker: str) -> str:
    if limit < 1:
        return ""
    if len(content) <= limit:
        return content
    if limit <= len(marker):
        return content[:limit]
    available = limit - len(marker)
    head_size = (available * 2) // 3
    tail_size = available - head_size
    return content[:head_size] + marker + content[-tail_size:]


def normalize_conversation_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_history_messages: int = 16,
    max_history_chars: int = 12_000,
    max_assistant_history_chars: int = 1_200,
) -> tuple[dict[str, str], ...]:
    """Validate and trim a conversation while always retaining the latest user turn."""

    if min(
        max_history_messages,
        max_history_chars,
        max_assistant_history_chars,
    ) < 1:
        raise RouterDataError("conversation limits must be positive")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise RouterDataError("messages must be a sequence")
    if not messages:
        raise RouterDataError("messages cannot be empty")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise RouterDataError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise RouterDataError(f"messages[{index}] role/content must be strings")
        role = role.strip()
        content = content.strip()
        if role not in ALLOWED_MESSAGE_ROLES:
            raise RouterDataError(f"messages[{index}] has unsupported role {role!r}")
        if not content:
            raise RouterDataError(f"messages[{index}].content cannot be empty")
        # Source system messages are untrusted conversation content.  The
        # router's own fixed system prompt is added separately.
        if role != "system":
            normalized.append({"role": role, "content": content})

    if not normalized:
        raise RouterDataError("conversation has no messages after dropping system turns")
    if normalized[-1]["role"] != "user":
        raise RouterDataError("the final non-system message must have role 'user'")

    current = dict(normalized[-1])
    current["content"] = _truncate_middle(
        current["content"], max_history_chars, LATEST_TRUNCATION_MARKER
    )
    remaining_chars = max_history_chars - len(current["content"])
    history_slots = max_history_messages - 1
    recent_history = normalized[:-1][-history_slots:] if history_slots else []
    kept: list[dict[str, str]] = []
    for original in reversed(recent_history):
        if remaining_chars <= 0:
            break
        message = dict(original)
        limit = remaining_chars
        if message["role"] in {"assistant", "tool"}:
            limit = min(limit, max_assistant_history_chars)
        message["content"] = _truncate_middle(
            message["content"], limit, HISTORY_TRUNCATION_MARKER
        )
        if message["content"]:
            kept.append(message)
            remaining_chars -= len(message["content"])
    return tuple([*reversed(kept), current])


def messages_from_row(row: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    """Read canonical multi-turn messages or wrap a legacy single query."""

    raw_messages = row.get("messages")
    if raw_messages is not None:
        return normalize_conversation_messages(raw_messages)
    for field in ("query", "input_text", "instruction"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return ({"role": "user", "content": value.strip()},)
    raise RouterDataError("row must contain messages or a non-empty query")


def conversation_query_group(messages: Sequence[Mapping[str, str]]) -> str:
    """Return a stable group id so duplicate conversations cannot cross splits."""

    normalized = normalize_conversation_messages(messages)
    canonical = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(
        unicodedata.normalize("NFKC", canonical).casefold().encode("utf-8")
    ).hexdigest()
    return f"conversation:{digest}"


def build_conversation_user_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    """Serialize history and current request without flattening role boundaries."""

    normalized = normalize_conversation_messages(messages)
    payload: dict[str, Any] = {}
    if len(normalized) > 1:
        payload["history"] = list(normalized[:-1])
    payload["current_user_request"] = normalized[-1]["content"]
    conversation = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        f"<conversation_json>{conversation}</conversation_json>\n"
        "输出候选名称："
    )


def render_candidate_router_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    system_prompt: str,
) -> str:
    """Render the fixed router prompt for a normalized multi-turn conversation."""

    system_prompt = _nonempty_string(system_prompt, field="system_prompt")
    user_prompt = build_conversation_user_prompt(messages)
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    chat_template = getattr(tokenizer, "chat_template", None)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if chat_template and callable(apply_template):
        return apply_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    kwargs = {"add_special_tokens": False}
    try:
        values = tokenizer.encode(text, verbose=False, **kwargs)
    except TypeError:
        values = tokenizer.encode(text, **kwargs)
    return [int(value) for value in values]


def fit_candidate_router_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: str,
    *,
    max_prompt_tokens: int,
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Fit a prompt by dropping old history before truncating the current turn."""

    if max_prompt_tokens < 1:
        raise RouterDataError("max_prompt_tokens must be positive")
    normalized = list(normalize_conversation_messages(messages))

    def render(candidate_messages: Sequence[Mapping[str, str]]) -> tuple[str, list[int]]:
        prompt = render_candidate_router_prompt(
            tokenizer, candidate_messages, system_prompt
        )
        return prompt, _encode_text(tokenizer, prompt)

    prompt, prompt_ids = render(normalized)
    while len(prompt_ids) > max_prompt_tokens and len(normalized) > 1:
        normalized.pop(0)
        prompt, prompt_ids = render(normalized)
    if len(prompt_ids) <= max_prompt_tokens:
        return prompt, tuple(normalized)

    original = normalized[-1]["content"]
    low, high = 1, len(original)
    best: tuple[str, tuple[dict[str, str], ...]] | None = None
    while low <= high:
        limit = (low + high) // 2
        current = {
            "role": "user",
            "content": _truncate_middle(original, limit, LATEST_TRUNCATION_MARKER),
        }
        candidate_prompt, candidate_ids = render((current,))
        if len(candidate_ids) <= max_prompt_tokens:
            best = (candidate_prompt, (current,))
            low = limit + 1
        else:
            high = limit - 1
    if best is None:
        raise RouterDataError(
            "system prompt and chat template exceed the available prompt-token budget"
        )
    return best


def candidate_token_sequences(
    tokenizer: Any,
    candidate_names: Iterable[str],
) -> dict[str, tuple[int, ...]]:
    """Tokenize legal names and reject ambiguous token-level aliases."""

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise RouterDataError("causal tokenizer must define eos_token_id")
    result: dict[str, tuple[int, ...]] = {}
    used: dict[tuple[int, ...], str] = {}
    for raw_name in candidate_names:
        name = _nonempty_string(raw_name, field="candidate name")
        if name in result:
            raise RouterDataError(f"duplicate candidate name: {name!r}")
        ids = tuple(_encode_text(tokenizer, name))
        if not ids:
            raise RouterDataError(f"candidate name {name!r} tokenizes to an empty path")
        if eos_token_id in ids:
            raise RouterDataError(f"candidate name {name!r} contains EOS")
        previous = used.get(ids)
        if previous is not None:
            raise RouterDataError(
                f"candidate names {previous!r} and {name!r} share one token sequence"
            )
        result[name] = ids
        used[ids] = name
    if not result:
        raise RouterDataError("candidate name set cannot be empty")
    return result


def target_candidate_name(row: Mapping[str, Any]) -> str:
    """Read the canonical direct-routing label from a training/evaluation row."""

    for field in ("target_candidate_name", "candidate_name", "target"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RouterDataError("training row has no target_candidate_name")


def standard_candidate_sft_row(
    row: Mapping[str, Any],
    *,
    legal_candidate_names: set[str],
    system_prompt: str,
    tokenizer: Any | None = None,
    candidate_name_tokens: Mapping[str, Sequence[int]] | None = None,
    max_length: int = 1024,
) -> dict[str, list[dict[str, str]]]:
    """Represent one direct-router example as standard conversational SFT data."""

    name = target_candidate_name(row)
    if name not in legal_candidate_names:
        raise RouterDataError(f"unknown target candidate name: {name!r}")
    source_messages = messages_from_row(row)
    fitted_messages = source_messages
    if tokenizer is not None:
        if candidate_name_tokens is None or name not in candidate_name_tokens:
            raise RouterDataError("candidate token sequences are required for fitting")
        target_length = len(candidate_name_tokens[name]) + 1
        if max_length <= target_length:
            raise RouterDataError("max_length leaves no room for a router prompt")
        _, fitted_messages = fit_candidate_router_prompt(
            tokenizer,
            source_messages,
            system_prompt,
            max_prompt_tokens=max_length - target_length,
        )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_conversation_user_prompt(fitted_messages),
            },
            {"role": "assistant", "content": name},
        ]
    }


def encode_candidate_name_example(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    candidate_names: Iterable[str],
    candidate_name_tokens: Mapping[str, Sequence[int]] | None = None,
    max_length: int,
    system_prompt: str,
) -> dict[str, list[int]]:
    """Encode ``candidate name + EOS`` with loss only on the generated target."""

    sequences = (
        candidate_token_sequences(tokenizer, candidate_names)
        if candidate_name_tokens is None
        else {
            str(name): tuple(int(value) for value in values)
            for name, values in candidate_name_tokens.items()
        }
    )
    name = target_candidate_name(row)
    if name not in sequences:
        raise RouterDataError(f"unknown target candidate name: {name!r}")
    eos_token_id = int(tokenizer.eos_token_id)
    target_ids = [*sequences[name], eos_token_id]
    if max_length <= len(target_ids):
        raise RouterDataError("max_length leaves no room for a router prompt")
    prompt, _ = fit_candidate_router_prompt(
        tokenizer,
        messages_from_row(row),
        system_prompt,
        max_prompt_tokens=max_length - len(target_ids),
    )
    prompt_ids = _encode_text(tokenizer, prompt)
    input_ids = [*prompt_ids, *target_ids]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


class CandidateNameTokenTrie:
    """Variable-length grammar for exactly one legal candidate name then EOS."""

    _NAME = object()

    def __init__(
        self,
        name_to_tokens: Mapping[str, Sequence[int]],
        *,
        eos_token_id: int,
    ) -> None:
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise RouterDataError("eos_token_id must be a non-negative integer")
        if not name_to_tokens:
            raise RouterDataError("cannot build a candidate trie without names")
        self.eos_token_id = eos_token_id
        self._root: dict[Any, Any] = {}
        self._name_to_tokens: dict[str, tuple[int, ...]] = {}
        self._tokens_to_name: dict[tuple[int, ...], str] = {}
        for raw_name, raw_tokens in name_to_tokens.items():
            name = _nonempty_string(raw_name, field="candidate name")
            tokens = tuple(int(value) for value in raw_tokens)
            if not tokens or any(value < 0 for value in tokens):
                raise RouterDataError(f"candidate {name!r} has an invalid token path")
            if eos_token_id in tokens:
                raise RouterDataError(f"candidate {name!r} token path contains EOS")
            previous = self._tokens_to_name.get(tokens)
            if previous is not None:
                raise RouterDataError(
                    f"candidate names {previous!r} and {name!r} share one token path"
                )
            node = self._root
            for token_id in tokens:
                node = node.setdefault(token_id, {})
            if self._NAME in node:
                raise RouterDataError(f"duplicate candidate token path for {name!r}")
            node[self._NAME] = name
            self._name_to_tokens[name] = tokens
            self._tokens_to_name[tokens] = name

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._name_to_tokens)

    @property
    def max_name_tokens(self) -> int:
        return max(len(tokens) for tokens in self._name_to_tokens.values())

    def allowed_next(self, generated: Sequence[int]) -> tuple[int, ...]:
        node = self._root
        for raw_token_id in generated:
            token_id = int(raw_token_id)
            if token_id == self.eos_token_id:
                return ()
            child = node.get(token_id)
            if not isinstance(child, dict):
                return ()
            node = child
        allowed = sorted(key for key in node if key is not self._NAME)
        if self._NAME in node:
            allowed.append(self.eos_token_id)
        return tuple(allowed)

    def resolve(self, generated: Sequence[int]) -> str:
        tokens = tuple(int(value) for value in generated)
        try:
            return self._tokens_to_name[tokens]
        except KeyError as exc:
            raise RouterDataError(
                f"generated token sequence is not a complete candidate name: {tokens!r}"
            ) from exc
