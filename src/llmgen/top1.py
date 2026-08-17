"""Pure data and prompt logic for direct candidate-name Top1 training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape as escape_xml_text


ROUTING_MODE = "candidate_name_top1"
CONVERSATION_TEMPLATE = "routing_envelope_xml_v1"
TARGET_CONTRACT = "candidate_name_tokens_plus_eos"
INFERENCE_DECISION_RULE = "candidate_path_sum_logprob"
MEMORIZATION_SOURCE_TYPE = "label_description"
MEMORIZATION_DESCRIPTION_TYPES = (
    "label_term",
    "related_term",
    "concise_definition",
    "extended_definition",
)
MAX_HISTORY_MESSAGES = 16
MAX_HISTORY_CHARACTERS = 12_000
MAX_ASSISTANT_HISTORY_CHARACTERS = 1_200
LATEST_TRUNCATION_MARKER = "\n...[当前用户消息中间内容已截断]...\n"
HISTORY_TRUNCATION_MARKER = "\n...[历史消息中间内容已截断]...\n"
ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


class Top1DataError(ValueError):
    """Raised when a Top1 training contract is invalid."""


@dataclass(frozen=True)
class PreparedExample:
    """One encoded training example and its equivalent messages-only SFT row."""

    encoded: dict[str, list[int]]
    sft_row: dict[str, list[dict[str, str]]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PreparedPrompt:
    """The canonical label-independent prompt used by training and inference."""

    text: str
    input_ids: tuple[int, ...]
    fitted_messages: tuple[dict[str, str], ...]
    reserved_target_tokens: int


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file whose rows must be objects."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise Top1DataError(
                    f"invalid JSON at {source}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise Top1DataError(f"row at {source}:{line_number} must be an object")
            rows.append(row)
    return rows


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON object."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write JSON objects as UTF-8 JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Top1DataError(f"{field} must be a non-empty string")
    return value.strip()


def _escape_xml_content(content: str) -> str:
    """Escape untrusted conversation text without changing its natural layout."""

    return escape_xml_text(content)


def load_candidate_names(path: str | Path) -> tuple[str, ...]:
    """Load the ordered closed set of legal generated candidate names."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Top1DataError(f"invalid candidate registry JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError("candidate registry must be a JSON object")
    if payload.get("routing_mode") != ROUTING_MODE:
        raise Top1DataError(f"candidate registry routing_mode must be {ROUTING_MODE!r}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise Top1DataError("candidate registry must contain a non-empty candidates list")

    names: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(candidates):
        name = _nonempty_string(value, field=f"candidates[{index}]")
        if name in seen:
            raise Top1DataError(f"duplicate candidate name: {name!r}")
        names.append(name)
        seen.add(name)
    return tuple(names)


def candidate_registry_payload(candidate_names: Iterable[str]) -> dict[str, Any]:
    """Build the portable candidate-name registry stored with a checkpoint."""

    names = tuple(candidate_names)
    if not names:
        raise Top1DataError("candidate name set cannot be empty")
    return {
        "schema_version": 1,
        "routing_mode": ROUTING_MODE,
        "candidates": list(names),
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


def normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    """Validate and trim source messages while retaining the latest user turn."""

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise Top1DataError("messages must be a sequence")
    if not messages:
        raise Top1DataError("messages cannot be empty")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise Top1DataError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise Top1DataError(f"messages[{index}] role/content must be strings")
        role = role.strip()
        content = content.strip()
        if role not in ALLOWED_MESSAGE_ROLES:
            raise Top1DataError(f"messages[{index}] has unsupported role {role!r}")
        if not content:
            raise Top1DataError(f"messages[{index}].content cannot be empty")
        if role != "system":
            normalized.append({"role": role, "content": content})

    if not normalized:
        raise Top1DataError("conversation has no messages after dropping system turns")
    if normalized[-1]["role"] != "user":
        raise Top1DataError("the final non-system message must have role 'user'")

    current = dict(normalized[-1])
    current["content"] = _truncate_middle(
        current["content"], MAX_HISTORY_CHARACTERS, LATEST_TRUNCATION_MARKER
    )
    remaining_characters = MAX_HISTORY_CHARACTERS - len(current["content"])
    recent_history = normalized[:-1][-(MAX_HISTORY_MESSAGES - 1) :]
    kept: list[dict[str, str]] = []
    for original in reversed(recent_history):
        if remaining_characters <= 0:
            break
        message = dict(original)
        limit = remaining_characters
        if message["role"] in {"assistant", "tool"}:
            limit = min(limit, MAX_ASSISTANT_HISTORY_CHARACTERS)
        message["content"] = _truncate_middle(
            message["content"], limit, HISTORY_TRUNCATION_MARKER
        )
        if message["content"]:
            kept.append(message)
            remaining_characters -= len(message["content"])
    return tuple([*reversed(kept), current])


def messages_from_row(row: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    """Read the canonical messages field from one training row."""

    if "messages" not in row:
        raise Top1DataError("training row must contain messages")
    return normalize_messages(row["messages"])


def target_candidate_name(row: Mapping[str, Any]) -> str:
    """Read the canonical candidate-name label from one training row."""

    return _nonempty_string(
        row.get("target_candidate_name"), field="target_candidate_name"
    )


def build_user_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    """Serialize conversation data using the fixed XML routing envelope."""

    normalized = normalize_messages(messages)
    lines = ["<routing_input>", "<history>"]
    lines.extend(
        f'<message role="{message["role"]}">'
        f'{_escape_xml_content(message["content"])}</message>'
        for message in normalized[:-1]
    )
    lines.extend(
        (
            "</history>",
            "<current_user_request>"
            f'{_escape_xml_content(normalized[-1]["content"])}'
            "</current_user_request>",
            "</routing_input>",
        )
    )
    return "\n".join(lines)


def render_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: str,
) -> str:
    """Apply the model chat template to one normalized Top1 request."""

    system_prompt = _nonempty_string(system_prompt, field="system_prompt")
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(messages)},
    ]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if getattr(tokenizer, "chat_template", None) and callable(apply_template):
        return apply_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return (
        f"System: {system_prompt}\n"
        f"User: {chat_messages[1]['content']}\n"
        "Assistant:"
    )


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Encode text without adding tokenizer-level special tokens."""

    try:
        values = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    except TypeError:
        values = tokenizer.encode(text, add_special_tokens=False)
    return [int(value) for value in values]


def fit_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    system_prompt: str,
    *,
    max_prompt_tokens: int,
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Fit a prompt by dropping old history before truncating the latest turn."""

    if max_prompt_tokens < 1:
        raise Top1DataError("max_prompt_tokens must be positive")
    normalized = list(normalize_messages(messages))

    def render(candidate_messages: Sequence[Mapping[str, Any]]) -> tuple[str, list[int]]:
        prompt = render_prompt(tokenizer, candidate_messages, system_prompt)
        return prompt, encode_text(tokenizer, prompt)

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
            best = candidate_prompt, (current,)
            low = limit + 1
        else:
            high = limit - 1
    if best is None:
        raise Top1DataError(
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
        raise Top1DataError("tokenizer must define a non-negative eos_token_id")
    result: dict[str, tuple[int, ...]] = {}
    used: dict[tuple[int, ...], str] = {}
    for raw_name in candidate_names:
        name = _nonempty_string(raw_name, field="candidate name")
        ids = tuple(encode_text(tokenizer, name))
        if not ids:
            raise Top1DataError(f"candidate name {name!r} tokenizes to an empty path")
        if eos_token_id in ids:
            raise Top1DataError(f"candidate name {name!r} contains EOS")
        previous = used.get(ids)
        if previous is not None:
            raise Top1DataError(
                f"candidate names {previous!r} and {name!r} share one token sequence"
            )
        result[name] = ids
        used[ids] = name
    if not result:
        raise Top1DataError("candidate name set cannot be empty")
    return result


def prepare_router_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    candidate_tokens: Mapping[str, Sequence[int]],
    max_length: int,
    system_prompt: str,
) -> PreparedPrompt:
    """Build the sole prompt contract shared by training and inference.

    The prompt always reserves space for the longest legal candidate path plus EOS.
    Its fitted content therefore cannot depend on the target label.
    """

    if not candidate_tokens:
        raise Top1DataError("candidate token set cannot be empty")
    reserved_target_tokens = max(len(tokens) + 1 for tokens in candidate_tokens.values())
    if max_length <= reserved_target_tokens:
        raise Top1DataError("max_length leaves no room for a router prompt")
    prompt, fitted_messages = fit_prompt(
        tokenizer,
        messages,
        system_prompt,
        max_prompt_tokens=max_length - reserved_target_tokens,
    )
    prompt_ids = tuple(encode_text(tokenizer, prompt))
    if len(prompt_ids) + reserved_target_tokens > max_length:
        raise Top1DataError("fitted router prompt exceeds max_length")
    return PreparedPrompt(
        text=prompt,
        input_ids=prompt_ids,
        fitted_messages=fitted_messages,
        reserved_target_tokens=reserved_target_tokens,
    )


def prompt_implementation_sha256() -> str:
    """Fingerprint every function and constant that can change prompt tokens."""

    functions = (
        _nonempty_string,
        _escape_xml_content,
        _truncate_middle,
        normalize_messages,
        messages_from_row,
        build_user_prompt,
        render_prompt,
        encode_text,
        fit_prompt,
        prepare_router_prompt,
    )
    payload = {
        "constants": {
            "conversation_template": CONVERSATION_TEMPLATE,
            "max_history_messages": MAX_HISTORY_MESSAGES,
            "max_history_characters": MAX_HISTORY_CHARACTERS,
            "max_assistant_history_characters": MAX_ASSISTANT_HISTORY_CHARACTERS,
            "latest_truncation_marker": LATEST_TRUNCATION_MARKER,
            "history_truncation_marker": HISTORY_TRUNCATION_MARKER,
            "allowed_message_roles": sorted(ALLOWED_MESSAGE_ROLES),
        },
        "functions": {
            function.__name__: inspect.getsource(function) for function in functions
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def tokenizer_prompt_contract(
    tokenizer: Any,
    *,
    transformers_version: str,
) -> dict[str, Any]:
    """Describe tokenizer state that can alter prompt or candidate tokens."""

    chat_template = getattr(tokenizer, "chat_template", None)
    serialized_template = json.dumps(
        chat_template,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "transformers_version": transformers_version,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(serialized_template).hexdigest(),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
    }


def validate_training_rows(
    rows: Sequence[Mapping[str, Any]],
    candidate_names: Iterable[str],
    *,
    source: str | Path,
) -> dict[str, Any]:
    """Validate canonical Top1 rows and summarize their supervision."""

    if not rows:
        raise Top1DataError(f"training data is empty: {source}")
    ordered_names = tuple(candidate_names)
    legal_names = set(ordered_names)
    counts: Counter[str] = Counter()
    multi_turn = 0
    for row_number, row in enumerate(rows, start=1):
        try:
            messages = messages_from_row(row)
            name = target_candidate_name(row)
            if name not in legal_names:
                raise Top1DataError(f"unknown target candidate name: {name!r}")
        except Top1DataError as exc:
            raise Top1DataError(f"{source}:{row_number}: {exc}") from exc
        counts[name] += 1
        multi_turn += len(messages) > 1
    return {
        "rows": len(rows),
        "multi_turn_rows": multi_turn,
        "candidate_counts": {
            name: counts[name] for name in ordered_names if counts[name]
        },
    }


def validate_memorization_rows(
    rows: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
    *,
    source: str | Path,
) -> dict[str, Any]:
    """Validate the structured description-to-candidate memorization dataset."""

    report = validate_training_rows(rows, candidate_names, source=source)
    seen_ids: set[str] = set()
    description_counts: Counter[str] = Counter()
    candidate_description_counts: dict[str, Counter[str]] = {
        name: Counter() for name in candidate_names
    }
    for row_number, row in enumerate(rows, start=1):
        try:
            sample_id = _nonempty_string(row.get("id"), field="id")
            if sample_id in seen_ids:
                raise Top1DataError(f"duplicate id: {sample_id!r}")
            seen_ids.add(sample_id)
            source_type = _nonempty_string(
                row.get("source_type"),
                field="source_type",
            )
            if source_type != MEMORIZATION_SOURCE_TYPE:
                raise Top1DataError(
                    f"source_type must be {MEMORIZATION_SOURCE_TYPE!r}"
                )
            description_type = _nonempty_string(
                row.get("description_type"),
                field="description_type",
            )
            if description_type not in MEMORIZATION_DESCRIPTION_TYPES:
                raise Top1DataError(
                    f"unsupported description_type: {description_type!r}"
                )
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 1:
                raise Top1DataError(
                    "memorization rows must contain exactly one user message"
                )
            message = messages[0]
            if not isinstance(message, Mapping) or message.get("role") != "user":
                raise Top1DataError(
                    "memorization rows must contain exactly one user message"
                )
            _nonempty_string(message.get("content"), field="messages[0].content")
            target = target_candidate_name(row)
        except Top1DataError as exc:
            raise Top1DataError(f"{source}:{row_number}: {exc}") from exc
        description_counts[description_type] += 1
        candidate_description_counts[target][description_type] += 1

    missing_candidates = [
        name for name in candidate_names if not report["candidate_counts"].get(name)
    ]
    if missing_candidates:
        raise Top1DataError(
            "memorization data must cover every candidate: "
            + ", ".join(missing_candidates)
        )
    return {
        **report,
        "source_type": MEMORIZATION_SOURCE_TYPE,
        "description_type_counts": {
            name: description_counts[name]
            for name in MEMORIZATION_DESCRIPTION_TYPES
        },
        "candidate_description_type_counts": {
            candidate: {
                name: candidate_description_counts[candidate][name]
                for name in MEMORIZATION_DESCRIPTION_TYPES
            }
            for candidate in candidate_names
        },
    }


def prepare_example(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    candidate_tokens: Mapping[str, Sequence[int]],
    max_length: int,
    system_prompt: str,
) -> PreparedExample:
    """Create the sole training representation and its inspectable SFT form."""

    name = target_candidate_name(row)
    if name not in candidate_tokens:
        raise Top1DataError(f"unknown target candidate name: {name!r}")
    target_ids = [*map(int, candidate_tokens[name]), int(tokenizer.eos_token_id)]
    normalized_messages = messages_from_row(row)
    prepared_prompt = prepare_router_prompt(
        tokenizer,
        normalized_messages,
        candidate_tokens=candidate_tokens,
        max_length=max_length,
        system_prompt=system_prompt,
    )
    fitted_messages = prepared_prompt.fitted_messages
    prompt_ids = list(prepared_prompt.input_ids)
    input_ids = [*prompt_ids, *target_ids]
    source_messages = row["messages"]
    source_non_system = [
        message
        for message in source_messages
        if isinstance(message, Mapping) and message.get("role", "").strip() != "system"
    ]
    source_current = str(source_non_system[-1]["content"]).strip()
    return PreparedExample(
        encoded={
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + target_ids,
        },
        sft_row={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(fitted_messages)},
                {"role": "assistant", "content": name},
            ]
        },
        diagnostics={
            "target_candidate_name": name,
            "original_message_count": len(source_non_system),
            "normalized_message_count": len(normalized_messages),
            "fitted_message_count": len(fitted_messages),
            "history_messages_dropped": max(
                0,
                (len(source_non_system) - 1) - (len(fitted_messages) - 1),
            ),
            "current_user_truncated": fitted_messages[-1]["content"] != source_current,
            "prompt_tokens": len(prompt_ids),
            "target_tokens": len(target_ids),
            "reserved_target_tokens": prepared_prompt.reserved_target_tokens,
            "input_tokens": len(input_ids),
        },
    )
