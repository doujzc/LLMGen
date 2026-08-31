#!/usr/bin/env python
"""Probe constrained decoding on a running vLLM simple ``/generate`` API.

The probe distinguishes a static token whitelist from the request-level
logits processor required by LLMGen's dynamic Trie.  It imports no model,
Transformers, Torch, or torch_npu in the client process.

For the exact logits-processor probe, the vLLM server must be able to import
this file.  Start it with the scripts directory on ``PYTHONPATH``::

    PYTHONPATH=/path/to/LLMGen/scripts:$PYTHONPATH \
      python -m vllm.entrypoints.api_server ...

Then run::

    python scripts/probe_vllm_constrained_decoding.py \
      --model-dir /path/to/router/model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_DEFAULT_URL = "http://127.0.0.1:18000/generate"
_DEFAULT_PROCESSOR_QUALNAME = (
    "probe_vllm_constrained_decoding."
    "create_forced_sequence_logits_processor"
)


class ProbeError(RuntimeError):
    """Raised when the endpoint or local probe inputs are unusable."""


class ForcedSequenceLogitsProcessor:
    """Minimal JSON-constructible processor used by the remote probe."""

    def __init__(self, *, sequence: Sequence[int], eos_token_id: int) -> None:
        normalized = tuple(int(token_id) for token_id in sequence)
        if not normalized or any(token_id < 0 for token_id in normalized):
            raise ValueError("sequence must contain non-negative token IDs")
        if int(eos_token_id) < 0:
            raise ValueError("eos_token_id must be non-negative")
        self.sequence = normalized
        self.eos_token_id = int(eos_token_id)

    def clone(self) -> "ForcedSequenceLogitsProcessor":
        return self

    def __call__(self, *args: Any) -> Any:
        # Support both vLLM processor ABIs:
        #   (output_token_ids, scores)
        #   (prompt_token_ids, output_token_ids, scores)
        if len(args) not in {2, 3}:
            raise TypeError(
                "expected (output_ids, scores) or (prompt_ids, output_ids, scores)"
            )
        output_token_ids = args[-2]
        scores = args[-1]
        position = len(output_token_ids)
        forced_token_id = (
            self.sequence[position]
            if position < len(self.sequence)
            else self.eos_token_id
        )
        vocabulary_size = int(scores.shape[-1])
        if forced_token_id >= vocabulary_size:
            raise RuntimeError("forced token is outside the model vocabulary")
        masked = scores.new_full(scores.shape, -float("inf"))
        masked[forced_token_id] = scores[forced_token_id]
        return masked


def create_forced_sequence_logits_processor(
    *, sequence: Sequence[int], eos_token_id: int
) -> ForcedSequenceLogitsProcessor:
    """Factory resolved by servers supporting processor descriptors."""

    return ForcedSequenceLogitsProcessor(
        sequence=sequence,
        eos_token_id=eos_token_id,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=_DEFAULT_URL)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help=(
            "Router model directory; used only to read config.json, "
            "virtual_tokens.txt, and tokenizer.json"
        ),
    )
    parser.add_argument("--force-token-id", type=int)
    parser.add_argument("--force-token-text")
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument(
        "--prompt",
        default="请只输出约束解码探针指定的特殊 token。",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--static-tokens", type=int, default=3)
    parser.add_argument(
        "--processor-qualname",
        default=_DEFAULT_PROCESSOR_QUALNAME,
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Only test allowed_token_ids; do not test a processor descriptor.",
    )
    return parser.parse_args(argv)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProbeError(f"JSON root is not an object: {path}")
    return payload


def _first_integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            result = _first_integer(item)
            if result is not None:
                return result
    return None


def _token_id_from_tokenizer_json(path: Path, token_text: str) -> int:
    payload = _read_json_object(path)
    added_tokens = payload.get("added_tokens")
    if isinstance(added_tokens, list):
        for item in added_tokens:
            if (
                isinstance(item, Mapping)
                and item.get("content") == token_text
                and isinstance(item.get("id"), int)
            ):
                return int(item["id"])
    model = payload.get("model")
    vocab = model.get("vocab") if isinstance(model, Mapping) else None
    if isinstance(vocab, Mapping) and isinstance(vocab.get(token_text), int):
        return int(vocab[token_text])
    raise ProbeError(f"token is absent from tokenizer.json: {token_text!r}")


def _resolve_probe_tokens(args: argparse.Namespace) -> tuple[int, str | None, int]:
    force_token_id = args.force_token_id
    force_token_text = args.force_token_text
    eos_token_id = args.eos_token_id
    if args.model_dir is not None:
        model_dir = args.model_dir.expanduser().resolve()
        if force_token_text is None:
            virtual_tokens_path = model_dir / "virtual_tokens.txt"
            try:
                force_token_text = next(
                    line.strip()
                    for line in virtual_tokens_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
            except (OSError, StopIteration) as exc:
                raise ProbeError(
                    f"cannot select a virtual token from {virtual_tokens_path}"
                ) from exc
        if force_token_id is None:
            force_token_id = _token_id_from_tokenizer_json(
                model_dir / "tokenizer.json", force_token_text
            )
        if eos_token_id is None:
            for filename in ("generation_config.json", "config.json"):
                path = model_dir / filename
                if not path.is_file():
                    continue
                eos_token_id = _first_integer(
                    _read_json_object(path).get("eos_token_id")
                )
                if eos_token_id is not None:
                    break
    if force_token_id is None or eos_token_id is None:
        raise ProbeError(
            "provide --model-dir, or provide both --force-token-id and "
            "--eos-token-id"
        )
    if force_token_id < 0 or eos_token_id < 0:
        raise ProbeError("token IDs must be non-negative")
    if force_token_id == eos_token_id:
        raise ProbeError("the forced token must differ from EOS")
    return force_token_id, force_token_text, eos_token_id


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise ProbeError(f"HTTP {exc.code}: {details[:2000]}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ProbeError(f"cannot call {url}: {exc}") from exc
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"endpoint returned invalid JSON: {body[:500]!r}") from exc
    if not isinstance(result, Mapping):
        raise ProbeError("endpoint JSON root is not an object")
    return result


def _integer_sequence(value: Any) -> list[int] | None:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return [int(item) for item in value]
        for item in value:
            nested = _integer_sequence(item)
            if nested is not None:
                return nested
    return None


def _completion_token_ids(response: Mapping[str, Any]) -> list[int] | None:
    for key in ("token_ids", "output_token_ids", "completion_token_ids"):
        result = _integer_sequence(response.get(key))
        if result is not None:
            return result
    for container_name in ("outputs", "choices"):
        containers = response.get(container_name)
        if not isinstance(containers, list) or not containers:
            continue
        first = containers[0]
        if not isinstance(first, Mapping):
            continue
        for key in ("token_ids", "output_token_ids", "completion_token_ids"):
            result = _integer_sequence(first.get(key))
            if result is not None:
                return result
    return None


def _completion_text(response: Mapping[str, Any], prompt: str) -> str | None:
    value: Any = response.get("generated_text", response.get("text"))
    if value is None:
        for container_name in ("outputs", "choices"):
            containers = response.get(container_name)
            if (
                isinstance(containers, list)
                and containers
                and isinstance(containers[0], Mapping)
            ):
                value = containers[0].get("text")
                if value is not None:
                    break
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    return value[len(prompt) :] if value.startswith(prompt) else value


def _evidence(
    response: Mapping[str, Any],
    *,
    prompt: str,
) -> dict[str, Any]:
    token_ids = _completion_token_ids(response)
    text = _completion_text(response, prompt)
    return {
        "token_ids": token_ids,
        "completion_text": text,
        "response_keys": sorted(str(key) for key in response),
    }


def _run_static_probe(
    args: argparse.Namespace,
    *,
    force_token_id: int,
    force_token_text: str | None,
) -> dict[str, Any]:
    payload = {
        "prompt": args.prompt,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": args.static_tokens,
        "min_tokens": args.static_tokens,
        "skip_special_tokens": False,
        "allowed_token_ids": [force_token_id],
    }
    try:
        response = _post_json(args.url, payload, timeout=args.timeout)
    except ProbeError as exc:
        return {"supported": False, "enforced": False, "error": str(exc)}
    evidence = _evidence(response, prompt=args.prompt)
    token_ids = evidence["token_ids"]
    if token_ids is not None:
        enforced = (
            len(token_ids) == args.static_tokens
            and all(token_id == force_token_id for token_id in token_ids)
        )
        verification = "completion_token_ids"
    elif force_token_text is not None and evidence["completion_text"] is not None:
        enforced = evidence["completion_text"].count(force_token_text) == args.static_tokens
        verification = "completion_text"
    else:
        enforced = False
        verification = "inconclusive_no_token_ids_or_token_text"
    return {
        "supported": True,
        "enforced": enforced,
        "verification": verification,
        **evidence,
    }


def _run_processor_probe(
    args: argparse.Namespace,
    *,
    force_token_id: int,
    force_token_text: str | None,
    eos_token_id: int,
) -> dict[str, Any]:
    payload = {
        "prompt": args.prompt,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 2,
        "min_tokens": 1,
        "skip_special_tokens": False,
        "logits_processors": [
            {
                "qualname": args.processor_qualname,
                "kwargs": {
                    "sequence": [force_token_id],
                    "eos_token_id": eos_token_id,
                },
            }
        ],
    }
    try:
        response = _post_json(args.url, payload, timeout=args.timeout)
    except ProbeError as exc:
        return {"supported": False, "enforced": False, "error": str(exc)}
    evidence = _evidence(response, prompt=args.prompt)
    token_ids = evidence["token_ids"]
    if token_ids is not None:
        # Some engines retain the sampled EOS in token_ids; others remove it.
        enforced = token_ids in ([force_token_id], [force_token_id, eos_token_id])
        verification = "completion_token_ids"
    elif force_token_text is not None and evidence["completion_text"] is not None:
        enforced = evidence["completion_text"].count(force_token_text) == 1
        verification = "completion_text"
    else:
        enforced = False
        verification = "inconclusive_no_token_ids_or_token_text"
    return {
        "supported": True,
        "enforced": enforced,
        "verification": verification,
        **evidence,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.timeout <= 0 or args.static_tokens < 1:
        raise ProbeError("timeout and static-tokens must be positive")
    force_token_id, force_token_text, eos_token_id = _resolve_probe_tokens(args)
    static_result = _run_static_probe(
        args,
        force_token_id=force_token_id,
        force_token_text=force_token_text,
    )
    processor_result: dict[str, Any] | None = None
    if not args.static_only:
        processor_result = _run_processor_probe(
            args,
            force_token_id=force_token_id,
            force_token_text=force_token_text,
            eos_token_id=eos_token_id,
        )
    report = {
        "url": args.url,
        "force_token_id": force_token_id,
        "force_token_text": force_token_text,
        "eos_token_id": eos_token_id,
        "static_allowed_token_ids": static_result,
        "request_level_logits_processor": processor_result,
        "llmgen_trie_compatible": (
            None
            if processor_result is None
            else bool(
                processor_result.get("supported")
                and processor_result.get("enforced")
            )
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if processor_result is not None:
        return 0 if report["llmgen_trie_compatible"] else 1
    return 0 if static_result.get("enforced") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
