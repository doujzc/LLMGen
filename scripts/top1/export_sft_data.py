#!/usr/bin/env python3
"""Export Top1 router supervision as standard conversational SFT JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from llmgen.direct_router import (
    CURRENT_CONVERSATION_TEMPLATE,
    candidate_token_sequences,
    load_candidate_registry,
    standard_candidate_sft_row,
)
from llmgen.router import RouterDataError, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Top1 messages + target_candidate_name rows into standard "
            "system/user/assistant SFT JSONL."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--candidate-registry", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument(
        "--tokenizer-name-or-path",
        help=(
            "Optional tokenizer used to reproduce the router's max-length fitting. "
            "Without it, only message-count/character normalization is applied."
        ),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def export_sft_jsonl(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    candidate_registry: str | Path,
    system_prompt_file: str | Path,
    tokenizer: Any | None = None,
    max_length: int = 1024,
) -> dict[str, Any]:
    """Convert a complete Top1 JSONL file and return a compact export report."""

    source = Path(input_jsonl).expanduser().resolve()
    destination = Path(output_jsonl).expanduser().resolve()
    if source == destination:
        raise RouterDataError("input and output JSONL paths must differ")
    prompt_path = Path(system_prompt_file).expanduser()
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise RouterDataError("router system prompt is empty")
    routes = load_candidate_registry(candidate_registry)
    candidate_names = tuple(route.name for route in routes)
    legal_names = set(candidate_names)
    token_sequences = (
        candidate_token_sequences(tokenizer, candidate_names)
        if tokenizer is not None
        else None
    )
    rows = read_jsonl(source)
    if not rows:
        raise RouterDataError("input Top1 JSONL is empty")

    converted: list[dict[str, list[dict[str, str]]]] = []
    counts: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=1):
        try:
            converted_row = standard_candidate_sft_row(
                row,
                legal_candidate_names=legal_names,
                system_prompt=system_prompt,
                tokenizer=tokenizer,
                candidate_name_tokens=token_sequences,
                max_length=max_length,
                conversation_template=CURRENT_CONVERSATION_TEMPLATE,
            )
        except RouterDataError as exc:
            raise RouterDataError(f"{source}:{row_number}: {exc}") from exc
        converted.append(converted_row)
        counts[converted_row["messages"][-1]["content"]] += 1
    write_jsonl(destination, converted)
    return {
        "input": str(source),
        "output": str(destination),
        "rows": len(converted),
        "candidate_counts": {
            name: counts[name] for name in candidate_names if counts[name]
        },
        "token_length_fitted": tokenizer is not None,
        "max_length": max_length if tokenizer is not None else None,
        "conversation_template": CURRENT_CONVERSATION_TEMPLATE,
    }


def _load_tokenizer(args: argparse.Namespace) -> Any | None:
    if not args.tokenizer_name_or_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - real export environment
        raise SystemExit(
            "Tokenizer-aware export requires transformers; install training extras."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise RouterDataError("export tokenizer must define eos_token_id")
    return tokenizer


def main() -> None:
    args = parse_args()
    if args.max_length < 1:
        raise RouterDataError("max_length must be positive")
    report = export_sft_jsonl(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        candidate_registry=args.candidate_registry,
        system_prompt_file=args.system_prompt_file,
        tokenizer=_load_tokenizer(args),
        max_length=args.max_length,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
