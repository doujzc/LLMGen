#!/usr/bin/env python3
"""Merge a trained PEFT router adapter into a self-contained HF model."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


PORTABLE_ARTIFACTS = (
    "router_manifest.json",
    "skill_decode_map.json",
    "virtual_tokens.txt",
    "chat_template.jinja",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = args.adapter.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter_config.json is missing: {adapter}")
    if output.exists():
        raise FileExistsError(f"merge output already exists: {output}")

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - training environment
        raise SystemExit(
            "LoRA export requires the project's transformers and peft dependencies"
        ) from error

    tokenizer_source = (
        adapter
        if (adapter / "tokenizer_config.json").is_file()
        else args.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype="auto",
            **model_kwargs,
        )
    except TypeError:  # Transformers 4.x spelling.
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype="auto",
            **model_kwargs,
        )
    # LoRA routers preserve the resized embedding/lm_head in modules_to_save so
    # newly added virtual code tokens survive adapter checkpoints. Recreate that
    # vocabulary before PEFT loads those tensors from the adapter.
    base.resize_token_embeddings(len(tokenizer))
    adapter_model = PeftModel.from_pretrained(base, str(adapter))
    merged = adapter_model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True)
    merged.save_pretrained(output, safe_serialization=True)

    tokenizer.save_pretrained(output)
    for name in PORTABLE_ARTIFACTS:
        source = adapter / name
        if source.is_file():
            shutil.copy2(source, output / name)


if __name__ == "__main__":
    main()
