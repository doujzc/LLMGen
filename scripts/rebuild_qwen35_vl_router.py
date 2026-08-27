#!/usr/bin/env python
"""Rebuild a full Qwen3.5 VL Router from a fine-tuned text-only export.

The Router training flow intentionally uses ``AutoModelForCausalLM``.  For a
Qwen3.5 VL base this saves only ``Qwen3_5ForCausalLM`` and drops the visual
tower.  This script restores the original full-model container and visual
weights, replaces its language model and LM head with the trained Router, and
saves a deployable ``Qwen3_5ForConditionalGeneration`` bundle.

The original full base model is required.  A text-only checkpoint cannot
reconstruct missing visual weights by itself.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger("llmgen.rebuild_qwen35_vl_router")
_FULL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_FULL_MODEL_TYPE = "qwen3_5"
_TEXT_ARCHITECTURE = "Qwen3_5ForCausalLM"
_TEXT_MODEL_TYPE = "qwen3_5_text"
_STRUCTURAL_TEXT_FIELDS = (
    "attention_bias",
    "attention_dropout",
    "attn_output_gate",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "layer_types",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "max_position_embeddings",
    "mlp_only_layers",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_parameters",
)
_ROUTER_FILES_TO_PRESERVE = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "router_manifest.json",
    "skill_decode_map.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "virtual_tokens.txt",
)


class ReconstructionError(RuntimeError):
    """Raised when the two source bundles cannot be safely combined."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        required=True,
        help="Original complete Qwen3.5-2B VL model directory.",
    )
    parser.add_argument(
        "--router-model",
        required=True,
        help="Fine-tuned text-only Router model directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory; it must not already exist.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="CPU load/save dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum safetensors shard size passed to save_pretrained.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verify-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reload the reconstructed full model before publishing it.",
    )
    return parser.parse_args(argv)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReconstructionError(f"required JSON file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReconstructionError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionError(f"JSON root must be an object: {path}")
    return payload


def _validate_source_configs(
    base_config: Mapping[str, Any], router_config: Mapping[str, Any]
) -> None:
    """Require a full VL base and its structurally compatible text export."""

    base_architectures = base_config.get("architectures")
    if (
        base_config.get("model_type") != _FULL_MODEL_TYPE
        or not isinstance(base_config.get("vision_config"), Mapping)
        or not isinstance(base_config.get("text_config"), Mapping)
        or not isinstance(base_architectures, list)
        or _FULL_ARCHITECTURE not in base_architectures
    ):
        raise ReconstructionError(
            "base model must be a complete Qwen3.5 VL bundle with "
            f"architecture {_FULL_ARCHITECTURE!r}, text_config, and vision_config"
        )

    router_architectures = router_config.get("architectures")
    if (
        router_config.get("model_type") != _TEXT_MODEL_TYPE
        or not isinstance(router_architectures, list)
        or _TEXT_ARCHITECTURE not in router_architectures
    ):
        raise ReconstructionError(
            "Router model must be the text-only Qwen3.5 export with "
            f"architecture {_TEXT_ARCHITECTURE!r}"
        )

    base_text = base_config["text_config"]
    mismatches: list[str] = []
    for name in _STRUCTURAL_TEXT_FIELDS:
        base_value = base_text.get(name)
        router_value = router_config.get(name)
        if base_value != router_value:
            mismatches.append(
                f"{name}: base={base_value!r}, router={router_value!r}"
            )
    if mismatches:
        preview = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            preview += f"; ... and {len(mismatches) - 8} more"
        raise ReconstructionError(
            "base and Router text architectures are incompatible: " + preview
        )

    vision_config = base_config["vision_config"]
    if vision_config.get("out_hidden_size") != router_config.get("hidden_size"):
        raise ReconstructionError(
            "base vision output size does not match Router hidden size: "
            f"vision={vision_config.get('out_hidden_size')!r}, "
            f"router={router_config.get('hidden_size')!r}"
        )
    router_vocab_size = router_config.get("vocab_size")
    if not isinstance(router_vocab_size, int) or router_vocab_size < 1:
        raise ReconstructionError("Router config has an invalid vocab_size")
    for name in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        token_id = base_config.get(name)
        if not isinstance(token_id, int) or not 0 <= token_id < router_vocab_size:
            raise ReconstructionError(
                f"base {name}={token_id!r} is outside Router vocabulary "
                f"size {router_vocab_size}"
            )


def _nested_attribute(root: Any, path: str) -> Any:
    current = root
    for component in path.split("."):
        if not hasattr(current, component):
            raise ReconstructionError(
                f"model has no required attribute {path!r}; stopped at {component!r}"
            )
        current = getattr(current, component)
    return current


def _detach_base_language_modules(full_model: Any) -> tuple[Any, Any]:
    """Free the base language modules before loading the trained replacement."""

    model_container = _nested_attribute(full_model, "model")
    if not hasattr(model_container, "language_model"):
        raise ReconstructionError(
            "full Qwen3.5 model has no model.language_model submodule"
        )
    if not hasattr(model_container, "visual"):
        raise ReconstructionError("full Qwen3.5 model has no model.visual submodule")
    if not hasattr(full_model, "lm_head"):
        raise ReconstructionError("full Qwen3.5 model has no lm_head")
    old_language_model = model_container.language_model
    old_lm_head = full_model.lm_head
    model_container.language_model = None
    full_model.lm_head = None
    return old_language_model, old_lm_head


def _attach_router_language_modules(full_model: Any, router_model: Any) -> None:
    """Install trained text modules while preserving the base visual tower."""

    model_container = _nested_attribute(full_model, "model")
    router_language_model = _nested_attribute(router_model, "model")
    router_lm_head = _nested_attribute(router_model, "lm_head")
    router_config = _nested_attribute(router_model, "config")
    full_config = _nested_attribute(full_model, "config")

    model_container.language_model = router_language_model
    full_model.lm_head = router_lm_head
    full_config.text_config = router_config
    full_config.architectures = [_FULL_ARCHITECTURE]
    if hasattr(router_config, "tie_word_embeddings"):
        full_config.tie_word_embeddings = bool(router_config.tie_word_embeddings)
    if hasattr(model_container, "config"):
        model_container.config = full_config
    if hasattr(router_language_model, "config"):
        router_language_model.config = router_config
    if hasattr(full_model, "vocab_size") and hasattr(router_config, "vocab_size"):
        full_model.vocab_size = int(router_config.vocab_size)
    if hasattr(router_model, "generation_config"):
        full_model.generation_config = router_model.generation_config


def _copy_router_files(router_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in _ROUTER_FILES_TO_PRESERVE:
        source = router_dir / name
        if not source.is_file():
            continue
        shutil.copy2(source, output_dir / name)
        copied.append(name)
    for required in (
        "router_manifest.json",
        "skill_decode_map.json",
        "virtual_tokens.txt",
    ):
        if required not in copied:
            raise ReconstructionError(
                f"Router source is missing required deployment artifact: {required}"
            )
    return copied


def _transformers_stack() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ReconstructionError(
            "reconstruction requires torch and transformers"
        ) from exc
    return torch, transformers


def _load_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    value = getattr(torch, name, None)
    if value is None:
        raise ReconstructionError(f"torch does not expose dtype {name!r}")
    return value


def _full_model_class(transformers: Any) -> Any:
    model_class = getattr(transformers, _FULL_ARCHITECTURE, None)
    if model_class is not None:
        return model_class
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise ReconstructionError(
        "installed transformers has no Qwen3.5 multimodal model class"
    )


def _model_load_kwargs(args: argparse.Namespace, dtype: Any) -> dict[str, Any]:
    return {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(args.trust_remote_code),
    }


def _verify_reconstructed_config(output_dir: Path, tokenizer_size: int) -> None:
    config = _read_json_object(output_dir / "config.json")
    architectures = config.get("architectures")
    text_config = config.get("text_config")
    if (
        config.get("model_type") != _FULL_MODEL_TYPE
        or not isinstance(architectures, list)
        or _FULL_ARCHITECTURE not in architectures
        or not isinstance(config.get("vision_config"), Mapping)
        or not isinstance(text_config, Mapping)
    ):
        raise ReconstructionError(
            "saved output is not a complete Qwen3.5 conditional-generation model"
        )
    if int(text_config.get("vocab_size", -1)) != tokenizer_size:
        raise ReconstructionError(
            "saved text_config vocab_size disagrees with Router tokenizer: "
            f"config={text_config.get('vocab_size')!r}, tokenizer={tokenizer_size}"
        )


def rebuild(args: argparse.Namespace) -> Path:
    base_dir = Path(args.base_model).expanduser().resolve()
    router_dir = Path(args.router_model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for label, directory in (("base", base_dir), ("router", router_dir)):
        if not directory.is_dir():
            raise ReconstructionError(f"{label} model directory does not exist: {directory}")
    if output_dir.exists():
        raise ReconstructionError(f"output directory already exists: {output_dir}")

    base_config_dict = _read_json_object(base_dir / "config.json")
    router_config_dict = _read_json_object(router_dir / "config.json")
    _validate_source_configs(base_config_dict, router_config_dict)
    LOGGER.info("source configurations are structurally compatible")

    torch, transformers = _transformers_stack()
    dtype = _load_dtype(torch, args.dtype)
    load_kwargs = _model_load_kwargs(args, dtype)
    full_model_cls = _full_model_class(transformers)
    text_model_cls = getattr(transformers, "AutoModelForCausalLM", None)
    tokenizer_cls = getattr(transformers, "AutoTokenizer", None)
    processor_cls = getattr(transformers, "AutoProcessor", None)
    if text_model_cls is None or tokenizer_cls is None or processor_cls is None:
        raise ReconstructionError(
            "installed transformers lacks AutoModelForCausalLM/AutoTokenizer/AutoProcessor"
        )

    LOGGER.info("loading trained Router tokenizer and original VL processor")
    router_tokenizer = tokenizer_cls.from_pretrained(
        str(router_dir), trust_remote_code=bool(args.trust_remote_code)
    )
    processor = processor_cls.from_pretrained(
        str(base_dir), trust_remote_code=bool(args.trust_remote_code)
    )
    tokenizer_size = len(router_tokenizer)
    if int(router_config_dict.get("vocab_size", -1)) != tokenizer_size:
        raise ReconstructionError(
            "Router config vocab_size disagrees with its tokenizer: "
            f"config={router_config_dict.get('vocab_size')!r}, "
            f"tokenizer={tokenizer_size}"
        )

    LOGGER.info("loading original full VL model from %s", base_dir)
    full_model = full_model_cls.from_pretrained(str(base_dir), **load_kwargs)
    old_language_model, old_lm_head = _detach_base_language_modules(full_model)
    del old_language_model, old_lm_head
    gc.collect()

    LOGGER.info("loading trained text-only Router model from %s", router_dir)
    router_model = text_model_cls.from_pretrained(str(router_dir), **load_kwargs)
    _attach_router_language_modules(full_model, router_model)
    if not hasattr(processor, "tokenizer"):
        raise ReconstructionError("original VL processor has no tokenizer attribute")
    processor.tokenizer = router_tokenizer

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.rebuild-", dir=output_dir.parent)
    )
    try:
        LOGGER.info("saving reconstructed full VL model to staging directory %s", staging_dir)
        full_model.save_pretrained(
            str(staging_dir),
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        processor.save_pretrained(str(staging_dir))
        router_tokenizer.save_pretrained(str(staging_dir))
        copied = _copy_router_files(router_dir, staging_dir)
        _verify_reconstructed_config(staging_dir, tokenizer_size)
        LOGGER.info("preserved Router files: %s", ", ".join(copied))

        del router_model, full_model, processor, router_tokenizer
        gc.collect()

        if args.verify_load:
            LOGGER.info("reloading reconstructed model for verification")
            verified_model, loading_info = full_model_cls.from_pretrained(
                str(staging_dir),
                output_loading_info=True,
                **load_kwargs,
            )
            problems = {
                name: loading_info.get(name)
                for name in ("missing_keys", "unexpected_keys", "mismatched_keys")
                if loading_info.get(name)
            }
            if problems:
                raise ReconstructionError(
                    "reconstructed model did not reload cleanly: "
                    + json.dumps(problems, ensure_ascii=False, default=str)
                )
            if not hasattr(verified_model.config, "vision_config"):
                raise ReconstructionError(
                    "reloaded model configuration has no vision_config"
                )
            del verified_model
            gc.collect()

        os.replace(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    LOGGER.info("reconstruction complete: %s", output_dir)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[rebuild-qwen35-vl] %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    try:
        args = _parse_args(argv)
        rebuild(args)
    except ReconstructionError as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
