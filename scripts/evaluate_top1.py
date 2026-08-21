#!/usr/bin/env python3
"""Evaluate one immutable Top1 model artifact as an independent run."""

from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from llmgen.evaluation import (
    aggregate_predictions,
    load_backend_decision_policy,
    prediction_from_generation,
)
from llmgen.experiment import (
    EVALUATION_RUN_SCHEMA_VERSION,
    RunStore,
    append_jsonl,
    canonical_sha256,
    compact_beijing_now,
    directory_file_manifest,
    git_snapshot,
    load_and_verify_model_artifact,
    read_json_object,
    system_snapshot,
    utc_now,
)
from llmgen.top1 import (
    CONVERSATION_TEMPLATE,
    CandidateNameTokenTrie,
    INFERENCE_DECISION_RULE,
    MAX_ASSISTANT_HISTORY_CHARACTERS,
    MAX_HISTORY_CHARACTERS,
    MAX_HISTORY_MESSAGES,
    ROUTING_MODE,
    TARGET_CONTRACT,
    Top1DataError,
    INFERENCE_SCORING_RULE,
    candidate_token_sequences,
    load_candidate_names,
    messages_from_row,
    prepare_router_prompt,
    prompt_implementation_sha256,
    read_jsonl,
    sha256_file,
    target_candidate_name,
    tokenizer_prompt_contract,
    write_json,
)


DECODING_MODES = ("greedy", "beam_search")
DEVICE_MAP_MODES = ("auto", "balanced", "balanced_low_0", "sequential")


def _parse_route_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("route threshold must be a number") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("route threshold must be between 0 and 1")
    return threshold


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrained-generate one candidate and record one Top1 evaluation run."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation-root", default="runs/evaluations/top1")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--suite-id")
    parser.add_argument("--candidate-registry")
    parser.add_argument("--decision-policy")
    parser.add_argument("--route-threshold", type=_parse_route_threshold)
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--decoding-mode",
        choices=DECODING_MODES,
        default="greedy",
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--device-map",
        choices=DEVICE_MAP_MODES,
        help=(
            "Shard model layers across available devices with Transformers/Accelerate. "
            "Omit to retain single-device evaluation."
        ),
    )
    parser.add_argument(
        "--max-memory",
        action="append",
        default=[],
        metavar="DEVICE=LIMIT[,DEVICE=LIMIT...]",
        help=(
            "Optional per-device memory limits used with --device-map, for example "
            "0=22GiB,1=22GiB,cpu=64GiB. May be repeated."
        ),
    )
    parser.add_argument("--history-ablation", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def _parse_max_memory(values: Sequence[str]) -> dict[int | str, str]:
    """Parse Accelerate max-memory entries without depending on Accelerate."""

    parsed: dict[int | str, str] = {}
    for option in values:
        for raw_entry in option.split(","):
            entry = raw_entry.strip()
            device_text, separator, limit_text = entry.partition("=")
            device_text = device_text.strip()
            limit_text = limit_text.strip()
            if not separator or not device_text or not limit_text:
                raise Top1DataError(
                    "max_memory entries must use DEVICE=LIMIT, for example 0=22GiB"
                )
            if device_text.isdecimal():
                device: int | str = int(device_text)
            elif device_text == "cpu":
                device = device_text
            else:
                raise Top1DataError(
                    "max_memory device must be a non-negative GPU index or 'cpu'"
                )
            if device in parsed:
                raise Top1DataError(f"duplicate max_memory device: {device_text}")
            parsed[device] = limit_text
    return parsed


def _json_max_memory(max_memory: Mapping[int | str, str]) -> dict[str, str] | None:
    """Normalize max-memory keys for durable JSON metadata."""

    if not max_memory:
        return None
    return {str(device): limit for device, limit in max_memory.items()}


def _validate_single_process() -> None:
    """Reject torchrun-style replication; model sharding uses one writer process."""

    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as exc:
        raise Top1DataError("WORLD_SIZE must be an integer") from exc
    if world_size != 1:
        raise Top1DataError(
            "evaluation requires one process; expose multiple GPUs with "
            "CUDA_VISIBLE_DEVICES and use --device-map instead of torchrun"
        )


def _safe_component(value: str, *, label: str) -> str:
    cleaned = value.strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise Top1DataError(f"{label} must be one non-empty path component")
    return cleaned


def _resolve_bundle_file(
    explicit: str | None,
    model_dir: Path,
    filename: str,
    *,
    label: str,
) -> Path:
    path = Path(explicit).expanduser() if explicit else model_dir / filename
    if not path.is_file():
        raise Top1DataError(f"{label} does not exist: {path}")
    return path.resolve()


def _resolve_decision_policy(explicit: str | None, model_dir: Path) -> Path:
    """Resolve policy from CLI, bundle, or the repository compatibility default."""

    if explicit:
        candidates = (Path(explicit).expanduser(),)
    else:
        candidates = (
            model_dir / "decision_policy.json",
            Path(__file__).resolve().parents[1]
            / "configs"
            / "top1_decision_policy.json",
        )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise Top1DataError(
        "backend decision policy does not exist; pass --decision-policy"
    )


def _load_router_contract(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "router_manifest.json"
    if not manifest_path.is_file():
        raise Top1DataError(f"router manifest does not exist: {manifest_path}")
    manifest = read_json_object(manifest_path)
    schema_version = manifest.get("schema_version")
    if schema_version not in {2, 3, 4}:
        raise Top1DataError("router manifest schema_version must be 2, 3, or 4")
    if manifest.get("routing_mode") != ROUTING_MODE:
        raise Top1DataError("router manifest has an incompatible routing mode")
    if manifest.get("target") != TARGET_CONTRACT:
        raise Top1DataError("router manifest has an incompatible target contract")
    conversation = manifest.get("conversation")
    expected_conversation = {
        "template": CONVERSATION_TEMPLATE,
        "max_history_messages": MAX_HISTORY_MESSAGES,
        "max_history_characters": MAX_HISTORY_CHARACTERS,
        "max_assistant_history_characters": MAX_ASSISTANT_HISTORY_CHARACTERS,
        "implementation_sha256": prompt_implementation_sha256(),
    }
    if conversation != expected_conversation:
        raise Top1DataError("router manifest has an incompatible prompt contract")
    inference = manifest.get("inference")
    expected_inference = {
        2: {
            "decision_rule": "candidate_path_sum_logprob",
            "include_eos": True,
        },
        3: {
            "scoring_rule": "candidate_path_sum_logprob",
            "decision_rule": "backend_group_threshold_v1",
            "include_eos": True,
        },
        4: {
            "scoring_rule": INFERENCE_SCORING_RULE,
            "decision_rule": INFERENCE_DECISION_RULE,
            "include_eos": True,
            "decoding_modes": list(DECODING_MODES),
            "num_return_sequences": 1,
        },
    }[schema_version]
    if inference != expected_inference:
        raise Top1DataError("router manifest has an incompatible inference contract")
    if schema_version >= 3:
        decision_policy = manifest.get("backend_decision_policy")
        expected_policy_rule = (
            "backend_group_threshold_v1"
            if schema_version == 3
            else INFERENCE_DECISION_RULE
        )
        if (
            not isinstance(decision_policy, Mapping)
            or decision_policy.get("decision_rule") != expected_policy_rule
            or not isinstance(decision_policy.get("path"), str)
            or not isinstance(decision_policy.get("sha256"), str)
        ):
            raise Top1DataError(
                "router manifest has invalid backend decision policy metadata"
            )
    max_length = manifest.get("max_length")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise Top1DataError("router manifest max_length must be a positive integer")
    return manifest


def _validate_bundled_decision_policy(
    manifest: Mapping[str, Any],
    *,
    model_dir: Path,
    decision_policy_path: Path,
) -> None:
    """Verify the default policy when evaluation uses the bundled file."""

    metadata = manifest.get("backend_decision_policy")
    if not isinstance(metadata, Mapping):
        return
    bundled_path = (model_dir / str(metadata["path"])).resolve()
    if decision_policy_path == bundled_path and metadata["sha256"] != sha256_file(
        decision_policy_path
    ):
        raise Top1DataError("bundled backend decision policy differs from manifest")


def _apply_router_contract(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    *,
    registry_path: Path,
    prompt_path: Path,
    candidate_names: Sequence[str],
) -> None:
    registry = manifest.get("candidate_registry")
    prompt = manifest.get("system_prompt")
    training = manifest.get("training")
    if not isinstance(registry, Mapping) or not isinstance(prompt, Mapping):
        raise Top1DataError("router manifest is missing prompt bundle metadata")
    if not isinstance(training, Mapping):
        raise Top1DataError("router manifest is missing training metadata")
    if registry.get("sha256") != sha256_file(registry_path):
        raise Top1DataError("candidate registry differs from the training bundle")
    if registry.get("candidate_names") != list(candidate_names):
        raise Top1DataError("candidate order differs from the training bundle")
    if prompt.get("sha256") != sha256_file(prompt_path):
        raise Top1DataError("system prompt differs from the training bundle")

    trained_max_length = int(manifest["max_length"])
    if args.max_length is not None and args.max_length != trained_max_length:
        raise Top1DataError(
            "evaluation max_length must equal the training prompt contract: "
            f"{trained_max_length}"
        )
    args.max_length = trained_max_length

    trained_trust_remote_code = training.get("trust_remote_code")
    if not isinstance(trained_trust_remote_code, bool):
        raise Top1DataError("router manifest trust_remote_code must be boolean")
    if (
        args.trust_remote_code is not None
        and args.trust_remote_code != trained_trust_remote_code
    ):
        raise Top1DataError(
            "evaluation trust_remote_code must equal the training prompt contract"
        )
    args.trust_remote_code = trained_trust_remote_code


def _validate_loaded_tokenizer(
    manifest: Mapping[str, Any],
    *,
    tokenizer: Any,
    candidate_names: Sequence[str],
    candidate_tokens: Mapping[str, Sequence[int]],
    transformers_version: str,
) -> None:
    expected_tokenizer = manifest.get("tokenizer")
    actual_tokenizer = tokenizer_prompt_contract(
        tokenizer,
        transformers_version=transformers_version,
    )
    if expected_tokenizer != actual_tokenizer:
        raise Top1DataError(
            "loaded tokenizer or Transformers version differs from training"
        )
    registry = manifest.get("candidate_registry")
    if not isinstance(registry, Mapping):
        raise Top1DataError("router manifest is missing candidate token sequences")
    actual_sequences = {
        name: list(candidate_tokens[name]) for name in candidate_names
    }
    if registry.get("token_sequences") != actual_sequences:
        raise Top1DataError("candidate token sequences differ from training")


def _import_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - GPU evaluation environment
        raise SystemExit(
            "Top1 evaluation requires torch, transformers, and tqdm; "
            "install -e '.[train]'"
        ) from exc
    return torch, transformers, tqdm


def _load_model(
    *,
    model_dir: Path,
    transformers: Any,
    dtype: Any,
    trust_remote_code: bool,
    router_contract: Mapping[str, Any],
    device_map: str | None,
    max_memory: Mapping[int | str, str],
) -> Any:
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
    }
    if device_map is not None:
        model_kwargs.update(
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if max_memory:
            model_kwargs["max_memory"] = dict(max_memory)
    if (model_dir / "adapter_config.json").is_file():
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - LoRA evaluation environment
            raise SystemExit("LoRA evaluation requires peft") from exc
        dependency = router_contract["base_model_dependency"]
        base_kwargs = dict(model_kwargs)
        if dependency["kind"] == "hub_revision":
            base_kwargs["revision"] = dependency["revision"]
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            dependency["reference"],
            **base_kwargs,
        )
        adapter_kwargs: dict[str, Any] = {
            "is_trainable": False,
            "low_cpu_mem_usage": device_map is not None,
        }
        if device_map is not None:
            adapter_kwargs["device_map"] = device_map
            if max_memory:
                adapter_kwargs["max_memory"] = dict(max_memory)
        return PeftModel.from_pretrained(
            base_model,
            str(model_dir),
            **adapter_kwargs,
        )
    parameters = inspect.signature(
        transformers.AutoModelForCausalLM.from_pretrained
    ).parameters
    if "torch_dtype" not in parameters and not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        model_kwargs.pop("torch_dtype")
    return transformers.AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        **model_kwargs,
    )


def _dispatched_input_device(model: Any, torch: Any) -> Any:
    """Return the device hosting input embeddings for a dispatched model."""

    get_embeddings = getattr(model, "get_input_embeddings", None)
    embeddings = get_embeddings() if callable(get_embeddings) else None
    weight = getattr(embeddings, "weight", None)
    raw_device = getattr(weight, "device", None)
    if raw_device is None:
        raise Top1DataError(
            "device-mapped model does not expose its input embedding device"
        )
    device = torch.device(raw_device)
    if device.type == "meta":
        raise Top1DataError(
            "input embeddings were offloaded to a meta device; increase GPU/CPU "
            "max_memory so embeddings remain resident"
        )
    return device


def _resolved_device_map(model: Any) -> dict[str, str] | None:
    """Find and JSON-normalize Accelerate's resolved module placement."""

    pending = [model]
    visited: set[int] = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        mapping = getattr(current, "hf_device_map", None)
        if isinstance(mapping, Mapping):
            normalized = {}
            for module, device in mapping.items():
                value = f"cuda:{device}" if isinstance(device, int) else str(device)
                normalized[str(module)] = value
            return dict(sorted(normalized.items()))
        pending.extend(
            getattr(current, attribute, None)
            for attribute in ("base_model", "model")
        )
    return None


def _verify_base_model_dependency(
    manifest: Mapping[str, Any],
    *,
    model_dir: Path,
) -> None:
    finetune_mode = manifest.get("finetune_mode")
    dependency = manifest.get("base_model_dependency")
    has_adapter = (model_dir / "adapter_config.json").is_file()
    if finetune_mode == "full":
        if dependency != {"kind": "self_contained"} or has_adapter:
            raise Top1DataError("full-model bundle has an invalid base dependency")
        return
    if finetune_mode != "lora" or not has_adapter or not isinstance(dependency, Mapping):
        raise Top1DataError("LoRA bundle has an invalid base dependency")
    kind = dependency.get("kind")
    reference = dependency.get("reference")
    if not isinstance(reference, str) or not reference:
        raise Top1DataError("LoRA base model reference is invalid")
    if kind == "hub_revision":
        revision = dependency.get("revision")
        if not isinstance(revision, str) or not revision:
            raise Top1DataError("LoRA base model revision is missing")
        return
    if kind == "local_directory":
        base_path = Path(reference).expanduser()
        files = directory_file_manifest(base_path)
        identity = {"schema_version": 1, "files": files}
        if dependency.get("content_id") != canonical_sha256(identity):
            raise Top1DataError("local LoRA base model changed after training")
        return
    raise Top1DataError("LoRA base model dependency kind is unsupported")


def _device_and_dtype(args: argparse.Namespace, torch: Any) -> tuple[Any, Any, str]:
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    precision = args.precision
    if precision == "auto":
        precision = (
            "bf16"
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "fp16" if device.type == "cuda" else "fp32"
        )
    if device.type == "cpu" and precision == "fp16":
        raise Top1DataError("fp16 evaluation is not supported on CPU")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    return device, dtype, precision


def _prepare_prompts(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    candidate_names: Sequence[str],
    candidate_tokens: Mapping[str, Sequence[int]],
    system_prompt: str,
    max_length: int,
    history_ablation: bool,
) -> list[dict[str, Any]]:
    legal_names = set(candidate_names)
    prepared = []
    for row_index, row in enumerate(rows):
        try:
            messages = messages_from_row(row)
            target = None
            if "target_candidate_name" in row:
                target = target_candidate_name(row)
                if target not in legal_names:
                    raise Top1DataError(f"unknown target candidate name: {target!r}")
            prepared_prompt = prepare_router_prompt(
                tokenizer,
                messages,
                candidate_tokens=candidate_tokens,
                max_length=max_length,
                system_prompt=system_prompt,
            )
            fitted = prepared_prompt.fitted_messages
            prompt_ids = list(prepared_prompt.input_ids)
            source_non_system = [
                message
                for message in row["messages"]
                if isinstance(message, Mapping)
                and str(message.get("role", "")).strip() != "system"
            ]
            current_source = str(source_non_system[-1]["content"]).strip()
            diagnostics = {
                "original_message_count": len(source_non_system),
                "fitted_message_count": len(fitted),
                "history_messages_dropped": max(
                    0,
                    (len(source_non_system) - 1) - (len(fitted) - 1),
                ),
                "current_user_truncated": fitted[-1]["content"] != current_source,
                "prompt_tokens": len(prompt_ids),
                "reserved_target_tokens": prepared_prompt.reserved_target_tokens,
            }
            ablated_ids = None
            if history_ablation and len(messages) > 1:
                ablated_prompt = prepare_router_prompt(
                    tokenizer,
                    (messages[-1],),
                    candidate_tokens=candidate_tokens,
                    max_length=max_length,
                    system_prompt=system_prompt,
                )
                ablated_ids = list(ablated_prompt.input_ids)
        except Top1DataError as exc:
            raise Top1DataError(f"evaluation row {row_index + 1}: {exc}") from exc
        prepared.append(
            {
                "row_index": row_index,
                "target_candidate_name": target,
                "prompt_text": prepared_prompt.text,
                "prompt_ids": prompt_ids,
                "fitted_messages": list(fitted),
                "history_ablation_prompt_ids": ablated_ids,
                "diagnostics": diagnostics,
            }
        )
    return prepared


def _logits_processor_class(torch: Any, transformers: Any):
    class CandidateNameLogitsProcessor(transformers.LogitsProcessor):
        def __init__(self, trie: CandidateNameTokenTrie, prompt_width: int) -> None:
            self.trie = trie
            self.prompt_width = prompt_width

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            generated = input_ids[:, self.prompt_width :]
            masked = torch.full_like(scores, -float("inf"))
            for row_index, suffix in enumerate(generated.tolist()):
                allowed = (
                    (self.trie.eos_token_id,)
                    if self.trie.eos_token_id in suffix
                    else self.trie.allowed_next(suffix)
                )
                if not allowed:
                    raise RuntimeError(
                        "generation reached an invalid candidate prefix: "
                        f"{suffix!r}"
                    )
                masked[row_index, list(allowed)] = scores[row_index, list(allowed)]
            return masked

    return CandidateNameLogitsProcessor


def _generate_prepared(
    prepared: Sequence[Mapping[str, Any]],
    *,
    prompt_key: str,
    model: Any,
    tokenizer: Any,
    trie: CandidateNameTokenTrie,
    torch: Any,
    transformers: Any,
    device: Any,
    decoding_mode: str,
    num_beams: int,
    route_threshold: float | None,
) -> dict[int, dict[str, Any]]:
    items = [row for row in prepared if row[prompt_key] is not None]
    if not items:
        return {}
    prompts = [[*map(int, row[prompt_key])] for row in items]
    prompt_width = max(map(len, prompts))
    pad_token_id = int(tokenizer.pad_token_id)
    input_ids = [
        [*([pad_token_id] * (prompt_width - len(prompt))), *prompt]
        for prompt in prompts
    ]
    attention_mask = [
        [0] * (prompt_width - len(prompt)) + [1] * len(prompt)
        for prompt in prompts
    ]
    effective_beams = 1 if decoding_mode == "greedy" else num_beams
    if decoding_mode == "beam_search":
        root_width = len(trie.allowed_next(()))
        if effective_beams > root_width:
            raise Top1DataError(
                f"num_beams={effective_beams} exceeds the {root_width} legal "
                "first-token branches"
            )
    CandidateNameLogitsProcessor = _logits_processor_class(torch, transformers)
    generation_kwargs: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
            device=device,
        ),
        "do_sample": False,
        "num_beams": effective_beams,
        "num_return_sequences": 1,
        "max_new_tokens": trie.max_name_tokens + 1,
        "min_new_tokens": 1,
        "eos_token_id": trie.eos_token_id,
        "pad_token_id": pad_token_id,
        "logits_processor": transformers.LogitsProcessorList(
            [CandidateNameLogitsProcessor(trie, prompt_width)]
        ),
        "use_cache": True,
        "renormalize_logits": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if decoding_mode == "beam_search":
        generation_kwargs.update(early_stopping=True, length_penalty=0.0)
    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)

    transition_scores = None
    compute_scores = getattr(model, "compute_transition_scores", None)
    if callable(compute_scores):
        score_kwargs: dict[str, Any] = {"normalize_logits": True}
        beam_indices = getattr(generated, "beam_indices", None)
        if beam_indices is not None:
            score_kwargs["beam_indices"] = beam_indices
        transition_scores = compute_scores(
            generated.sequences,
            generated.scores,
            **score_kwargs,
        )
    if route_threshold is not None and transition_scores is None:
        raise Top1DataError(
            "route threshold requires normalized generation transition scores"
        )

    results = {}
    for index, row in enumerate(items):
        suffix = [
            int(value)
            for value in generated.sequences[index, prompt_width:].tolist()
        ]
        try:
            eos_position = suffix.index(trie.eos_token_id)
        except ValueError as exc:
            raise RuntimeError(
                "constrained candidate generation did not emit EOS"
            ) from exc
        candidate_name = trie.resolve(suffix[:eos_position])
        path_tokens = eos_position + 1
        path_logprob = (
            float(transition_scores[index, :path_tokens].sum().item())
            if transition_scores is not None
            else None
        )
        results[int(row["row_index"])] = {
            "candidate_name": candidate_name,
            "path_logprob": path_logprob,
            "path_tokens": path_tokens,
        }
    return results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_single_process()
    max_memory = _parse_max_memory(args.max_memory)
    if (args.max_length is not None and args.max_length <= 0) or args.batch_size <= 0:
        raise Top1DataError("max_length and batch_size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise Top1DataError("max_rows must be positive when specified")
    if args.decoding_mode == "beam_search" and args.num_beams < 2:
        raise Top1DataError("beam_search requires num_beams >= 2")
    if max_memory and args.device_map is None:
        raise Top1DataError("max_memory requires --device-map")
    if args.device_map is not None and args.device != "auto":
        raise Top1DataError("--device cannot be combined with --device-map")
    model_dir = Path(args.model_dir).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise Top1DataError(f"evaluation data does not exist: {data_path}")
    router_contract = _load_router_contract(model_dir)
    registry_path = _resolve_bundle_file(
        args.candidate_registry,
        model_dir,
        "candidate_registry.json",
        label="candidate registry",
    )
    prompt_path = _resolve_bundle_file(
        args.system_prompt_file,
        model_dir,
        "router_system_prompt.md",
        label="system prompt",
    )
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise Top1DataError("system prompt file is empty")
    candidate_names = load_candidate_names(registry_path)
    decision_policy_path = _resolve_decision_policy(args.decision_policy, model_dir)
    decision_policy = load_backend_decision_policy(
        decision_policy_path,
        candidate_names,
    )
    _validate_bundled_decision_policy(
        router_contract,
        model_dir=model_dir,
        decision_policy_path=decision_policy_path,
    )
    _apply_router_contract(
        args,
        router_contract,
        registry_path=registry_path,
        prompt_path=prompt_path,
        candidate_names=candidate_names,
    )
    _verify_base_model_dependency(router_contract, model_dir=model_dir)
    model_artifact = load_and_verify_model_artifact(
        model_dir,
        verify_files=True,
    )
    semantic_config = {
        "model_id": model_artifact["model_id"],
        "dataset_sha256": sha256_file(data_path),
        "candidate_registry_sha256": sha256_file(registry_path),
        "system_prompt_sha256": sha256_file(prompt_path),
        "routing_mode": ROUTING_MODE,
        "source_router_schema_version": router_contract["schema_version"],
        "conversation_template": CONVERSATION_TEMPLATE,
        "max_length": args.max_length,
        "max_rows": args.max_rows,
        "score_mode": "constrained_generate_path_logprob",
        "decoding_mode": args.decoding_mode,
        "num_beams": 1 if args.decoding_mode == "greedy" else args.num_beams,
        "route_threshold": args.route_threshold,
        "history_ablation": args.history_ablation,
        "backend_decision": decision_policy.payload(),
    }
    evaluation_signature = canonical_sha256(semantic_config)
    evaluation_id = _safe_component(
        args.evaluation_id or f"{compact_beijing_now()}-{evaluation_signature[:8]}",
        label="evaluation_id",
    )
    suite_id = (
        _safe_component(args.suite_id, label="suite_id") if args.suite_id else None
    )
    evaluation_root = Path(args.evaluation_root).expanduser().resolve()
    run_dir = (
        evaluation_root
        / str(model_artifact["model_id"])[:16]
        / evaluation_id
    )
    store = RunStore.evaluation(run_dir)
    repository = Path(__file__).resolve().parents[1]
    code_files = (
        Path(__file__).resolve(),
        repository / "src" / "llmgen" / "top1.py",
        repository / "src" / "llmgen" / "experiment.py",
        repository / "src" / "llmgen" / "evaluation.py",
    )
    manifest = {
        "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
        "run_signature": evaluation_signature,
        "evaluation_id": evaluation_id,
        "suite_id": suite_id,
        "evaluation_signature": evaluation_signature,
        "created_at": utc_now(),
        "model": {
            "model_id": model_artifact["model_id"],
            "path": str(model_dir),
            "verified": True,
            "training_run_id": model_artifact.get("training_run_id"),
        },
        "dataset": {
            "path": str(data_path),
            "sha256": semantic_config["dataset_sha256"],
        },
        "decision_policy": {
            "path": str(decision_policy_path),
            "sha256": sha256_file(decision_policy_path),
            "effective": decision_policy.payload(),
        },
        "semantic_inference": semantic_config,
        "execution": {
            "batch_size": args.batch_size,
            "precision": args.precision,
            "device": args.device,
            "device_map": args.device_map,
            "max_memory": _json_max_memory(max_memory),
            "trust_remote_code": args.trust_remote_code,
        },
        "code": {
            "git": git_snapshot(repository),
            "files": {
                path.relative_to(repository).as_posix(): sha256_file(path)
                for path in code_files
            },
        },
    }
    store.initialize(manifest)
    try:
        torch, transformers, tqdm = _import_dependencies()
        device, dtype, resolved_precision = _device_and_dtype(args, torch)
        write_json(run_dir / "logs" / "system.json", system_snapshot(torch))
        store.update_status("RUNNING", rows_completed=0)
        store.event("model_loading", model_id=model_artifact["model_id"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.eos_token_id is None:
            raise Top1DataError("model tokenizer must define an EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        candidate_tokens = candidate_token_sequences(tokenizer, candidate_names)
        trie = CandidateNameTokenTrie(
            candidate_tokens,
            eos_token_id=int(tokenizer.eos_token_id),
        )
        _validate_loaded_tokenizer(
            router_contract,
            tokenizer=tokenizer,
            candidate_names=candidate_names,
            candidate_tokens=candidate_tokens,
            transformers_version=str(transformers.__version__),
        )
        model = _load_model(
            model_dir=model_dir,
            transformers=transformers,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            router_contract=router_contract,
            device_map=args.device_map,
            max_memory=max_memory,
        )
        if args.device_map is None:
            model = model.to(device)
            input_device = device
            resolved_device_map = None
        else:
            input_device = _dispatched_input_device(model, torch)
            resolved_device_map = _resolved_device_map(model)
            if resolved_device_map is None:
                raise Top1DataError(
                    "model loading did not expose an Accelerate device map"
                )
        model_placement = {
            "requested_device": args.device,
            "requested_device_map": args.device_map,
            "max_memory": _json_max_memory(max_memory),
            "input_device": str(input_device),
            "resolved_device_map": resolved_device_map,
        }
        write_json(run_dir / "logs" / "model_placement.json", model_placement)
        store.event("model_loaded", placement=model_placement)
        model.eval()
        rows = read_jsonl(data_path)
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        if not rows:
            raise Top1DataError("evaluation dataset is empty")
        prepared = _prepare_prompts(
            rows,
            tokenizer=tokenizer,
            candidate_names=candidate_names,
            candidate_tokens=candidate_tokens,
            system_prompt=system_prompt,
            max_length=args.max_length,
            history_ablation=args.history_ablation,
        )
        predictions = []
        prediction_path = run_dir / "predictions.jsonl"
        rows_per_chunk = args.batch_size
        progress = tqdm(
            total=len(prepared),
            desc="[top1-eval] generating",
            unit="row",
            dynamic_ncols=True,
        )
        try:
            for start in range(0, len(prepared), rows_per_chunk):
                chunk = prepared[start : start + rows_per_chunk]
                generated = _generate_prepared(
                    chunk,
                    prompt_key="prompt_ids",
                    model=model,
                    tokenizer=tokenizer,
                    trie=trie,
                    torch=torch,
                    transformers=transformers,
                    device=input_device,
                    decoding_mode=args.decoding_mode,
                    num_beams=args.num_beams,
                    route_threshold=args.route_threshold,
                )
                ablation_generated = (
                    _generate_prepared(
                        chunk,
                        prompt_key="history_ablation_prompt_ids",
                        model=model,
                        tokenizer=tokenizer,
                        trie=trie,
                        torch=torch,
                        transformers=transformers,
                        device=input_device,
                        decoding_mode=args.decoding_mode,
                        num_beams=args.num_beams,
                        route_threshold=args.route_threshold,
                    )
                    if args.history_ablation
                    else {}
                )
                for row in chunk:
                    row_index = int(row["row_index"])
                    output = generated[row_index]
                    record = prediction_from_generation(
                        row_index=row_index,
                        candidate_names=candidate_names,
                        generated_candidate_name=output["candidate_name"],
                        path_logprob=output["path_logprob"],
                        path_tokens=output["path_tokens"],
                        target_candidate_name=row["target_candidate_name"],
                        diagnostics=row["diagnostics"],
                        decision_policy=decision_policy,
                        route_threshold=args.route_threshold,
                        decoding={
                            "mode": args.decoding_mode,
                            "num_beams": (
                                1
                                if args.decoding_mode == "greedy"
                                else args.num_beams
                            ),
                            "num_return_sequences": 1,
                            "scope": "candidate_name_top1",
                        },
                        history_ablation=ablation_generated.get(row_index),
                    )
                    append_jsonl(prediction_path, record)
                    predictions.append(record)
                completed = len(predictions)
                progress.update(len(chunk))
                if completed == len(prepared) or completed % 100 < len(chunk):
                    store.update_status("RUNNING", rows_completed=completed)
                    store.event(
                        "evaluation_progress",
                        rows_completed=completed,
                        rows_total=len(prepared),
                    )
        finally:
            progress.close()
        metrics = aggregate_predictions(
            predictions,
            candidate_names,
            decision_policy,
        )
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "confusion_matrix.json", metrics["confusion_matrix"])
        write_json(
            run_dir / "backend_confusion_matrix.json",
            metrics["backend"]["confusion_matrix"],
        )
        backend_metrics = metrics["backend"]
        summary = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "evaluation_signature": evaluation_signature,
            "model_id": model_artifact["model_id"],
            "rows": metrics["rows"],
            "top1_accuracy": backend_metrics["accuracy"],
            "backend_accuracy": backend_metrics["accuracy"],
            "raw_candidate_accuracy": metrics["top1_accuracy"],
            "macro_recall_observed_candidates": metrics[
                "macro_recall_observed_candidates"
            ],
            "expected_calibration_error": metrics["calibration"][
                "expected_calibration_error"
            ],
            "routing_policy": metrics["routing_policy"],
            "available_oos": backend_metrics["available_oos"],
            "resolved_precision": resolved_precision,
            "model_placement": model_placement,
            "completed_at": utc_now(),
        }
        write_json(run_dir / "summary.json", summary)
        index_record = {
            "timestamp": utc_now(),
            "state": "COMPLETED",
            "suite_id": suite_id,
            "model_id": model_artifact["model_id"],
            "evaluation_id": evaluation_id,
            "evaluation_signature": evaluation_signature,
            "run_dir": str(run_dir),
            "dataset_sha256": semantic_config["dataset_sha256"],
            "score_mode": "constrained_generate_path_logprob",
            "decoding_mode": args.decoding_mode,
            "rows": metrics["rows"],
            "top1_accuracy": backend_metrics["accuracy"],
            "backend_accuracy": backend_metrics["accuracy"],
            "raw_candidate_accuracy": metrics["top1_accuracy"],
            "route_threshold": args.route_threshold,
        }
        append_jsonl(evaluation_root / "evaluation_index.jsonl", index_record)
        if suite_id:
            append_jsonl(
                evaluation_root / "suites" / suite_id / "members.jsonl",
                index_record,
            )
        store.event("evaluation_completed", rows=metrics["rows"])
        store.update_status("COMPLETED", rows_completed=metrics["rows"])
        print(f"[top1-eval] {run_dir}", flush=True)
    except BaseException as exc:
        store.event("evaluation_failed", error_type=type(exc).__name__, error=str(exc))
        store.update_status("FAILED", error_type=type(exc).__name__, error=str(exc))
        try:
            failure_record = {
                "timestamp": utc_now(),
                "state": "FAILED",
                "suite_id": suite_id,
                "model_id": model_artifact["model_id"],
                "evaluation_id": evaluation_id,
                "evaluation_signature": evaluation_signature,
                "run_dir": str(run_dir),
                "dataset_sha256": semantic_config["dataset_sha256"],
                "score_mode": "constrained_generate_path_logprob",
                "decoding_mode": args.decoding_mode,
                "route_threshold": args.route_threshold,
                "error_type": type(exc).__name__,
            }
            append_jsonl(evaluation_root / "evaluation_index.jsonl", failure_record)
            if suite_id:
                append_jsonl(
                    evaluation_root / "suites" / suite_id / "members.jsonl",
                    failure_record,
                )
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
