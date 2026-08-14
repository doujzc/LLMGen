#!/usr/bin/env python3
"""Top1 inference for routers trained to emit candidate names directly."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

from llmgen.direct_router import (
    CURRENT_CONVERSATION_TEMPLATE,
    DIRECT_ROUTING_MODE,
    LEGACY_CONVERSATION_TEMPLATE,
    SUPPORTED_CONVERSATION_TEMPLATES,
    CandidateNameTokenTrie,
    CandidateRoute,
    candidate_token_sequences,
    fit_candidate_router_prompt,
    load_candidate_registry,
    messages_from_row,
    target_candidate_name,
)
from llmgen.router import RouterDataError, read_jsonl, write_jsonl
from llmgen.skillret import sha256_file


DECODING_MODES = ("greedy", "beam_search")


def _parse_route_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("route threshold must be a number") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("route threshold must be between 0 and 1")
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exactly one legal candidate name for each conversation."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--base-model-name-or-path",
        help="Base model when model-name-or-path is a PEFT adapter.",
    )
    parser.add_argument(
        "--candidate-registry",
        help="Defaults to candidate_registry.json bundled with the trained router.",
    )
    parser.add_argument(
        "--system-prompt-file",
        help="Defaults to router_system_prompt.md bundled with the trained router.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries", help="JSONL rows containing messages or query")
    source.add_argument("--query-txt", help="One independent query per non-empty line")
    source.add_argument("--query", help="One single-turn query")
    source.add_argument(
        "--messages-json",
        help="JSON file containing a messages array for one multi-turn request.",
    )
    parser.add_argument("--query-id", default="interactive")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-input-length",
        type=int,
        help="Prompt-token budget; inferred from the training manifest by default.",
    )
    parser.add_argument(
        "--decoding-mode", choices=DECODING_MODES, default="greedy"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam width; inference still returns only the best candidate name.",
    )
    parser.add_argument(
        "--route-threshold",
        type=_parse_route_threshold,
        help=(
            "Abstain when an executable candidate's constrained path probability "
            "is below this value (0 to 1). Virtual no-route candidates are unchanged."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _bundle_file(model_path: str | Path, filename: str) -> Path | None:
    path = Path(model_path).expanduser()
    candidates = [path / filename]
    if path.name.startswith("checkpoint-"):
        candidates.append(path.parent / filename)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _load_manifest(model_path: str | Path) -> tuple[dict[str, Any] | None, Path | None]:
    path = _bundle_file(model_path, "router_manifest.json")
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RouterDataError("router_manifest.json must be an object")
    if payload.get("routing_mode") != DIRECT_ROUTING_MODE:
        raise RouterDataError(
            "checkpoint is not a direct candidate-name Top1 router"
        )
    contract = payload.get("generation_contract")
    if not isinstance(contract, dict) or contract.get("mode") != DIRECT_ROUTING_MODE:
        raise RouterDataError("checkpoint has an incompatible generation contract")
    return payload, path


def _conversation_template_from_manifest(
    manifest: Mapping[str, Any] | None,
) -> str:
    """Preserve the prompt contract of checkpoints created before versioning."""

    if manifest is None:
        return CURRENT_CONVERSATION_TEMPLATE
    contract = manifest.get("generation_contract")
    value = (
        contract.get("conversation_template", LEGACY_CONVERSATION_TEMPLATE)
        if isinstance(contract, Mapping)
        else LEGACY_CONVERSATION_TEMPLATE
    )
    if value not in SUPPORTED_CONVERSATION_TEMPLATES:
        raise RouterDataError(
            f"checkpoint uses an unsupported conversation template: {value!r}"
        )
    return value


def _resolve_artifacts(
    args: argparse.Namespace,
) -> tuple[Path, str, dict[str, Any] | None, int | None]:
    manifest, manifest_path = _load_manifest(args.model_name_or_path)
    registry_path = (
        Path(args.candidate_registry).expanduser()
        if args.candidate_registry
        else _bundle_file(args.model_name_or_path, "candidate_registry.json")
    )
    if registry_path is None or not registry_path.is_file():
        raise RouterDataError(
            "candidate registry is required; pass --candidate-registry or use a "
            "complete trained-router directory"
        )
    prompt_path = (
        Path(args.system_prompt_file).expanduser()
        if args.system_prompt_file
        else _bundle_file(args.model_name_or_path, "router_system_prompt.md")
    )
    if prompt_path is None or not prompt_path.is_file():
        raise RouterDataError(
            "router system prompt is required; pass --system-prompt-file or use a "
            "complete trained-router directory"
        )
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise RouterDataError("router system prompt is empty")

    if manifest is not None:
        registry_meta = manifest.get("candidate_registry")
        if isinstance(registry_meta, dict):
            expected = registry_meta.get("sha256")
            if expected and expected != sha256_file(registry_path):
                raise RouterDataError("candidate registry differs from training artifact")
        prompt_meta = manifest.get("system_prompt_artifact")
        if isinstance(prompt_meta, dict):
            expected = prompt_meta.get("sha256")
            if expected and expected != sha256_file(prompt_path):
                raise RouterDataError("system prompt differs from training artifact")
        trained_names = manifest.get("generation_contract", {}).get("candidate_names")
        actual_names = [route.name for route in load_candidate_registry(registry_path)]
        if isinstance(trained_names, list) and trained_names != actual_names:
            raise RouterDataError("candidate registry order differs from training contract")
    trained_max_length = None if manifest is None else manifest.get("max_length")
    if not isinstance(trained_max_length, int) or trained_max_length < 1:
        trained_max_length = None
    del manifest_path
    return registry_path, system_prompt, manifest, trained_max_length


def _normalize_input_row(row: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    row_id = row.get("query_id")
    if row_id is None:
        row_id = row.get("id")
    if row_id is None:
        row_id = f"row-{row_number:06d}"
    messages = messages_from_row(row)
    return {**dict(row), "id": row_id, "messages": list(messages)}


def _load_queries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.query is not None:
        return [
            {
                "id": args.query_id,
                "messages": [{"role": "user", "content": args.query}],
            }
        ]
    if args.messages_json is not None:
        payload = json.loads(Path(args.messages_json).read_text(encoding="utf-8"))
        messages = payload.get("messages") if isinstance(payload, dict) else payload
        return [
            _normalize_input_row(
                {"id": args.query_id, "messages": messages},
                1,
            )
        ]
    if args.query_txt is not None:
        rows: list[dict[str, Any]] = []
        with Path(args.query_txt).open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                query = raw_line.strip()
                if query:
                    rows.append(
                        {
                            "id": f"line-{line_number:06d}",
                            "source_line": line_number,
                            "messages": [{"role": "user", "content": query}],
                        }
                    )
        if not rows:
            raise RouterDataError("query TXT contains no non-empty lines")
        return rows
    rows = read_jsonl(args.queries)
    normalized = [
        _normalize_input_row(row, row_number)
        for row_number, row in enumerate(rows, start=1)
    ]
    return normalized


def _dtype(torch: Any, value: str):
    return {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _vocabulary_capacity(module: Any) -> int | None:
    """Return the token dimension of an embedding/projection module."""

    for attribute in ("num_embeddings", "out_features"):
        value = getattr(module, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", ())
    if shape and isinstance(shape[0], int) and shape[0] > 0:
        return int(shape[0])
    return None


def _validate_model_tokenizer_vocabulary(model: Any, tokenizer: Any) -> None:
    """Ensure every tokenizer ID is representable by the loaded model.

    Model vocabulary capacity may legitimately exceed ``len(tokenizer)``. Qwen3,
    for example, pads its embedding matrix with unused rows. Compatibility is
    therefore an ID-boundary check rather than a size-equality check.
    """

    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise RouterDataError("router tokenizer has an empty or invalid vocabulary")
    token_ids = tuple(vocabulary.values())
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in token_ids
    ):
        raise RouterDataError("router tokenizer contains an invalid token id")
    max_token_id = max(token_ids)

    input_capacity = _vocabulary_capacity(model.get_input_embeddings())
    if input_capacity is None:
        raise RouterDataError("cannot determine model input vocabulary capacity")
    output_embeddings = model.get_output_embeddings()
    output_capacity = (
        _vocabulary_capacity(output_embeddings)
        if output_embeddings is not None
        else None
    )
    capacities = [("input", input_capacity)]
    if output_capacity is not None:
        capacities.append(("output", output_capacity))
    for name, capacity in capacities:
        if max_token_id >= capacity:
            raise RouterDataError(
                "model/tokenizer vocabulary mismatch: tokenizer max token id "
                f"{max_token_id} is outside the {name} vocabulary with {capacity} "
                f"rows (maximum id {capacity - 1})"
            )


def _load_model_and_tokenizer(
    args: argparse.Namespace,
    *,
    trained_max_length: int | None,
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - real inference environment
        raise SystemExit(
            "Inference requires torch and transformers. Install training dependencies."
        ) from exc

    model_path = Path(args.model_name_or_path)
    has_adapter_config = (model_path / "adapter_config.json").is_file()
    has_full_model_config = (model_path / "config.json").is_file()
    is_adapter = has_adapter_config or (
        bool(args.base_model_name_or_path) and not has_full_model_config
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise RouterDataError("router tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    requested_dtype = _dtype(torch, args.dtype)
    if requested_dtype is not None:
        model_kwargs["torch_dtype"] = requested_dtype
    if is_adapter:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("PEFT adapter inference requires peft.") from exc
        adapter_config = PeftConfig.from_pretrained(args.model_name_or_path)
        base_name = args.base_model_name_or_path or adapter_config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, **model_kwargs
        )
    _validate_model_tokenizer_vocabulary(model, tokenizer)
    model.to(args.device)
    model.eval()

    if args.max_input_length is None:
        context_length = getattr(model.config, "max_position_embeddings", None)
        limits = [
            int(value)
            for value in (trained_max_length, context_length)
            if isinstance(value, (int, float)) and int(value) > 0
        ]
        args._total_length = min(limits) if limits else 1024
    else:
        args._total_length = None
    return torch, tokenizer, model


def _logits_processor_class(torch: Any):
    from transformers import LogitsProcessor

    class CandidateNameLogitsProcessor(LogitsProcessor):
        def __init__(self, trie: CandidateNameTokenTrie, prompt_width: int) -> None:
            self.trie = trie
            self.prompt_width = prompt_width

        def __call__(self, input_ids, scores):
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
                        f"generation reached an invalid candidate name: {suffix!r}"
                    )
                masked[row_index, list(allowed)] = scores[row_index, list(allowed)]
            return masked

    return CandidateNameLogitsProcessor


def _chunks(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _candidate_confidence(score: float | None) -> float | None:
    """Convert a constrained candidate-path log probability into probability."""

    if score is None:
        return None
    if math.isnan(score):
        raise RouterDataError("candidate generation produced a NaN path score")
    if score == -math.inf:
        return 0.0
    if score == math.inf:
        raise RouterDataError("candidate generation produced an infinite path score")
    return max(0.0, min(1.0, math.exp(min(0.0, score))))


def _route_decision(
    *,
    route: CandidateRoute,
    score: float | None,
    route_threshold: float | None,
) -> dict[str, Any]:
    """Apply optional abstention without obscuring the model's raw prediction."""

    confidence = _candidate_confidence(score)
    raw_should_route = route.intent_label is not None
    if route_threshold is not None and confidence is None:
        raise RouterDataError(
            "--route-threshold requires normalized generation transition scores"
        )
    threshold_triggered = bool(
        raw_should_route
        and route_threshold is not None
        and confidence is not None
        and confidence < route_threshold
    )
    return {
        "raw_selected_candidate_id": route.candidate_id,
        "raw_intent_label": route.intent_label,
        "raw_should_route": raw_should_route,
        "candidate_confidence": confidence,
        "route_threshold": route_threshold,
        "threshold_triggered": threshold_triggered,
        "selected_candidate_id": None if threshold_triggered else route.candidate_id,
        "intent_label": None if threshold_triggered else route.intent_label,
        "should_route": raw_should_route and not threshold_triggered,
        "status": (
            "abstained"
            if threshold_triggered
            else "routed"
            if raw_should_route
            else "no_route"
        ),
    }


def _generate_batch(
    *,
    batch: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    torch: Any,
    trie: CandidateNameTokenTrie,
    routes_by_name: Mapping[str, Any],
    system_prompt: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompts = [
        fit_candidate_router_prompt(
            tokenizer,
            row["messages"],
            system_prompt,
            max_prompt_tokens=args.max_input_length,
            conversation_template=getattr(
                args,
                "conversation_template",
                CURRENT_CONVERSATION_TEMPLATE,
            ),
        )[0]
        for row in batch
    ]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    num_beams = 1 if args.decoding_mode == "greedy" else args.num_beams
    if args.decoding_mode == "beam_search":
        root_width = len(trie.allowed_next(()))
        if num_beams > root_width:
            raise RouterDataError(
                f"num_beams={num_beams} exceeds the {root_width} legal first-token "
                "branches; lower --num-beams"
            )

    CandidateNameLogitsProcessor = _logits_processor_class(torch)
    from transformers import LogitsProcessorList

    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "num_beams": num_beams,
        "num_return_sequences": 1,
        "max_new_tokens": trie.max_name_tokens + 1,
        "min_new_tokens": 1,
        "eos_token_id": trie.eos_token_id,
        "pad_token_id": int(tokenizer.pad_token_id),
        "logits_processor": LogitsProcessorList(
            [CandidateNameLogitsProcessor(trie, prompt_width)]
        ),
        "use_cache": True,
        "renormalize_logits": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if args.decoding_mode == "beam_search":
        generation_kwargs.update(early_stopping=True, length_penalty=0.0)
    with torch.inference_mode():
        generated = model.generate(**encoded, **generation_kwargs)

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
    route_threshold = getattr(args, "route_threshold", None)
    if route_threshold is not None and transition_scores is None:
        raise RouterDataError(
            "--route-threshold requires a model with normalized transition scores"
        )

    results: list[dict[str, Any]] = []
    for index, row in enumerate(batch):
        suffix = [
            int(value)
            for value in generated.sequences[index, prompt_width:].tolist()
        ]
        try:
            eos_position = suffix.index(trie.eos_token_id)
        except ValueError as exc:
            raise RuntimeError("constrained candidate generation did not emit EOS") from exc
        name = trie.resolve(suffix[:eos_position])
        route = routes_by_name[name]
        score = None
        if transition_scores is not None:
            score = float(transition_scores[index, : eos_position + 1].sum().item())
        decision = _route_decision(
            route=route,
            score=score,
            route_threshold=route_threshold,
        )
        results.append(
            {
                "query_id": row["id"],
                "messages": row["messages"],
                "candidate_name": name,
                **decision,
                "generated_text": name,
                "score": score,
                "decoding": {
                    "mode": args.decoding_mode,
                    "num_beams": num_beams,
                    "num_return_sequences": 1,
                    "scope": "candidate_name_top1",
                },
            }
        )
    return results


def _metrics(
    queries: list[dict[str, Any]],
    results: list[dict[str, Any]],
    routes_by_name: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    labeled: list[tuple[str, dict[str, Any], str | None]] = []
    for query, result in zip(queries, results, strict=True):
        try:
            expected_name = target_candidate_name(query)
        except RouterDataError:
            continue
        labeled.append((expected_name, result, query.get("expected_system_output")))
    if not labeled:
        return None
    per_name_total: Counter[str] = Counter()
    per_name_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for expected, result, _ in labeled:
        predicted = result["candidate_name"]
        per_name_total[expected] += 1
        per_name_correct[expected] += expected == predicted
        confusion[expected][predicted] += 1
    candidate_correct = sum(
        expected == result["candidate_name"] for expected, result, _ in labeled
    )
    output_correct = sum(
        expected_output == result["intent_label"]
        for _, result, expected_output in labeled
    )
    metrics: dict[str, Any] = {
        "examples": len(labeled),
        "candidate_accuracy": candidate_correct / len(labeled),
        "system_output_accuracy": output_correct / len(labeled),
        "per_candidate": {
            name: {
                "examples": per_name_total[name],
                "accuracy": per_name_correct[name] / per_name_total[name],
            }
            for name in sorted(per_name_total)
        },
        "confusion": {
            expected: dict(sorted(predictions.items()))
            for expected, predictions in sorted(confusion.items())
        },
    }
    if routes_by_name is not None:
        policy_rows = [
            (expected, result, routes_by_name[expected].intent_label is not None)
            for expected, result, _ in labeled
            if expected in routes_by_name
        ]
        if policy_rows:
            routed = sum(bool(result["should_route"]) for _, result, _ in policy_rows)
            triggered = sum(
                bool(result.get("threshold_triggered"))
                for _, result, _ in policy_rows
            )
            accepted = len(policy_rows) - triggered
            accepted_correct = sum(
                expected == result["candidate_name"]
                and not result.get("threshold_triggered", False)
                for expected, result, _ in policy_rows
            )
            true_positive = sum(
                expected_route and bool(result["should_route"])
                for _, result, expected_route in policy_rows
            )
            false_positive = sum(
                not expected_route and bool(result["should_route"])
                for _, result, expected_route in policy_rows
            )
            false_negative = sum(
                expected_route and not bool(result["should_route"])
                for _, result, expected_route in policy_rows
            )
            true_negative = (
                len(policy_rows) - true_positive - false_positive - false_negative
            )
            correct_routed = sum(
                expected == result["candidate_name"] and bool(result["should_route"])
                for expected, result, _ in policy_rows
            )
            expected_routed = true_positive + false_negative
            expected_no_route = false_positive + true_negative
            route_precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else None
            )
            route_recall = (
                true_positive / expected_routed if expected_routed else None
            )
            route_f1 = (
                2 * route_precision * route_recall / (route_precision + route_recall)
                if route_precision is not None
                and route_recall is not None
                and route_precision + route_recall
                else None
            )
            threshold_values = {
                result.get("route_threshold") for _, result, _ in policy_rows
            }
            metrics["routing_policy"] = {
                "route_threshold": (
                    next(iter(threshold_values)) if len(threshold_values) == 1 else None
                ),
                "examples": len(policy_rows),
                "output_route_coverage": routed / len(policy_rows),
                "threshold_triggered_examples": triggered,
                "threshold_abstention_rate": triggered / len(policy_rows),
                "selective_candidate_accuracy": (
                    accepted_correct / accepted if accepted else None
                ),
                "binary_route_precision": route_precision,
                "binary_route_recall": route_recall,
                "binary_route_f1": route_f1,
                "false_route_rate": (
                    false_positive / expected_no_route if expected_no_route else None
                ),
                "false_no_route_rate": (
                    false_negative / expected_routed if expected_routed else None
                ),
                "routed_candidate_precision": (
                    correct_routed / routed if routed else None
                ),
                "end_to_end_route_recall": (
                    correct_routed / expected_routed if expected_routed else None
                ),
                "decision_confusion": {
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "true_negative": true_negative,
                },
            }
    return metrics


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise RouterDataError("batch_size must be positive")
    if args.decoding_mode == "beam_search" and args.num_beams < 2:
        raise RouterDataError("beam_search requires num_beams >= 2")

    registry_path, system_prompt, manifest, trained_max_length = _resolve_artifacts(
        args
    )
    args.conversation_template = _conversation_template_from_manifest(manifest)
    routes = load_candidate_registry(registry_path)
    routes_by_name = {route.name: route for route in routes}
    queries = _load_queries(args)
    if not queries:
        raise RouterDataError("query input is empty")
    torch, tokenizer, model = _load_model_and_tokenizer(
        args, trained_max_length=trained_max_length
    )
    sequences = candidate_token_sequences(tokenizer, routes_by_name)
    trie = CandidateNameTokenTrie(
        sequences,
        eos_token_id=int(tokenizer.eos_token_id),
    )
    if args.max_input_length is None:
        args.max_input_length = args._total_length - trie.max_name_tokens - 1
    if args.max_input_length < 1:
        raise RouterDataError("max_input_length leaves no room for a candidate name")

    results: list[dict[str, Any]] = []
    for batch in _chunks(queries, args.batch_size):
        results.extend(
            _generate_batch(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                trie=trie,
                routes_by_name=routes_by_name,
                system_prompt=system_prompt,
                args=args,
            )
        )
        print(f"[inference] {len(results)}/{len(queries)}", flush=True)
    write_jsonl(args.output_jsonl, results)

    metrics = _metrics(queries, results, routes_by_name)
    if metrics is not None:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if args.metrics_output:
            destination = Path(args.metrics_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
