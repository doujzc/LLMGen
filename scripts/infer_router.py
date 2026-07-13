#!/usr/bin/env python3
"""Constrained-beam inference and SkillRet evaluation for the causal router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llmgen.router import (
    GeneratedPath,
    RouterDataError,
    TokenTrie,
    active_skill_ids_from_registry,
    aggregate_retrieval_metrics,
    buckets_from_codes,
    code_token_id_map,
    load_virtual_tokens,
    normalize_code_rows,
    qrels_by_query,
    query_code_path_metrics,
    query_retrieval_metrics,
    rank_bucket_candidates,
    read_jsonl,
    render_router_prompt,
    validate_registry_assignments,
    write_jsonl,
)
from llmgen.skillret import sha256_file


DEFAULT_SYSTEM_PROMPT = (
    "Select the Agent Skill code that best matches the user request. "
    "Answer with code tokens only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-length skill codes using an active-registry trie."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--base-model-name-or-path",
        help="Base model when --model-name-or-path is a PEFT adapter directory.",
    )
    parser.add_argument("--virtual-tokens", required=True)
    parser.add_argument("--codes", required=True, help="index/test_codes.jsonl")
    parser.add_argument("--registry", required=True, help="index/test_registry.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries", help="queries_test.jsonl")
    source.add_argument("--query", help="One interactive query")
    parser.add_argument("--query-id", default="interactive")
    parser.add_argument("--qrels", help="qrels_test.jsonl; enables metrics")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="Prompt-token budget; defaults to the router training manifest/model context.",
    )
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--num-code-paths", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=(1, 5, 10))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RouterDataError("registry JSON must be an object")
    return payload


def _load_queries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.query is not None:
        return [{"id": args.query_id, "query": args.query}]
    rows = read_jsonl(args.queries)
    normalized = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        query_id = row.get("query_id", row.get("id"))
        query = row.get("query", row.get("input_text"))
        if not isinstance(query_id, str) or not query_id:
            raise RouterDataError(f"query row {row_number} has no id")
        if not isinstance(query, str) or not query.strip():
            raise RouterDataError(f"query {query_id!r} has no text")
        if query_id in seen:
            raise RouterDataError(f"duplicate query id: {query_id!r}")
        seen.add(query_id)
        normalized.append({**row, "id": query_id, "query": query.strip()})
    return normalized


def _dtype(torch: Any, value: str):
    return {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _validate_training_contract(args: argparse.Namespace) -> int | None:
    """Reject router/index lineage mismatches before loading a large model."""

    training_manifest = Path(args.model_name_or_path) / "router_manifest.json"
    if not training_manifest.is_file():
        return None
    payload = json.loads(training_manifest.read_text(encoding="utf-8"))
    expected_tokens_hash = payload.get("virtual_tokens_sha256")
    if expected_tokens_hash and expected_tokens_hash != sha256_file(args.virtual_tokens):
        raise RouterDataError(
            "virtual_tokens.txt differs from the router training artifact"
        )
    expected_stage1_sha256 = payload.get("stage1_checkpoint_sha256")
    if expected_stage1_sha256:
        index_manifest_path = Path(args.codes).resolve().parent / "manifest.json"
        if not index_manifest_path.is_file():
            raise RouterDataError(
                "index manifest is required to verify router/codebook lineage"
            )
        index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        actual_stage1_sha256 = index_manifest.get("checkpoint_sha256")
        if actual_stage1_sha256 != expected_stage1_sha256:
            raise RouterDataError(
                "router checkpoint and supplied index use different Stage-1 codebooks"
            )
        actual_codes_sha256 = sha256_file(args.codes)
        indexed_splits = index_manifest.get("splits")
        if not isinstance(indexed_splits, dict):
            raise RouterDataError("index manifest has no split artifacts")
        indexed_code_hashes = {
            details.get("codes_sha256")
            for details in indexed_splits.values()
            if isinstance(details, dict)
        }
        if actual_codes_sha256 not in indexed_code_hashes:
            raise RouterDataError(
                "codes artifact is not recorded by its adjacent index manifest"
            )
    trained_max_length = payload.get("max_length")
    return (
        int(trained_max_length)
        if isinstance(trained_max_length, (int, float)) and trained_max_length > 0
        else None
    )


def _load_model_and_tokenizer(args: argparse.Namespace, virtual_tokens: tuple[str, ...]):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - real inference environment
        raise SystemExit(
            "Inference requires torch and transformers. Install training dependencies."
        ) from exc
    trained_max_length = _validate_training_contract(args)

    model_path = Path(args.model_name_or_path)
    has_adapter_config = (model_path / "adapter_config.json").exists()
    has_full_model_config = (model_path / "config.json").exists()
    # A base override helps remote adapter IDs, but must not turn a local full
    # model into a PEFT adapter merely because the wrapper supplied the option.
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
    tokenizer.truncation_side = "right"
    token_ids = code_token_id_map(tokenizer, virtual_tokens)

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    requested_dtype = _dtype(torch, args.dtype)
    if requested_dtype is not None:
        model_kwargs["torch_dtype"] = requested_dtype

    if is_adapter:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("PEFT adapter inference requires the peft package.") from exc
        adapter_config = PeftConfig.from_pretrained(args.model_name_or_path)
        base_name = (
            args.base_model_name_or_path or adapter_config.base_model_name_or_path
        )
        model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, **model_kwargs
        )
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            raise RouterDataError(
                "model/tokenizer vocabulary mismatch; load the tokenizer saved with "
                "the router checkpoint"
            )
    model.to(args.device)
    model.eval()
    if args.max_input_length is None:
        context_length = getattr(model.config, "max_position_embeddings", None)
        if context_length is None:
            context_length = getattr(model.config, "n_positions", None)
        limits = [
            int(value)
            for value in (trained_max_length, context_length)
            if isinstance(value, (int, float)) and int(value) > 0
        ]
        total_limit = min(limits) if limits else 1024
        args.max_input_length = total_limit - args._num_levels - 1
    if args.max_input_length < 1:
        raise RouterDataError("max_input_length leaves no room for the prompt")
    return torch, tokenizer, model, token_ids


def _logits_processor_class(torch: Any):
    from transformers import LogitsProcessor

    class TrieLogitsProcessor(LogitsProcessor):
        def __init__(self, trie: TokenTrie, prompt_width: int) -> None:
            self.trie = trie
            self.prompt_width = prompt_width

        def __call__(self, input_ids, scores):
            generated = input_ids[:, self.prompt_width :]
            masked = torch.full_like(scores, -float("inf"))
            for row_index, suffix in enumerate(generated.tolist()):
                allowed = self.trie.allowed_next(suffix)
                if not allowed:
                    raise RuntimeError(
                        f"beam reached an invalid code prefix: {suffix!r}"
                    )
                masked[row_index, list(allowed)] = scores[row_index, list(allowed)]
            return masked

    return TrieLogitsProcessor


def _chunks(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _generate_batch(
    *,
    batch: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    torch: Any,
    trie: TokenTrie,
    id_to_token: dict[int, str],
    buckets: dict[tuple[str, ...], tuple[str, ...]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompts = [
        render_router_prompt(tokenizer, row["query"], args.system_prompt)
        for row in batch
    ]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
        return_tensors="pt",
    )
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    active_path_count = len(trie.paths)
    num_beams = min(args.beam_size, active_path_count)
    num_return_sequences = min(args.num_code_paths, num_beams)
    TrieLogitsProcessor = _logits_processor_class(torch)
    processor = TrieLogitsProcessor(trie, prompt_width)
    from transformers import LogitsProcessorList

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=trie.num_levels + 1,
            # EOS becomes legal after the Lth code token.  Setting L+1 here
            # would have Hugging Face mask the only token allowed by the trie.
            min_new_tokens=trie.num_levels,
            eos_token_id=trie.eos_token_id,
            pad_token_id=int(tokenizer.pad_token_id),
            logits_processor=LogitsProcessorList([processor]),
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
            early_stopping=True,
        )

    sequence_scores = getattr(generated, "sequences_scores", None)
    results: list[dict[str, Any]] = []
    for batch_index, query_row in enumerate(batch):
        paths: list[GeneratedPath] = []
        path_payloads: list[dict[str, Any]] = []
        offset = batch_index * num_return_sequences
        for beam_offset in range(num_return_sequences):
            row_index = offset + beam_offset
            suffix = generated.sequences[row_index, prompt_width:].tolist()
            if len(suffix) < trie.num_levels + 1:
                raise RuntimeError("constrained generation ended before L tokens plus EOS")
            path_ids = tuple(int(value) for value in suffix[: trie.num_levels])
            if not trie.is_active_path(path_ids):
                raise RuntimeError(f"model returned a path outside the active trie: {path_ids}")
            if int(suffix[trie.num_levels]) != trie.eos_token_id:
                raise RuntimeError("model did not emit EOS immediately after the Lth token")
            path_tokens = tuple(id_to_token[token_id] for token_id in path_ids)
            score = (
                float(sequence_scores[row_index].item())
                if sequence_scores is not None
                else 0.0
            )
            if any(existing.tokens == path_tokens for existing in paths):
                continue
            paths.append(GeneratedPath(path_tokens, score))
            path_payloads.append(
                {
                    "code_tokens": list(path_tokens),
                    "code_text": "".join(path_tokens),
                    "score": score,
                    "skill_ids": list(buckets[path_tokens]),
                }
            )
        paths.sort(key=lambda item: (-item.score, item.tokens))
        # Keep the serialized path order consistent with candidate path_rank.
        payload_by_tokens = {
            tuple(item["code_tokens"]): item for item in path_payloads
        }
        path_payloads = [payload_by_tokens[path.tokens] for path in paths]
        candidates = rank_bucket_candidates(paths, buckets, limit=args.top_k)
        results.append(
            {
                "query_id": query_row["id"],
                "query": query_row["query"],
                "paths": path_payloads,
                "candidates": candidates,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.beam_size < 1 or args.num_code_paths < 1:
        raise RouterDataError("batch/beam/path counts must be positive")
    if args.num_code_paths > args.beam_size:
        raise RouterDataError("num_code_paths cannot exceed beam_size")
    if args.top_k < 1:
        raise RouterDataError("top_k must be positive")
    if any(cutoff < 1 for cutoff in args.cutoffs):
        raise RouterDataError("metric cutoffs must be positive")
    if args.qrels and args.top_k < max(args.cutoffs):
        raise RouterDataError("top_k must cover the largest requested metric cutoff")

    code_rows = read_jsonl(args.codes)
    skill_to_code, num_levels = normalize_code_rows(code_rows)
    args._num_levels = num_levels
    registry = _load_registry(args.registry)
    validate_registry_assignments(registry, code_rows)
    registry_levels = registry.get("num_levels")
    if registry_levels is not None and registry_levels != num_levels:
        raise RouterDataError(
            f"registry num_levels={registry_levels} disagrees with codes={num_levels}"
        )
    active_skill_ids = active_skill_ids_from_registry(registry)
    buckets = buckets_from_codes(skill_to_code, active_skill_ids)

    virtual_tokens = load_virtual_tokens(args.virtual_tokens)
    torch, tokenizer, model, token_ids = _load_model_and_tokenizer(
        args, virtual_tokens
    )
    try:
        active_token_paths = [
            tuple(token_ids[token] for token in path) for path in buckets
        ]
    except KeyError as exc:
        raise RouterDataError(
            f"active code uses token absent from virtual_tokens.txt: {exc.args[0]!r}"
        ) from exc
    trie = TokenTrie(active_token_paths, eos_token_id=int(tokenizer.eos_token_id))
    if trie.num_levels != num_levels:
        raise RuntimeError("active trie changed the configured number of levels")
    id_to_token = {token_id: token for token, token_id in token_ids.items()}
    queries = _load_queries(args)

    results: list[dict[str, Any]] = []
    for batch in _chunks(queries, args.batch_size):
        results.extend(
            _generate_batch(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                trie=trie,
                id_to_token=id_to_token,
                buckets=buckets,
                args=args,
            )
        )

    metrics_summary = None
    if args.qrels:
        relevance = qrels_by_query(read_jsonl(args.qrels))
        per_query = []
        for result in results:
            query_id = result["query_id"]
            relevant = relevance.get(query_id)
            if not relevant:
                raise RouterDataError(f"no positive qrels for query {query_id!r}")
            ranked = [candidate["skill_id"] for candidate in result["candidates"]]
            metrics = query_retrieval_metrics(
                ranked,
                relevant,
                cutoffs=args.cutoffs,
            )
            metrics.update(
                query_code_path_metrics(
                    [path["code_tokens"] for path in result["paths"]],
                    relevant,
                    skill_to_code,
                    buckets,
                    cutoffs=args.cutoffs,
                )
            )
            result["relevant_skill_ids"] = list(relevant)
            result["metrics"] = metrics
            per_query.append(metrics)
        metrics_summary = {
            "num_queries": len(results),
            "num_active_skills": len(active_skill_ids),
            "num_active_paths": len(buckets),
            "num_levels": num_levels,
            "cutoffs": sorted(set(args.cutoffs)),
            "metrics": aggregate_retrieval_metrics(per_query),
        }

    write_jsonl(args.output_jsonl, results)
    if metrics_summary is not None:
        metrics_path = Path(args.metrics_output) if args.metrics_output else Path(
            args.output_jsonl
        ).with_suffix(".metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(metrics_summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(results[0] if len(results) == 1 else {"queries": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
