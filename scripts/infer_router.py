#!/usr/bin/env python3
"""Constrained autoregressive multi-skill inference and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llmgen.router import (
    CODE_PATH_SEPARATOR,
    GeneratedPath,
    MultiPathTokenTrie,
    RouterDataError,
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
    "Select every Agent Skill needed for the user request in execution order. "
    "Output one hierarchical skill code per line, with no other text."
)
DECODING_MODES = ("greedy", "beam_search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoregressively generate newline-delimited skill codes."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--base-model-name-or-path",
        help="Base model when --model-name-or-path is a PEFT adapter directory.",
    )
    parser.add_argument("--virtual-tokens")
    parser.add_argument("--codes", help="index/test_codes.jsonl")
    parser.add_argument("--registry", help="index/test_registry.json")
    parser.add_argument(
        "--candidate-state-dir",
        help=(
            "Candidate overlay/model bundle containing skill_decode_map.json and "
            "virtual_tokens.txt. Replaces --codes/--registry for incremental inference."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries", help="queries_test.jsonl")
    source.add_argument(
        "--query-txt",
        help="UTF-8 text file containing one query per non-empty line.",
    )
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
    parser.add_argument(
        "--max-code-paths",
        type=int,
        default=8,
        help="Safety cap; EOS may stop generation after any complete path.",
    )
    parser.add_argument(
        "--decoding-mode",
        choices=DECODING_MODES,
        default="greedy",
        help=(
            "Generate an ordered multi-path sequence (greedy) or return the top "
            "single-path codes from constrained beam search (beam_search)."
        ),
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help=(
            "Beam width and number of single-path codes returned by "
            "--decoding-mode beam_search; ignored by greedy decoding."
        ),
    )
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
    query_txt = getattr(args, "query_txt", None)
    if query_txt is not None:
        rows = []
        with Path(query_txt).open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                query = raw_line.strip()
                if not query:
                    continue
                rows.append(
                    {
                        "id": f"line-{line_number:06d}",
                        "query": query,
                        "source_line": line_number,
                    }
                )
        if not rows:
            raise RouterDataError("query TXT contains no non-empty lines")
        return rows
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
    generation_contract = payload.get("generation_contract")
    if not isinstance(generation_contract, dict) or generation_contract.get(
        "mode"
    ) != "autoregressive_multi_path":
        raise RouterDataError(
            "router checkpoint predates autoregressive multi-path generation; "
            "rebuild router data and retrain Stage 2"
        )
    if generation_contract.get("path_separator") != CODE_PATH_SEPARATOR:
        raise RouterDataError("router checkpoint uses a different path separator")
    expected_tokens_hash = payload.get("virtual_tokens_sha256")
    if expected_tokens_hash and expected_tokens_hash != sha256_file(args.virtual_tokens):
        raise RouterDataError(
            "virtual_tokens.txt differs from the router training artifact"
        )
    decode_map_path = getattr(args, "decode_map", None)
    if decode_map_path:
        decoder_artifacts = payload.get("decoder_artifacts")
        expected_decode_hash = (
            decoder_artifacts.get("decode_map_sha256")
            if isinstance(decoder_artifacts, dict)
            else None
        )
        actual_decode_hash = sha256_file(decode_map_path)
        if (
            expected_decode_hash
            and actual_decode_hash != expected_decode_hash
        ):
            from llmgen.incremental import incremental_ancestor_hashes
            from llmgen.router_bundle import load_skill_decode_map

            incremental_map = load_skill_decode_map(decode_map_path)
            if expected_decode_hash not in incremental_ancestor_hashes(incremental_map):
                raise RouterDataError(
                    "skill_decode_map.json differs from the router manifest and "
                    "does not declare that bundled map as an incremental ancestor"
                )
    expected_stage1_sha256 = payload.get("stage1_checkpoint_sha256")
    if expected_stage1_sha256:
        if decode_map_path:
            from llmgen.router_bundle import load_skill_decode_map

            decode_map = load_skill_decode_map(decode_map_path)
            actual_stage1_sha256 = decode_map.get("provenance", {}).get(
                "stage1_checkpoint_sha256"
            )
            if actual_stage1_sha256 != expected_stage1_sha256:
                raise RouterDataError(
                    "router checkpoint and bundled decode map use different "
                    "Stage-1 codebooks"
                )
        else:
            codes_path = getattr(args, "codes", None)
            if not codes_path:
                raise RouterDataError(
                    "codes or a bundled decode map is required to verify codebook lineage"
                )
            index_manifest_path = Path(codes_path).resolve().parent / "manifest.json"
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
            actual_codes_sha256 = sha256_file(codes_path)
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
    separator_token_ids = tuple(
        int(value)
        for value in tokenizer.encode(
            CODE_PATH_SEPARATOR,
            add_special_tokens=False,
            verbose=False,
        )
    )
    if not separator_token_ids:
        raise RouterDataError("tokenizer encodes the code-path separator as empty")
    if int(tokenizer.eos_token_id) in separator_token_ids:
        raise RouterDataError("code-path separator contains the tokenizer EOS token")
    args._separator_token_ids = separator_token_ids

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
        output_budget = (
            args.max_code_paths * args._num_levels
            + (args.max_code_paths - 1) * len(separator_token_ids)
            + 1
        )
        args.max_input_length = total_limit - output_budget
    if args.max_input_length < 1:
        raise RouterDataError("max_input_length leaves no room for the prompt")
    return torch, tokenizer, model, token_ids


def _logits_processor_class(torch: Any):
    from transformers import LogitsProcessor

    class MultiPathTrieLogitsProcessor(LogitsProcessor):
        def __init__(self, trie: MultiPathTokenTrie, prompt_width: int) -> None:
            self.trie = trie
            self.prompt_width = prompt_width

        def __call__(self, input_ids, scores):
            generated = input_ids[:, self.prompt_width :]
            masked = torch.full_like(scores, -float("inf"))
            for row_index, suffix in enumerate(generated.tolist()):
                # Finished rows remain in a padded batch while other rows keep
                # decoding. Their pad token is EOS, so keep EOS legal without
                # asking the active grammar to parse beyond termination.
                allowed = (
                    (self.trie.eos_token_id,)
                    if self.trie.eos_token_id in suffix
                    else self.trie.allowed_next(suffix)
                )
                if not allowed:
                    raise RuntimeError(
                        f"generation reached an invalid code sequence: {suffix!r}"
                    )
                masked[row_index, list(allowed)] = scores[row_index, list(allowed)]
            return masked

    return MultiPathTrieLogitsProcessor


def _chunks(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _resolve_decoding(args: argparse.Namespace) -> tuple[str, int]:
    """Return the normalized decoding mode and effective beam width."""

    mode = getattr(args, "decoding_mode", "greedy")
    if mode not in DECODING_MODES:
        raise RouterDataError(
            f"decoding_mode must be one of {', '.join(DECODING_MODES)}"
        )
    raw_num_beams = getattr(args, "num_beams", 1)
    if isinstance(raw_num_beams, bool):
        raise RouterDataError("num_beams must be an integer")
    try:
        num_beams = int(raw_num_beams)
    except (TypeError, ValueError) as exc:
        raise RouterDataError("num_beams must be an integer") from exc
    if num_beams < 1:
        raise RouterDataError("num_beams must be positive")
    if mode == "beam_search" and num_beams < 2:
        raise RouterDataError("beam_search requires num_beams >= 2")
    return mode, num_beams if mode == "beam_search" else 1


def _generate_batch(
    *,
    batch: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    torch: Any,
    trie: MultiPathTokenTrie,
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
    decoding_mode, num_beams = _resolve_decoding(args)
    if decoding_mode == "beam_search":
        # Beam Search is a retrieval mode: every beam represents exactly one
        # fixed-length code. It does not search for alternative full multi-line
        # outputs produced by Greedy decoding.
        decoding_trie = MultiPathTokenTrie(
            trie.paths,
            eos_token_id=trie.eos_token_id,
            separator_token_ids=trie.separator_token_ids,
            max_paths=1,
        )
        num_return_sequences = num_beams
        decoding_scope = "single_code_top_k"
    else:
        decoding_trie = trie
        num_return_sequences = 1
        decoding_scope = "autoregressive_multi_path"

    MultiPathTrieLogitsProcessor = _logits_processor_class(torch)
    processor = MultiPathTrieLogitsProcessor(decoding_trie, prompt_width)
    from transformers import LogitsProcessorList

    if decoding_mode == "beam_search":
        # Stop immediately after the fixed-length code. Generating EOS here
        # would mix the model's "stop vs. another line" probability into code
        # retrieval scores.
        max_new_tokens = decoding_trie.num_levels
    else:
        max_new_tokens = (
            decoding_trie.max_paths * decoding_trie.num_levels
            + (decoding_trie.max_paths - 1)
            * len(decoding_trie.separator_token_ids)
            + 1
        )
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": decoding_trie.num_levels,
        "eos_token_id": decoding_trie.eos_token_id,
        "pad_token_id": int(tokenizer.pad_token_id),
        "logits_processor": LogitsProcessorList([processor]),
        # Training checkpoints may persist use_cache=False from gradient
        # checkpointing; decoding should use the KV cache, especially for beams.
        "use_cache": True,
        "renormalize_logits": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if decoding_mode == "beam_search":
        generation_kwargs["early_stopping"] = True
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            **generation_kwargs,
        )

    transition_scores = None
    compute_transition_scores = getattr(model, "compute_transition_scores", None)
    if callable(compute_transition_scores):
        transition_kwargs: dict[str, Any] = {"normalize_logits": True}
        beam_indices = getattr(generated, "beam_indices", None)
        if beam_indices is not None:
            # Beam rows are reordered at every step. Passing the ancestry map is
            # required to recover scores for the selected final sequence.
            transition_kwargs["beam_indices"] = beam_indices
        transition_scores = compute_transition_scores(
            generated.sequences,
            generated.scores,
            **transition_kwargs,
        )
    expected_sequences = len(batch) * num_return_sequences
    if int(generated.sequences.shape[0]) != expected_sequences:
        raise RuntimeError(
            "generation returned an unexpected number of sequences: "
            f"expected {expected_sequences}, got {generated.sequences.shape[0]}"
        )

    results: list[dict[str, Any]] = []
    for batch_index, query_row in enumerate(batch):
        paths: list[GeneratedPath] = []
        path_payloads: list[dict[str, Any]] = []
        seen_paths: set[tuple[int, ...]] = set()
        first_sequence = batch_index * num_return_sequences
        for sequence_index in range(
            first_sequence,
            first_sequence + num_return_sequences,
        ):
            suffix = generated.sequences[sequence_index, prompt_width:].tolist()
            if decoding_mode == "beam_search":
                if len(suffix) != decoding_trie.num_levels:
                    raise RuntimeError(
                        "beam search did not return one fixed-length code"
                    )
                sequence_ids = tuple(int(value) for value in suffix)
            else:
                try:
                    eos_position = suffix.index(decoding_trie.eos_token_id)
                except ValueError as exc:
                    raise RuntimeError(
                        "constrained generation did not emit EOS"
                    ) from exc
                sequence_ids = tuple(int(value) for value in suffix[:eos_position])
            path_id_sequence = decoding_trie.parse_complete(sequence_ids)
            if decoding_mode == "beam_search" and len(path_id_sequence) != 1:
                raise RuntimeError("beam search returned a multi-path sequence")

            cursor = 0
            for path_ids in path_id_sequence:
                if path_ids in seen_paths:
                    cursor += decoding_trie.num_levels + len(
                        decoding_trie.separator_token_ids
                    )
                    continue
                seen_paths.add(path_ids)
                path_tokens = tuple(id_to_token[token_id] for token_id in path_ids)
                score = 0.0
                if transition_scores is not None:
                    score = float(
                        transition_scores[
                            sequence_index,
                            cursor : cursor + decoding_trie.num_levels,
                        ]
                        .sum()
                        .item()
                    )
                paths.append(GeneratedPath(path_tokens, score))
                path_payloads.append(
                    {
                        "code_tokens": list(path_tokens),
                        "code_text": "".join(path_tokens),
                        "score": score,
                        "skill_ids": list(buckets[path_tokens]),
                    }
                )
                cursor += decoding_trie.num_levels + len(
                    decoding_trie.separator_token_ids
                )
        candidates = rank_bucket_candidates(paths, buckets, limit=args.top_k)
        results.append(
            {
                "query_id": query_row["id"],
                "query": query_row["query"],
                "generated_text": CODE_PATH_SEPARATOR.join(
                    path["code_text"] for path in path_payloads
                ),
                "decoding": {
                    "mode": decoding_mode,
                    "num_beams": num_beams,
                    "scope": decoding_scope,
                    "num_return_sequences": num_return_sequences,
                },
                "paths": path_payloads,
                "candidates": candidates,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_code_paths < 1:
        raise RouterDataError("batch_size and max_code_paths must be positive")
    if args.top_k < 1:
        raise RouterDataError("top_k must be positive")
    decoding_mode, num_beams = _resolve_decoding(args)
    if any(cutoff < 1 for cutoff in args.cutoffs):
        raise RouterDataError("metric cutoffs must be positive")
    if args.qrels and args.top_k < max(args.cutoffs):
        raise RouterDataError("top_k must cover the largest requested metric cutoff")

    if args.candidate_state_dir:
        from llmgen.incremental import load_candidate_state

        if args.codes or args.registry:
            raise RouterDataError(
                "--candidate-state-dir replaces --codes and --registry"
            )
        decode_map, decode_path, state_tokens_path = load_candidate_state(
            args.candidate_state_dir
        )
        if (
            args.virtual_tokens
            and sha256_file(args.virtual_tokens) != sha256_file(state_tokens_path)
        ):
            raise RouterDataError(
                "--virtual-tokens differs from the candidate-state namespace"
            )
        args.virtual_tokens = str(state_tokens_path)
        args.decode_map = str(decode_path)
        num_levels = int(decode_map["num_levels"])
        skill_to_code = {
            skill_id: tuple(details["tokens"])
            for skill_id, details in decode_map["skill_to_code"].items()
        }
        active_skill_ids = tuple(sorted(decode_map["skills"]))
        buckets = {
            tuple(path["tokens"]): tuple(path["skill_ids"])
            for path in decode_map["paths"]
        }
    else:
        if not args.virtual_tokens or not args.codes or not args.registry:
            raise RouterDataError(
                "set --candidate-state-dir, or set --virtual-tokens, --codes, "
                "and --registry together"
            )
        code_rows = read_jsonl(args.codes)
        skill_to_code, num_levels = normalize_code_rows(code_rows)
        registry = _load_registry(args.registry)
        validate_registry_assignments(registry, code_rows)
        registry_levels = registry.get("num_levels")
        if registry_levels is not None and registry_levels != num_levels:
            raise RouterDataError(
                f"registry num_levels={registry_levels} disagrees with codes={num_levels}"
            )
        active_skill_ids = active_skill_ids_from_registry(registry)
        buckets = buckets_from_codes(skill_to_code, active_skill_ids)
    args._num_levels = num_levels

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
    trie = MultiPathTokenTrie(
        active_token_paths,
        eos_token_id=int(tokenizer.eos_token_id),
        separator_token_ids=args._separator_token_ids,
        max_paths=args.max_code_paths,
    )
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
            "generation": {
                "mode": (
                    "single_code_top_k"
                    if decoding_mode == "beam_search"
                    else "autoregressive_multi_path"
                ),
                "separator": CODE_PATH_SEPARATOR,
                "max_code_paths": (
                    1 if decoding_mode == "beam_search" else trie.max_paths
                ),
                "decoding_mode": decoding_mode,
                "num_beams": num_beams,
                "num_return_sequences": (
                    num_beams if decoding_mode == "beam_search" else 1
                ),
            },
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
