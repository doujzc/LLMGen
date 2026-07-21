#!/usr/bin/env python3
"""Diagnose low-recall hierarchical skill-router checkpoints.

The default path is lightweight: it inspects data coverage, code assignments,
Trainer logs, and an existing ``predictions.jsonl`` without loading the LLM.
Passing ``--model-name-or-path`` additionally measures teacher-forced raw and
trie-constrained token accuracy on sampled train/evaluation examples.  The
latter separates optimization/checkpoint failures from free-running decoding
and generalization failures.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from llmgen.router import (
    CODE_PATH_SEPARATOR,
    MultiPathTokenTrie,
    RouterDataError,
    active_skill_ids_from_registry,
    aggregate_retrieval_metrics,
    buckets_from_codes,
    build_retrieval_examples,
    code_token_id_map,
    encode_target_only_example,
    load_virtual_tokens,
    normalize_code_rows,
    qrels_by_query,
    query_code_path_metrics,
    query_retrieval_metrics,
    read_jsonl,
    validate_registry_assignments,
)


RETRIEVAL_SYSTEM_PROMPT = (
    "Select every Agent Skill needed for the user request in execution order. "
    "Output one hierarchical skill code per line, with no other text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--train-qrels", required=True)
    parser.add_argument("--eval-qrels", required=True)
    parser.add_argument("--eval-queries", required=True)
    parser.add_argument("--router-train", required=True)
    parser.add_argument(
        "--teacher-eval-data",
        help="Optional prebuilt router JSONL used for teacher-forced eval.",
    )
    parser.add_argument("--predictions")
    parser.add_argument("--router-manifest")
    parser.add_argument("--stage1-history")
    parser.add_argument("--source-train-queries")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=(1, 5, 10))

    parser.add_argument(
        "--model-name-or-path",
        help="Optional final router checkpoint; enables teacher-forced diagnostics.",
    )
    parser.add_argument("--base-model-name-or-path")
    parser.add_argument("--virtual-tokens")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--min-train-code-accuracy", type=float, default=0.75)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--system-prompt", default=RETRIEVAL_SYSTEM_PROMPT)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RouterDataError(f"expected JSON object: {path}")
    return value


def _quantiles(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)

    def interpolate(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        name: round(interpolate(fraction), 6)
        for name, fraction in (
            ("min", 0.0),
            ("p10", 0.1),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p90", 0.9),
            ("p95", 0.95),
            ("max", 1.0),
        )
    }


def _distribution(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(Counter(int(value) for value in values).items())
    }


def _normalized_entropy(counts: Sequence[int | float], namespace_size: int) -> float:
    total = float(sum(counts))
    if total <= 0 or namespace_size <= 1:
        return 0.0
    entropy = -sum(
        (float(value) / total) * math.log(float(value) / total)
        for value in counts
        if value > 0
    )
    return entropy / math.log(namespace_size)


def analyze_codes(
    code_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    train_frequency: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]], dict[tuple[str, ...], tuple[str, ...]]]:
    validate_registry_assignments(registry, code_rows)
    skill_to_code, num_levels = normalize_code_rows(code_rows)
    active = active_skill_ids_from_registry(registry)
    buckets = buckets_from_codes(skill_to_code, active)
    branching = registry.get("branching_factors")
    if not isinstance(branching, list) or len(branching) != num_levels:
        branching = [len({path[level] for path in skill_to_code.values()}) for level in range(num_levels)]

    level_details = []
    for level in range(num_levels):
        candidate_counts = Counter(skill_to_code[skill_id][level] for skill_id in active)
        target_counts: Counter[str] = Counter()
        for skill_id, frequency in train_frequency.items():
            if skill_id in skill_to_code and frequency > 0:
                target_counts[skill_to_code[skill_id][level]] += int(frequency)
        namespace_size = int(branching[level])
        level_details.append(
            {
                "level": level + 1,
                "namespace_size": namespace_size,
                "candidate_used_tokens": len(candidate_counts),
                "candidate_utilization": len(candidate_counts) / max(namespace_size, 1),
                "candidate_usage_quantiles": _quantiles(list(candidate_counts.values())),
                "candidate_normalized_entropy": _normalized_entropy(
                    list(candidate_counts.values()), namespace_size
                ),
                "positive_used_tokens": len(target_counts),
                "positive_usage_quantiles": _quantiles(list(target_counts.values())),
                "positive_normalized_entropy": _normalized_entropy(
                    list(target_counts.values()), namespace_size
                ),
                "top_positive_tokens": target_counts.most_common(10),
            }
        )

    collision_members = sum(len(members) for members in buckets.values() if len(members) > 1)
    result = {
        "num_active_skills": len(active),
        "num_levels": num_levels,
        "num_active_paths": len(buckets),
        "path_capacity": math.prod(int(value) for value in branching),
        "collision_count": len(active) - len(buckets),
        "collision_rate": (len(active) - len(buckets)) / max(len(active), 1),
        "collision_member_rate": collision_members / max(len(active), 1),
        "max_bucket_size": max((len(value) for value in buckets.values()), default=0),
        "bucket_size_distribution": _distribution(len(value) for value in buckets.values()),
        "levels": level_details,
    }
    return result, skill_to_code, buckets


def _frequency_bin(value: int) -> str:
    for lower, upper, label in (
        (0, 0, "0"),
        (1, 2, "1-2"),
        (3, 5, "3-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 10**18, "21+"),
    ):
        if lower <= value <= upper:
            return label
    raise AssertionError(value)


def analyze_dataset(
    *,
    train_relevance: Mapping[str, Sequence[str]],
    eval_relevance: Mapping[str, Sequence[str]],
    active_skill_ids: Sequence[str],
    router_train_rows: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
    source_train_queries: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], Counter[str], dict[str, dict[str, float]]]:
    source_train_frequency: Counter[str] = Counter(
        skill_id for values in train_relevance.values() for skill_id in values
    )
    train_frequency: Counter[str] = Counter()
    for row_number, row in enumerate(router_train_rows, start=1):
        raw_targets = row.get("positive_skill_ids", row.get("target_skill_ids"))
        if not isinstance(raw_targets, list) or not raw_targets:
            raise RouterDataError(
                f"router training row {row_number} has no positive_skill_ids"
            )
        train_frequency.update(str(skill_id) for skill_id in raw_targets)
    eval_frequency: Counter[str] = Counter(
        skill_id for values in eval_relevance.values() for skill_id in values
    )
    eval_target_train_frequencies = [
        train_frequency[skill_id]
        for values in eval_relevance.values()
        for skill_id in values
    ]
    unseen_eval_associations = sum(value == 0 for value in eval_target_train_frequencies)
    unseen_eval_skills = sorted(
        skill_id for skill_id in eval_frequency if train_frequency[skill_id] == 0
    )

    popularity = sorted(
        active_skill_ids,
        key=lambda skill_id: (-train_frequency[skill_id], skill_id),
    )
    popularity_per_query = [
        query_retrieval_metrics(popularity, relevant, cutoffs=cutoffs)
        for relevant in eval_relevance.values()
    ]
    popularity_metrics = aggregate_retrieval_metrics(popularity_per_query)
    random_expected = {
        f"recall@{cutoff}": min(cutoff, len(active_skill_ids))
        / max(len(active_skill_ids), 1)
        for cutoff in sorted(set(cutoffs))
    }

    workflow_details = None
    if source_train_queries:
        workflows = {str(row.get("workflow_id") or "") for row in source_train_queries}
        workflows.discard("")
        workflows_by_skill: dict[str, set[str]] = defaultdict(set)
        for row in source_train_queries:
            workflow_id = str(row.get("workflow_id") or "")
            raw_skills = row.get("skill_ids")
            if workflow_id and isinstance(raw_skills, list):
                for skill_id in raw_skills:
                    workflows_by_skill[str(skill_id)].add(workflow_id)
        counts = [len(workflows_by_skill[skill_id]) for skill_id in train_frequency]
        workflow_details = {
            "num_unique_workflows": len(workflows),
            "per_positive_skill_quantiles": _quantiles(counts),
            "skills_with_at_most_5_workflows": sum(value <= 5 for value in counts),
        }

    result = {
        "num_source_train_queries": len(train_relevance),
        "num_source_train_associations": sum(source_train_frequency.values()),
        "num_train_queries": len(router_train_rows),
        "num_train_associations": sum(train_frequency.values()),
        "num_train_positive_skills": len(train_frequency),
        "num_zero_positive_active_skills": sum(
            train_frequency[skill_id] == 0 for skill_id in active_skill_ids
        ),
        "train_associations_per_active_skill": _quantiles(
            [train_frequency[skill_id] for skill_id in active_skill_ids]
        ),
        "train_associations_per_positive_skill": _quantiles(
            list(train_frequency.values())
        ),
        "num_eval_queries": len(eval_relevance),
        "num_eval_associations": sum(eval_frequency.values()),
        "num_eval_target_skills": len(eval_frequency),
        "unseen_eval_target_skills": unseen_eval_skills,
        "unseen_eval_association_count": unseen_eval_associations,
        "eval_target_train_frequency_quantiles": _quantiles(
            eval_target_train_frequencies
        ),
        "eval_associations_by_train_frequency": dict(
            sorted(Counter(_frequency_bin(value) for value in eval_target_train_frequencies).items())
        ),
        "router_train_examples": len(router_train_rows),
        "router_train_target_path_distribution": _distribution(
            len(row.get("target_paths", [])) for row in router_train_rows
        ),
        "popularity_top_skills": popularity[:10],
        "popularity_baseline": popularity_metrics,
        "uniform_random_expected": random_expected,
        "workflow_diversity": workflow_details,
    }
    return result, train_frequency, popularity_metrics


def analyze_predictions(
    *,
    predictions: Sequence[Mapping[str, Any]],
    eval_relevance: Mapping[str, Sequence[str]],
    train_frequency: Mapping[str, int],
    skill_to_code: Mapping[str, Sequence[str]],
    buckets: Mapping[tuple[str, ...], Sequence[str]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    by_query: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(predictions, start=1):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise RouterDataError(f"prediction row {row_number} has no query_id")
        if query_id in by_query:
            raise RouterDataError(f"duplicate prediction query: {query_id}")
        by_query[query_id] = row
    missing = sorted(set(eval_relevance).difference(by_query))
    extra = sorted(set(by_query).difference(eval_relevance))
    if missing or extra:
        raise RouterDataError(
            f"prediction/qrels mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    per_query: list[dict[str, float]] = []
    generated_counts: list[int] = []
    candidate_counts: list[int] = []
    gold_path_counts: list[int] = []
    path_frequency: Counter[tuple[str, ...]] = Counter()
    skill_frequency: Counter[str] = Counter()
    level_prefix_recall: dict[int, list[float]] = defaultdict(list)
    top1_prefix_hit: dict[int, list[float]] = defaultdict(list)
    association_bins: dict[str, list[float]] = defaultdict(list)
    largest_cutoff = max(cutoffs)

    for query_id, relevant_raw in eval_relevance.items():
        row = by_query[query_id]
        relevant = tuple(relevant_raw)
        raw_candidates = row.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise RouterDataError(f"prediction {query_id} candidates must be a list")
        ranked = [str(value["skill_id"]) for value in raw_candidates]
        raw_paths = row.get("paths", [])
        if not isinstance(raw_paths, list):
            raise RouterDataError(f"prediction {query_id} paths must be a list")
        predicted_paths = [tuple(str(token) for token in value["code_tokens"]) for value in raw_paths]
        gold_paths = list(dict.fromkeys(tuple(skill_to_code[skill_id]) for skill_id in relevant))

        metrics = query_retrieval_metrics(ranked, relevant, cutoffs=cutoffs)
        metrics.update(
            query_code_path_metrics(
                predicted_paths,
                relevant,
                skill_to_code,
                buckets,
                cutoffs=cutoffs,
            )
        )
        per_query.append(metrics)
        generated_counts.append(len(predicted_paths))
        candidate_counts.append(len(ranked))
        gold_path_counts.append(len(gold_paths))
        path_frequency.update(predicted_paths)
        skill_frequency.update(ranked)

        for level in range(1, len(gold_paths[0]) + 1):
            predicted_prefixes = {path[:level] for path in predicted_paths}
            gold_prefixes = {path[:level] for path in gold_paths}
            level_prefix_recall[level].append(
                len(predicted_prefixes.intersection(gold_prefixes)) / len(gold_prefixes)
            )
            top1_prefix_hit[level].append(
                float(bool(predicted_paths) and predicted_paths[0][:level] in gold_prefixes)
            )

        retrieved = set(ranked[:largest_cutoff])
        for skill_id in relevant:
            association_bins[_frequency_bin(int(train_frequency.get(skill_id, 0)))].append(
                float(skill_id in retrieved)
            )

    aggregate = aggregate_retrieval_metrics(per_query)
    total_generated = sum(path_frequency.values())
    top_path_count = path_frequency.most_common(1)[0][1] if path_frequency else 0
    return {
        "num_queries": len(predictions),
        "metrics": aggregate,
        "generated_path_count_distribution": _distribution(generated_counts),
        "generated_path_count_mean": mean(generated_counts),
        "gold_path_count_distribution": _distribution(gold_path_counts),
        "gold_path_count_mean": mean(gold_path_counts),
        "candidate_count_distribution": _distribution(candidate_counts),
        "candidate_count_mean": mean(candidate_counts),
        "unique_generated_paths": len(path_frequency),
        "top_generated_path_share": top_path_count / max(total_generated, 1),
        "top_generated_paths": [
            {"code_tokens": list(path), "count": count}
            for path, count in path_frequency.most_common(10)
        ],
        "top_predicted_skills": skill_frequency.most_common(10),
        "prefix_recall": {
            f"level_{level}": mean(values)
            for level, values in sorted(level_prefix_recall.items())
        },
        "top1_prefix_hit": {
            f"level_{level}": mean(values)
            for level, values in sorted(top1_prefix_hit.items())
        },
        f"association_recall@{largest_cutoff}_by_train_frequency": {
            label: {
                "count": len(values),
                "recall": mean(values),
            }
            for label, values in sorted(association_bins.items())
        },
    }


def analyze_training_logs(
    router_manifest_path: str | None,
    stage1_history_path: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if stage1_history_path and Path(stage1_history_path).is_file():
        history = read_jsonl(stage1_history_path)
        if history:
            best_collision = min(
                history,
                key=lambda row: (
                    float(row.get("collision_rate", float("inf"))),
                    float(row.get("loss", float("inf"))),
                ),
            )
            result["stage1"] = {
                "epochs_logged": len(history),
                "last": history[-1],
                "best_collision_epoch": best_collision,
            }

    if router_manifest_path and Path(router_manifest_path).is_file():
        manifest_path = Path(router_manifest_path)
        manifest = _load_json(manifest_path)
        trainer_states = []
        for path in manifest_path.parent.glob("checkpoint-*/trainer_state.json"):
            try:
                trainer_states.append(_load_json(path))
            except (OSError, json.JSONDecodeError, RouterDataError):
                continue
        trainer_state = max(
            trainer_states,
            key=lambda value: int(value.get("global_step", 0)),
            default=None,
        )
        phase: dict[str, Any] = {"manifest": manifest}
        if trainer_state is not None:
            logs = trainer_state.get("log_history", [])
            train_logs = [row for row in logs if "loss" in row]
            eval_logs = [row for row in logs if "eval_loss" in row]
            grad_norms = [float(row["grad_norm"]) for row in train_logs if "grad_norm" in row]
            phase["trainer_state"] = {
                "global_step": trainer_state.get("global_step"),
                "best_global_step": trainer_state.get("best_global_step"),
                "best_metric": trainer_state.get("best_metric"),
                "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
                "last_epoch": max(
                    (float(row.get("epoch", 0.0)) for row in logs), default=0.0
                ),
                "last_train_log": train_logs[-1] if train_logs else None,
                "last_eval_log": eval_logs[-1] if eval_logs else None,
                "minimum_eval_loss": min(
                    (float(row["eval_loss"]) for row in eval_logs), default=None
                ),
                "grad_norm_quantiles": _quantiles(grad_norms),
            }
        result["retrieval"] = phase
    return result


def _load_model(args: argparse.Namespace, virtual_tokens: tuple[str, ...]):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - GPU environment only
        raise SystemExit("--model-name-or-path requires torch and transformers") from exc

    model_path = Path(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise RouterDataError("router tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = code_token_id_map(tokenizer, virtual_tokens)
    requested_dtype = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if requested_dtype is not None:
        model_kwargs["torch_dtype"] = requested_dtype

    is_adapter = (model_path / "adapter_config.json").is_file() or (
        bool(args.base_model_name_or_path) and not (model_path / "config.json").is_file()
    )
    if is_adapter:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("adapter diagnostics require peft") from exc
        adapter_config = PeftConfig.from_pretrained(args.model_name_or_path)
        base_name = args.base_model_name_or_path or adapter_config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.to(args.device)
    model.eval()
    return torch, tokenizer, model, token_ids


def _sample_rows(rows: Sequence[Mapping[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    if len(copied) <= size:
        return copied
    indices = sorted(random.Random(seed).sample(range(len(copied)), size))
    return [copied[index] for index in indices]


def _teacher_forced_split(
    *,
    rows: Sequence[Mapping[str, Any]],
    torch: Any,
    tokenizer: Any,
    model: Any,
    token_ids: Mapping[str, int],
    skill_to_code: Mapping[str, Sequence[str]],
    trie: MultiPathTokenTrie,
    max_length: int,
    system_prompt: str,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch.nn.functional as F

    token_level: dict[int, int] = {}
    for path in skill_to_code.values():
        for level, token in enumerate(path, start=1):
            token_level[token_ids[token]] = level
    separator_ids = set(trie.separator_token_ids)
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    nll: dict[str, list[float]] = defaultdict(list)
    raw_exact_examples = 0
    constrained_exact_examples = 0

    encoded = [
        encode_target_only_example(
            tokenizer,
            row,
            code_token_ids=token_ids,
            num_levels=trie.num_levels,
            max_length=max_length,
            system_prompt=system_prompt,
        )
        for row in rows
    ]
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        attention_mask = []
        for row in batch:
            padding = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [int(tokenizer.pad_token_id)] * padding)
            attention_mask.append(row["attention_mask"] + [0] * padding)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                use_cache=False,
            ).logits

        for batch_index, row in enumerate(batch):
            labels = row["labels"]
            supervised = [position for position, value in enumerate(labels) if value != -100]
            prefix: list[int] = []
            raw_example_correct = True
            constrained_example_correct = True
            for position in supervised:
                if position == 0:
                    raise RouterDataError("supervised target unexpectedly starts at position zero")
                gold = int(labels[position])
                scores = logits[batch_index, position - 1].float()
                raw_prediction = int(torch.argmax(scores).item())
                raw_loss = float(F.cross_entropy(scores.unsqueeze(0), torch.tensor([gold], device=device)).item())
                allowed = trie.allowed_next(prefix)
                if gold not in allowed:
                    raise RouterDataError(
                        f"gold token {gold} is illegal after teacher prefix {prefix}"
                    )
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
                allowed_scores = scores.index_select(0, allowed_tensor)
                allowed_gold_index = allowed.index(gold)
                constrained_prediction = int(
                    allowed[int(torch.argmax(allowed_scores).item())]
                )
                constrained_loss = float(
                    F.cross_entropy(
                        allowed_scores.unsqueeze(0),
                        torch.tensor([allowed_gold_index], device=device),
                    ).item()
                )
                if gold in token_level:
                    categories = ("all", "code", f"level_{token_level[gold]}")
                elif gold == trie.eos_token_id:
                    categories = ("all", "boundary", "eos")
                elif gold in separator_ids:
                    categories = ("all", "boundary", "separator")
                else:
                    categories = ("all", "other")
                for category in categories:
                    totals[category]["count"] += 1
                    totals[category]["raw_correct"] += int(raw_prediction == gold)
                    totals[category]["constrained_correct"] += int(
                        constrained_prediction == gold
                    )
                    nll[f"{category}:raw"].append(raw_loss)
                    nll[f"{category}:constrained"].append(constrained_loss)
                raw_example_correct &= raw_prediction == gold
                constrained_example_correct &= constrained_prediction == gold
                if gold != trie.eos_token_id:
                    prefix.append(gold)
            raw_exact_examples += int(raw_example_correct)
            constrained_exact_examples += int(constrained_example_correct)

    category_metrics = {}
    for category, counts in sorted(totals.items()):
        count = counts["count"]
        category_metrics[category] = {
            "tokens": count,
            "raw_accuracy": counts["raw_correct"] / max(count, 1),
            "constrained_accuracy": counts["constrained_correct"] / max(count, 1),
            "raw_nll": mean(nll[f"{category}:raw"]),
            "constrained_nll": mean(nll[f"{category}:constrained"]),
        }
    return {
        "examples": len(rows),
        "raw_sequence_exact_match": raw_exact_examples / max(len(rows), 1),
        "constrained_sequence_exact_match": constrained_exact_examples / max(len(rows), 1),
        "categories": category_metrics,
    }


def analyze_teacher_forcing(
    *,
    args: argparse.Namespace,
    router_train_rows: Sequence[Mapping[str, Any]],
    eval_queries: Sequence[Mapping[str, Any]],
    eval_relevance: Mapping[str, Sequence[str]],
    skill_to_code: Mapping[str, Sequence[str]],
    buckets: Mapping[tuple[str, ...], Sequence[str]],
    router_manifest: Mapping[str, Any] | None,
    teacher_eval_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if not args.virtual_tokens:
        raise RouterDataError("--virtual-tokens is required with --model-name-or-path")
    virtual_tokens = load_virtual_tokens(args.virtual_tokens)
    torch, tokenizer, model, token_ids = _load_model(args, virtual_tokens)
    separator_ids = tuple(
        int(value)
        for value in tokenizer.encode(
            CODE_PATH_SEPARATOR,
            add_special_tokens=False,
            verbose=False,
        )
    )
    active_paths = [tuple(token_ids[token] for token in path) for path in buckets]
    eval_rows = (
        [dict(row) for row in teacher_eval_rows]
        if teacher_eval_rows is not None
        else build_retrieval_examples(eval_queries, skill_to_code, eval_relevance)
    )
    all_teacher_rows = [*router_train_rows, *eval_rows]
    trie = MultiPathTokenTrie(
        active_paths,
        eos_token_id=int(tokenizer.eos_token_id),
        separator_token_ids=separator_ids,
        max_paths=max(len(row.get("target_paths", [])) for row in all_teacher_rows),
    )
    max_length = args.max_length
    if max_length is None and router_manifest:
        value = router_manifest.get("max_length")
        if isinstance(value, (int, float)):
            max_length = int(value)
    max_length = max_length or 1024
    train_sample = _sample_rows(router_train_rows, args.sample_size, args.seed)
    eval_sample = _sample_rows(eval_rows, args.sample_size, args.seed + 1)
    return {
        "eval_source": "prebuilt" if teacher_eval_rows is not None else "retrieval_qrels",
        "train": _teacher_forced_split(
            rows=train_sample,
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            token_ids=token_ids,
            skill_to_code=skill_to_code,
            trie=trie,
            max_length=max_length,
            system_prompt=args.system_prompt,
            batch_size=args.batch_size,
            device=args.device,
        ),
        "eval": _teacher_forced_split(
            rows=eval_sample,
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            token_ids=token_ids,
            skill_to_code=skill_to_code,
            trie=trie,
            max_length=max_length,
            system_prompt=args.system_prompt,
            batch_size=args.batch_size,
            device=args.device,
        ),
    }


def build_findings(
    report: Mapping[str, Any],
    cutoffs: Sequence[int],
    min_train_code_accuracy: float = 0.75,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    dataset = report["dataset"]
    codes = report["codes"]
    predictions = report.get("predictions")
    cutoff = 5 if 5 in cutoffs else max(cutoffs)

    if dataset["unseen_eval_association_count"]:
        findings.append(
            {
                "severity": "critical",
                "code": "unseen_eval_targets",
                "message": "Some evaluation targets have no positive retrieval supervision.",
            }
        )
    if codes["collision_rate"] > 0.05:
        findings.append(
            {
                "severity": "critical",
                "code": "code_collisions",
                "message": (
                    f"{codes['collision_rate']:.1%} of active skills lose a unique path; "
                    "exact skill retrieval is impossible inside collision buckets."
                ),
            }
        )
    for level in codes["levels"]:
        if level["candidate_utilization"] < 0.8 or level["candidate_normalized_entropy"] < 0.85:
            findings.append(
                {
                    "severity": "high",
                    "code": f"level_{level['level']}_codebook_collapse",
                    "message": (
                        f"Level {level['level']} uses {level['candidate_used_tokens']}/"
                        f"{level['namespace_size']} tokens with normalized entropy "
                        f"{level['candidate_normalized_entropy']:.3f}."
                    ),
                }
            )

    if predictions:
        model_recall = float(predictions["metrics"].get(f"recall@{cutoff}", 0.0))
        popularity_recall = float(
            dataset["popularity_baseline"].get(f"recall@{cutoff}", 0.0)
        )
        if model_recall <= popularity_recall:
            findings.append(
                {
                    "severity": "critical",
                    "code": "below_popularity_baseline",
                    "message": (
                        f"Model Recall@{cutoff}={model_recall:.4f} does not beat the "
                        f"train-popularity baseline {popularity_recall:.4f}."
                    ),
                }
            )
        if predictions["generated_path_count_mean"] < 0.75 * predictions["gold_path_count_mean"]:
            findings.append(
                {
                    "severity": "high",
                    "code": "premature_eos",
                    "message": (
                        "The model generates fewer skill paths than the gold workflows "
                        f"({predictions['generated_path_count_mean']:.2f} vs "
                        f"{predictions['gold_path_count_mean']:.2f})."
                    ),
                }
            )
        path_coverage = predictions["unique_generated_paths"] / max(codes["num_active_paths"], 1)
        if path_coverage < 0.05 or predictions["top_generated_path_share"] > 0.15:
            findings.append(
                {
                    "severity": "critical",
                    "code": "prediction_collapse",
                    "message": (
                        f"Generated-path coverage is {path_coverage:.1%}; the most common "
                        f"path occupies {predictions['top_generated_path_share']:.1%} of outputs."
                    ),
                }
            )
        prefix = predictions.get("prefix_recall", {})
        first = float(prefix.get("level_1", 0.0))
        full = float(prefix.get(f"level_{codes['num_levels']}", 0.0))
        if first >= 0.25 and full < first * 0.4:
            findings.append(
                {
                    "severity": "high",
                    "code": "leaf_disambiguation_failure",
                    "message": (
                        f"Level-1 prefix recall is {first:.3f}, but full-path recall is "
                        f"only {full:.3f}; the later code level is the bottleneck."
                    ),
                }
            )

    teacher = report.get("teacher_forcing")
    if teacher:
        train_code = teacher["train"]["categories"]["code"]["constrained_accuracy"]
        eval_code = teacher["eval"]["categories"]["code"]["constrained_accuracy"]
        if train_code < min_train_code_accuracy:
            findings.append(
                {
                    "severity": "critical",
                    "code": "router_underfit_or_checkpoint_failure",
                    "message": (
                        f"Teacher-forced constrained code accuracy on train is only "
                        f"{train_code:.1%} (required {min_train_code_accuracy:.1%}); "
                        "inspect optimization, saved weights, and token embeddings."
                    ),
                }
            )
        elif train_code > 0.95 and eval_code < 0.6:
            findings.append(
                {
                    "severity": "high",
                    "code": "workflow_overfit",
                    "message": (
                        f"Train teacher-forced code accuracy is {train_code:.1%}, but eval is "
                        f"{eval_code:.1%}; supervision lacks compositional workflow diversity."
                    ),
                }
            )
        elif predictions and eval_code > 0.8 and predictions["metrics"].get(f"recall@{cutoff}", 0.0) < 0.1:
            findings.append(
                {
                    "severity": "critical",
                    "code": "free_running_decode_failure",
                    "message": (
                        "Teacher-forced eval accuracy is high while autoregressive recall is low; "
                        "boundary/EOS decisions or exposure error dominate."
                    ),
                }
            )

    if not findings:
        findings.append(
            {
                "severity": "info",
                "code": "no_threshold_triggered",
                "message": "No automatic threshold fired; inspect the detailed report.",
            }
        )
    return findings


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.sample_size < 1:
        raise RouterDataError("batch-size and sample-size must be positive")
    if not 0.0 <= args.min_train_code_accuracy <= 1.0:
        raise RouterDataError("min-train-code-accuracy must be in [0, 1]")
    if not args.cutoffs or any(value < 1 for value in args.cutoffs):
        raise RouterDataError("cutoffs must be positive")

    code_rows = read_jsonl(args.codes)
    registry = _load_json(args.registry)
    train_relevance = qrels_by_query(read_jsonl(args.train_qrels))
    eval_relevance = qrels_by_query(read_jsonl(args.eval_qrels))
    eval_queries = read_jsonl(args.eval_queries)
    router_train_rows = read_jsonl(args.router_train)
    teacher_eval_rows = (
        read_jsonl(args.teacher_eval_data) if args.teacher_eval_data else None
    )
    source_train_queries = (
        read_jsonl(args.source_train_queries)
        if args.source_train_queries and Path(args.source_train_queries).is_file()
        else None
    )
    active = active_skill_ids_from_registry(registry)
    dataset, train_frequency, _ = analyze_dataset(
        train_relevance=train_relevance,
        eval_relevance=eval_relevance,
        active_skill_ids=active,
        router_train_rows=router_train_rows,
        cutoffs=args.cutoffs,
        source_train_queries=source_train_queries,
    )
    codes, skill_to_code, buckets = analyze_codes(
        code_rows, registry, train_frequency
    )
    router_manifest = (
        _load_json(args.router_manifest)
        if args.router_manifest and Path(args.router_manifest).is_file()
        else None
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "codes": codes,
        "training": analyze_training_logs(args.router_manifest, args.stage1_history),
    }
    if args.predictions and Path(args.predictions).is_file():
        report["predictions"] = analyze_predictions(
            predictions=read_jsonl(args.predictions),
            eval_relevance=eval_relevance,
            train_frequency=train_frequency,
            skill_to_code=skill_to_code,
            buckets=buckets,
            cutoffs=args.cutoffs,
        )
    if args.model_name_or_path:
        report["teacher_forcing"] = analyze_teacher_forcing(
            args=args,
            router_train_rows=router_train_rows,
            eval_queries=eval_queries,
            eval_relevance=eval_relevance,
            skill_to_code=skill_to_code,
            buckets=buckets,
            router_manifest=router_manifest,
            teacher_eval_rows=teacher_eval_rows,
        )
    report["findings"] = build_findings(
        report, args.cutoffs, args.min_train_code_accuracy
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    key_metrics = {
        "popularity_recall@5": dataset["popularity_baseline"].get("recall@5"),
        "model_recall@5": report.get("predictions", {}).get("metrics", {}).get("recall@5"),
        "collision_rate": codes["collision_rate"],
        "generated_paths_mean": report.get("predictions", {}).get("generated_path_count_mean"),
        "gold_paths_mean": report.get("predictions", {}).get("gold_path_count_mean"),
    }
    print(
        json.dumps(
            {
                "report": str(output.resolve()),
                "key_metrics": key_metrics,
                "findings": report["findings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
