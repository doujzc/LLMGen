#!/usr/bin/env python3
"""Generate controlled, independently reviewed multi-turn Top1 training data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from llmgen.synthesis import (
    DIRECTNESS_AUDIT_VERSION,
    SYNTHESIS_VERSION,
    DialogueBlueprint,
    ModelCall,
    OpenAICompatibleClient,
    acceptance_reasons,
    build_dialogue_blueprints,
    combine_directness_audits,
    content_sha256,
    directness_messages,
    generation_messages,
    judgment_messages,
    load_api_credentials,
    load_taxonomy_descriptions,
    parse_generated_samples,
    parse_directness_audits,
    parse_judgments,
    taxonomy_prompt,
)
from llmgen.top1 import (
    Top1DataError,
    load_candidate_names,
    normalize_messages,
    read_jsonl,
    sha256_file,
    validate_training_rows,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIG = "configs/top1_synthesis_v1.json"
DEFAULT_CREDENTIALS = "~/Codes/api_keys/llm_api.txt"
DEFAULT_OUTPUT_DIR = "data_top1/generated/top1_controlled_multiturn_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse independent synthesis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--credentials-file", default=DEFAULT_CREDENTIALS)
    parser.add_argument(
        "--candidate-registry",
        default="configs/top1_candidates.json",
    )
    parser.add_argument(
        "--taxonomy-data",
        default="data_top1/top1_labeldesc_paper_v1.jsonl",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--intent-change-per-pair", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-sample-attempts", type=int)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--judgment-batch-size", type=int)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the immutable manifest and plans without calling an LLM",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit successfully even if quality gates exhaust some scenario attempts",
    )
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Top1DataError(f"invalid JSON config: {path}") from exc
    if not isinstance(payload, dict):
        raise Top1DataError(f"JSON object required: {path}")
    return payload


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise Top1DataError("batch size must be positive")
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _request_hash(messages: Sequence[Mapping[str, str]]) -> str:
    return content_sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def _canonical_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (message["role"], message["content"])
        for message in normalize_messages(messages)
    )


def _run_model_batches(
    *,
    stage: str,
    batches: Sequence[Sequence[Any]],
    model: str,
    client: OpenAICompatibleClient,
    max_workers: int,
    temperature: float,
    max_tokens: int,
    build_messages: Callable[[Sequence[Any]], list[dict[str, str]]],
    item_id: Callable[[Any], str],
    item_attempt: Callable[[Any], int],
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Run independent batches concurrently while logging only secret-free responses."""

    if not batches:
        return []
    prepared: list[tuple[int, Sequence[Any], list[dict[str, str]]]] = []
    for index, batch in enumerate(batches, start=1):
        prepared.append((index, batch, build_messages(batch)))

    expected_by_hash = {
        _request_hash(messages): {
            "scenario_ids": [item_id(item) for item in batch],
            "sample_attempts": {
                item_id(item): item_attempt(item) for item in batch
            },
        }
        for _, batch, messages in prepared
    }
    cached: list[dict[str, Any]] = []
    cached_hashes: set[str] = set()
    if raw_path.is_file():
        for record in read_jsonl(raw_path):
            request_sha256 = record.get("request_sha256")
            expected = expected_by_hash.get(str(request_sha256))
            if (
                expected is None
                or request_sha256 in cached_hashes
                or record.get("status") != "completed"
                or record.get("model") != model
                or record.get("scenario_ids") != expected["scenario_ids"]
            ):
                continue
            recorded_attempts = record.get("sample_attempts")
            legacy_attempt_two = (
                recorded_attempts is None
                and set(expected["sample_attempts"].values()) == {2}
            )
            if recorded_attempts != expected["sample_attempts"] and not legacy_attempt_two:
                continue
            cached_record = dict(record)
            cached_record["cache_hit"] = True
            cached.append(cached_record)
            cached_hashes.add(str(request_sha256))

    pending_prepared = [
        value
        for value in prepared
        if _request_hash(value[2]) not in cached_hashes
    ]
    results: list[dict[str, Any]] = list(cached)
    if cached:
        print(f"[synthesis] {stage}: reused {len(cached)}/{len(batches)} cached batches")
    if not pending_prepared:
        return results
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                client.chat_json,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ): (batch_index, batch, messages)
            for batch_index, batch, messages in pending_prepared
        }
        for future in as_completed(futures):
            batch_index, batch, messages = futures[future]
            identifiers = [item_id(item) for item in batch]
            record: dict[str, Any] = {
                "timestamp": _now(),
                "stage": stage,
                "batch_index": batch_index,
                "scenario_ids": identifiers,
                "sample_attempts": {
                    item_id(item): item_attempt(item) for item in batch
                },
                "model": model,
                "request_sha256": _request_hash(messages),
            }
            try:
                call: ModelCall = future.result()
                record.update(
                    {
                        "status": "completed",
                        "content": call.content,
                        "finish_reason": call.finish_reason,
                        "usage": dict(call.usage),
                        "elapsed_seconds": round(call.elapsed_seconds, 6),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve batch failure for retry
                record.update({"status": "failed", "error": str(exc)})
            _append_jsonl(raw_path, (record,))
            results.append(record)
            completed += 1
            done_total = len(cached) + completed
            if completed == len(pending_prepared) or completed % 10 == 0:
                print(f"[synthesis] {stage}: {done_total}/{len(batches)} batches", flush=True)
    return results


def _accepted_attempt(row: Mapping[str, Any]) -> int:
    synthesis = row.get("synthesis")
    if not isinstance(synthesis, Mapping) or not isinstance(synthesis.get("attempt"), int):
        raise Top1DataError("accepted record has no synthesis attempt")
    return int(synthesis["attempt"])


def _load_invalidations(path: Path) -> dict[str, int]:
    invalidated: dict[str, int] = {}
    if not path.is_file():
        return invalidated
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        attempt = row.get("invalidated_attempt")
        if isinstance(scenario_id, str) and isinstance(attempt, int):
            invalidated[scenario_id] = max(invalidated.get(scenario_id, 0), attempt)
    return invalidated


def _load_accepted(
    path: Path,
    invalidation_path: Path,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        if not isinstance(scenario_id, str):
            raise Top1DataError("accepted record has no scenario_id")
        previous = result.get(scenario_id)
        if previous is None or _accepted_attempt(row) > _accepted_attempt(previous):
            result[scenario_id] = row
    invalidated = _load_invalidations(invalidation_path)
    return {
        scenario_id: row
        for scenario_id, row in result.items()
        if _accepted_attempt(row) > invalidated.get(scenario_id, 0)
    }


def _load_attempt_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.is_file():
        return counts
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        attempt = row.get("attempt")
        if isinstance(scenario_id, str) and isinstance(attempt, int):
            counts[scenario_id] = max(counts[scenario_id], attempt)
    return counts


def _usage_summary(raw_directory: Path) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    for path in sorted(raw_directory.glob("*_responses.jsonl")):
        for row in read_jsonl(path):
            stage = str(row.get("stage", "unknown"))
            bucket = summary[stage]
            bucket["calls"] += 1
            if row.get("status") != "completed":
                bucket["failed_calls"] += 1
            usage = row.get("usage")
            if isinstance(usage, Mapping):
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(field)
                    if isinstance(value, int):
                        bucket[field] += value
    return dict(sorted(summary.items()))


def _count(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def _write_summary(
    *,
    path: Path,
    plans: Sequence[DialogueBlueprint],
    accepted: Mapping[str, Mapping[str, Any]],
    attempt_path: Path,
    raw_directory: Path,
    train_path: Path,
    complete: bool,
) -> None:
    attempts = read_jsonl(attempt_path) if attempt_path.is_file() else []
    rejected_attempts = [row for row in attempts if row.get("status") != "accepted"]
    rejection_reasons = Counter(
        str(reason)
        for row in rejected_attempts
        for reason in row.get("reasons", [])
        if isinstance(reason, str)
    )
    accepted_rows = list(accepted.values())
    output: dict[str, Any] = {
        "path": str(train_path),
        "exists": train_path.is_file(),
    }
    if train_path.is_file():
        output["sha256"] = sha256_file(train_path)
    write_json(
        path,
        {
            "schema_version": 1,
            "pipeline_version": SYNTHESIS_VERSION,
            "updated_at": _now(),
            "complete": complete,
            "planned_rows": len(plans),
            "accepted_rows": len(accepted_rows),
            "unresolved_rows": len(plans) - len(accepted_rows),
            "attempts": len(attempts),
            "rejected_attempts": len(rejected_attempts),
            "acceptance_rate_per_attempt": (
                len(accepted_rows) / len(attempts) if attempts else 0.0
            ),
            "candidate_counts": _count(accepted_rows, "target_candidate_name"),
            "phenomenon_counts": _count(accepted_rows, "conversation_phenomenon"),
            "source_candidate_counts": _count(
                [row for row in accepted_rows if row.get("source_candidate_name")],
                "source_candidate_name",
            ),
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
            "model_usage": _usage_summary(raw_directory),
            "output": output,
        },
    )


def _manifest_signature(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "created_at"}
    return content_sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True))


def _prepare_run(
    *,
    output_directory: Path,
    config: Mapping[str, Any],
    candidate_path: Path,
    taxonomy_path: Path,
    endpoint: str,
    plans: Sequence[DialogueBlueprint],
    taxonomy: str,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    plans_path = output_directory / "plans.jsonl"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_version": SYNTHESIS_VERSION,
        "created_at": _now(),
        "config": dict(config),
        "inputs": {
            "candidate_registry": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "taxonomy_data": {
                "path": str(taxonomy_path),
                "sha256": sha256_file(taxonomy_path),
            },
        },
        "endpoint_sha256": content_sha256(endpoint),
        "taxonomy_prompt_sha256": content_sha256(taxonomy),
        "planned_rows": len(plans),
    }
    signature = _manifest_signature(payload)
    payload["run_signature"] = signature
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("run_signature") != signature:
            raise Top1DataError(
                f"output directory belongs to a different synthesis run: {output_directory}"
            )
    else:
        write_json(manifest_path, payload)

    expected_plans = [plan.to_dict() for plan in plans]
    if plans_path.is_file():
        if read_jsonl(plans_path) != expected_plans:
            raise Top1DataError("existing synthesis plans differ from the current run")
    else:
        write_jsonl(plans_path, expected_plans)


def _raw_record_by_ids(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(scenario_id): record
        for record in records
        for scenario_id in record.get("scenario_ids", [])
    }


def _generate_attempt(
    *,
    blueprints: Sequence[DialogueBlueprint],
    attempt_numbers: Mapping[str, int],
    taxonomy: str,
    candidate_names: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    batches = _chunks(blueprints, int(config["generation_batch_size"]))
    records = _run_model_batches(
        stage="generation",
        batches=batches,
        model=str(config["generator_model"]),
        client=client,
        max_workers=int(config["max_workers"]),
        temperature=float(config["generator_temperature"]),
        max_tokens=int(config["generation_max_tokens"]),
        build_messages=lambda batch: generation_messages(batch, taxonomy),
        item_id=lambda item: item.scenario_id,
        item_attempt=lambda item: attempt_numbers[item.scenario_id],
        raw_path=raw_directory / "generation_responses.jsonl",
    )
    generated: dict[str, dict[str, Any]] = {}
    errors = {blueprint.scenario_id: [] for blueprint in blueprints}
    blueprint_by_id = {blueprint.scenario_id: blueprint for blueprint in blueprints}
    for record in records:
        identifiers = [str(value) for value in record["scenario_ids"]]
        if record.get("status") != "completed":
            for scenario_id in identifiers:
                errors[scenario_id].append("generation_api_failure")
            continue
        assigned = {scenario_id: blueprint_by_id[scenario_id] for scenario_id in identifiers}
        parsed, parse_errors = parse_generated_samples(
            str(record["content"]),
            assigned,
            candidate_names,
        )
        for scenario_id, values in parse_errors.items():
            errors[scenario_id].extend(values)
        for scenario_id, sample in parsed.items():
            sample["attempt"] = attempt_numbers[scenario_id]
            generated[scenario_id] = sample
    return generated, errors


def _judge_attempt(
    *,
    stage: str,
    samples: Sequence[Mapping[str, Any]],
    taxonomy: str,
    candidate_names: Sequence[str],
    model: str,
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    batches = _chunks(samples, int(config["judgment_batch_size"]))
    records = _run_model_batches(
        stage=stage,
        batches=batches,
        model=model,
        client=client,
        max_workers=int(config["max_workers"]),
        temperature=float(config["judge_temperature"]),
        max_tokens=int(config["judgment_max_tokens"]),
        build_messages=lambda batch: judgment_messages(batch, taxonomy),
        item_id=lambda item: str(item["scenario_id"]),
        item_attempt=lambda item: int(item["attempt"]),
        raw_path=raw_directory / f"{stage}_responses.jsonl",
    )
    judgments: dict[str, dict[str, Any]] = {}
    errors = {str(sample["scenario_id"]): [] for sample in samples}
    for record in records:
        identifiers = [str(value) for value in record["scenario_ids"]]
        if record.get("status") != "completed":
            for scenario_id in identifiers:
                errors[scenario_id].append(f"{stage}_api_failure")
            continue
        parsed, parse_errors = parse_judgments(
            str(record["content"]),
            identifiers,
            candidate_names,
        )
        judgments.update(parsed)
        for scenario_id, values in parse_errors.items():
            errors[scenario_id].extend(values)
    return judgments, errors


def _directness_attempt(
    *,
    samples: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    batches = _chunks(samples, int(config["judgment_batch_size"]))
    if len(models) < 2 or len(set(models)) != len(models):
        raise Top1DataError("directness audit requires at least two distinct models")
    identifiers = [str(sample["scenario_id"]) for sample in samples]
    errors = {scenario_id: [] for scenario_id in identifiers}
    audits_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for index, model in enumerate(models):
        stage = "directness" if index == 0 else f"directness_crosscheck_{index}"
        records = _run_model_batches(
            stage=stage,
            batches=batches,
            model=model,
            client=client,
            max_workers=int(config["max_workers"]),
            temperature=float(config["judge_temperature"]),
            max_tokens=int(config["judgment_max_tokens"]),
            build_messages=directness_messages,
            item_id=lambda item: str(item["scenario_id"]),
            item_attempt=lambda item: int(item["attempt"]),
            raw_path=raw_directory / f"{stage}_responses.jsonl",
        )
        model_audits: dict[str, dict[str, Any]] = {}
        for record in records:
            batch_ids = [str(value) for value in record["scenario_ids"]]
            if record.get("status") != "completed":
                for scenario_id in batch_ids:
                    errors[scenario_id].append(f"{stage}_api_failure")
                continue
            parsed, parse_errors = parse_directness_audits(
                str(record["content"]),
                batch_ids,
            )
            model_audits.update(parsed)
            for scenario_id, values in parse_errors.items():
                errors[scenario_id].extend(f"{stage}:{value}" for value in values)
        audits_by_model[model] = model_audits

    consensus: dict[str, dict[str, Any]] = {}
    for scenario_id in identifiers:
        judgments = [audits_by_model[model].get(scenario_id) for model in models]
        if errors[scenario_id] or any(judgment is None for judgment in judgments):
            continue
        model_judgments = {
            model: judgment
            for model, judgment in zip(models, judgments)
            if judgment is not None
        }
        consensus[scenario_id] = {
            "scenario_id": scenario_id,
            **combine_directness_audits(model_judgments),
        }
    return consensus, errors


def _load_directness_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return records
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        attempt = row.get("attempt")
        if (
            isinstance(scenario_id, str)
            and isinstance(attempt, int)
            and row.get("audit_version") == DIRECTNESS_AUDIT_VERSION
        ):
            records[(scenario_id, attempt)] = row
    return records


def _audit_existing_directness(
    *,
    accepted: dict[str, dict[str, Any]],
    directness_path: Path,
    invalidation_path: Path,
    models: Sequence[str],
    config: Mapping[str, Any],
    client: OpenAICompatibleClient,
    raw_directory: Path,
) -> dict[str, dict[str, Any]]:
    existing = _load_directness_records(directness_path)
    samples = []
    for row in accepted.values():
        if row.get("conversation_phenomenon") != "intent_change":
            continue
        attempt = _accepted_attempt(row)
        if (str(row["scenario_id"]), attempt) in existing:
            continue
        samples.append(
            {
                "scenario_id": str(row["scenario_id"]),
                "messages": row["messages"],
                "attempt": attempt,
            }
        )
    if not samples:
        return accepted
    print(f"[synthesis] strict directness backfill: {len(samples)} IntentChange rows", flush=True)
    audits, errors = _directness_attempt(
        samples=samples,
        models=models,
        config=config,
        client=client,
        raw_directory=raw_directory,
    )
    audit_records: list[dict[str, Any]] = []
    invalidations: list[dict[str, Any]] = []
    for sample in samples:
        scenario_id = str(sample["scenario_id"])
        audit = audits.get(scenario_id)
        audit_errors = errors[scenario_id]
        passed = (
            not audit_errors
            and audit is not None
            and audit.get("contains_only_new_request") is True
            and audit.get("references_previous_exchange") is False
            and audit.get("uses_transition_or_acknowledgment") is False
            and audit.get("direct_final_request") is True
            and audit.get("has_switch_meta_language") is False
        )
        audit_records.append(
            {
                "timestamp": _now(),
                "audit_version": DIRECTNESS_AUDIT_VERSION,
                "scenario_id": scenario_id,
                "attempt": int(sample["attempt"]),
                "source": "backfill",
                "passed": passed,
                "errors": audit_errors,
                "audit": audit,
            }
        )
        if not passed:
            invalidations.append(
                {
                    "timestamp": _now(),
                    "audit_version": DIRECTNESS_AUDIT_VERSION,
                    "scenario_id": scenario_id,
                    "invalidated_attempt": int(sample["attempt"]),
                    "reason": "strict_directness_backfill_failed",
                }
            )
    _append_jsonl(directness_path, audit_records)
    _append_jsonl(invalidation_path, invalidations)
    for row in invalidations:
        accepted.pop(str(row["scenario_id"]), None)
    print(
        f"[synthesis] strict directness backfill: "
        f"{len(samples) - len(invalidations)} passed, {len(invalidations)} invalidated",
        flush=True,
    )
    return accepted


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    candidate_path = Path(args.candidate_registry).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy_data).expanduser().resolve()
    output_directory = Path(args.output_dir).expanduser().resolve()
    config = _read_json(config_path)
    if config.get("pipeline_version") != SYNTHESIS_VERSION:
        raise Top1DataError("synthesis config pipeline_version mismatch")
    runtime_config = dict(config)
    for argument, field in (
        (args.max_workers, "max_workers"),
        (args.max_sample_attempts, "max_sample_attempts"),
        (args.generation_batch_size, "generation_batch_size"),
        (args.judgment_batch_size, "judgment_batch_size"),
    ):
        if argument is not None:
            runtime_config[field] = argument
    for argument, field in (
        (args.target_count, "target_count"),
        (args.intent_change_per_pair, "intent_change_per_pair"),
    ):
        if argument is not None:
            config[field] = argument
    if int(config["target_count"]) <= 0:
        raise Top1DataError("target_count must be positive")
    if (
        int(runtime_config["max_workers"]) <= 0
        or int(runtime_config["max_sample_attempts"]) <= 0
        or int(runtime_config["generation_batch_size"]) <= 0
        or int(runtime_config["judgment_batch_size"]) <= 0
    ):
        raise Top1DataError("worker and attempt counts must be positive")

    candidate_names = load_candidate_names(candidate_path)
    descriptions = load_taxonomy_descriptions(taxonomy_path, candidate_names)
    taxonomy = taxonomy_prompt(descriptions)
    plans = build_dialogue_blueprints(
        candidate_names,
        target_count=int(config["target_count"]),
        intent_change_per_pair=int(config["intent_change_per_pair"]),
        seed=int(config["seed"]),
    )
    base_url, api_key = load_api_credentials(args.credentials_file)
    _prepare_run(
        output_directory=output_directory,
        config=config,
        candidate_path=candidate_path,
        taxonomy_path=taxonomy_path,
        endpoint=base_url,
        plans=plans,
        taxonomy=taxonomy,
    )
    print(f"[synthesis] planned {len(plans)} controlled dialogues: {output_directory}")
    if args.plan_only:
        print("[synthesis] plan-only mode completed; no model calls were made")
        return

    raw_directory = output_directory / "raw"
    attempt_path = output_directory / "attempts.jsonl"
    accepted_path = output_directory / "accepted_records.jsonl"
    directness_path = output_directory / "directness_records.jsonl"
    invalidation_path = output_directory / "invalidated_records.jsonl"
    train_path = output_directory / "train.jsonl"
    rejected_path = output_directory / "rejected.jsonl"
    summary_path = output_directory / "summary.json"
    accepted = _load_accepted(accepted_path, invalidation_path)
    attempt_counts = _load_attempt_counts(attempt_path)
    plan_by_id = {plan.scenario_id: plan for plan in plans}
    unknown_accepted = set(accepted) - set(plan_by_id)
    if unknown_accepted:
        raise Top1DataError("accepted records contain scenarios outside the immutable plan")
    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=float(config["request_timeout_seconds"]),
        request_attempts=int(config["request_attempts"]),
    )
    accepted = _audit_existing_directness(
        accepted=accepted,
        directness_path=directness_path,
        invalidation_path=invalidation_path,
        models=(str(config["reviewer_model"]), str(config["labeler_model"])),
        config=runtime_config,
        client=client,
        raw_directory=raw_directory,
    )
    accepted_conversations = {
        _canonical_messages(row["messages"]): scenario_id
        for scenario_id, row in accepted.items()
    }

    while True:
        pending = [
            plan
            for plan in plans
            if plan.scenario_id not in accepted
            and attempt_counts[plan.scenario_id]
            < int(runtime_config["max_sample_attempts"])
        ]
        if not pending:
            break
        attempt_numbers = {
            plan.scenario_id: attempt_counts[plan.scenario_id] + 1 for plan in pending
        }
        round_number = min(attempt_numbers.values())
        print(
            f"[synthesis] attempt round {round_number}: {len(pending)} pending, "
            f"{len(accepted)}/{len(plans)} accepted"
        )
        generated, generation_errors = _generate_attempt(
            blueprints=pending,
            attempt_numbers=attempt_numbers,
            taxonomy=taxonomy,
            candidate_names=candidate_names,
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        samples = list(generated.values())
        labeler, labeler_errors = _judge_attempt(
            stage="labeler",
            samples=samples,
            taxonomy=taxonomy,
            candidate_names=candidate_names,
            model=str(config["labeler_model"]),
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        reviewer, reviewer_errors = _judge_attempt(
            stage="reviewer",
            samples=samples,
            taxonomy=taxonomy,
            candidate_names=candidate_names,
            model=str(config["reviewer_model"]),
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        intent_change_samples = [
            sample
            for sample in samples
            if plan_by_id[str(sample["scenario_id"])].phenomenon == "intent_change"
        ]
        directness, directness_errors = _directness_attempt(
            samples=intent_change_samples,
            models=(str(config["reviewer_model"]), str(config["labeler_model"])),
            config=runtime_config,
            client=client,
            raw_directory=raw_directory,
        )
        _append_jsonl(
            directness_path,
            (
                {
                    "timestamp": _now(),
                    "audit_version": DIRECTNESS_AUDIT_VERSION,
                    "scenario_id": str(sample["scenario_id"]),
                    "attempt": int(sample["attempt"]),
                    "source": "generation_attempt",
                    "passed": (
                        not directness_errors[str(sample["scenario_id"])]
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "contains_only_new_request"
                        )
                        is True
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "references_previous_exchange"
                        )
                        is False
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "uses_transition_or_acknowledgment"
                        )
                        is False
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "direct_final_request"
                        )
                        is True
                        and directness.get(str(sample["scenario_id"]), {}).get(
                            "has_switch_meta_language"
                        )
                        is False
                    ),
                    "errors": directness_errors[str(sample["scenario_id"])],
                    "audit": directness.get(str(sample["scenario_id"])),
                }
                for sample in intent_change_samples
            ),
        )

        attempt_records: list[dict[str, Any]] = []
        newly_accepted: list[dict[str, Any]] = []
        round_conversations: dict[tuple[tuple[str, str], ...], str] = {}
        for blueprint in pending:
            scenario_id = blueprint.scenario_id
            reasons = list(generation_errors[scenario_id])
            sample = generated.get(scenario_id)
            labeler_judgment = labeler.get(scenario_id)
            reviewer_judgment = reviewer.get(scenario_id)
            directness_judgment = directness.get(scenario_id)
            if sample is not None:
                reasons.extend(labeler_errors[scenario_id])
                reasons.extend(reviewer_errors[scenario_id])
                if blueprint.phenomenon == "intent_change":
                    reasons.extend(directness_errors[scenario_id])
                reasons.extend(
                    acceptance_reasons(
                        blueprint,
                        labeler_judgment,
                        reviewer_judgment,
                        directness_judgment,
                    )
                )
                canonical = _canonical_messages(sample["messages"])
                if canonical in accepted_conversations or canonical in round_conversations:
                    reasons.append("duplicate_conversation")
            reasons = list(dict.fromkeys(reasons))
            status = "accepted" if not reasons and sample is not None else "rejected"
            attempt_record: dict[str, Any] = {
                "timestamp": _now(),
                "scenario_id": scenario_id,
                "attempt": attempt_numbers[scenario_id],
                "status": status,
                "reasons": reasons,
                "planned_target_candidate_name": blueprint.target_candidate_name,
                "planned_source_candidate_name": blueprint.source_candidate_name,
                "planned_phenomenon": blueprint.phenomenon,
                "labeler": labeler_judgment,
                "reviewer": reviewer_judgment,
                "directness": directness_judgment,
            }
            if sample is not None:
                attempt_record["messages"] = sample["messages"]
            attempt_records.append(attempt_record)
            attempt_counts[scenario_id] = attempt_numbers[scenario_id]
            if status != "accepted" or sample is None:
                continue
            row: dict[str, Any] = {
                "id": scenario_id,
                "dataset_version": SYNTHESIS_VERSION,
                "source_type": "llm_controlled_multiturn",
                "scenario_id": scenario_id,
                "conversation_phenomenon": blueprint.phenomenon,
                "messages": sample["messages"],
                "target_candidate_name": blueprint.target_candidate_name,
                "synthesis": {
                    "blueprint_seed": blueprint.seed,
                    "attempt": attempt_numbers[scenario_id],
                    "generator_model": config["generator_model"],
                    "labeler_model": config["labeler_model"],
                    "reviewer_model": config["reviewer_model"],
                    "labeler_predicted_candidate_name": labeler_judgment[
                        "predicted_candidate_name"
                    ],
                    "reviewer_predicted_candidate_name": reviewer_judgment[
                        "predicted_candidate_name"
                    ],
                    "directness_audit": directness_judgment,
                },
            }
            if blueprint.source_candidate_name is not None:
                row["source_candidate_name"] = blueprint.source_candidate_name
            newly_accepted.append(row)
            canonical = _canonical_messages(sample["messages"])
            round_conversations[canonical] = scenario_id

        _append_jsonl(attempt_path, attempt_records)
        _append_jsonl(accepted_path, newly_accepted)
        for row in newly_accepted:
            scenario_id = str(row["scenario_id"])
            accepted[scenario_id] = row
            accepted_conversations[_canonical_messages(row["messages"])] = scenario_id
        ordered_rows = [accepted[plan.scenario_id] for plan in plans if plan.scenario_id in accepted]
        validate_training_rows(ordered_rows, candidate_names, source=SYNTHESIS_VERSION)
        write_jsonl(train_path, ordered_rows)
        _write_summary(
            path=summary_path,
            plans=plans,
            accepted=accepted,
            attempt_path=attempt_path,
            raw_directory=raw_directory,
            train_path=train_path,
            complete=len(accepted) == len(plans),
        )
        print(
            f"[synthesis] round completed: +{len(newly_accepted)} accepted; "
            f"total {len(accepted)}/{len(plans)}"
        )

    unresolved = [plan for plan in plans if plan.scenario_id not in accepted]
    ordered_rows = [accepted[plan.scenario_id] for plan in plans if plan.scenario_id in accepted]
    validate_training_rows(ordered_rows, candidate_names, source=SYNTHESIS_VERSION)
    write_jsonl(train_path, ordered_rows)
    last_rejection: dict[str, dict[str, Any]] = {}
    if attempt_path.is_file():
        for row in read_jsonl(attempt_path):
            if row.get("status") != "accepted":
                last_rejection[str(row["scenario_id"])] = row
    write_jsonl(
        rejected_path,
        [last_rejection[plan.scenario_id] for plan in unresolved if plan.scenario_id in last_rejection],
    )
    _write_summary(
        path=summary_path,
        plans=plans,
        accepted=accepted,
        attempt_path=attempt_path,
        raw_directory=raw_directory,
        train_path=train_path,
        complete=not unresolved,
    )
    print(f"[synthesis] training data: {train_path}")
    print(f"[synthesis] quality summary: {summary_path}")
    if unresolved and not args.allow_partial:
        raise RuntimeError(
            f"quality gates exhausted for {len(unresolved)} scenarios; see {rejected_path}"
        )


if __name__ == "__main__":
    main()
