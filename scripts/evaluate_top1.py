#!/usr/bin/env python3
"""Evaluate one immutable Top1 model artifact as an independent run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

from llmgen.evaluation import SCORE_MODES, aggregate_predictions, prediction_from_scores
from llmgen.experiment import (
    EVALUATION_RUN_SCHEMA_VERSION,
    RunStore,
    append_jsonl,
    canonical_sha256,
    compact_utc_now,
    git_snapshot,
    load_and_verify_model_artifact,
    system_snapshot,
    utc_now,
)
from llmgen.top1 import (
    CONVERSATION_TEMPLATE,
    ROUTING_MODE,
    Top1DataError,
    candidate_token_sequences,
    encode_text,
    fit_prompt,
    load_candidate_names,
    messages_from_row,
    read_jsonl,
    sha256_file,
    target_candidate_name,
    write_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score closed-set candidate paths and record one Top1 evaluation run."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation-root", default="runs/evaluations/top1")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--suite-id")
    parser.add_argument("--candidate-registry")
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--score-mode", choices=SCORE_MODES, default="sum_logprob")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--history-ablation", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-model-verification", action="store_true")
    return parser.parse_args(argv)


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


def _import_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - GPU evaluation environment
        raise SystemExit(
            "Top1 evaluation requires torch and transformers; install -e '.[train]'"
        ) from exc
    return torch, transformers


def _load_model(
    *,
    model_dir: Path,
    transformers: Any,
    dtype: Any,
    trust_remote_code: bool,
) -> Any:
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
    }
    if (model_dir / "adapter_config.json").is_file():
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:  # pragma: no cover - LoRA evaluation environment
            raise SystemExit("LoRA evaluation requires peft") from exc
        return AutoPeftModelForCausalLM.from_pretrained(
            str(model_dir),
            **model_kwargs,
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
    max_target_tokens = max(len(tokens) + 1 for tokens in candidate_tokens.values())
    if max_length <= max_target_tokens:
        raise Top1DataError("max_length leaves no room for an evaluation prompt")
    prepared = []
    for row_index, row in enumerate(rows):
        try:
            messages = messages_from_row(row)
            target = None
            if "target_candidate_name" in row:
                target = target_candidate_name(row)
                if target not in legal_names:
                    raise Top1DataError(f"unknown target candidate name: {target!r}")
            prompt, fitted = fit_prompt(
                tokenizer,
                messages,
                system_prompt,
                max_prompt_tokens=max_length - max_target_tokens,
            )
            prompt_ids = encode_text(tokenizer, prompt)
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
            }
            ablated_ids = None
            if history_ablation and len(messages) > 1:
                ablated_prompt, _ = fit_prompt(
                    tokenizer,
                    (messages[-1],),
                    system_prompt,
                    max_prompt_tokens=max_length - max_target_tokens,
                )
                ablated_ids = encode_text(tokenizer, ablated_prompt)
        except Top1DataError as exc:
            raise Top1DataError(f"evaluation row {row_index + 1}: {exc}") from exc
        prepared.append(
            {
                "row_index": row_index,
                "target_candidate_name": target,
                "prompt_ids": prompt_ids,
                "history_ablation_prompt_ids": ablated_ids,
                "diagnostics": diagnostics,
            }
        )
    return prepared


def _score_prompt_batch(
    prompt_items: Sequence[tuple[int, str, Sequence[int]]],
    *,
    model: Any,
    tokenizer: Any,
    candidate_tokens: Mapping[str, Sequence[int]],
    torch: Any,
    device: Any,
) -> list[tuple[int, str, dict[str, float | int]]]:
    sequences = []
    path_ids = []
    prompt_lengths = []
    for _, candidate, prompt_ids in prompt_items:
        path = [*map(int, candidate_tokens[candidate]), int(tokenizer.eos_token_id)]
        sequences.append([*map(int, prompt_ids), *path])
        path_ids.append(path)
        prompt_lengths.append(len(prompt_ids))
    max_length = max(map(len, sequences))
    input_ids = [
        [*sequence, *([int(tokenizer.pad_token_id)] * (max_length - len(sequence)))]
        for sequence in sequences
    ]
    attention_mask = [
        [1] * len(sequence) + [0] * (max_length - len(sequence))
        for sequence in sequences
    ]
    with torch.inference_mode():
        outputs = model(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long, device=device),
        )
        logits = outputs.logits.float()
        results = []
        for index, (row_index, candidate, _) in enumerate(prompt_items):
            start = prompt_lengths[index] - 1
            path = path_ids[index]
            token_logits = logits[index, start : start + len(path), :]
            token_logprob = torch.log_softmax(token_logits, dim=-1).gather(
                1,
                torch.tensor(path, dtype=torch.long, device=device).unsqueeze(1),
            )
            total = float(token_logprob.sum().item())
            results.append(
                (
                    row_index,
                    candidate,
                    {
                        "sum_logprob": total,
                        "mean_logprob": total / len(path),
                        "path_tokens": len(path),
                    },
                )
            )
    return results


def _score_prepared(
    prepared: Sequence[Mapping[str, Any]],
    *,
    prompt_key: str,
    model: Any,
    tokenizer: Any,
    candidate_names: Sequence[str],
    candidate_tokens: Mapping[str, Sequence[int]],
    torch: Any,
    device: Any,
    batch_size: int,
) -> dict[int, dict[str, dict[str, float | int]]]:
    tasks = [
        (int(row["row_index"]), candidate, row[prompt_key])
        for row in prepared
        if row[prompt_key] is not None
        for candidate in candidate_names
    ]
    grouped: dict[int, dict[str, dict[str, float | int]]] = defaultdict(dict)
    for start in range(0, len(tasks), batch_size):
        for row_index, candidate, score in _score_prompt_batch(
            tasks[start : start + batch_size],
            model=model,
            tokenizer=tokenizer,
            candidate_tokens=candidate_tokens,
            torch=torch,
            device=device,
        ):
            grouped[row_index][candidate] = score
    return dict(grouped)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_length <= 0 or args.batch_size <= 0:
        raise Top1DataError("max_length and batch_size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise Top1DataError("max_rows must be positive when specified")
    model_dir = Path(args.model_dir).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise Top1DataError(f"evaluation data does not exist: {data_path}")
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
    model_artifact = load_and_verify_model_artifact(
        model_dir,
        verify_files=not args.skip_model_verification,
    )
    semantic_config = {
        "model_id": model_artifact["model_id"],
        "dataset_sha256": sha256_file(data_path),
        "candidate_registry_sha256": sha256_file(registry_path),
        "system_prompt_sha256": sha256_file(prompt_path),
        "routing_mode": ROUTING_MODE,
        "conversation_template": CONVERSATION_TEMPLATE,
        "max_length": args.max_length,
        "max_rows": args.max_rows,
        "score_mode": args.score_mode,
        "history_ablation": args.history_ablation,
    }
    evaluation_signature = canonical_sha256(semantic_config)
    evaluation_id = _safe_component(
        args.evaluation_id or f"{compact_utc_now()}-{evaluation_signature[:8]}",
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
            "verified": not args.skip_model_verification,
            "training_run_id": model_artifact.get("training_run_id"),
        },
        "dataset": {
            "path": str(data_path),
            "sha256": semantic_config["dataset_sha256"],
        },
        "semantic_inference": semantic_config,
        "execution": {
            "batch_size": args.batch_size,
            "precision": args.precision,
            "device": args.device,
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
        torch, transformers = _import_dependencies()
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
        model = _load_model(
            model_dir=model_dir,
            transformers=transformers,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        model.eval()
        rows = read_jsonl(data_path)
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        if not rows:
            raise Top1DataError("evaluation dataset is empty")
        candidate_tokens = candidate_token_sequences(tokenizer, candidate_names)
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
        rows_per_chunk = max(1, args.batch_size // len(candidate_names))
        for start in range(0, len(prepared), rows_per_chunk):
            chunk = prepared[start : start + rows_per_chunk]
            scores = _score_prepared(
                chunk,
                prompt_key="prompt_ids",
                model=model,
                tokenizer=tokenizer,
                candidate_names=candidate_names,
                candidate_tokens=candidate_tokens,
                torch=torch,
                device=device,
                batch_size=args.batch_size,
            )
            ablation_scores = (
                _score_prepared(
                    chunk,
                    prompt_key="history_ablation_prompt_ids",
                    model=model,
                    tokenizer=tokenizer,
                    candidate_names=candidate_names,
                    candidate_tokens=candidate_tokens,
                    torch=torch,
                    device=device,
                    batch_size=args.batch_size,
                )
                if args.history_ablation
                else {}
            )
            for row in chunk:
                row_index = int(row["row_index"])
                record = prediction_from_scores(
                    row_index=row_index,
                    candidate_names=candidate_names,
                    scores=scores[row_index],
                    score_mode=args.score_mode,
                    target_candidate_name=row["target_candidate_name"],
                    diagnostics=row["diagnostics"],
                    history_ablation_scores=ablation_scores.get(row_index),
                )
                append_jsonl(prediction_path, record)
                predictions.append(record)
            completed = len(predictions)
            if completed == len(prepared) or completed % 100 < len(chunk):
                store.update_status("RUNNING", rows_completed=completed)
                store.event(
                    "evaluation_progress",
                    rows_completed=completed,
                    rows_total=len(prepared),
                )
        metrics = aggregate_predictions(predictions, candidate_names)
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "confusion_matrix.json", metrics["confusion_matrix"])
        summary = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "evaluation_signature": evaluation_signature,
            "model_id": model_artifact["model_id"],
            "rows": metrics["rows"],
            "top1_accuracy": metrics["top1_accuracy"],
            "macro_recall_observed_candidates": metrics[
                "macro_recall_observed_candidates"
            ],
            "expected_calibration_error": metrics["calibration"][
                "expected_calibration_error"
            ],
            "resolved_precision": resolved_precision,
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
            "score_mode": args.score_mode,
            "rows": metrics["rows"],
            "top1_accuracy": metrics["top1_accuracy"],
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
                "score_mode": args.score_mode,
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
