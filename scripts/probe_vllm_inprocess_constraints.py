#!/usr/bin/env python
"""Probe request-level constrained decoding inside the custom vLLM process.

Running without ``--model`` only inspects the installed vLLM API and constructs
``SamplingParams``.  It does not load a model or occupy an NPU::

    python scripts/probe_vllm_inprocess_constraints.py

Supplying ``--model`` starts the model with the same core sequence used by the
910B SkillHub integration (``from_engine_args`` -> ``load_model`` ->
``start_background_loop`` -> ``generate``), then tries to force output token
IDs ``[0, 1]`` with a real Python logits processor::

    python scripts/probe_vllm_inprocess_constraints.py --model /path/to/model

The processor appends one JSON record per invocation to a trace file.  This is
deliberate: it proves whether the callback ran even when a custom vLLM moves or
copies it into another process.  The script itself does not import ``torch``,
``torch_npu``, or any LLMGen repository module.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, time
import traceback
from typing import Any, Mapping, Sequence
import uuid


_MARKER = "[[LLMGEN-VLLM-INPROCESS-PROBE]]"
_IMPORTABLE_MODULE_NAME = "probe_vllm_inprocess_constraints"
_FORCED_SEQUENCE = (0, 1)

# A worker may need to import the processor by its module-qualified name.  A
# script normally runs as ``__main__``; this alias gives pickle/import-based
# custom workers a stable, importable module name as long as this file remains
# in the script directory (which Python automatically puts on sys.path).
_CURRENT_MODULE = sys.modules.get(__name__)
if _CURRENT_MODULE is not None:
    sys.modules.setdefault(_IMPORTABLE_MODULE_NAME, _CURRENT_MODULE)


def _emit(event: str, **fields: Any) -> None:
    rendered = " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
        for key, value in fields.items()
    )
    print(f"{_MARKER} event={event} {rendered}".rstrip(), flush=True)


def _type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def _source_file(value: Any) -> str | None:
    try:
        return inspect.getsourcefile(value) or inspect.getfile(value)
    except (TypeError, OSError):
        return None


def _error_details(exc: BaseException) -> dict[str, str]:
    return {
        "type": _type_name(exc),
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _callable_accepts_keyword(callable_obj: Any, keyword: str) -> bool | None:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return True
    return keyword in signature.parameters


def _filter_callable_kwargs(
    callable_obj: Any, kwargs: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], bool]:
    """Return accepted kwargs, skipped names, and signature availability."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs), [], False
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs), [], True
    supported = set(signature.parameters)
    accepted = {key: value for key, value in kwargs.items() if key in supported}
    skipped = sorted(set(kwargs).difference(accepted))
    return accepted, skipped, True


class ForcedSequenceLogitsProcessor:
    """Force a deterministic token sequence and leave cross-process evidence."""

    def __init__(
        self,
        sequence: Sequence[int] = _FORCED_SEQUENCE,
        trace_file: str | None = None,
    ) -> None:
        normalized = tuple(int(token_id) for token_id in sequence)
        if not normalized or any(token_id < 0 for token_id in normalized):
            raise ValueError("sequence must contain non-negative token IDs")
        self.sequence = normalized
        self.trace_file = trace_file

    def clone(self) -> "ForcedSequenceLogitsProcessor":
        return self

    def __call__(self, *args: Any) -> Any:
        # Support the two processor ABIs seen in vLLM variants:
        #   (output_token_ids, scores)
        #   (prompt_token_ids, output_token_ids, scores)
        if len(args) not in {2, 3}:
            raise TypeError(
                "expected (output_ids, scores) or "
                "(prompt_ids, output_ids, scores)"
            )
        output_token_ids = args[-2]
        scores = args[-1]
        position = len(output_token_ids)
        forced_token_id = (
            self.sequence[position]
            if position < len(self.sequence)
            else self.sequence[-1]
        )
        vocabulary_size = int(scores.shape[-1])
        if forced_token_id >= vocabulary_size:
            raise RuntimeError(
                f"forced token {forced_token_id} is outside vocabulary "
                f"size {vocabulary_size}"
            )

        self._write_trace(
            {
                "timestamp": time(),
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "arg_count": len(args),
                "generated_count": position,
                "forced_token_id": forced_token_id,
                "score_shape": list(scores.shape),
            }
        )
        masked = scores.new_full(scores.shape, -float("inf"))
        # Ellipsis works for both a 1-D score vector and a batched final axis.
        masked[..., forced_token_id] = scores[..., forced_token_id]
        return masked

    def _write_trace(self, record: Mapping[str, Any]) -> None:
        if not self.trace_file:
            return
        payload = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            self.trace_file,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)


# Keep the qualified name stable when this file was launched as a script.
ForcedSequenceLogitsProcessor.__module__ = _IMPORTABLE_MODULE_NAME


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Model directory. Omit it for the zero-NPU inspection phase.",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer directory (default: --model).",
    )
    parser.add_argument(
        "--prompt",
        default="约束解码进程内探针。",
    )
    parser.add_argument(
        "--force-token-ids",
        default="0,1",
        help="Comma-separated token sequence to force (default: 0,1).",
    )
    parser.add_argument(
        "--trace-file",
        help="Append-only processor trace (default: unique file under /tmp).",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--health-timeout", type=float, default=1000.0)
    parser.add_argument("--health-interval", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decode-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--scheduler-budget-len", type=int, default=10240)
    parser.add_argument("--first-token-timeout", type=float, default=1000.0)
    parser.add_argument("--max-log-len", type=int, default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--swap-space", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--architectures")
    parser.add_argument("--model-vision")
    parser.add_argument(
        "--engine-role",
        choices=("m", "omit"),
        default="m",
        help=(
            "Use EngineRole.M like service_910b/SkillHub, or omit the argument "
            "to resemble the standalone api_server (default: m)."
        ),
    )
    parser.add_argument(
        "--engine-kwargs-json",
        default="{}",
        help="Additional/overriding AsyncEngineArgs JSON object.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-log-requests",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _parse_forced_sequence(raw_value: str) -> tuple[int, ...]:
    try:
        sequence = tuple(
            int(part.strip())
            for part in raw_value.split(",")
            if part.strip()
        )
    except ValueError as exc:
        raise ValueError("--force-token-ids must contain integers") from exc
    if not sequence or any(token_id < 0 for token_id in sequence):
        raise ValueError("--force-token-ids must contain non-negative integers")
    return sequence


def _parse_json_object(raw_value: str, option_name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{option_name} must be a JSON object")
    return value


def _runtime_environment() -> dict[str, Any]:
    relevant_names = sorted(
        name
        for name in os.environ
        if name.startswith(
            (
                "ASCEND_",
                "NPU_",
                "RAY_",
                "VLLM_",
                "HCCL_",
                "LCAL_",
                "LCCL_",
            )
        )
        or name
        in {
            "LOG_LEVEL",
            "LONG_CONTEXT",
            "USE_YR",
            "tensor_parallel_size",
            "LLM_TENSOR_PARALLEL_SIZE",
        }
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "multiprocessing_start_method": multiprocessing.get_start_method(
            allow_none=True
        ),
        "script": str(Path(__file__).resolve()),
        "sys_path_head": sys.path[:8],
        "environment": {name: os.environ[name] for name in relevant_names},
    }


def _inspect_vllm(
    *, sequence: Sequence[int], trace_file: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = perf_counter()
    import vllm
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    try:
        from vllm.global_consts import EngineRole
    except (ImportError, AttributeError) as exc:
        EngineRole = None  # type: ignore[assignment,misc]
        engine_role_error = _error_details(exc)
    else:
        engine_role_error = None

    processor = ForcedSequenceLogitsProcessor(sequence, trace_file)
    construction: dict[str, Any] = {}
    base_sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": len(sequence),
        "min_tokens": len(sequence),
        "detokenize": False,
        "skip_special_tokens": False,
    }
    for name, extra_kwargs in (
        ("logits_processors", {"logits_processors": [processor]}),
        ("allowed_token_ids", {"allowed_token_ids": list(sequence)}),
    ):
        kwargs = {**base_sampling_kwargs, **extra_kwargs}
        try:
            instance = SamplingParams(**kwargs)
        except BaseException as exc:
            construction[name] = {
                "constructed": False,
                "error": _error_details(exc),
            }
        else:
            stored_value = getattr(instance, name, None)
            construction[name] = {
                "constructed": True,
                "stored_attribute_present": hasattr(instance, name),
                "stored_type": (
                    _type_name(stored_value) if stored_value is not None else None
                ),
                "stored_count": (
                    len(stored_value)
                    if isinstance(stored_value, Sequence)
                    and not isinstance(stored_value, (str, bytes, bytearray))
                    else None
                ),
                "repr": repr(instance)[:2000],
            }

    report = {
        "elapsed_ms": (perf_counter() - started) * 1000.0,
        "vllm_version": getattr(vllm, "__version__", None),
        "vllm_file": getattr(vllm, "__file__", None),
        "async_engine_args": {
            "type": _type_name(AsyncEngineArgs),
            "signature": _safe_signature(AsyncEngineArgs),
            "source_file": _source_file(AsyncEngineArgs),
        },
        "async_llm_engine": {
            "type": _type_name(AsyncLLMEngine),
            "source_file": _source_file(AsyncLLMEngine),
            "from_engine_args_signature": _safe_signature(
                AsyncLLMEngine.from_engine_args
            ),
            "generate_signature": _safe_signature(
                getattr(AsyncLLMEngine, "generate", None)
            ),
        },
        "sampling_params": {
            "type": _type_name(SamplingParams),
            "signature": _safe_signature(SamplingParams),
            "source_file": _source_file(SamplingParams),
            "accepts_logits_processors": _callable_accepts_keyword(
                SamplingParams, "logits_processors"
            ),
            "accepts_allowed_token_ids": _callable_accepts_keyword(
                SamplingParams, "allowed_token_ids"
            ),
            "construction": construction,
        },
        "engine_role": {
            "available": EngineRole is not None,
            "m_repr": repr(getattr(EngineRole, "M", None)),
            "error": engine_role_error,
        },
        "processor": {
            "type": _type_name(processor),
            "module": type(processor).__module__,
            "module_importable": _IMPORTABLE_MODULE_NAME in sys.modules,
            "sequence": list(sequence),
            "trace_file": trace_file,
        },
    }
    symbols = {
        "AsyncEngineArgs": AsyncEngineArgs,
        "AsyncLLMEngine": AsyncLLMEngine,
        "SamplingParams": SamplingParams,
        "EngineRole": EngineRole,
    }
    return report, symbols


def _build_engine_kwargs(
    args: argparse.Namespace,
    *,
    engine_role: Any,
) -> dict[str, Any]:
    tokenizer_path = args.tokenizer or args.model
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "tokenizer_mode": "auto",
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "decode_tensor_parallel_size": args.decode_tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "context_parallel_size": 1,
        "decode_pipeline_parallel_size": 1,
        "decode_data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": False,
        "decode_enable_expert_parallel": False,
        "max_num_seqs": args.max_num_seqs,
        "scheduler_budget_len": args.scheduler_budget_len,
        "first_token_timeout": args.first_token_timeout,
        "max_log_len": args.max_log_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_requests": bool(args.disable_log_requests),
        "swap_space": args.swap_space,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.architectures:
        kwargs["architectures"] = args.architectures
    if args.model_vision:
        kwargs["model_vision"] = args.model_vision
    if args.engine_role == "m":
        if engine_role is None:
            raise RuntimeError(
                "--engine-role=m requested but vllm.global_consts.EngineRole.M "
                "is unavailable; retry with --engine-role omit"
            )
        kwargs["engine_role"] = engine_role
    kwargs.update(
        _parse_json_object(args.engine_kwargs_json, "--engine-kwargs-json")
    )
    return kwargs


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _start_engine(
    engine: Any,
    *,
    health_timeout: float,
    health_interval: float,
    timings: dict[str, float],
) -> None:
    startup_started = perf_counter()
    load_model = getattr(getattr(engine, "engine", None), "load_model", None)
    if callable(load_model):
        started = perf_counter()
        _emit("engine.load_model.begin")
        await _maybe_await(load_model())
        timings["load_model_ms"] = (perf_counter() - started) * 1000.0
        _emit("engine.load_model.complete", elapsed_ms=timings["load_model_ms"])
    else:
        timings["load_model_ms"] = 0.0
        _emit("engine.load_model.skipped", reason="not_callable")

    start_background_loop = getattr(engine, "start_background_loop", None)
    if callable(start_background_loop):
        started = perf_counter()
        _emit("engine.background_loop.begin")
        await _maybe_await(start_background_loop())
        timings["background_loop_ms"] = (perf_counter() - started) * 1000.0
        _emit(
            "engine.background_loop.complete",
            elapsed_ms=timings["background_loop_ms"],
        )
    else:
        timings["background_loop_ms"] = 0.0
        _emit("engine.background_loop.skipped", reason="not_callable")

    is_health = getattr(engine, "is_health", None)
    if callable(is_health):
        deadline = perf_counter() + health_timeout
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError("vLLM health check timed out")
            healthy = await asyncio.wait_for(
                _maybe_await(is_health()), timeout=remaining
            )
            _emit("engine.health", attempt=attempt, healthy=bool(healthy))
            if healthy:
                break
            await asyncio.sleep(min(max(health_interval, 0.1), remaining))
    else:
        _emit("engine.health.skipped", reason="not_callable")
    timings["startup_ms"] = (perf_counter() - startup_started) * 1000.0


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer_list(value: Any) -> list[int] | None:
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            value = value.tolist()
        except Exception:
            return None
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return [int(item) for item in value]
    return None


def _completion_evidence(request_output: Any) -> dict[str, Any]:
    outputs = _field(request_output, "outputs")
    first_output = None
    if isinstance(outputs, Sequence) and not isinstance(
        outputs, (str, bytes, bytearray)
    ) and outputs:
        first_output = outputs[0]
    candidates = [first_output, request_output]
    token_ids = None
    text = None
    finish_reason = None
    for candidate in candidates:
        if candidate is None:
            continue
        for name in (
            "token_ids",
            "output_tokens",
            "output_token_ids",
            "completion_token_ids",
        ):
            token_ids = _integer_list(_field(candidate, name))
            if token_ids is not None:
                break
        if text is None:
            value = _field(candidate, "text")
            if isinstance(value, str):
                text = value
        if finish_reason is None:
            finish_reason = _field(candidate, "finish_reason")
        if token_ids is not None:
            break
    return {
        "token_ids": token_ids,
        "text": text,
        "finish_reason": finish_reason,
        "request_output_type": _type_name(request_output),
        "first_output_type": (
            _type_name(first_output) if first_output is not None else None
        ),
    }


async def _consume_generation(stream: Any) -> tuple[Any, int]:
    if inspect.isawaitable(stream):
        stream = await stream
    if not hasattr(stream, "__aiter__"):
        return stream, 1
    final_output = None
    frame_count = 0
    async for request_output in stream:
        final_output = request_output
        frame_count += 1
        _emit(
            "engine.generate.frame",
            frame=frame_count,
            evidence=_completion_evidence(request_output),
        )
    if final_output is None:
        raise RuntimeError("engine.generate returned no RequestOutput frames")
    return final_output, frame_count


async def _abort_request(engine: Any, request_id: str) -> dict[str, Any]:
    nested = getattr(engine, "engine", None)
    for method_name, method in (
        ("engine.abort", getattr(engine, "abort", None)),
        ("engine.abort_request", getattr(engine, "abort_request", None)),
        ("engine.engine.abort", getattr(nested, "abort", None)),
        ("engine.engine.abort_request", getattr(nested, "abort_request", None)),
    ):
        if not callable(method):
            continue
        try:
            await _maybe_await(method(request_id))
        except BaseException as exc:
            return {
                "attempted": True,
                "method": method_name,
                "acknowledged": False,
                "error": _error_details(exc),
            }
        return {
            "attempted": True,
            "method": method_name,
            "acknowledged": True,
        }
    return {"attempted": False, "reason": "no_request_abort_method"}


async def _shutdown_engine(engine: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    nested = getattr(engine, "llm_engine", None)
    for name, method in (
        ("shutdown_background_loop", getattr(engine, "shutdown_background_loop", None)),
        ("llm_engine.shutdown", getattr(nested, "shutdown", None)),
    ):
        if not callable(method):
            results.append({"method": name, "called": False})
            continue
        try:
            await _maybe_await(method())
        except BaseException as exc:
            results.append(
                {"method": name, "called": True, "error": _error_details(exc)}
            )
        else:
            results.append({"method": name, "called": True, "succeeded": True})
    return results


def _read_trace_records(trace_file: str, start_offset: int) -> list[Any]:
    path = Path(trace_file)
    if not path.is_file():
        return []
    records: list[Any] = []
    with path.open("rb") as stream:
        stream.seek(start_offset)
        for raw_line in stream:
            try:
                records.append(json.loads(raw_line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                records.append(
                    {
                        "invalid_trace_line": raw_line.decode(
                            "utf-8", errors="replace"
                        ),
                        "error": str(exc),
                    }
                )
    return records


async def _run_model_probe(
    args: argparse.Namespace,
    *,
    sequence: tuple[int, ...],
    trace_file: str,
    trace_start_offset: int,
    symbols: Mapping[str, Any],
) -> dict[str, Any]:
    AsyncEngineArgs = symbols["AsyncEngineArgs"]
    AsyncLLMEngine = symbols["AsyncLLMEngine"]
    SamplingParams = symbols["SamplingParams"]
    EngineRole = symbols["EngineRole"]
    engine_role_m = getattr(EngineRole, "M", None) if EngineRole is not None else None

    report: dict[str, Any] = {
        "attempted": True,
        "completed": False,
        "forced_sequence": list(sequence),
        "trace_file": trace_file,
    }
    engine = None
    request_id = f"llmgen-constraint-probe-{uuid.uuid4().hex}"
    try:
        raw_engine_kwargs = _build_engine_kwargs(
            args,
            engine_role=engine_role_m,
        )
        engine_kwargs, skipped, signature_available = _filter_callable_kwargs(
            AsyncEngineArgs, raw_engine_kwargs
        )
        report["engine_args"] = {
            "raw": {key: repr(value) for key, value in raw_engine_kwargs.items()},
            "accepted_keys": sorted(engine_kwargs),
            "skipped_keys": skipped,
            "signature_available": signature_available,
        }
        _emit(
            "engine.construct.begin",
            accepted_keys=sorted(engine_kwargs),
            skipped_keys=skipped,
        )
        started = perf_counter()
        engine_args = AsyncEngineArgs(**engine_kwargs)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        report["timings"] = {
            "construct_engine_ms": (perf_counter() - started) * 1000.0
        }
        _emit(
            "engine.construct.complete",
            elapsed_ms=report["timings"]["construct_engine_ms"],
            engine_type=_type_name(engine),
        )
        await _start_engine(
            engine,
            health_timeout=args.health_timeout,
            health_interval=args.health_interval,
            timings=report["timings"],
        )

        from transformers import AutoTokenizer

        tokenizer_started = perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or args.model,
            trust_remote_code=bool(args.trust_remote_code),
        )
        prompt_token_ids = tokenizer.encode(
            args.prompt,
            add_special_tokens=False,
        )
        report["timings"]["tokenizer_ms"] = (
            perf_counter() - tokenizer_started
        ) * 1000.0
        report["prompt"] = {
            "text": args.prompt,
            "token_ids": list(prompt_token_ids),
            "token_count": len(prompt_token_ids),
        }

        processor = ForcedSequenceLogitsProcessor(sequence, trace_file)
        sampling_kwargs = {
            "temperature": 0.0,
            "max_tokens": len(sequence),
            "min_tokens": len(sequence),
            "detokenize": False,
            "skip_special_tokens": False,
            "logits_processors": [processor],
        }
        sampling_params = SamplingParams(**sampling_kwargs)
        report["sampling_params"] = {
            "kwargs": {
                **sampling_kwargs,
                "logits_processors": [_type_name(processor)],
            },
            "stored_processor_count": len(
                getattr(sampling_params, "logits_processors", ()) or ()
            ),
            "repr": repr(sampling_params)[:2000],
        }

        raw_generate_kwargs = {
            "prompt": args.prompt,
            "sampling_params": sampling_params,
            "request_id": request_id,
            "prompt_token_ids": list(prompt_token_ids),
            "tag": None,
            "arrival_time": None,
            "multi_modal_data": None,
            "scheduler_result": None,
            "is_stream": False,
        }
        generate_kwargs, generate_skipped, generate_signature_available = (
            _filter_callable_kwargs(engine.generate, raw_generate_kwargs)
        )
        report["generate_call"] = {
            "signature": _safe_signature(engine.generate),
            "passed_keys": sorted(generate_kwargs),
            "skipped_keys": generate_skipped,
            "signature_available": generate_signature_available,
            "request_id": request_id,
        }
        _emit(
            "engine.generate.begin",
            request_id=request_id,
            passed_keys=sorted(generate_kwargs),
            skipped_keys=generate_skipped,
        )
        generation_started = perf_counter()
        try:
            final_output, frame_count = await asyncio.wait_for(
                _consume_generation(engine.generate(**generate_kwargs)),
                timeout=args.request_timeout,
            )
        except asyncio.TimeoutError as exc:
            report["abort"] = await _abort_request(engine, request_id)
            raise TimeoutError(
                f"generation exceeded {args.request_timeout} seconds"
            ) from exc
        report["timings"]["generation_ms"] = (
            perf_counter() - generation_started
        ) * 1000.0
        evidence = _completion_evidence(final_output)
        records = _read_trace_records(trace_file, trace_start_offset)
        report["output"] = evidence
        report["frame_count"] = frame_count
        report["processor_trace_records"] = records
        report["processor_callback_executed"] = bool(records)
        report["constraint_enforced"] = evidence["token_ids"] == list(sequence)
        report["completed"] = True
        _emit(
            "engine.generate.complete",
            elapsed_ms=report["timings"]["generation_ms"],
            output_token_ids=evidence["token_ids"],
            trace_records=len(records),
            constraint_enforced=report["constraint_enforced"],
        )
    except BaseException as exc:
        report["error"] = _error_details(exc)
        report["processor_trace_records"] = _read_trace_records(
            trace_file, trace_start_offset
        )
        report["processor_callback_executed"] = bool(
            report["processor_trace_records"]
        )
        _emit(
            "model_probe.failed",
            error_type=_type_name(exc),
            message=str(exc),
            processor_callback_executed=report["processor_callback_executed"],
        )
    finally:
        if engine is not None:
            report["shutdown"] = await _shutdown_engine(engine)
    return report


def run(argv: Sequence[str] | None = None) -> tuple[dict[str, Any], int]:
    args = _parse_args(argv)
    sequence = _parse_forced_sequence(args.force_token_ids)
    trace_file = args.trace_file or (
        f"/tmp/llmgen-vllm-constraint-probe-{os.getpid()}-"
        f"{uuid.uuid4().hex[:8]}.jsonl"
    )
    trace_path = Path(trace_file).expanduser().resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_start_offset = trace_path.stat().st_size if trace_path.is_file() else 0

    report: dict[str, Any] = {
        "probe_version": 1,
        "mode": "model" if args.model else "inspect_only",
        "runtime": _runtime_environment(),
    }
    _emit("probe.begin", mode=report["mode"], model=args.model)
    try:
        inspection, symbols = _inspect_vllm(
            sequence=sequence,
            trace_file=str(trace_path),
        )
    except BaseException as exc:
        report["inspection_error"] = _error_details(exc)
        _emit("inspection.failed", error_type=_type_name(exc), message=str(exc))
        return report, 1
    report["inspection"] = inspection
    _emit(
        "inspection.complete",
        vllm_version=inspection["vllm_version"],
        logits_processors=(
            inspection["sampling_params"]["construction"]["logits_processors"][
                "constructed"
            ]
        ),
        allowed_token_ids=(
            inspection["sampling_params"]["construction"]["allowed_token_ids"][
                "constructed"
            ]
        ),
    )

    if args.model:
        report["model_probe"] = asyncio.run(
            _run_model_probe(
                args,
                sequence=sequence,
                trace_file=str(trace_path),
                trace_start_offset=trace_start_offset,
                symbols=symbols,
            )
        )
    else:
        report["next_step"] = (
            "Re-run with --model /path/to/model to execute the processor during "
            "real generation."
        )
    _emit("probe.complete", mode=report["mode"])
    return report, 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report, exit_code = run(argv)
    except Exception as exc:
        report = {
            "probe_version": 1,
            "fatal_error": _error_details(exc),
        }
        exit_code = 1
        _emit("probe.fatal", error_type=_type_name(exc), message=str(exc))
    print(
        f"{_MARKER} FINAL_JSON_BEGIN\n"
        f"{json.dumps(report, ensure_ascii=False, indent=2, default=str)}\n"
        f"{_MARKER} FINAL_JSON_END",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
