#!/usr/bin/env python
"""Minimal HTTP benchmark for a running vLLM ``/generate`` endpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import sys
from time import perf_counter
from typing import Any, Mapping, NamedTuple, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_ROUTER_PROMPT = (
    "System: Select every Agent Skill needed for the user request in execution "
    "order. Output one hierarchical skill code per line, with no other text.\n"
    "User: 帮我查询北京明天的天气，并规划从公司到机场的路线。\n"
    "Assistant:"
)
_PROMPT_PADDING = " 查询天气地图导航日程搜索文件处理"


class BenchmarkError(RuntimeError):
    """Raised when a benchmark request fails."""


class StreamObservation(NamedTuple):
    """Timing and token-count evidence collected from one streamed response."""

    response_bytes: int
    first_token_at: float | None
    last_token_at: float | None
    completion_tokens: int | None
    token_count_source: str | None
    generated_updates: int


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:18000/generate",
        help="vLLM generation endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--input-chars",
        type=int,
        default=256,
        help="Exact prompt character count used for every request (default: 256).",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=8,
        help="Requested and minimum generated token count (default: 8).",
    )
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def _build_prompt(input_chars: int) -> str:
    if input_chars < 1:
        raise BenchmarkError("input-chars must be positive")
    if input_chars <= len(_ROUTER_PROMPT):
        return _ROUTER_PROMPT[:input_chars]
    remaining = input_chars - len(_ROUTER_PROMPT)
    repeats = (remaining + len(_PROMPT_PADDING) - 1) // len(_PROMPT_PADDING)
    return (_ROUTER_PROMPT + _PROMPT_PADDING * repeats)[:input_chars]


def _post_generate(
    *,
    url: str,
    prompt: str,
    output_tokens: int,
    timeout: float,
) -> tuple[float, int, float, float, float, int, str]:
    payload = {
        "prompt": prompt,
        "stream": True,
        "temperature": 0.0,
        "max_tokens": output_tokens,
        # Prevent early EOS so every request performs comparable decode work.
        "min_tokens": output_tokens,
        # Keep special router tokens visible so text-based TTFT detection does
        # not silently skip the first generated token.
        "skip_special_tokens": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            observation = _consume_stream(
                response, prompt=prompt, started=started
            )
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(
            f"vLLM returned HTTP {exc.code}: {details[:1000]}"
        ) from exc
    except URLError as exc:
        raise BenchmarkError(f"cannot reach {url}: {exc}") from exc
    finished = perf_counter()
    if observation.first_token_at is None:
        raise BenchmarkError("vLLM stream contained no generated token")
    if observation.last_token_at is None:
        raise BenchmarkError("vLLM stream contained no final generated token")

    if observation.completion_tokens is None:
        # min_tokens == max_tokens makes this an exact count for a conforming
        # vLLM server. Keep the source explicit instead of presenting it as a
        # count observed in the response.
        completion_tokens = output_tokens
        token_count_source = "requested_min_equals_max"
    else:
        completion_tokens = observation.completion_tokens
        token_count_source = observation.token_count_source or "response"
        if completion_tokens != output_tokens:
            raise BenchmarkError(
                "vLLM generated an unexpected number of completion tokens: "
                f"requested={output_tokens}, observed={completion_tokens}, "
                f"source={token_count_source}"
            )

    latency_ms = (finished - started) * 1000.0
    ttft_ms = (observation.first_token_at - started) * 1000.0
    tpot_ms = (
        (observation.last_token_at - observation.first_token_at)
        * 1000.0
        / (completion_tokens - 1)
        if completion_tokens > 1
        else 0.0
    )
    response_tail_ms = max(0.0, (finished - observation.last_token_at) * 1000.0)
    return (
        latency_ms,
        observation.response_bytes,
        ttft_ms,
        tpot_ms,
        response_tail_ms,
        completion_tokens,
        token_count_source,
    )


def _parse_stream_payload(frame: bytes) -> Mapping[str, Any] | None:
    text = frame.decode("utf-8", errors="replace").strip()
    if text.startswith("data:"):
        text = text[5:].strip()
    if not text or text == "[DONE]":
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid vLLM stream frame: {text[:500]}") from exc
    if not isinstance(payload, Mapping):
        raise BenchmarkError("vLLM stream frame is not a JSON object")
    return payload


def _generated_text(payload: Mapping[str, Any], prompt: str) -> str | None:
    value: Any = payload.get("text")
    if value is None:
        value = payload.get("generated_text")
    if value is None and isinstance(payload.get("outputs"), list):
        outputs = payload["outputs"]
        if outputs and isinstance(outputs[0], Mapping):
            value = outputs[0].get("text")
    if value is None and isinstance(payload.get("choices"), list):
        choices = payload["choices"]
        if choices and isinstance(choices[0], Mapping):
            value = choices[0].get("text")
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    return value[len(prompt) :] if value.startswith(prompt) else value


def _completion_token_count(
    payload: Mapping[str, Any],
) -> tuple[int, str] | None:
    """Extract an unambiguous cumulative completion-token count when present."""

    usage = payload.get("usage")
    if isinstance(usage, Mapping):
        value = usage.get("completion_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value, "usage.completion_tokens"

    for key in ("completion_tokens", "num_output_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value, key

    outputs = payload.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], Mapping):
        token_ids = outputs[0].get("token_ids")
        if isinstance(token_ids, list) and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in token_ids
        ):
            return len(token_ids), "outputs[0].token_ids"
    return None


def _consume_stream(
    response: Any, *, prompt: str, started: float
) -> StreamObservation:
    """Consume NUL-delimited vLLM frames or standard SSE frames."""

    del started  # Documents the timestamp domain used by the caller.
    buffer = b""
    response_bytes = 0
    first_token_at: float | None = None
    last_token_at: float | None = None
    latest_generated: str | None = None
    completion_tokens: int | None = None
    token_count_source: str | None = None
    generated_updates = 0
    read_chunk = getattr(response, "read1", None)
    if not callable(read_chunk):
        read_chunk = response.read

    def consume(frame: bytes) -> None:
        nonlocal first_token_at, last_token_at, latest_generated
        nonlocal completion_tokens, token_count_source, generated_updates
        payload = _parse_stream_payload(frame)
        if payload is None:
            return
        count_evidence = _completion_token_count(payload)
        count_increased = False
        if count_evidence is not None:
            observed_count, observed_source = count_evidence
            if completion_tokens is None or observed_count > completion_tokens:
                count_increased = observed_count > 0
                completion_tokens = observed_count
                token_count_source = observed_source
        generated = _generated_text(payload, prompt)
        text_changed = generated is not None and generated != latest_generated
        if generated is not None:
            latest_generated = generated
        # Cumulative token IDs are valid arrival evidence even when a special
        # token decodes to an empty string. Conversely, some protocols publish
        # usage only in a duplicate final frame; that late aggregate must not
        # move the last-token timestamp and put response finalization in TPOT.
        count_is_arrival_evidence = (
            count_evidence is not None
            and count_evidence[1]
            in {"outputs[0].token_ids", "num_output_tokens"}
        )
        if (generated and text_changed) or (
            count_increased
            and (generated is None or count_is_arrival_evidence)
        ):
            observed_at = perf_counter()
            if first_token_at is None:
                first_token_at = observed_at
            last_token_at = observed_at
            generated_updates += 1

    while True:
        chunk = read_chunk(4096)
        if not chunk:
            break
        response_bytes += len(chunk)
        buffer += chunk
        while True:
            separators = [
                (position, separator)
                for separator in (b"\0", b"\n\n")
                if (position := buffer.find(separator)) >= 0
            ]
            if not separators:
                break
            position, separator = min(separators, key=lambda item: item[0])
            frame = buffer[:position]
            buffer = buffer[position + len(separator) :]
            consume(frame)
    if buffer.strip():
        consume(buffer)
    return StreamObservation(
        response_bytes=response_bytes,
        first_token_at=first_token_at,
        last_token_at=last_token_at,
        completion_tokens=completion_tokens,
        token_count_source=token_count_source,
        generated_updates=generated_updates,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(
    *,
    latencies: Sequence[float],
    ttfts: Sequence[float],
    tpots: Sequence[float],
    response_tails: Sequence[float],
    completion_tokens: Sequence[int],
    token_count_sources: Sequence[str],
    wall_seconds: float,
    input_chars: int,
    output_tokens: int,
    concurrency: int,
    response_bytes: int,
) -> dict[str, Any]:
    count = len(latencies)
    def latency_stats(values: Sequence[float]) -> dict[str, float]:
        return {
            "min": round(min(values), 3),
            "mean": round(statistics.fmean(values), 3),
            "p50": round(_percentile(values, 0.50), 3),
            "p95": round(_percentile(values, 0.95), 3),
            "p99": round(_percentile(values, 0.99), 3),
            "max": round(max(values), 3),
        }

    return {
        "requests": count,
        "concurrency": concurrency,
        "input_chars_per_request": input_chars,
        "output_tokens_per_request": output_tokens,
        "wall_seconds": round(wall_seconds, 6),
        "qps": round(count / wall_seconds, 4),
        "requested_output_tokens_per_second": round(
            count * output_tokens / wall_seconds, 4
        ),
        "completion_tokens": {
            "min": min(completion_tokens),
            "mean": round(statistics.fmean(completion_tokens), 3),
            "max": max(completion_tokens),
        },
        "completion_token_count_sources": sorted(set(token_count_sources)),
        "completion_tokens_per_second": round(
            sum(completion_tokens) / wall_seconds, 4
        ),
        "response_bytes": response_bytes,
        "latency_ms": latency_stats(latencies),
        "ttft_ms": latency_stats(ttfts),
        "tpot_ms": latency_stats(tpots),
        "response_tail_ms": latency_stats(response_tails),
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.output_tokens < 1 or args.requests < 1 or args.concurrency < 1:
            raise BenchmarkError(
                "output-tokens, requests, and concurrency must be positive"
            )
        if args.warmup < 0 or args.timeout <= 0:
            raise BenchmarkError("warmup must be non-negative and timeout positive")
        prompt = _build_prompt(args.input_chars)

        for _ in range(args.warmup):
            _post_generate(
                url=args.url,
                prompt=prompt,
                output_tokens=args.output_tokens,
                timeout=args.timeout,
            )

        started = perf_counter()
        latencies: list[float] = []
        ttfts: list[float] = []
        tpots: list[float] = []
        response_tails: list[float] = []
        completion_tokens: list[int] = []
        token_count_sources: list[str] = []
        response_bytes = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    _post_generate,
                    url=args.url,
                    prompt=prompt,
                    output_tokens=args.output_tokens,
                    timeout=args.timeout,
                )
                for _ in range(args.requests)
            ]
            for future in as_completed(futures):
                (
                    latency_ms,
                    body_size,
                    ttft_ms,
                    tpot_ms,
                    response_tail_ms,
                    actual_completion_tokens,
                    token_count_source,
                ) = future.result()
                latencies.append(latency_ms)
                ttfts.append(ttft_ms)
                tpots.append(tpot_ms)
                response_tails.append(response_tail_ms)
                completion_tokens.append(actual_completion_tokens)
                token_count_sources.append(token_count_source)
                response_bytes += body_size
        wall_seconds = perf_counter() - started
        print(
            json.dumps(
                _summary(
                    latencies=latencies,
                    ttfts=ttfts,
                    tpots=tpots,
                    response_tails=response_tails,
                    completion_tokens=completion_tokens,
                    token_count_sources=token_count_sources,
                    wall_seconds=wall_seconds,
                    input_chars=len(prompt),
                    output_tokens=args.output_tokens,
                    concurrency=args.concurrency,
                    response_bytes=response_bytes,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
