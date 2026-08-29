#!/usr/bin/env python
"""Minimal HTTP benchmark for a running vLLM ``/generate`` endpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence
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
) -> tuple[float, int]:
    payload = {
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": output_tokens,
        # Prevent early EOS so every request performs comparable decode work.
        "min_tokens": output_tokens,
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
            response_body = response.read()
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(
            f"vLLM returned HTTP {exc.code}: {details[:1000]}"
        ) from exc
    except URLError as exc:
        raise BenchmarkError(f"cannot reach {url}: {exc}") from exc
    latency_ms = (perf_counter() - started) * 1000.0
    try:
        decoded: Any = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise BenchmarkError("vLLM returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise BenchmarkError("vLLM response JSON is not an object")
    return latency_ms, len(response_body)


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
    wall_seconds: float,
    input_chars: int,
    output_tokens: int,
    concurrency: int,
    response_bytes: int,
) -> dict[str, Any]:
    count = len(latencies)
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
        "response_bytes": response_bytes,
        "latency_ms": {
            "min": round(min(latencies), 3),
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "p99": round(_percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3),
        },
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
                latency_ms, body_size = future.result()
                latencies.append(latency_ms)
                response_bytes += body_size
        wall_seconds = perf_counter() - started
        print(
            json.dumps(
                _summary(
                    latencies=latencies,
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
