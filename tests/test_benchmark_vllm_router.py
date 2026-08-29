from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_vllm_router.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_vllm_router", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


class BenchmarkVllmRouterTest(unittest.TestCase):
    def test_prompt_has_exact_requested_character_count(self) -> None:
        for length in (1, 32, 256, 1024):
            self.assertEqual(len(benchmark._build_prompt(length)), length)

    def test_summary_reports_latency_and_requested_throughput(self) -> None:
        summary = benchmark._summary(
            latencies=[10.0, 20.0],
            wall_seconds=0.5,
            input_chars=256,
            output_tokens=8,
            concurrency=2,
            response_bytes=100,
        )

        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["qps"], 4.0)
        self.assertEqual(summary["requested_output_tokens_per_second"], 32.0)
        self.assertEqual(summary["latency_ms"]["p50"], 15.0)


if __name__ == "__main__":
    unittest.main()
