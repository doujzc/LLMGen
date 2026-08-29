from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


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
            ttfts=[4.0, 6.0],
            tpots=[1.0, 3.0],
            response_tails=[0.5, 1.5],
            completion_tokens=[8, 8],
            token_count_sources=[
                "requested_min_equals_max",
                "requested_min_equals_max",
            ],
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
        self.assertEqual(summary["ttft_ms"]["p50"], 5.0)
        self.assertEqual(summary["tpot_ms"]["p50"], 2.0)
        self.assertEqual(summary["response_tail_ms"]["p50"], 1.0)
        self.assertEqual(summary["completion_tokens"]["mean"], 8.0)
        self.assertEqual(summary["completion_tokens_per_second"], 32.0)

    def test_parses_vllm_and_sse_stream_frames(self) -> None:
        self.assertEqual(
            benchmark._generated_text(
                benchmark._parse_stream_payload(
                    b'{"text":["prompt-token"]}'
                ),
                "prompt-",
            ),
            "token",
        )
        self.assertIsNone(benchmark._parse_stream_payload(b"data: [DONE]"))

    def test_tpot_uses_last_generated_frame_not_response_close(self) -> None:
        prompt = "prompt-"
        frames = [
            json.dumps(
                {
                    "outputs": [
                        {"text": "a", "token_ids": [101]}
                    ]
                }
            ).encode("utf-8")
            + b"\0",
            json.dumps(
                {
                    "outputs": [
                        {"text": "ab", "token_ids": [101, 102]}
                    ]
                }
            ).encode("utf-8")
            + b"\0",
            # A duplicate final frame must not move the last-token timestamp.
            json.dumps(
                {
                    "outputs": [
                        {"text": "ab", "token_ids": [101, 102]}
                    ]
                }
            ).encode("utf-8")
            + b"\0",
        ]

        class FakeResponse:
            def __init__(self) -> None:
                self._chunks = iter([*frames, b""])

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                return None

            def read1(self, size: int) -> bytes:
                del size
                return next(self._chunks)

        with (
            patch.object(
                benchmark,
                "urlopen",
                return_value=FakeResponse(),
            ) as open_mock,
            patch.object(
                benchmark,
                "perf_counter",
                side_effect=[0.0, 0.100, 0.200, 0.300],
            ),
        ):
            result = benchmark._post_generate(
                url="http://127.0.0.1:18000/generate",
                prompt=prompt,
                output_tokens=2,
                timeout=1.0,
            )

        request = open_mock.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertFalse(request_payload["skip_special_tokens"])
        self.assertAlmostEqual(result[0], 300.0)
        self.assertAlmostEqual(result[2], 100.0)
        self.assertAlmostEqual(result[3], 100.0)
        self.assertAlmostEqual(result[4], 100.0)
        self.assertEqual(result[5], 2)
        self.assertEqual(result[6], "outputs[0].token_ids")

    def test_rejects_observed_completion_count_mismatch(self) -> None:
        class FakeResponse:
            def __init__(self) -> None:
                self._chunks = iter(
                    [
                        b'{"outputs":[{"text":"x","token_ids":[1,2]}]}\0',
                        b"",
                    ]
                )

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                return None

            def read1(self, size: int) -> bytes:
                del size
                return next(self._chunks)

        with (
            patch.object(benchmark, "urlopen", return_value=FakeResponse()),
            patch.object(benchmark, "perf_counter", side_effect=[0.0, 0.1, 0.2]),
        ):
            with self.assertRaisesRegex(
                benchmark.BenchmarkError,
                "unexpected number of completion tokens",
            ):
                benchmark._post_generate(
                    url="http://127.0.0.1:18000/generate",
                    prompt="prompt-",
                    output_tokens=5,
                    timeout=1.0,
                )

    def test_token_ids_time_special_tokens_with_empty_decoded_text(self) -> None:
        class FakeResponse:
            def __init__(self) -> None:
                self._chunks = iter(
                    [
                        b'{"outputs":[{"text":"","token_ids":[1]}]}\0',
                        b'{"outputs":[{"text":"","token_ids":[1,2]}]}\0',
                        b"",
                    ]
                )

            def read1(self, size: int) -> bytes:
                del size
                return next(self._chunks)

        with patch.object(
            benchmark,
            "perf_counter",
            side_effect=[0.1, 0.2],
        ):
            observation = benchmark._consume_stream(
                FakeResponse(),
                prompt="prompt-",
                started=0.0,
            )

        self.assertAlmostEqual(observation.first_token_at, 0.1)
        self.assertAlmostEqual(observation.last_token_at, 0.2)
        self.assertEqual(observation.completion_tokens, 2)
        self.assertEqual(
            observation.token_count_source,
            "outputs[0].token_ids",
        )


if __name__ == "__main__":
    unittest.main()
