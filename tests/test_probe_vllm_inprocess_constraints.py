from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_vllm_inprocess_constraints.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "probe_vllm_inprocess_constraints", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


class FakeScores:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.shape = (len(values),)

    def new_full(self, shape: object, value: float) -> "FakeScores":
        assert shape == self.shape
        return FakeScores([value for _ in self.values])

    @staticmethod
    def _index(value: object) -> int:
        if isinstance(value, tuple):
            return int(value[-1])
        return int(value)

    def __getitem__(self, index: object) -> float:
        return self.values[self._index(index)]

    def __setitem__(self, index: object, value: float) -> None:
        self.values[self._index(index)] = value


class ProbeVllmInProcessConstraintsTest(unittest.TestCase):
    def test_processor_forces_each_position_and_writes_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_file = str(Path(temporary_directory) / "trace.jsonl")
            processor = probe.ForcedSequenceLogitsProcessor(
                (0, 1), trace_file
            )

            first = processor([], FakeScores([2.0, 5.0, 9.0]))
            second = processor([0], FakeScores([2.0, 5.0, 9.0]))

            self.assertEqual(first.values, [2.0, -float("inf"), -float("inf")])
            self.assertEqual(second.values, [-float("inf"), 5.0, -float("inf")])
            records = [
                json.loads(line)
                for line in Path(trace_file).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["forced_token_id"] for record in records], [0, 1]
            )
            self.assertEqual(
                [record["generated_count"] for record in records], [0, 1]
            )

    def test_inspection_distinguishes_supported_sampling_parameters(self) -> None:
        class FakeSamplingParams:
            def __init__(
                self,
                *,
                temperature: float,
                max_tokens: int,
                min_tokens: int,
                detokenize: bool,
                skip_special_tokens: bool,
                logits_processors: list[object],
            ) -> None:
                del temperature, max_tokens, min_tokens, detokenize
                del skip_special_tokens
                self.logits_processors = logits_processors

        class FakeAsyncEngineArgs:
            pass

        class FakeAsyncLLMEngine:
            @classmethod
            def from_engine_args(cls, engine_args: object) -> object:
                del cls, engine_args
                return object()

            def generate(self) -> None:
                return None

        fake_vllm = ModuleType("vllm")
        fake_vllm.__version__ = "probe-test"  # type: ignore[attr-defined]
        fake_vllm.AsyncEngineArgs = FakeAsyncEngineArgs  # type: ignore[attr-defined]
        fake_vllm.AsyncLLMEngine = FakeAsyncLLMEngine  # type: ignore[attr-defined]
        fake_vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
        fake_global_consts = ModuleType("vllm.global_consts")
        fake_global_consts.EngineRole = SimpleNamespace(M="role-m")  # type: ignore[attr-defined]

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.dict(
                sys.modules,
                {
                    "vllm": fake_vllm,
                    "vllm.global_consts": fake_global_consts,
                },
            ),
        ):
            report, _ = probe._inspect_vllm(
                sequence=(0, 1),
                trace_file=str(Path(temporary_directory) / "trace.jsonl"),
            )

        sampling = report["sampling_params"]
        self.assertTrue(sampling["accepts_logits_processors"])
        self.assertFalse(sampling["accepts_allowed_token_ids"])
        self.assertTrue(
            sampling["construction"]["logits_processors"]["constructed"]
        )
        self.assertFalse(
            sampling["construction"]["allowed_token_ids"]["constructed"]
        )

    def test_model_probe_verifies_callback_and_output_ids(self) -> None:
        events: list[str] = []

        class FakeSamplingParams:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class FakeAsyncEngineArgs:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FakeEngine:
            def __init__(self) -> None:
                self.engine = SimpleNamespace(load_model=self.load_model)
                self.llm_engine = SimpleNamespace(shutdown=self.shutdown)

            @staticmethod
            def load_model() -> None:
                events.append("load_model")

            @staticmethod
            def start_background_loop() -> None:
                events.append("start_background_loop")

            @staticmethod
            async def is_health() -> bool:
                return True

            @staticmethod
            def generate(**kwargs: object) -> object:
                processor = kwargs["sampling_params"].logits_processors[0]  # type: ignore[union-attr]

                async def frames() -> object:
                    processor([], FakeScores([4.0, 3.0, 2.0]))
                    processor([0], FakeScores([4.0, 3.0, 2.0]))
                    yield SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                token_ids=[0, 1],
                                text="",
                                finish_reason="length",
                            )
                        ]
                    )

                return frames()

            @staticmethod
            def shutdown_background_loop() -> None:
                events.append("shutdown_background_loop")

            @staticmethod
            def shutdown() -> None:
                events.append("shutdown")

        engine = FakeEngine()

        class FakeAsyncLLMEngine:
            @classmethod
            def from_engine_args(cls, engine_args: object) -> FakeEngine:
                del cls, engine_args
                return engine

        class FakeTokenizer:
            @staticmethod
            def encode(text: str, *, add_special_tokens: bool) -> list[int]:
                del text, add_special_tokens
                return [7, 8]

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, path: str, **kwargs: object) -> FakeTokenizer:
                del cls, path, kwargs
                return FakeTokenizer()

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
        args = probe._parse_args(["--model", "/model"])
        symbols = {
            "AsyncEngineArgs": FakeAsyncEngineArgs,
            "AsyncLLMEngine": FakeAsyncLLMEngine,
            "SamplingParams": FakeSamplingParams,
            "EngineRole": SimpleNamespace(M="role-m"),
        }

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.dict(sys.modules, {"transformers": fake_transformers}),
        ):
            trace_file = str(Path(temporary_directory) / "trace.jsonl")
            report = asyncio.run(
                probe._run_model_probe(
                    args,
                    sequence=(0, 1),
                    trace_file=trace_file,
                    trace_start_offset=0,
                    symbols=symbols,
                )
            )

        self.assertTrue(report["completed"])
        self.assertTrue(report["processor_callback_executed"])
        self.assertTrue(report["constraint_enforced"])
        self.assertEqual(report["output"]["token_ids"], [0, 1])
        self.assertEqual(events[:2], ["load_model", "start_background_loop"])
        self.assertEqual(events[-2:], ["shutdown_background_loop", "shutdown"])


if __name__ == "__main__":
    unittest.main()
