from __future__ import annotations

import asyncio
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import service_910b


def _write_router_bundle(directory: Path) -> None:
    (directory / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"]}) + "\n",
        encoding="utf-8",
    )
    (directory / "model.safetensors").write_bytes(b"fake-full-weights")

    virtual_tokens = [
        "<SK_L1_0>",
        "<SK_L2_0>",
        "<SK_L1_1>",
        "<SK_L2_1>",
    ]
    (directory / "virtual_tokens.txt").write_text(
        "\n".join(virtual_tokens) + "\n", encoding="utf-8"
    )
    decode_map = {
        "schema_version": 1,
        "num_levels": 2,
        "num_skills": 2,
        "num_paths": 2,
        "virtual_tokens": virtual_tokens,
        "skills": {
            "weather": {"name": "天气查询"},
            "maps": {"name": "地图导航"},
        },
        "skill_to_code": {
            "weather": {
                "tokens": virtual_tokens[:2],
                "code_text": "".join(virtual_tokens[:2]),
            },
            "maps": {
                "tokens": virtual_tokens[2:],
                "code_text": "".join(virtual_tokens[2:]),
            },
        },
        "paths": [
            {
                "tokens": virtual_tokens[:2],
                "skill_ids": ["weather"],
            },
            {
                "tokens": virtual_tokens[2:],
                "skill_ids": ["maps"],
            },
        ],
    }
    (directory / "skill_decode_map.json").write_text(
        json.dumps(decode_map, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "router_manifest.json").write_text(
        json.dumps(
            {
                "system_prompt": "测试专用检索提示",
                "max_length": 64,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _service_environment(directory: Path) -> dict[str, str]:
    return {
        "MODEL_PATH": str(directory),
        "TOKENIZER_PATH": str(directory),
        "CANDIDATE_STATE_PATH": str(directory),
        "TOP_K": "2",
        "MAX_CODE_PATHS": "2",
        "SERVED_MODEL_NAME": "router-910b-test",
        "GENERATION_DTYPE": "float16",
        "VLLM_TENSOR_PARALLEL_SIZE": "1",
        "VLLM_MAX_MODEL_LEN": "64",
        "VLLM_MAX_NUM_SEQS": "3",
        "VLLM_TRUST_REMOTE_CODE": "0",
        "VLLM_HEALTH_CHECK_TIMEOUT": "1",
        "VLLM_HEALTH_CHECK_INTERVAL": "0.1",
        "PROGRESSIVE_REQUEST_TIMEOUT": "2",
    }


class _FakeDependencyState:
    def __init__(
        self,
        *,
        health_error: BaseException | None = None,
        health_delay: float = 0.0,
        load_model_delay: float = 0.0,
    ) -> None:
        self.health_error = health_error
        self.health_delay = health_delay
        self.load_model_delay = load_model_delay
        self.events: list[str] = []
        self.tokenizer_load_calls: list[tuple[str, dict[str, object]]] = []
        self.tokenizer_encode_calls: list[tuple[str, bool]] = []
        self.chat_template_calls: list[
            tuple[list[dict[str, str]], dict[str, object]]
        ] = []
        self.engine_args_calls: list[dict[str, object]] = []
        self.from_engine_args_calls: list[object] = []
        self.sampling_params_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.engine: object | None = None
        self.tokenizer: object | None = None
        self.engine_role_m = object()


def _fake_dependency_modules(
    state: _FakeDependencyState,
) -> dict[str, ModuleType]:
    class FakeTokenizer:
        eos_token_id = 0
        chat_template = "fake-chat-template"

        def encode(
            self, text: str, add_special_tokens: bool = False
        ) -> list[int]:
            state.tokenizer_encode_calls.append((text, add_special_tokens))
            atomic = {
                "<SK_L1_0>": [1],
                "<SK_L2_0>": [2],
                "<SK_L1_1>": [4],
                "<SK_L2_1>": [5],
                "\n": [9],
            }
            return atomic.get(text, [40, 41])

        @staticmethod
        def apply_chat_template(
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            state.chat_template_calls.append(
                (
                    [dict(message) for message in messages],
                    {
                        "tokenize": tokenize,
                        "add_generation_prompt": add_generation_prompt,
                        "enable_thinking": enable_thinking,
                    },
                )
            )
            return "<rendered-router-prompt>"

    tokenizer = FakeTokenizer()
    state.tokenizer = tokenizer

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(
            cls, tokenizer_path: str, **kwargs: object
        ) -> FakeTokenizer:
            del cls
            state.tokenizer_load_calls.append((tokenizer_path, dict(kwargs)))
            return tokenizer

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            state.sampling_params_calls.append(dict(kwargs))
            self.__dict__.update(kwargs)

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)
            state.engine_args_calls.append(self.kwargs)

    class FakeEngine:
        def __init__(self) -> None:
            self.engine = SimpleNamespace(load_model=self.load_model)
            self.llm_engine = SimpleNamespace(shutdown=self.shutdown)

        @staticmethod
        def load_model() -> None:
            state.events.append("load_model")
            if state.load_model_delay:
                threading.Event().wait(state.load_model_delay)

        @staticmethod
        def start_background_loop() -> None:
            state.events.append("start_background_loop")

        @staticmethod
        async def is_health() -> bool:
            state.events.append("is_health")
            if state.health_delay:
                await asyncio.sleep(state.health_delay)
            if state.health_error is not None:
                raise state.health_error
            return True

        @staticmethod
        def generate(
            *,
            prompt: str,
            sampling_params: object,
            request_id: str,
            prompt_token_ids: list[int],
            tag: object,
            arrival_time: object,
            multi_modal_data: object,
            scheduler_result: object,
            is_stream: bool,
        ) -> object:
            call = {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
                "prompt_token_ids": prompt_token_ids,
                "tag": tag,
                "arrival_time": arrival_time,
                "multi_modal_data": multi_modal_data,
                "scheduler_result": scheduler_result,
                "is_stream": is_stream,
            }
            state.generate_calls.append(call)
            state.events.append("generate")

            async def output_frames() -> object:
                # The first frame is intentionally unusable: successful decoding
                # proves that the service consumes the final streamed frame.
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="partial-text-must-not-be-parsed",
                            token_ids=[999],
                            finish_reason=None,
                        )
                    ]
                )
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="text-must-not-be-used-for-routing",
                            token_ids=[1, 2, 9, 4, 5, 0],
                            finish_reason="stop",
                        )
                    ]
                )

            return output_frames()

        @staticmethod
        def shutdown_background_loop() -> None:
            state.events.append("shutdown_background_loop")

        @staticmethod
        def shutdown() -> None:
            state.events.append("llm_engine.shutdown")

    engine = FakeEngine()
    state.engine = engine

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args: object) -> FakeEngine:
            del cls
            state.from_engine_args_calls.append(engine_args)
            return engine

    class FakeEngineRole:
        M = state.engine_role_m

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []  # type: ignore[attr-defined]
    fake_vllm.AsyncEngineArgs = FakeAsyncEngineArgs  # type: ignore[attr-defined]
    fake_vllm.AsyncLLMEngine = FakeAsyncLLMEngine  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]

    fake_global_consts = ModuleType("vllm.global_consts")
    fake_global_consts.EngineRole = FakeEngineRole  # type: ignore[attr-defined]
    fake_vllm.global_consts = fake_global_consts  # type: ignore[attr-defined]

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    return {
        "vllm": fake_vllm,
        "vllm.global_consts": fake_global_consts,
        "transformers": fake_transformers,
    }


class SelfContainedService910BTest(unittest.TestCase):
    def test_imports_only_standard_library_and_lazy_runtime_dependencies(self) -> None:
        source_path = Path(service_910b.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])

        self.assertEqual(
            imported_roots,
            {
                "__future__",
                "asyncio",
                "concurrent",
                "dataclasses",
                "gc",
                "inspect",
                "json",
                "logging",
                "os",
                "pathlib",
                "threading",
                "time",
                "transformers",
                "typing",
                "uuid",
                "vllm",
            },
        )

    def test_single_copied_file_runs_mock_mode_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            copied_service = directory / "service_910b.py"
            shutil.copy2(Path(service_910b.__file__).resolve(), copied_service)
            environment = {
                "MOCK_MODE": "1",
                "MOCK_RESPONSES_JSON": json.dumps(
                    {"查天气": ["天气查询", "空气质量"]},
                    ensure_ascii=False,
                ),
                "TOP_K": "2",
            }

            completed = subprocess.run(
                [sys.executable, "-I", str(copied_service)],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=10,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout.strip()),
            ["天气查询", "空气质量"],
        )


class CustomVllm910BServiceTest(unittest.TestCase):
    def test_json_options_cannot_replace_owned_model_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            environment = {
                "VLLM_KWARGS_JSON": json.dumps(
                    {
                        "model": "/wrong/model",
                        "tokenizer": "/wrong/tokenizer",
                        "engine_role": "wrong-role",
                    }
                )
            }
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(
                    service_910b.ServiceConfigurationError,
                    "cannot override.*engine_role.*model.*tokenizer",
                ):
                    service_910b._build_vllm_kwargs(model_path=directory)
                with self.assertRaisesRegex(
                    service_910b.ServiceConfigurationError,
                    "cannot override.*model",
                ):
                    service_910b.load_vllm_model(
                        model_path=directory,
                        tokenizer_path=directory,
                        vllm_kwargs={"model": "/wrong/model"},
                    )

    def test_load_calc_close_preserves_skillhub_and_llmgen_protocols(self) -> None:
        state = _FakeDependencyState()
        fake_modules = _fake_dependency_modules(state)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            environment = _service_environment(directory)
            runtime = service_910b.RetriverTest()

            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, fake_modules),
            ):
                runtime.load()
                runtime.load()
                owned_runtime = runtime.llm
                try:
                    result = runtime.calc(
                        {
                            "data": {
                                "query": "出门前查天气和路线",
                                "top_k": 2,
                            }
                        }
                    )
                    self.assertEqual(
                        json.loads(result), ["天气查询", "地图导航"]
                    )
                    self.assertEqual(runtime.backend, "vllm_910b")
                    self.assertEqual(runtime.system_prompt, "测试专用检索提示")
                    self.assertEqual(runtime.max_input_length, 58)

                    self.assertEqual(len(state.tokenizer_load_calls), 1)
                    tokenizer_path, tokenizer_kwargs = state.tokenizer_load_calls[0]
                    self.assertEqual(tokenizer_path, str(directory.resolve()))
                    self.assertEqual(tokenizer_kwargs, {"trust_remote_code": False})

                    self.assertEqual(len(state.engine_args_calls), 1)
                    engine_kwargs = state.engine_args_calls[0]
                    expected_engine_keys = set(
                        service_910b._VLLM_ENGINE_ARG_DEFAULTS
                    ) | {
                        "model",
                        "tokenizer",
                        "dtype",
                        "prefix_sharing_type",
                        "engine_role",
                    }
                    self.assertEqual(set(engine_kwargs), expected_engine_keys)
                    self.assertEqual(engine_kwargs["model"], str(directory.resolve()))
                    self.assertEqual(
                        engine_kwargs["tokenizer"], str(directory.resolve())
                    )
                    self.assertEqual(
                        engine_kwargs["model_vision"], "facebook/opt-125m"
                    )
                    self.assertEqual(
                        engine_kwargs["architectures"], "Qwen3ForCausalLM"
                    )
                    self.assertEqual(engine_kwargs["dtype"], "float16")
                    self.assertIs(
                        engine_kwargs["engine_role"], state.engine_role_m
                    )
                    self.assertEqual(engine_kwargs["tensor_parallel_size"], 1)
                    self.assertEqual(
                        engine_kwargs["decode_tensor_parallel_size"], 1
                    )
                    self.assertEqual(engine_kwargs["max_model_len"], 64)
                    self.assertEqual(engine_kwargs["max_num_seqs"], 3)
                    self.assertFalse(engine_kwargs["trust_remote_code"])
                    self.assertEqual(engine_kwargs["prefix_sharing_type"], "auto")
                    self.assertTrue(engine_kwargs["enable_datasystem"])
                    self.assertEqual(len(state.from_engine_args_calls), 1)
                    self.assertEqual(
                        state.from_engine_args_calls[0].kwargs, engine_kwargs
                    )

                    self.assertEqual(len(state.sampling_params_calls), 1)
                    sampling_kwargs = state.sampling_params_calls[0]
                    self.assertEqual(
                        set(sampling_kwargs),
                        {
                            "temperature",
                            "max_tokens",
                            "min_tokens",
                            "detokenize",
                            "skip_special_tokens",
                            "logits_processors",
                        },
                    )
                    self.assertEqual(sampling_kwargs["temperature"], 0.0)
                    self.assertEqual(sampling_kwargs["max_tokens"], 6)
                    self.assertEqual(sampling_kwargs["min_tokens"], 2)
                    self.assertFalse(sampling_kwargs["detokenize"])
                    self.assertFalse(sampling_kwargs["skip_special_tokens"])
                    processors = sampling_kwargs["logits_processors"]
                    self.assertEqual(len(processors), 1)
                    self.assertIsInstance(
                        processors[0], service_910b.TrieLogitsProcessor
                    )
                    self.assertIs(processors[0].trie, runtime.trie)

                    self.assertEqual(
                        state.chat_template_calls,
                        [
                            (
                                [
                                    {
                                        "role": "system",
                                        "content": "测试专用检索提示",
                                    },
                                    {
                                        "role": "user",
                                        "content": "出门前查天气和路线",
                                    },
                                ],
                                {
                                    "tokenize": False,
                                    "add_generation_prompt": True,
                                    "enable_thinking": False,
                                },
                            )
                        ],
                    )
                    self.assertEqual(len(state.generate_calls), 1)
                    generate_call = state.generate_calls[0]
                    self.assertEqual(
                        set(generate_call),
                        {
                            "prompt",
                            "sampling_params",
                            "request_id",
                            "prompt_token_ids",
                            "tag",
                            "arrival_time",
                            "multi_modal_data",
                            "scheduler_result",
                            "is_stream",
                        },
                    )
                    self.assertEqual(
                        generate_call["prompt"], "<rendered-router-prompt>"
                    )
                    self.assertIs(
                        generate_call["sampling_params"], runtime.sampling_params
                    )
                    self.assertEqual(generate_call["prompt_token_ids"], [40, 41])
                    request_uuid = uuid.UUID(str(generate_call["request_id"]))
                    self.assertEqual(request_uuid.version, 4)
                    for key in (
                        "tag",
                        "arrival_time",
                        "multi_modal_data",
                        "scheduler_result",
                    ):
                        self.assertIsNone(generate_call[key])
                    self.assertFalse(generate_call["is_stream"])
                finally:
                    runtime.close()
                    runtime.close()

        self.assertIsNotNone(owned_runtime)
        self.assertTrue(owned_runtime._closed)
        self.assertEqual(
            state.events,
            [
                "load_model",
                "start_background_loop",
                "is_health",
                "generate",
                "shutdown_background_loop",
                "llm_engine.shutdown",
            ],
        )
        self.assertFalse(runtime._loaded)
        self.assertIsNone(runtime.llm)
        self.assertIsNone(runtime.tokenizer)
        self.assertIsNone(runtime.sampling_params)
        self.assertIsNone(runtime.bundle)
        self.assertIsNone(runtime.trie)
        self.assertFalse(
            any(
                thread.name == "local-vllm-910b-async-loop"
                and thread.is_alive()
                for thread in threading.enumerate()
            )
        )

    def test_startup_failure_closes_engine_and_async_loop(self) -> None:
        state = _FakeDependencyState(
            health_error=RuntimeError("fake custom engine health failure")
        )
        fake_modules = _fake_dependency_modules(state)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            environment = _service_environment(directory)
            runtime = service_910b.RetriverTest()

            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, fake_modules),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "fake custom engine health failure"
                ):
                    runtime.load()
                runtime.close()
                runtime.close()

        self.assertEqual(len(state.engine_args_calls), 1)
        self.assertEqual(
            state.events,
            [
                "load_model",
                "start_background_loop",
                "is_health",
                "shutdown_background_loop",
                "llm_engine.shutdown",
            ],
        )
        self.assertFalse(runtime._loaded)
        self.assertIsNone(runtime.llm)
        self.assertIsNone(runtime.tokenizer)
        self.assertIsNone(runtime.sampling_params)
        self.assertIsNone(runtime.bundle)
        self.assertIsNone(runtime.trie)
        self.assertEqual(runtime.path_skill_ids, {})
        self.assertFalse(
            any(
                thread.name == "local-vllm-910b-async-loop"
                and thread.is_alive()
                for thread in threading.enumerate()
            )
        )

    def test_slow_model_load_times_out_before_background_start(self) -> None:
        state = _FakeDependencyState(load_model_delay=0.03)
        fake_modules = _fake_dependency_modules(state)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            environment = _service_environment(directory)
            environment["VLLM_HEALTH_CHECK_TIMEOUT"] = "0.01"
            runtime = service_910b.RetriverTest()

            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, fake_modules),
            ):
                with self.assertRaisesRegex(
                    TimeoutError, "startup timed out while loading model"
                ):
                    runtime.load()
                runtime.close()
                runtime.close()

        self.assertEqual(
            state.events,
            [
                "load_model",
                "shutdown_background_loop",
                "llm_engine.shutdown",
            ],
        )
        self.assertFalse(runtime._loaded)
        self.assertIsNone(runtime.llm)
        self.assertIsNone(runtime.tokenizer)
        self.assertIsNone(runtime.bundle)
        self.assertFalse(
            any(
                thread.name == "local-vllm-910b-async-loop"
                and thread.is_alive()
                for thread in threading.enumerate()
            )
        )

    def test_slow_health_check_obeys_startup_deadline(self) -> None:
        state = _FakeDependencyState(health_delay=0.05)
        fake_modules = _fake_dependency_modules(state)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            environment = _service_environment(directory)
            environment["VLLM_HEALTH_CHECK_TIMEOUT"] = "0.01"
            runtime = service_910b.RetriverTest()

            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, fake_modules),
            ):
                with self.assertRaisesRegex(
                    TimeoutError, "timed out during health check"
                ):
                    runtime.load()
                runtime.close()

        self.assertEqual(
            state.events,
            [
                "load_model",
                "start_background_loop",
                "is_health",
                "shutdown_background_loop",
                "llm_engine.shutdown",
            ],
        )
        self.assertFalse(runtime._loaded)
        self.assertFalse(
            any(
                thread.name == "local-vllm-910b-async-loop"
                and thread.is_alive()
                for thread in threading.enumerate()
            )
        )


if __name__ == "__main__":
    unittest.main()
