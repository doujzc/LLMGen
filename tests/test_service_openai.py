from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import service_openai


def _write_router_bundle(directory: Path) -> None:
    (directory / "config.json").write_text("{}\n", encoding="utf-8")
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
        "SERVED_MODEL_NAME": "router-openai-test",
        "OPENAI_BASE_URL": "http://127.0.0.1:8123/v1",
        "OPENAI_API_KEY": "local-test-key",
        "OPENAI_TIMEOUT_SECONDS": "12.5",
        "OPENAI_MAX_RETRIES": "1",
        "VLLM_TRUST_REMOTE_CODE": "0",
        "TOKENIZER_LOCAL_FILES_ONLY": "1",
    }


class _FakeDependencyState:
    def __init__(self, *, finish_reason: str = "stop") -> None:
        self.finish_reason = finish_reason
        self.completion_tokens = [1, 2, 9, 4, 5, 0]
        self.tokenizer_load_calls: list[tuple[str, dict[str, object]]] = []
        self.tokenizer_encode_calls: list[tuple[str, bool]] = []
        self.openai_init_calls: list[dict[str, object]] = []
        self.completion_calls: list[dict[str, object]] = []
        self.close_calls = 0


class _FakeRunningProcess:
    @staticmethod
    def poll() -> None:
        return None


class _FakeVllmServerHandle:
    def __init__(self) -> None:
        self.process = _FakeRunningProcess()
        self.base_url = "http://127.0.0.1:8123/v1"
        self.api_key = "local-test-key"
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        if self.closed:
            return
        self.close_calls += 1
        self.closed = True


class _FakePopen:
    def __init__(self) -> None:
        self.pid = 4321
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


def _fake_dependency_modules(
    state: _FakeDependencyState,
) -> dict[str, ModuleType]:
    class FakeTokenizer:
        eos_token_id = 0
        chat_template = None

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

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(
            cls, tokenizer_path: str, **kwargs: object
        ) -> FakeTokenizer:
            del cls
            state.tokenizer_load_calls.append((tokenizer_path, kwargs))
            return FakeTokenizer()

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> SimpleNamespace:
            state.completion_calls.append(kwargs)
            completion = SimpleNamespace(
                text="",
                finish_reason=state.finish_reason,
                logprobs=SimpleNamespace(
                    tokens=[
                        f"token_id:{token_id}"
                        for token_id in state.completion_tokens
                    ]
                ),
            )
            return SimpleNamespace(choices=[completion])

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            state.openai_init_calls.append(kwargs)
            self.completions = FakeCompletions()

        @staticmethod
        def close() -> None:
            state.close_calls += 1

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    return {
        "transformers": fake_transformers,
        "openai": fake_openai,
    }


class IndependentVllmLoaderTest(unittest.TestCase):
    def _launch_environment(self) -> dict[str, str]:
        return {
            "GENERATION_DTYPE": "float16",
            "VLLM_TENSOR_PARALLEL_SIZE": "1",
            "VLLM_TRUST_REMOTE_CODE": "0",
            "OPENAI_API_KEY": "launcher-test-key",
            "VLLM_SHUTDOWN_TIMEOUT_SECONDS": "2",
        }

    def test_load_vllm_model_starts_server_and_waits_until_ready(self) -> None:
        process = _FakePopen()

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            with (
                patch.dict(
                    os.environ,
                    self._launch_environment(),
                    clear=True,
                ),
                patch.object(
                    service_openai,
                    "_ensure_vllm_port_available",
                ) as ensure_port,
                patch.object(
                    service_openai.subprocess,
                    "Popen",
                    return_value=process,
                ) as popen,
                patch.object(
                    service_openai,
                    "_wait_for_vllm_ready",
                ) as wait_ready,
                patch.object(
                    service_openai,
                    "_stop_vllm_process_group",
                    return_value=True,
                ) as stop_group,
                patch.object(
                    service_openai.os,
                    "getpgid",
                    return_value=process.pid,
                ),
                patch.object(service_openai.os, "killpg") as kill_group,
            ):
                handle = service_openai.load_vllm_model(
                    directory,
                    directory,
                    served_model_name="router-launcher-test",
                    base_url="http://127.0.0.1:8123/v1",
                    max_num_seqs=3,
                )

                ensure_port.assert_called_once_with("127.0.0.1", 8123)
                wait_ready.assert_called_once_with(
                    process,
                    health_url="http://127.0.0.1:8123/health",
                    models_url="http://127.0.0.1:8123/v1/models",
                    served_model_name="router-launcher-test",
                    api_key="launcher-test-key",
                )
                popen.assert_called_once()
                command = popen.call_args.args[0]
                popen_kwargs = popen.call_args.kwargs

                self.assertEqual(
                    command[:3],
                    [
                        sys.executable,
                        "-m",
                        "vllm.entrypoints.openai.api_server",
                    ],
                )
                self.assertEqual(
                    command[command.index("--model") + 1],
                    str(directory.resolve()),
                )
                self.assertEqual(
                    command[command.index("--tokenizer") + 1],
                    str(directory.resolve()),
                )
                self.assertEqual(
                    command[command.index("--served-model-name") + 1],
                    "router-launcher-test",
                )
                self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
                self.assertEqual(command[command.index("--port") + 1], "8123")
                self.assertEqual(command[command.index("--dtype") + 1], "float16")
                self.assertEqual(
                    command[command.index("--tensor-parallel-size") + 1],
                    "1",
                )
                self.assertEqual(
                    command[command.index("--max-num-seqs") + 1], "3"
                )
                self.assertEqual(
                    command[command.index("--logits-processor-pattern") + 1],
                    r"^service_openai\.create_trie_logits_processor$",
                )
                self.assertEqual(
                    popen_kwargs["env"]["VLLM_USE_V1"], "0"
                )
                self.assertEqual(
                    popen_kwargs["env"]["VLLM_API_KEY"],
                    "launcher-test-key",
                )
                self.assertIn(
                    str(Path(service_openai.__file__).resolve().parent),
                    popen_kwargs["env"]["PYTHONPATH"].split(os.pathsep),
                )
                self.assertEqual(
                    popen_kwargs["start_new_session"], os.name == "posix"
                )

                self.assertIs(handle.process, process)
                self.assertEqual(handle.base_url, "http://127.0.0.1:8123/v1")
                self.assertEqual(handle.api_key, "launcher-test-key")
                handle.close()
                handle.close()
                stop_group.assert_called_once_with(4321, timeout=2.0)
                kill_group.assert_called_once_with(4321, service_openai.signal.SIGTERM)

        self.assertTrue(handle.closed)
        self.assertEqual(process.terminate_calls, 0)
        self.assertEqual(len(process.wait_calls), 1)
        self.assertEqual(process.kill_calls, 0)

    def test_load_vllm_model_reaps_process_when_readiness_fails(self) -> None:
        process = _FakePopen()

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            with (
                patch.dict(
                    os.environ,
                    self._launch_environment(),
                    clear=True,
                ),
                patch.object(
                    service_openai,
                    "_ensure_vllm_port_available",
                ),
                patch.object(
                    service_openai.subprocess,
                    "Popen",
                    return_value=process,
                ),
                patch.object(
                    service_openai,
                    "_wait_for_vllm_ready",
                    side_effect=TimeoutError("not ready"),
                ),
                patch.object(
                    service_openai,
                    "_stop_vllm_process_group",
                    return_value=True,
                ) as stop_group,
                patch.object(
                    service_openai.os,
                    "getpgid",
                    return_value=process.pid,
                ),
                patch.object(service_openai.os, "killpg") as kill_group,
            ):
                with self.assertRaisesRegex(TimeoutError, "not ready"):
                    service_openai.load_vllm_model(
                        directory,
                        directory,
                        served_model_name="router-launcher-test",
                        base_url="http://127.0.0.1:8123/v1",
                    )
                stop_group.assert_called_once_with(4321, timeout=2.0)
                kill_group.assert_called_once_with(4321, service_openai.signal.SIGTERM)

        self.assertEqual(process.terminate_calls, 0)
        self.assertEqual(len(process.wait_calls), 1)
        self.assertEqual(process.kill_calls, 0)

    def test_invalid_shutdown_timeout_still_terminates_server(self) -> None:
        process = _FakePopen()
        handle = service_openai.VllmServerHandle(
            process,
            base_url="http://127.0.0.1:8123/v1",
            api_key="test-key",
            process_group=False,
        )

        with (
            patch.dict(
                os.environ,
                {"VLLM_SHUTDOWN_TIMEOUT_SECONDS": "not-a-number"},
                clear=True,
            ),
            patch.object(service_openai.logger, "exception") as log_exception,
        ):
            handle.close()

        self.assertTrue(handle.closed)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(len(process.wait_calls), 1)
        log_exception.assert_called_once()

    def test_exited_leader_never_signals_stale_process_group(self) -> None:
        process = _FakePopen()
        process.returncode = 1
        handle = service_openai.VllmServerHandle(
            process,
            base_url="http://127.0.0.1:8123/v1",
            api_key="test-key",
            process_group=True,
        )

        with (
            patch.object(
                service_openai,
                "_process_group_exists",
                return_value=True,
            ),
            patch.object(service_openai, "_stop_vllm_process_group") as stop_group,
            patch.object(service_openai.os, "killpg") as kill_group,
        ):
            handle.close()

        self.assertTrue(handle.closed)
        stop_group.assert_not_called()
        kill_group.assert_not_called()
        self.assertEqual(process.terminate_calls, 0)
        self.assertEqual(process.kill_calls, 0)

    def test_server_extra_args_reject_protected_underscore_spelling(self) -> None:
        with patch.dict(
            os.environ,
            {"VLLM_SERVER_ARGS_JSON": '["--api_key", "unsafe"]'},
            clear=True,
        ):
            with self.assertRaisesRegex(
                service_openai.ServiceConfigurationError,
                "cannot override: --api-key",
            ):
                service_openai._load_vllm_server_extra_args()

    def test_server_extra_args_reject_protected_abbreviations(self) -> None:
        for payload, expected in (
            ('["--api-k", "unsafe"]', "--api-k"),
            ('["-tp", "9"]', "-tp"),
        ):
            with self.subTest(payload=payload), patch.dict(
                os.environ,
                {"VLLM_SERVER_ARGS_JSON": payload},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    service_openai.ServiceConfigurationError,
                    f"cannot override: {expected}",
                ):
                    service_openai._load_vllm_server_extra_args()


class SelfContainedOpenAIServiceTest(unittest.TestCase):
    def test_imports_only_standard_library_and_declared_runtimes(self) -> None:
        source_path = Path(service_openai.__file__).resolve()
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
                "dataclasses",
                "gc",
                "json",
                "logging",
                "openai",
                "os",
                "pathlib",
                "re",
                "secrets",
                "signal",
                "socket",
                "subprocess",
                "sys",
                "threading",
                "time",
                "transformers",
                "typing",
                "urllib",
            },
        )

    def test_single_copied_file_runs_mock_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            copied_service = directory / "service_openai.py"
            shutil.copy2(Path(service_openai.__file__).resolve(), copied_service)
            completed = subprocess.run(
                [sys.executable, "-I", str(copied_service)],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                env={
                    "MOCK_MODE": "1",
                    "MOCK_RESPONSES_JSON": json.dumps(
                        {"查天气": ["天气查询", "空气质量"]},
                        ensure_ascii=False,
                    ),
                    "TOP_K": "2",
                },
                timeout=10,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout.strip()),
            ["天气查询", "空气质量"],
        )


class OpenAIServiceEndToEndTest(unittest.TestCase):
    def test_load_closes_started_server_when_client_creation_fails(self) -> None:
        state = _FakeDependencyState()
        server_handle = _FakeVllmServerHandle()

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            with (
                patch.dict(
                    os.environ,
                    _service_environment(directory),
                    clear=True,
                ),
                patch.dict(sys.modules, _fake_dependency_modules(state)),
                patch.object(
                    service_openai,
                    "load_vllm_model",
                    return_value=server_handle,
                ),
                patch.object(
                    service_openai,
                    "_create_openai_client",
                    side_effect=RuntimeError("client setup failed"),
                ),
                patch.object(service_openai.logger, "exception"),
            ):
                runtime = service_openai.RetriverTest()
                with self.assertRaisesRegex(RuntimeError, "client setup failed"):
                    runtime.load()
                runtime.close()

        self.assertEqual(server_handle.close_calls, 1)
        self.assertTrue(server_handle.closed)
        self.assertFalse(runtime._loaded)
        self.assertIsNone(runtime.vllm_server)
        self.assertIsNone(runtime.openai_client)
        self.assertIsNone(runtime.bundle)

    def test_fake_openai_exercises_load_calc_and_idempotent_close(self) -> None:
        state = _FakeDependencyState()
        server_handle = _FakeVllmServerHandle()

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            with (
                patch.dict(
                    os.environ,
                    _service_environment(directory),
                    clear=True,
                ),
                patch.dict(sys.modules, _fake_dependency_modules(state)),
                patch.object(
                    service_openai,
                    "load_vllm_model",
                    return_value=server_handle,
                ) as launcher,
            ):
                runtime = service_openai.RetriverTest()
                runtime.load()
                runtime.load()
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
                    self.assertEqual(runtime.backend, "openai")
                    self.assertEqual(runtime.system_prompt, "测试专用检索提示")
                    self.assertEqual(runtime.output_budget, 6)
                    self.assertEqual(len(state.tokenizer_load_calls), 1)
                    self.assertEqual(len(state.openai_init_calls), 1)
                    self.assertEqual(len(state.completion_calls), 1)
                    launcher.assert_called_once_with(
                        directory.resolve(),
                        directory.resolve(),
                        served_model_name="router-openai-test",
                        base_url="http://127.0.0.1:8123/v1",
                        logits_processor_qualname=(
                            "service_openai.create_trie_logits_processor"
                        ),
                    )

                    tokenizer_path, tokenizer_kwargs = (
                        state.tokenizer_load_calls[0]
                    )
                    self.assertEqual(tokenizer_path, str(directory.resolve()))
                    self.assertEqual(
                        tokenizer_kwargs,
                        {
                            "trust_remote_code": False,
                            "local_files_only": True,
                        },
                    )
                    self.assertEqual(
                        state.openai_init_calls[0],
                        {
                            "base_url": "http://127.0.0.1:8123/v1",
                            "api_key": "local-test-key",
                            "timeout": 12.5,
                            "max_retries": 1,
                        },
                    )

                    request = state.completion_calls[0]
                    self.assertEqual(request["model"], "router-openai-test")
                    self.assertEqual(request["prompt"], [40, 41])
                    self.assertEqual(request["temperature"], 0.0)
                    self.assertEqual(request["top_p"], 1.0)
                    self.assertEqual(request["frequency_penalty"], 0.0)
                    self.assertEqual(request["presence_penalty"], 0.0)
                    self.assertEqual(request["max_tokens"], 6)
                    self.assertEqual(request["logprobs"], 0)
                    self.assertNotIn("stop", request)
                    self.assertEqual(
                        request["extra_body"],
                        {
                            "add_special_tokens": False,
                            "min_tokens": 2,
                            "top_k": -1,
                            "min_p": 0.0,
                            "repetition_penalty": 1.0,
                            "skip_special_tokens": False,
                            "spaces_between_special_tokens": False,
                            "return_tokens_as_token_ids": True,
                            "logits_processors": [
                                {
                                    "qualname": (
                                        "service_openai."
                                        "create_trie_logits_processor"
                                    ),
                                    "kwargs": {
                                        "paths": [[1, 2], [4, 5]],
                                        "eos_token_id": 0,
                                        "separator_token_ids": [9],
                                        "max_paths": 2,
                                    },
                                }
                            ],
                        },
                    )

                    descriptor = request["extra_body"]["logits_processors"][0]
                    processor = service_openai.create_trie_logits_processor(
                        **descriptor["kwargs"]
                    )
                    self.assertEqual(
                        processor.trie.parse_complete((1, 2, 9, 4, 5)),
                        ((1, 2), (4, 5)),
                    )
                    rendered_prompt = state.tokenizer_encode_calls[-1][0]
                    self.assertIn("测试专用检索提示", rendered_prompt)
                    self.assertIn("出门前查天气和路线", rendered_prompt)
                finally:
                    runtime.close()
                    runtime.close()

        self.assertEqual(state.close_calls, 1)
        self.assertEqual(server_handle.close_calls, 1)
        self.assertTrue(server_handle.closed)
        self.assertFalse(runtime._loaded)
        self.assertIsNone(runtime.vllm_server)
        self.assertIsNone(runtime.openai_client)
        self.assertIsNone(runtime.bundle)

    def test_length_finish_reason_is_rejected(self) -> None:
        state = _FakeDependencyState(finish_reason="length")
        server_handle = _FakeVllmServerHandle()

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            with (
                patch.dict(
                    os.environ,
                    _service_environment(directory),
                    clear=True,
                ),
                patch.dict(sys.modules, _fake_dependency_modules(state)),
                patch.object(
                    service_openai,
                    "load_vllm_model",
                    return_value=server_handle,
                ) as launcher,
            ):
                runtime = service_openai.RetriverTest()
                runtime.load()
                try:
                    with self.assertRaisesRegex(
                        RuntimeError, "exhausted its token budget"
                    ):
                        runtime.calc(
                            {"data": {"query": "查天气", "top_k": 2}}
                        )
                    self.assertTrue(runtime._loaded)
                    self.assertEqual(len(state.completion_calls), 1)
                    launcher.assert_called_once()
                finally:
                    runtime.close()

        self.assertEqual(state.close_calls, 1)
        self.assertEqual(server_handle.close_calls, 1)
        self.assertFalse(runtime._loaded)


if __name__ == "__main__":
    unittest.main()
