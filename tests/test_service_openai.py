from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
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
        "VLLM_REQUEST_TIMEOUT_SECONDS": "12.5",
        "VLLM_TRUST_REMOTE_CODE": "0",
        "TOKENIZER_LOCAL_FILES_ONLY": "1",
    }


class _FakeDependencyState:
    def __init__(self) -> None:
        self.completion_tokens = [1, 2, 9, 4, 5, 0]
        self.tokenizer_load_calls: list[tuple[str, dict[str, object]]] = []
        self.tokenizer_encode_calls: list[tuple[str, bool]] = []
        self.client_init_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.close_calls = 0


class _FakeRunningProcess:
    @staticmethod
    def poll() -> None:
        return None


class _FakeVllmServerHandle:
    def __init__(self) -> None:
        self.process = _FakeRunningProcess()
        self.origin = "http://127.0.0.1:8123"
        self.generate_url = self.origin + "/generate"
        self.health_url = self.origin + "/health"
        self.host = "127.0.0.1"
        self.port = 8123
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

        @staticmethod
        def decode(
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            del token_ids, skip_special_tokens, clean_up_tokenization_spaces
            return "<truncated-router-prompt>"

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(
            cls, tokenizer_path: str, **kwargs: object
        ) -> FakeTokenizer:
            del cls
            state.tokenizer_load_calls.append((tokenizer_path, kwargs))
            return FakeTokenizer()

    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "4.57.1"  # type: ignore[attr-defined]
    fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    return {"transformers": fake_transformers}


class _FakeGenerateClient:
    state: _FakeDependencyState

    def __init__(self, generate_url: str, *, timeout: float) -> None:
        self.state.client_init_calls.append(
            {"generate_url": generate_url, "timeout": timeout}
        )

    def generate(self, payload: dict[str, object]) -> dict[str, object]:
        self.state.generate_calls.append(dict(payload))
        max_tokens = int(payload.get("max_tokens", len(self.state.completion_tokens)))
        return {
            "outputs": [
                {
                    "token_ids": list(self.state.completion_tokens[:max_tokens]),
                    "text": "",
                }
            ]
        }

    def close(self) -> None:
        self.state.close_calls += 1


class IndependentVllmLoaderTest(unittest.TestCase):
    def _launch_environment(self) -> dict[str, str]:
        return {
            "GENERATION_DTYPE": "float16",
            "VLLM_TENSOR_PARALLEL_SIZE": "1",
            "VLLM_TRUST_REMOTE_CODE": "0",
            "VLLM_SCHEDULER_BUDGET_LEN": "10240",
            "VLLM_FIRST_TOKEN_TIMEOUT": "1000",
            "VLLM_MAX_LOG_LEN": "10",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.8",
            "VLLM_SERVER_ARGS_JSON": '["--custom-flag", "custom-value"]',
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
                    "_find_available_vllm_port",
                    return_value=8123,
                ) as find_port,
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
                    max_num_seqs=3,
                )

                find_port.assert_called_once_with(
                    "127.0.0.1", preferred_port=0
                )
                wait_ready.assert_called_once_with(
                    process,
                    health_url="http://127.0.0.1:8123/health",
                )
                popen.assert_called_once()
                command = popen.call_args.args[0]
                popen_kwargs = popen.call_args.kwargs

                self.assertEqual(
                    command[:3],
                    [
                        sys.executable,
                        "-m",
                        "vllm.entrypoints.api_server",
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
                    command[command.index("--scheduler-budget-len") + 1],
                    "10240",
                )
                self.assertEqual(
                    command[command.index("--first-token-timeout") + 1],
                    "1000",
                )
                self.assertEqual(
                    command[command.index("--max-log-len") + 1], "10"
                )
                self.assertEqual(
                    command[command.index("--gpu-memory-utilization") + 1],
                    "0.8",
                )
                self.assertEqual(
                    command[command.index("--swap-space") + 1], "0"
                )
                self.assertIn("--disable-log-requests", command)
                self.assertEqual(command[-2:], ["--custom-flag", "custom-value"])
                self.assertNotIn("--served-model-name", command)
                self.assertNotIn("--logits-processor-pattern", command)
                self.assertEqual(
                    popen_kwargs["start_new_session"], os.name == "posix"
                )

                self.assertIs(handle.process, process)
                self.assertEqual(handle.origin, "http://127.0.0.1:8123")
                self.assertEqual(
                    handle.generate_url, "http://127.0.0.1:8123/generate"
                )
                handle.close()
                handle.close()
                stop_group.assert_called_once_with(4321, timeout=2.0)
                kill_group.assert_called_once_with(4321, service_openai.signal.SIGTERM)

        self.assertTrue(handle.closed)
        self.assertEqual(process.terminate_calls, 0)
        self.assertEqual(len(process.wait_calls), 1)
        self.assertEqual(process.kill_calls, 0)

    def test_occupied_preferred_port_falls_back_to_free_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            occupied_port = int(occupied.getsockname()[1])
            selected_port = service_openai._find_available_vllm_port(
                "127.0.0.1",
                preferred_port=occupied_port,
            )

        self.assertNotEqual(selected_port, occupied_port)
        self.assertGreater(selected_port, 0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as verify:
            verify.bind(("127.0.0.1", selected_port))

    def test_generate_client_posts_json_to_simple_api(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                del args

            @staticmethod
            def read() -> bytes:
                return b'{"outputs":[{"token_ids":[1,2]}]}'

        with patch.object(
            service_openai,
            "urlopen",
            return_value=FakeResponse(),
        ) as open_mock:
            client = service_openai.VllmGenerateClient(
                "http://127.0.0.1:8123/generate",
                timeout=12.5,
            )
            result = client.generate(
                {"prompt": "router prompt", "stream": False, "max_tokens": 2}
            )

        request = open_mock.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8123/generate")
        self.assertEqual(open_mock.call_args.kwargs["timeout"], 12.5)
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {"prompt": "router prompt", "stream": False, "max_tokens": 2},
        )
        self.assertEqual(result["outputs"][0]["token_ids"], [1, 2])

    def test_model_output_debug_logging_is_bounded_and_opt_in(self) -> None:
        response = {"text": ["prompt" + "x" * 100]}
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(service_openai.logger, "info") as log_info,
        ):
            service_openai._log_vllm_raw_response(
                response,
                instance_id="instance",
                request_id="request",
            )
            service_openai._log_vllm_token_ids(
                [1, 2, 3],
                stage="raw_generation",
                instance_id="instance",
                request_id="request",
            )
        log_info.assert_not_called()

        with (
            patch.dict(
                os.environ,
                {
                    "SERVICE_OPENAI_LOG_MODEL_OUTPUT": "1",
                    "SERVICE_OPENAI_LOG_TOKEN_IDS": "1",
                    "SERVICE_OPENAI_LOG_PREVIEW_CHARS": "20",
                    "SERVICE_OPENAI_LOG_TOKEN_ITEMS": "2",
                },
                clear=True,
            ),
            patch.object(service_openai.logger, "info") as log_info,
        ):
            service_openai._log_vllm_raw_response(
                response,
                instance_id="instance",
                request_id="request",
            )
            service_openai._log_vllm_token_ids(
                [1, 2, 3],
                stage="raw_generation",
                instance_id="instance",
                request_id="request",
            )

        self.assertEqual(log_info.call_count, 2)
        raw_log_args = log_info.call_args_list[0].args
        self.assertIn("event=search.model_raw_output", raw_log_args[0])
        self.assertIn("<truncated chars=", raw_log_args[-1])
        token_log_args = log_info.call_args_list[1].args
        self.assertIn("event=search.model_token_ids", token_log_args[0])
        self.assertEqual(token_log_args[-1], "[1, 2]<truncated items=1>")

    def test_existing_single_card_kwargs_json_maps_to_cli(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VLLM_KWARGS_JSON": json.dumps(
                    {
                        "scheduler_budget_len": 50000,
                        "max_model_len": 65536,
                        "gpu_memory_utilization": 0.95,
                        "max_num_batched_tokens": 24576,
                        "tensor_parallel_size": 1,
                        "decode_tensor_parallel_size": 1,
                    }
                )
            },
            clear=True,
        ):
            command = service_openai._build_vllm_server_command(
                model_path=Path("/model"),
                tokenizer_path=Path("/model"),
                host="127.0.0.1",
                port=18000,
                vllm_overrides={},
            )

        self.assertEqual(
            command[command.index("--scheduler-budget-len") + 1], "50000"
        )
        self.assertEqual(
            command[command.index("--max-num-batched-tokens") + 1], "24576"
        )
        self.assertEqual(
            command[command.index("--tensor-parallel-size") + 1], "1"
        )
        self.assertEqual(
            command[command.index("--decode-tensor-parallel-size") + 1], "1"
        )

    def test_swap_space_json_float_zero_is_rendered_as_integer_zero(self) -> None:
        with patch.dict(
            os.environ,
            {"VLLM_KWARGS_JSON": '{"swap_space": 0.0}'},
            clear=True,
        ):
            command = service_openai._build_vllm_server_command(
                model_path=Path("/model"),
                tokenizer_path=Path("/model"),
                host="127.0.0.1",
                port=18000,
                vllm_overrides={},
            )

        self.assertEqual(command[command.index("--swap-space") + 1], "0")

    def test_nonzero_swap_space_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"VLLM_SWAP_SPACE": "1"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                service_openai.ServiceConfigurationError,
                "must be exactly 0",
            ):
                service_openai._build_vllm_server_command(
                    model_path=Path("/model"),
                    tokenizer_path=Path("/model"),
                    host="127.0.0.1",
                    port=18000,
                    vllm_overrides={},
                )

    def test_all_integer_server_options_drop_json_float_suffix(self) -> None:
        option_flags = {
            "tensor_parallel_size": "--tensor-parallel-size",
            "pipeline_parallel_size": "--pipeline-parallel-size",
            "max_num_seqs": "--max-num-seqs",
            "seed": "--seed",
            "max_model_len": "--max-model-len",
            "scheduler_budget_len": "--scheduler-budget-len",
            "max_num_batched_tokens": "--max-num-batched-tokens",
            "first_token_timeout": "--first-token-timeout",
            "max_log_len": "--max-log-len",
            "block_size": "--block-size",
            "decode_tensor_parallel_size": "--decode-tensor-parallel-size",
        }
        for option_name, flag in option_flags.items():
            with self.subTest(option_name=option_name):
                with patch.dict(
                    os.environ,
                    {"VLLM_KWARGS_JSON": json.dumps({option_name: 1.0})},
                    clear=True,
                ):
                    command = service_openai._build_vllm_server_command(
                        model_path=Path("/model"),
                        tokenizer_path=Path("/model"),
                        host="127.0.0.1",
                        port=18000,
                        vllm_overrides={},
                    )
                self.assertEqual(command[command.index(flag) + 1], "1")

    def test_fractional_integer_server_option_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"VLLM_KWARGS_JSON": '{"max_model_len": 8192.5}'},
            clear=True,
        ):
            with self.assertRaisesRegex(
                service_openai.ServiceConfigurationError,
                "max_model_len must be an integer",
            ):
                service_openai._build_vllm_server_command(
                    model_path=Path("/model"),
                    tokenizer_path=Path("/model"),
                    host="127.0.0.1",
                    port=18000,
                    vllm_overrides={},
                )

    def test_transformers_457_adapts_v5_extra_special_token_list(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            config_path = directory / "tokenizer_config.json"
            original = {
                "extra_special_tokens": ["<vision>", "<SK_L1_0>"],
                "additional_special_tokens": ["<SK_L1_0>", "<SK_L2_0>"],
            }
            config_path.write_text(
                json.dumps(original, ensure_ascii=False),
                encoding="utf-8",
            )

            kwargs = (
                service_openai._transformers_tokenizer_compatibility_kwargs(
                    directory,
                    transformers_version="4.57.1",
                )
            )
            persisted = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(kwargs["extra_special_tokens"], {})
        self.assertEqual(
            kwargs["additional_special_tokens"],
            ["<SK_L1_0>", "<SK_L2_0>", "<vision>"],
        )
        self.assertEqual(persisted, original)

    def test_transformers_v5_keeps_v5_extra_special_token_list(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "tokenizer_config.json").write_text(
                json.dumps({"extra_special_tokens": ["<SK_L1_0>"]}),
                encoding="utf-8",
            )
            kwargs = (
                service_openai._transformers_tokenizer_compatibility_kwargs(
                    directory,
                    transformers_version="5.0.0",
                )
            )

        self.assertEqual(kwargs, {})

    def test_local_tokenizer_load_applies_transformers_457_compatibility(self) -> None:
        state = _FakeDependencyState()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "extra_special_tokens": ["<SK_L1_0>"],
                        "additional_special_tokens": ["<SK_L2_0>"],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(sys.modules, _fake_dependency_modules(state)),
                patch.dict(os.environ, {}, clear=True),
            ):
                service_openai._load_tokenizer(directory)

        self.assertEqual(len(state.tokenizer_load_calls), 1)
        _, kwargs = state.tokenizer_load_calls[0]
        self.assertEqual(kwargs["extra_special_tokens"], {})
        self.assertEqual(
            kwargs["additional_special_tokens"],
            ["<SK_L2_0>", "<SK_L1_0>"],
        )

    def test_loader_requires_model_and_tokenizer_to_share_directory(self) -> None:
        with (
            tempfile.TemporaryDirectory() as model_raw,
            tempfile.TemporaryDirectory() as tokenizer_raw,
        ):
            model_dir = Path(model_raw)
            _write_router_bundle(model_dir)
            with self.assertRaisesRegex(
                service_openai.ServiceConfigurationError,
                "same resolved model directory",
            ):
                service_openai.load_vllm_model(model_dir, tokenizer_raw)

    def test_generate_response_text_falls_back_to_local_tokenizer(self) -> None:
        calls: list[tuple[str, bool]] = []

        class FakeTokenizer:
            @staticmethod
            def encode(text: str, *, add_special_tokens: bool) -> list[int]:
                calls.append((text, add_special_tokens))
                return [1, 2]

        result = service_openai._vllm_generate_token_ids(
            {"text": ["prompt<SK_L1_0><SK_L2_0>"]},
            tokenizer=FakeTokenizer(),
            prompt="prompt",
        )

        self.assertEqual(result, [1, 2])
        self.assertEqual(calls, [("<SK_L1_0><SK_L2_0>", False)])

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
                    "_find_available_vllm_port",
                    return_value=8123,
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
            origin="http://127.0.0.1:8123",
            host="127.0.0.1",
            port=8123,
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
            origin="http://127.0.0.1:8123",
            host="127.0.0.1",
            port=8123,
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
            {"VLLM_SERVER_ARGS_JSON": '["--tensor_parallel_size", "9"]'},
            clear=True,
        ):
            with self.assertRaisesRegex(
                service_openai.ServiceConfigurationError,
                "cannot override: --tensor-parallel-size",
            ):
                service_openai._load_vllm_server_extra_args()

    def test_server_extra_args_reject_protected_abbreviations(self) -> None:
        for payload, expected in (
            ('["--tensor-p", "9"]', "--tensor-p"),
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
    def test_model_directory_uses_platform_sfs_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            base = Path(raw_directory)
            model_object_id = "router-object-id"
            expected = (base / model_object_id / "model").resolve()
            expected.mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "MODEL_OBJECT_ID": model_object_id,
                    "MODEL_SFS": json.dumps({"sfsBasePath": str(base)}),
                    "MODEL_PATH": "/must/not/win",
                },
                clear=True,
            ):
                resolved = service_openai._resolve_model_directory(
                    Path("/component")
                )

        self.assertEqual(resolved, expected)

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
                "os",
                "pathlib",
                "signal",
                "socket",
                "subprocess",
                "sys",
                "threading",
                "time",
                "transformers",
                "typing",
                "urllib",
                "uuid",
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
        output_lines = [
            line for line in completed.stdout.splitlines() if line.strip()
        ]
        self.assertEqual(
            json.loads(output_lines[-1]),
            ["天气查询", "空气质量"],
        )


class OpenAIServiceEndToEndTest(unittest.TestCase):
    def test_calc_requests_execute_concurrently(self) -> None:
        runtime = service_openai.RetriverTest()
        runtime._loaded = True
        entered_lock = threading.Lock()
        both_entered = threading.Event()
        release = threading.Event()
        entered_queries: list[str] = []
        results: list[str] = []

        def blocking_search(
            query: str,
            *,
            request_id: str | None = None,
            requested_max_paths: int | None = None,
        ) -> list[str]:
            del request_id, requested_max_paths
            with entered_lock:
                entered_queries.append(query)
                if len(entered_queries) == 2:
                    both_entered.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release concurrent requests")
            return [query]

        runtime._search_names = blocking_search  # type: ignore[method-assign]

        def call(query: str) -> None:
            results.append(runtime.calc({"data": {"query": query}}))

        threads = [
            threading.Thread(target=call, args=(query,))
            for query in ("query-a", "query-b")
        ]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(
                both_entered.wait(timeout=1),
                "calc serialized requests before the model call",
            )
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertCountEqual(entered_queries, ["query-a", "query-b"])
        self.assertCountEqual(
            [json.loads(result) for result in results],
            [["query-a"], ["query-b"]],
        )
        self.assertEqual(runtime._active_calcs, 0)

    def test_close_waits_for_an_inflight_calc(self) -> None:
        runtime = service_openai.RetriverTest()
        runtime._loaded = True
        inference_entered = threading.Event()
        release_inference = threading.Event()
        close_finished = threading.Event()
        results: list[str] = []

        def blocking_search(
            query: str,
            *,
            request_id: str | None = None,
            requested_max_paths: int | None = None,
        ) -> list[str]:
            del request_id, requested_max_paths
            inference_entered.set()
            if not release_inference.wait(timeout=2):
                raise TimeoutError("test did not release the inflight request")
            return [query]

        runtime._search_names = blocking_search  # type: ignore[method-assign]
        calc_thread = threading.Thread(
            target=lambda: results.append(
                runtime.calc({"data": {"query": "inflight"}})
            )
        )

        def close_runtime() -> None:
            runtime.close()
            close_finished.set()

        calc_thread.start()
        self.assertTrue(inference_entered.wait(timeout=1))
        close_thread = threading.Thread(target=close_runtime)
        close_thread.start()
        try:
            self.assertFalse(close_finished.wait(timeout=0.1))
            self.assertTrue(runtime._loaded)
            self.assertEqual(runtime._active_calcs, 1)
        finally:
            release_inference.set()
            calc_thread.join(timeout=2)
            close_thread.join(timeout=2)

        self.assertFalse(calc_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertEqual([json.loads(result) for result in results], [["inflight"]])
        self.assertEqual(runtime._active_calcs, 0)
        self.assertFalse(runtime._loaded)

    def test_recoverable_calc_failure_releases_inflight_registration(self) -> None:
        runtime = service_openai.RetriverTest()
        runtime._loaded = True

        def failing_search(
            query: str,
            *,
            request_id: str | None = None,
            requested_max_paths: int | None = None,
        ) -> list[str]:
            del query, request_id, requested_max_paths
            raise RuntimeError("recoverable parse failure")

        runtime._search_names = failing_search  # type: ignore[method-assign]
        with patch.object(service_openai.logger, "error"):
            result = runtime.calc({"data": {"query": "broken"}})

        self.assertEqual(json.loads(result), [])
        self.assertEqual(runtime._active_calcs, 0)
        runtime.close()

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
                    "VllmGenerateClient",
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
        self.assertIsNone(runtime.vllm_client)
        self.assertIsNone(runtime.bundle)

    def test_fake_openai_exercises_load_calc_and_idempotent_close(self) -> None:
        state = _FakeDependencyState()
        _FakeGenerateClient.state = state
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
                patch.object(
                    service_openai,
                    "VllmGenerateClient",
                    _FakeGenerateClient,
                ),
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
                    self.assertEqual(runtime.backend, "vllm_http")
                    self.assertEqual(runtime.system_prompt, "测试专用检索提示")
                    self.assertEqual(runtime.output_budget, 5)
                    self.assertEqual(len(state.tokenizer_load_calls), 1)
                    self.assertEqual(len(state.client_init_calls), 1)
                    self.assertEqual(len(state.generate_calls), 1)
                    launcher.assert_called_once_with(
                        directory.resolve(),
                        directory.resolve(),
                        served_model_name="router-openai-test",
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
                        state.client_init_calls[0],
                        {
                            "generate_url": "http://127.0.0.1:8123/generate",
                            "timeout": 12.5,
                        },
                    )

                    request = state.generate_calls[0]
                    self.assertIn("测试专用检索提示", request["prompt"])
                    self.assertIn("出门前查天气和路线", request["prompt"])
                    self.assertFalse(request["stream"])
                    self.assertEqual(request["temperature"], 0.0)
                    self.assertEqual(request["max_tokens"], 5)
                    self.assertEqual(request["min_tokens"], 2)
                    self.assertFalse(request["skip_special_tokens"])
                    rendered_prompt = state.tokenizer_encode_calls[-1][0]
                    self.assertIn("测试专用检索提示", rendered_prompt)
                    self.assertIn("出门前查天气和路线", rendered_prompt)

                    top_one_result = runtime.calc(
                        {
                            "data": {
                                "query": "只查天气",
                                "top_k": 1,
                            }
                        }
                    )
                    self.assertEqual(json.loads(top_one_result), ["天气查询"])
                    self.assertEqual(len(state.generate_calls), 2)
                    self.assertEqual(
                        state.generate_calls[1]["max_tokens"],
                        2,
                    )
                    self.assertEqual(
                        state.generate_calls[1]["min_tokens"],
                        2,
                    )
                finally:
                    runtime.close()
                    runtime.close()

        self.assertEqual(state.close_calls, 1)
        self.assertEqual(server_handle.close_calls, 1)
        self.assertTrue(server_handle.closed)
        self.assertFalse(runtime._loaded)
        self.assertIsNone(runtime.vllm_server)
        self.assertIsNone(runtime.vllm_client)
        self.assertIsNone(runtime.bundle)

    def test_max_token_boundary_without_eos_is_accepted(self) -> None:
        state = _FakeDependencyState()
        state.completion_tokens = [1, 2, 9, 4, 5]
        _FakeGenerateClient.state = state
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
                patch.object(
                    service_openai,
                    "VllmGenerateClient",
                    _FakeGenerateClient,
                ),
            ):
                runtime = service_openai.RetriverTest()
                runtime.load()
                try:
                    result = runtime.calc(
                        {"data": {"query": "查天气", "top_k": 2}}
                    )
                    self.assertEqual(
                        json.loads(result), ["天气查询", "地图导航"]
                    )
                    self.assertTrue(runtime._loaded)
                    self.assertEqual(len(state.generate_calls), 1)
                    launcher.assert_called_once()
                finally:
                    runtime.close()

        self.assertEqual(state.close_calls, 1)
        self.assertEqual(server_handle.close_calls, 1)
        self.assertFalse(runtime._loaded)

    def test_invalid_model_suffix_recovers_complete_skill_path(self) -> None:
        state = _FakeDependencyState()
        # The first two IDs form the registered weather path. The final ID is
        # another level-2 token where the grammar expects EOS or a separator.
        state.completion_tokens = [1, 2, 5]
        _FakeGenerateClient.state = state
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
                    "VllmGenerateClient",
                    _FakeGenerateClient,
                ),
            ):
                runtime = service_openai.RetriverTest()
                runtime.load()
                try:
                    result = runtime.calc(
                        {"data": {"query": "查天气", "top_k": 2}}
                    )
                finally:
                    runtime.close()

        self.assertEqual(json.loads(result), ["天气查询"])
        self.assertEqual(state.close_calls, 1)
        self.assertEqual(server_handle.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
