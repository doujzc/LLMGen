from __future__ import annotations

import asyncio
import ast
import io
import json
import logging
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
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]})
        + "\n",
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
                            token_ids=[1, 2, 9, 4, 5],
                            finish_reason="length",
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
                "sys",
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
        stdout_lines = [line for line in completed.stdout.splitlines() if line]
        log_lines = [
            line
            for line in stdout_lines
            if line.startswith(service_910b._LOG_MARKER)
        ]
        payload_lines = [
            line
            for line in stdout_lines
            if not line.startswith(service_910b._LOG_MARKER)
        ]
        self.assertTrue(log_lines)
        self.assertEqual(len(payload_lines), 1)
        self.assertEqual(
            json.loads(payload_lines[0]),
            ["天气查询", "空气质量"],
        )
        self.assertTrue(
            all(line.count(service_910b._LOG_MARKER) == 1 for line in log_lines)
        )
        stdout_logs = "\n".join(log_lines)
        self.assertIn(service_910b._LOG_MARKER, stdout_logs)
        for event in (
            "event=service.load_complete",
            "event=service.calc_complete",
            "event=service.close_complete",
        ):
            self.assertEqual(stdout_logs.count(event), 1)
        self.assertEqual(completed.stderr, "")

    def test_service_logger_is_bound_only_to_stdout(self) -> None:
        self.assertFalse(service_910b.logger.propagate)
        self.assertFalse(service_910b.logger.disabled)
        owned_handlers = [
            handler
            for handler in service_910b.logger.handlers
            if getattr(handler, service_910b._STDOUT_HANDLER_TAG, False)
        ]
        self.assertEqual(len(owned_handlers), 1)
        self.assertEqual(len(service_910b.logger.handlers), 1)
        handler = owned_handlers[0]
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.level, logging.NOTSET)
        owned_filters = [
            log_filter
            for log_filter in service_910b.logger.filters
            if getattr(log_filter, service_910b._LOG_FILTER_TAG, False)
        ]
        self.assertEqual(len(owned_filters), 1)

    def test_debug_logs_have_marker_trace_id_and_hidden_text(self) -> None:
        sensitive_query = "SENSITIVE-QUERY-SHOULD-STAY-HIDDEN"
        sensitive_result = "SENSITIVE-SKILL-SHOULD-STAY-HIDDEN"
        environment = {
            "MOCK_MODE": "1",
            "MOCK_RESPONSES_JSON": json.dumps(
                {sensitive_query: [sensitive_result]}
            ),
            "SERVICE_910B_LOG_LEVEL": "DEBUG",
            "TOP_K": "1",
        }
        runtime = service_910b.RetriverTest()

        with patch.dict(os.environ, environment, clear=True), self.assertLogs(
            service_910b.logger, level="DEBUG"
        ) as captured:
            runtime.load()
            result = runtime.calc(
                {"data": {"query": sensitive_query, "top_k": 1}}
            )
            runtime.close()

        self.assertEqual(json.loads(result), [sensitive_result])
        messages = [record.getMessage() for record in captured.records]
        self.assertTrue(messages)
        self.assertTrue(
            all(message.startswith(service_910b._LOG_MARKER) for message in messages)
        )
        combined = "\n".join(messages)
        self.assertNotIn(sensitive_query, combined)
        self.assertNotIn(sensitive_result, combined)
        for event in (
            "event=service.load_begin",
            "event=service.calc_begin",
            "event=search.begin",
            "event=service.calc_complete",
            "event=service.close_complete",
        ):
            self.assertIn(event, combined)
        calc_begin = next(
            message
            for message in messages
            if "event=service.calc_begin" in message
        )
        request_id = next(
            field.partition("=")[2]
            for field in calc_begin.split()
            if field.startswith("request_id=")
        )
        self.assertGreaterEqual(
            sum(f"request_id={request_id}" in message for message in messages),
            4,
        )

    def test_startup_info_explains_candidate_override_before_path_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            model_directory = Path(raw_directory).resolve()
            missing_candidate_directory = model_directory / "missing-candidates"
            environment = {
                "MODEL_PATH": str(model_directory),
                "SKILL_INDEX_PATH": str(missing_candidate_directory),
                "SERVICE_910B_LOG_LEVEL": "INFO",
            }
            with patch.dict(os.environ, environment, clear=True), self.assertLogs(
                service_910b.logger, level="INFO"
            ) as captured:
                runtime = service_910b.RetriverTest()
                with self.assertRaisesRegex(
                    service_910b.ServiceConfigurationError,
                    "candidate state directory does not exist",
                ):
                    runtime.load()

        messages = [record.getMessage() for record in captured.records]
        combined = "\n".join(messages)
        self.assertIn("event=service.process_context", combined)
        self.assertIn("event=service.deployment_environment", combined)
        self.assertIn("model_source=MODEL_PATH", combined)
        self.assertIn("tokenizer_source=model", combined)
        self.assertIn("candidate_source=SKILL_INDEX_PATH", combined)
        self.assertIn("event=service.paths_resolved", combined)
        self.assertIn(f"model={model_directory}", combined)
        self.assertIn(f"tokenizer={model_directory}", combined)
        self.assertIn(f"candidate={missing_candidate_directory}", combined)
        path_checks = [
            message
            for message in messages
            if "event=service.path_check" in message
        ]
        self.assertEqual(len(path_checks), 3)
        candidate_probe = next(
            message
            for message in messages
            if "role=candidate_decode_map" in message
        )
        self.assertIn("requirement=required", candidate_probe)
        self.assertIn("exists=False", candidate_probe)
        self.assertIn("is_file=False", candidate_probe)
        self.assertIn("event=service.path_invalid", combined)

    def test_startup_info_shows_sfs_model_directory_as_candidate_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            sfs_base = Path(raw_directory).resolve()
            model_object_id = "router-object-123"
            model_directory = sfs_base / model_object_id / "model"
            model_directory.mkdir(parents=True)
            environment = {
                "MODEL_OBJECT_ID": model_object_id,
                "MODEL_SFS": json.dumps({"sfsBasePath": str(sfs_base)}),
                "SERVICE_910B_LOG_LEVEL": "INFO",
            }
            with patch.dict(os.environ, environment, clear=True), self.assertLogs(
                service_910b.logger, level="INFO"
            ) as captured:
                runtime = service_910b.RetriverTest()
                with self.assertRaisesRegex(
                    service_910b.ServiceConfigurationError,
                    "no consolidated Hugging Face inference weights",
                ):
                    runtime.load()

        combined = "\n".join(
            record.getMessage() for record in captured.records
        )
        self.assertIn("model_source=MODEL_SFS+MODEL_OBJECT_ID", combined)
        self.assertIn("model_sfs_status=valid", combined)
        self.assertIn(f"model_sfs_base_path={sfs_base}", combined)
        self.assertIn("tokenizer_source=model", combined)
        self.assertIn("candidate_source=model", combined)
        self.assertIn(f"model={model_directory}", combined)
        self.assertIn(f"tokenizer={model_directory}", combined)
        self.assertIn(f"candidate={model_directory}", combined)
        self.assertIn("role=candidate_decode_map", combined)
        self.assertIn("role=candidate_virtual_tokens", combined)

    def test_debug_metadata_accepts_non_string_request_keys(self) -> None:
        environment = {
            "MOCK_MODE": "1",
            "MOCK_RESPONSES_JSON": json.dumps({"hello": ["Skill A"]}),
            "SERVICE_910B_LOG_LEVEL": "INFO",
            "TOP_K": "1",
        }
        runtime = service_910b.RetriverTest()

        with patch.dict(os.environ, environment, clear=True):
            try:
                self.assertEqual(
                    json.loads(
                        runtime.calc(
                            {"data": {"query": "hello", 1: "ignored"}}
                        )
                    ),
                    ["Skill A"],
                )
            finally:
                runtime.close()

    def test_marker_covers_traceback_lines_and_preserves_structured_exception(
        self,
    ) -> None:
        stream = io.StringIO()

        class CapturingHandler(logging.StreamHandler):
            def __init__(self) -> None:
                super().__init__(stream)
                self.records: list[logging.LogRecord] = []

            def emit(self, record: logging.LogRecord) -> None:
                self.records.append(record)
                super().emit(record)

        handler = CapturingHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        original_level = service_910b.logger.level
        original_handlers = list(service_910b.logger.handlers)
        original_propagate = service_910b.logger.propagate
        try:
            service_910b.logger.handlers = [handler]
            service_910b.logger.setLevel(logging.ERROR)
            service_910b.logger.propagate = False
            with patch.dict(
                os.environ,
                {"SERVICE_910B_LOG_TRACEBACKS": "1"},
                clear=True,
            ):
                try:
                    raise RuntimeError("fake marked traceback")
                except RuntimeError:
                    service_910b.logger.exception("event=test.traceback")
        finally:
            service_910b.logger.handlers = original_handlers
            service_910b.logger.setLevel(original_level)
            service_910b.logger.propagate = original_propagate

        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertGreater(len(lines), 1)
        self.assertTrue(
            all(line.startswith(service_910b._LOG_MARKER) for line in lines)
        )
        self.assertEqual(len(handler.records), 1)
        self.assertIsNotNone(handler.records[0].exc_info)

    def test_log_sanitizer_handles_cycles_nested_secrets_and_newlines(self) -> None:
        cycle: dict[str, object] = {}
        cycle["self"] = cycle
        payload = {
            "api_key": "TOP-SECRET-API-KEY",
            "query": "PRIVATE QUERY",
            "nested": [
                {
                    "client_secret_value": "TOP-SECRET-CLIENT-VALUE",
                    "note": "first line\nFORGED LOG LINE",
                }
            ],
            "cycle": cycle,
        }

        with patch.dict(os.environ, {}, clear=True):
            sanitized = service_910b._safe_log_value("payload", payload)
        rendered = repr(sanitized)

        self.assertNotIn("TOP-SECRET", rendered)
        self.assertNotIn("PRIVATE QUERY", rendered)
        self.assertNotIn("\nFORGED LOG LINE", rendered)
        self.assertIn("<redacted>", rendered)
        self.assertIn("<cycle type=", rendered)

    def test_traceback_is_private_by_default(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        original_level = service_910b.logger.level
        original_handlers = list(service_910b.logger.handlers)
        original_propagate = service_910b.logger.propagate
        try:
            service_910b.logger.handlers = [handler]
            service_910b.logger.setLevel(logging.ERROR)
            service_910b.logger.propagate = False
            with patch.dict(os.environ, {}, clear=True):
                try:
                    raise RuntimeError("SENSITIVE-QUERY-IN-TRACEBACK")
                except RuntimeError:
                    service_910b.logger.exception("event=test.private_traceback")
        finally:
            service_910b.logger.handlers = original_handlers
            service_910b.logger.setLevel(original_level)
            service_910b.logger.propagate = original_propagate

        output = stream.getvalue()
        self.assertIn(service_910b._LOG_MARKER, output)
        self.assertIn("event=test.private_traceback", output)
        self.assertNotIn("SENSITIVE-QUERY-IN-TRACEBACK", output)
        self.assertNotIn("Traceback", output)

    def test_token_ids_are_hidden_without_explicit_opt_in(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                service_910b._token_ids_log_value([40, 41]),
                "<hidden count=2>",
            )
        with patch.dict(
            os.environ, {"SERVICE_910B_LOG_TOKEN_IDS": "1"}, clear=True
        ):
            self.assertEqual(
                service_910b._token_ids_log_value([40, 41]), [40, 41]
            )

    def test_directory_debug_sample_is_best_effort(self) -> None:
        class EnumerationDenied:
            @staticmethod
            def iterdir() -> object:
                raise PermissionError("directory enumeration denied")

        sample = service_910b._directory_entries_log_value(EnumerationDenied())
        self.assertIn("diagnostic-error PermissionError", str(sample))

    def test_context_limit_diagnostics_ignore_explosive_engine_properties(self) -> None:
        class ExplosiveEngine:
            @property
            def llm_engine(self) -> object:
                raise RuntimeError("diagnostic llm_engine property was read")

            @property
            def engine(self) -> object:
                raise RuntimeError("diagnostic engine property was read")

        runtime = SimpleNamespace(
            engine=ExplosiveEngine(),
            engine_kwargs={},
            runtime_id="diagnostic-context-test",
        )
        self.assertIsNone(service_910b._custom_engine_max_model_len(runtime))

    def test_marker_filter_does_not_mutate_shared_parent_logger(self) -> None:
        parent = logging.getLogger("web_demo")
        original_child_level = service_910b.logger.level
        self.addCleanup(service_910b.logger.setLevel, original_child_level)
        parent_state = (
            parent.level,
            list(parent.handlers),
            list(parent.filters),
            parent.propagate,
            parent.disabled,
        )
        with patch.dict(
            os.environ, {"SERVICE_910B_LOG_LEVEL": "DEBUG"}, clear=True
        ):
            service_910b._configure_service_logging()
        parent_record = logging.LogRecord(
            name=parent.name,
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="event=parent.unrelated",
            args=(),
            exc_info=None,
        )
        child_record = logging.LogRecord(
            name=service_910b.logger.name,
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="event=child.service",
            args=(),
            exc_info=None,
        )

        for log_filter in service_910b.logger.filters:
            log_filter.filter(child_record)

        self.assertEqual(parent_record.getMessage(), "event=parent.unrelated")
        self.assertTrue(
            child_record.getMessage().startswith(service_910b._LOG_MARKER)
        )
        self.assertEqual(
            (
                parent.level,
                list(parent.handlers),
                list(parent.filters),
                parent.propagate,
                parent.disabled,
            ),
            parent_state,
        )


class CustomVllm910BServiceTest(unittest.TestCase):
    def test_dense_qwen35_2b_defaults_use_one_npu(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)

            with patch.dict(os.environ, {}, clear=True):
                service_defaults = service_910b._vllm_engine_arg_defaults(
                    model_path=directory
                )
                (directory / "config.json").write_text(
                    json.dumps({}), encoding="utf-8"
                )
                fallback_defaults = service_910b._vllm_engine_arg_defaults(
                    model_path=directory
                )
                direct_loader_defaults = (
                    service_910b._build_custom_engine_kwargs(
                        model_path=str(directory),
                        tokenizer_path=str(directory),
                        dtype="bfloat16",
                        trust_remote_code=True,
                        engine_role=object(),
                        options={},
                    )
                )

        for engine_kwargs in (
            service_defaults,
            fallback_defaults,
            direct_loader_defaults,
        ):
            self.assertEqual(
                engine_kwargs["architectures"],
                "Qwen3_5ForConditionalGeneration_OnlyLLM",
            )
            for key in (
                "pipeline_parallel_size",
                "tensor_parallel_size",
                "data_parallel_size",
                "context_parallel_size",
                "decode_pipeline_parallel_size",
                "decode_tensor_parallel_size",
                "decode_data_parallel_size",
                "decode_context_parallel_size",
            ):
                self.assertEqual(engine_kwargs[key], 1, key)
            self.assertFalse(engine_kwargs["enable_expert_parallel"])
            self.assertFalse(engine_kwargs["decode_enable_expert_parallel"])

    def test_dense_profile_preserves_explicit_architecture_and_tp_override(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_router_bundle(directory)
            (directory / "config.json").write_text(
                json.dumps({"architectures": ["DenseOverrideArchitecture"]}),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"VLLM_TENSOR_PARALLEL_SIZE": "2"},
                clear=True,
            ):
                defaults = service_910b._vllm_engine_arg_defaults(
                    model_path=directory
                )

        self.assertEqual(
            defaults["architectures"], "DenseOverrideArchitecture"
        )
        self.assertEqual(defaults["tensor_parallel_size"], 2)
        self.assertEqual(defaults["decode_tensor_parallel_size"], 2)

    def test_info_logging_does_not_read_diagnostic_tokenizer_properties(
        self,
    ) -> None:
        state = _FakeDependencyState()
        fake_modules = _fake_dependency_modules(state)

        class DiagnosticExplodesTokenizer:
            @property
            def vocab_size(self) -> int:
                raise RuntimeError("diagnostic vocab_size property was read")

            @property
            def chat_template(self) -> str:
                raise RuntimeError("diagnostic chat_template property was read")

        with tempfile.TemporaryDirectory() as raw_directory:
            environment = {
                "SERVICE_910B_LOG_LEVEL": "INFO",
                "VLLM_HEALTH_CHECK_TIMEOUT": "1",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, fake_modules),
                patch.object(
                    service_910b,
                    "_load_chat_template_tokenizer",
                    return_value=DiagnosticExplodesTokenizer(),
                ),
            ):
                runtime = service_910b.load_vllm_model(
                    model_path=raw_directory,
                    tokenizer_path=raw_directory,
                    vllm_kwargs={"health_check_timeout": 1},
                )
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

    def test_info_logging_does_not_inspect_nonfinal_generation_frames(self) -> None:
        class DiagnosticFrame:
            @property
            def outputs(self) -> object:
                raise RuntimeError("diagnostic outputs property was read")

        final_frame = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 0])])

        class FakeEngine:
            @staticmethod
            def generate(**kwargs: object) -> object:
                del kwargs

                async def frames() -> object:
                    yield DiagnosticFrame()
                    yield final_frame

                return frames()

        original_level = service_910b.logger.level
        try:
            service_910b.logger.setLevel(logging.INFO)
            result = asyncio.run(
                service_910b._generate_on_910b_loop(
                    engine=FakeEngine(),
                    prompt="prompt",
                    prompt_token_ids=[40, 41],
                    sampling_params=object(),
                    request_id="diagnostic-frame-test",
                )
            )
        finally:
            service_910b.logger.setLevel(original_level)

        self.assertIs(result, final_frame)

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
                self.assertLogs(service_910b.logger, level="DEBUG") as debug_logs,
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
                    self.assertEqual(runtime.max_input_length, 59)

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
                        engine_kwargs["architectures"],
                        "Qwen3_5ForConditionalGeneration_OnlyLLM",
                    )
                    self.assertEqual(engine_kwargs["dtype"], "float16")
                    self.assertIs(
                        engine_kwargs["engine_role"], state.engine_role_m
                    )
                    self.assertEqual(engine_kwargs["tensor_parallel_size"], 1)
                    self.assertEqual(
                        engine_kwargs["decode_tensor_parallel_size"], 1
                    )
                    for key in (
                        "pipeline_parallel_size",
                        "data_parallel_size",
                        "context_parallel_size",
                        "decode_pipeline_parallel_size",
                        "decode_data_parallel_size",
                        "decode_context_parallel_size",
                    ):
                        self.assertEqual(engine_kwargs[key], 1, key)
                    self.assertFalse(engine_kwargs["enable_expert_parallel"])
                    self.assertFalse(
                        engine_kwargs["decode_enable_expert_parallel"]
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
                    self.assertEqual(sampling_kwargs["max_tokens"], 5)
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

            debug_messages = [
                record.getMessage() for record in debug_logs.records
            ]
            self.assertTrue(
                all(
                    message.startswith(service_910b._LOG_MARKER)
                    for message in debug_messages
                )
            )
            debug_output = "\n".join(debug_messages)
            for event in (
                "event=runtime.dependencies_loaded",
                "event=engine.load_model_begin",
                "event=engine.health_poll_complete",
                "event=engine.generate_frame",
                "event=search.paths_decoded",
                "event=runtime.close_complete",
            ):
                self.assertIn(event, debug_output)
            self.assertIn("token_ids=<hidden count=2>", debug_output)
            self.assertNotIn("[40, 41]", debug_output)
            calc_begin = next(
                message
                for message in debug_messages
                if "event=service.calc_begin" in message
            )
            logged_request_id = next(
                field.partition("=")[2]
                for field in calc_begin.split()
                if field.startswith("request_id=")
            )
            self.assertEqual(
                str(generate_call["request_id"]), logged_request_id
            )

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
