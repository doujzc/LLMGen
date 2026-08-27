from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace

import service


class SelfContainedServiceTest(unittest.TestCase):
    def test_imports_only_standard_library_and_vllm(self) -> None:
        source_path = Path(service.__file__).resolve()
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
                "threading",
                "time",
                "typing",
                "vllm",
            },
        )

    def test_single_copied_file_runs_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            copied_service = directory / "service.py"
            shutil.copy2(Path(service.__file__).resolve(), copied_service)
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

    def test_platform_sfs_model_path_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            base_path = Path(raw_directory)
            environment = {
                "MODEL_OBJECT_ID": "router-object",
                "MODEL_SFS": json.dumps({"sfsBasePath": str(base_path)}),
                "MODEL_PATH": "/unexpected/model/override",
            }
            with patch.dict(os.environ, environment, clear=True):
                resolved = service._resolve_model_directory(
                    base_path / "service-package"
                )

        self.assertEqual(
            resolved,
            (base_path / "router-object" / "model").resolve(),
        )


class MultiPathTokenTrieTest(unittest.TestCase):
    def test_enforces_active_unique_paths_and_separator(self) -> None:
        trie = service.MultiPathTokenTrie(
            ((1, 2), (1, 3), (4, 5)),
            eos_token_id=0,
            separator_token_ids=(9,),
            max_paths=2,
        )

        self.assertEqual(trie.allowed_next(()), (1, 4))
        self.assertEqual(trie.allowed_next((1,)), (2, 3))
        self.assertEqual(trie.allowed_next((1, 2)), (0, 9))
        self.assertEqual(trie.allowed_next((1, 2, 9, 1)), (3,))
        self.assertEqual(trie.allowed_next((1, 2, 9, 4, 5)), (0,))
        self.assertEqual(
            trie.parse_complete((1, 2, 9, 4, 5)),
            ((1, 2), (4, 5)),
        )
        self.assertEqual(trie.allowed_next((1, 2, 9, 1, 2)), ())


class CandidateBundleTest(unittest.TestCase):
    def test_loads_self_contained_decode_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "virtual_tokens.txt").write_text(
                "<SK_L1_0>\n<SK_L2_0>\n", encoding="utf-8"
            )
            payload = {
                "schema_version": 1,
                "num_levels": 2,
                "num_skills": 1,
                "num_paths": 1,
                "virtual_tokens": ["<SK_L1_0>", "<SK_L2_0>"],
                "skills": {"weather": {"name": "天气"}},
                "skill_to_code": {
                    "weather": {
                        "tokens": ["<SK_L1_0>", "<SK_L2_0>"],
                        "code_text": "<SK_L1_0><SK_L2_0>",
                    }
                },
                "paths": [
                    {
                        "tokens": ["<SK_L1_0>", "<SK_L2_0>"],
                        "skill_ids": ["weather"],
                    }
                ],
            }
            (directory / "skill_decode_map.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            bundle = service._load_candidate_bundle(directory)

        self.assertEqual(bundle.num_levels, 2)
        self.assertEqual(bundle.skills["weather"]["name"], "天气")
        self.assertEqual(
            bundle.token_paths[("<SK_L1_0>", "<SK_L2_0>")],
            ("weather",),
        )

    def test_rejects_malformed_candidate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "virtual_tokens.txt").write_text(
                "<SK_L1_0>\n<SK_L2_0>\n", encoding="utf-8"
            )
            payload = {
                "schema_version": 0,
                "num_levels": 2,
                "num_skills": 1,
                "num_paths": 1,
                "virtual_tokens": ["<SK_L1_0>", "<SK_L2_0>"],
                "skills": {"weather": {"name": "天气"}},
                "skill_to_code": {
                    "weather": {
                        "tokens": ["<SK_L1_0>", "<SK_L2_0>"],
                    }
                },
                "paths": [
                    {
                        "tokens": ["<SK_L1_0>", "<SK_L2_0>"],
                        "skill_ids": ["weather"],
                    }
                ],
            }
            decode_path = directory / "skill_decode_map.json"
            decode_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                service.ServiceConfigurationError, "unsupported.*schema"
            ):
                service._load_candidate_bundle(directory)

            payload["schema_version"] = 1
            payload["paths"][0]["skill_ids"] = ["weather", "weather"]
            decode_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                service.ServiceConfigurationError, "duplicate Skill IDs"
            ):
                service._load_candidate_bundle(directory)

            payload["paths"][0]["skill_ids"] = ["weather"]
            payload["supervision"] = {
                "phase": "retrieval",
                "target_counts": {"weather": 0},
            }
            decode_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                service.ServiceConfigurationError, "without train positives"
            ):
                service._load_candidate_bundle(directory)


class FullModelBundleTest(unittest.TestCase):
    def test_rejects_adapter_even_when_config_json_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "config.json").write_text("{}\n", encoding="utf-8")
            (directory / "adapter_config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (directory / "adapter_model.safetensors").write_bytes(b"adapter")

            with self.assertRaisesRegex(
                service.ServiceConfigurationError, "only a PEFT adapter"
            ):
                service._validate_full_model_bundle(directory)

    def test_accepts_valid_sharded_full_model_and_rejects_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "config.json").write_text("{}\n", encoding="utf-8")
            first_shard = directory / "model-00001-of-00002.safetensors"
            second_shard = directory / "model-00002-of-00002.safetensors"
            first_shard.write_bytes(b"first")
            second_shard.write_bytes(b"second")
            index_path = directory / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": first_shard.name,
                            "lm_head.weight": second_shard.name,
                        }
                    }
                ),
                encoding="utf-8",
            )

            service._validate_full_model_bundle(directory)
            second_shard.unlink()
            with self.assertRaisesRegex(
                service.ServiceConfigurationError, "missing, empty, or unsafe"
            ):
                service._validate_full_model_bundle(directory)


class RouterManifestTest(unittest.TestCase):
    def test_reads_only_runtime_prompt_settings(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest = {
                "schema_version": 999,
                "system_prompt": "自定义训练提示",
                "max_length": 128,
            }
            manifest_path = directory / "router_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            settings = service._load_router_settings(directory)
            self.assertEqual(settings.max_length, 128)
            self.assertEqual(settings.system_prompt, "自定义训练提示")

    def test_requires_manifest_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            with self.assertRaisesRegex(
                service.ServiceConfigurationError, "missing required prompt manifest"
            ):
                service._load_router_settings(directory)

            (directory / "router_manifest.json").write_text(
                json.dumps({"max_length": 128}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                service.ServiceConfigurationError, "system_prompt"
            ):
                service._load_router_settings(directory)


class ServiceProtocolTest(unittest.TestCase):
    def _loaded_service(self) -> service.RetriverTest:
        runtime = service.RetriverTest()
        runtime._loaded = True
        runtime.llm = object()
        runtime.bundle = object()  # type: ignore[assignment]
        runtime.default_top_k = 2
        runtime._search_names = lambda query: [  # type: ignore[method-assign]
            f"{query}-天气",
            "地图",
            "日历",
        ]
        return runtime

    def test_calc_matches_reference_request_and_json_string_response(self) -> None:
        runtime = self._loaded_service()

        result = runtime.calc({"data": {"query": "出行", "top_k": 1}})

        self.assertIsInstance(result, str)
        self.assertEqual(json.loads(result), ["出行-天气"])

    def test_calc_accepts_topk_alias_and_clamps_to_initialized_limit(self) -> None:
        runtime = self._loaded_service()

        result = runtime.calc({"data": {"query": "出行", "topk": 10}})

        self.assertEqual(json.loads(result), ["出行-天气", "地图"])

    def test_calc_returns_empty_list_for_blank_query_without_loading(self) -> None:
        runtime = service.RetriverTest()

        result = runtime.calc({"data": {"query": "  "}})

        self.assertEqual(json.loads(result), [])
        self.assertFalse(runtime._loaded)

    def test_search_decodes_vllm_token_ids_to_skill_names(self) -> None:
        class FakeTokenizer:
            chat_template = None

            @staticmethod
            def encode(text: str, add_special_tokens: bool = False) -> list[int]:
                del text, add_special_tokens
                return [40, 41]

        class FakeLLM:
            @staticmethod
            def generate(inputs, sampling_params, use_tqdm):
                self.assertEqual(inputs, [{"prompt_token_ids": [40, 41]}])
                self.assertIs(sampling_params, runtime.sampling_params)
                self.assertFalse(use_tqdm)
                generation = SimpleNamespace(token_ids=[1, 2, 9, 4, 5, 0])
                return [SimpleNamespace(outputs=[generation])]

        runtime = service.RetriverTest()
        runtime.llm = FakeLLM()
        runtime.tokenizer = FakeTokenizer()
        runtime.sampling_params = object()
        runtime.bundle = service.CandidateBundle(
            decode_map={},
            virtual_tokens=("a", "b", "c", "d"),
            skills={
                "weather": {"name": "天气"},
                "maps": {"name": "地图"},
            },
            token_paths={
                ("a", "b"): ("weather",),
                ("c", "d"): ("maps",),
            },
            num_levels=2,
        )
        runtime.trie = service.MultiPathTokenTrie(
            ((1, 2), (4, 5)),
            eos_token_id=0,
            separator_token_ids=(9,),
            max_paths=2,
        )
        runtime.path_skill_ids = {
            (1, 2): ("weather",),
            (4, 5): ("maps",),
        }
        runtime.max_input_length = 10

        self.assertEqual(runtime._search_names("出门"), ["天气", "地图"])

        class MissingEOSLLM:
            @staticmethod
            def generate(inputs, sampling_params, use_tqdm):
                del inputs, sampling_params, use_tqdm
                generation = SimpleNamespace(token_ids=[1, 2])
                return [SimpleNamespace(outputs=[generation])]

        runtime.llm = MissingEOSLLM()
        with self.assertRaisesRegex(RuntimeError, "did not emit EOS"):
            runtime._search_names("出门")

        class LengthLimitedLLM:
            @staticmethod
            def generate(inputs, sampling_params, use_tqdm):
                del inputs, sampling_params, use_tqdm
                generation = SimpleNamespace(
                    token_ids=[1, 2, 0], finish_reason="length"
                )
                return [SimpleNamespace(outputs=[generation])]

        runtime.llm = LengthLimitedLLM()
        with self.assertRaisesRegex(RuntimeError, "exhausted its token budget"):
            runtime._search_names("出门")

    def test_calc_and_close_do_not_race(self) -> None:
        runtime = service.RetriverTest()
        runtime._loaded = True
        runtime.default_top_k = 1
        search_started = threading.Event()
        release_search = threading.Event()
        close_completed = threading.Event()
        results: list[str] = []
        failures: list[BaseException] = []

        def slow_search(query: str) -> list[str]:
            del query
            search_started.set()
            if not release_search.wait(timeout=2):
                raise RuntimeError("test did not release search")
            return ["天气"]

        runtime._search_names = slow_search  # type: ignore[method-assign]

        def calculate() -> None:
            try:
                results.append(runtime.calc({"data": {"query": "查天气"}}))
            except BaseException as exc:  # pragma: no cover - assertion aid
                failures.append(exc)

        def close_runtime() -> None:
            runtime.close()
            close_completed.set()

        calc_thread = threading.Thread(target=calculate)
        close_thread = threading.Thread(target=close_runtime)
        calc_thread.start()
        self.assertTrue(search_started.wait(timeout=2))
        close_thread.start()
        self.assertFalse(close_completed.wait(timeout=0.05))
        release_search.set()
        calc_thread.join(timeout=2)
        close_thread.join(timeout=2)

        self.assertFalse(calc_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertFalse(failures)
        self.assertEqual(json.loads(results[0]), ["天气"])
        self.assertFalse(runtime._loaded)


class MockServiceEndToEndTest(unittest.TestCase):
    def test_load_calc_close_without_model_or_vllm(self) -> None:
        mock_responses = {
            "查天气": ["天气查询", "空气质量"],
            "*": ["默认技能"],
        }
        environment = {
            "MOCK_MODE": "1",
            "MOCK_RESPONSES_JSON": json.dumps(
                mock_responses, ensure_ascii=False
            ),
            "TOP_K": "2",
        }
        with patch.dict(os.environ, environment, clear=True):
            runtime = service.RetriverTest()
            runtime.load()
            try:
                self.assertTrue(runtime._loaded)
                self.assertEqual(runtime.backend, "mock")
                self.assertIsNone(runtime.llm)
                self.assertEqual(
                    json.loads(
                        runtime.calc(
                            {"data": {"query": "查天气", "top_k": 1}}
                        )
                    ),
                    ["天气查询"],
                )
                self.assertEqual(
                    json.loads(runtime.calc({"data": {"query": "未知请求"}})),
                    ["默认技能"],
                )
            finally:
                runtime.close()
            self.assertFalse(runtime._loaded)

    def test_service_entrypoint_runs_in_mock_mode(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "MOCK_MODE": "1",
                "MOCK_RESPONSES_JSON": json.dumps(
                    {"查天气": ["天气查询", "空气质量"]},
                    ensure_ascii=False,
                ),
                "TOP_K": "2",
            }
        )

        completed = subprocess.run(
            [sys.executable, str(Path(service.__file__).resolve())],
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


class VllmServiceEndToEndTest(unittest.TestCase):
    def test_fake_vllm_exercises_bundle_load_generation_and_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
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
            manifest = {
                "system_prompt": "测试专用检索提示",
                "max_length": 64,
            }
            (directory / "router_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )

            constructor_calls: list[dict[str, object]] = []
            shutdown_calls: list[bool] = []

            class FakeTokenizer:
                eos_token_id = 0
                chat_template = None

                @staticmethod
                def encode(
                    text: str, add_special_tokens: bool = False
                ) -> list[int]:
                    del add_special_tokens
                    atomic = {
                        "<SK_L1_0>": [1],
                        "<SK_L2_0>": [2],
                        "<SK_L1_1>": [4],
                        "<SK_L2_1>": [5],
                        "\n": [9],
                    }
                    return atomic.get(text, [40, 41])

            class FakeSamplingParams:
                def __init__(self, **kwargs: object) -> None:
                    self.__dict__.update(kwargs)

            class FakeLLM:
                def __init__(self, **kwargs: object) -> None:
                    constructor_calls.append(kwargs)
                    self.llm_engine = SimpleNamespace(
                        model_config=SimpleNamespace(max_model_len=64)
                    )

                @staticmethod
                def get_tokenizer() -> FakeTokenizer:
                    return FakeTokenizer()

                @staticmethod
                def generate(inputs, sampling_params, use_tqdm):
                    self.assertEqual(inputs, [{"prompt_token_ids": [40, 41]}])
                    self.assertFalse(use_tqdm)
                    self.assertIsInstance(sampling_params, FakeSamplingParams)
                    generation = SimpleNamespace(
                        token_ids=[1, 2, 9, 4, 5, 0]
                    )
                    return [SimpleNamespace(outputs=[generation])]

                @staticmethod
                def shutdown() -> None:
                    shutdown_calls.append(True)

            fake_vllm = ModuleType("vllm")
            fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
            fake_vllm.SamplingParams = (  # type: ignore[attr-defined]
                FakeSamplingParams
            )
            environment = {
                "MODEL_PATH": str(directory),
                "TOKENIZER_PATH": str(directory),
                "CANDIDATE_STATE_PATH": str(directory),
                "TOP_K": "2",
                "MAX_CODE_PATHS": "2",
            }

            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, {"vllm": fake_vllm}),
            ):
                runtime = service.RetriverTest()
                runtime.load()
                runtime.load()
                try:
                    result = runtime.calc(
                        {"data": {"query": "出门前查天气和路线", "top_k": 2}}
                    )
                    self.assertEqual(
                        json.loads(result), ["天气查询", "地图导航"]
                    )
                    self.assertEqual(runtime.backend, "vllm")
                    self.assertEqual(runtime.system_prompt, "测试专用检索提示")
                    self.assertEqual(len(constructor_calls), 1)
                    self.assertFalse(constructor_calls[0]["trust_remote_code"])
                    processors = runtime.sampling_params.logits_processors
                    self.assertIsInstance(
                        processors[0], service.TrieLogitsProcessor
                    )
                finally:
                    runtime.close()
                    runtime.close()

            self.assertEqual(shutdown_calls, [True])
            self.assertFalse(runtime._loaded)
            self.assertIsNone(runtime.llm)
            self.assertIsNone(runtime.bundle)


if __name__ == "__main__":
    unittest.main()
