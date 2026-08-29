from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
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
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


class BenchmarkVllmRouterTest(unittest.TestCase):
    def test_loads_project_bundle_and_decodes_skill_names(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            model_dir = Path(raw_directory)
            tokens = ["<L1A>", "<L2A>", "<L1B>", "<L2B>"]
            (model_dir / "virtual_tokens.txt").write_text(
                "\n".join(tokens) + "\n", encoding="utf-8"
            )
            (model_dir / "router_manifest.json").write_text(
                json.dumps({"system_prompt": "router prompt"}),
                encoding="utf-8",
            )
            (model_dir / "skill_decode_map.json").write_text(
                json.dumps(
                    {
                        "virtual_tokens": tokens,
                        "num_levels": 2,
                        "skills": {
                            "weather": {"name": "天气查询"},
                            "maps": {"name": "地图导航"},
                        },
                        "paths": [
                            {
                                "tokens": tokens[:2],
                                "skill_ids": ["weather"],
                            },
                            {
                                "tokens": tokens[2:],
                                "skill_ids": ["maps"],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            bundle = benchmark._load_bundle(model_dir)
            decoded = benchmark._decode_skill_names(
                "<L1A><L2A>\n<L1B><L2B>", bundle, top_k=2
            )

            self.assertEqual(decoded, ["天气查询", "地图导航"])
            self.assertEqual(bundle.system_prompt, "router prompt")

    def test_extracts_prompt_prefixed_api_server_response(self) -> None:
        prompt = "System: router\nUser: query\nAssistant:"
        response = {"text": [prompt + "<L1A><L2A>"]}

        self.assertEqual(
            benchmark._extract_completion_text(response, prompt),
            "<L1A><L2A>",
        )

    def test_percentile_interpolates_small_samples(self) -> None:
        self.assertEqual(benchmark._percentile([10.0, 20.0], 0.5), 15.0)


if __name__ == "__main__":
    unittest.main()
