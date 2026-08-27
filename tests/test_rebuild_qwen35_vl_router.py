from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rebuild_qwen35_vl_router.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "rebuild_qwen35_vl_router", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
rebuild_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rebuild_script)


def _base_config() -> dict[str, object]:
    text_config = {
        name: f"value-{name}"
        for name in rebuild_script._STRUCTURAL_TEXT_FIELDS
    }
    text_config["hidden_size"] = 2048
    text_config["vocab_size"] = 248064
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": text_config,
        "vision_config": {"out_hidden_size": 2048, "spatial_merge_size": 2},
        "image_token_id": 9,
        "video_token_id": 10,
        "vision_start_token_id": 7,
        "vision_end_token_id": 8,
    }


def _router_config() -> dict[str, object]:
    config = dict(_base_config()["text_config"])
    config.update(
        {
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_text",
            "vocab_size": 248125,
        }
    )
    return config


class RebuildQwen35VlRouterTest(unittest.TestCase):
    def test_source_validation_accepts_vocab_growth_only(self) -> None:
        rebuild_script._validate_source_configs(
            _base_config(), _router_config()
        )

        incompatible = _router_config()
        incompatible["hidden_size"] = 4096
        with self.assertRaisesRegex(
            rebuild_script.ReconstructionError, "hidden_size"
        ):
            rebuild_script._validate_source_configs(
                _base_config(), incompatible
            )

    def test_attach_replaces_language_modules_and_preserves_visual(self) -> None:
        base_language = object()
        base_visual = object()
        base_head = object()
        full_config = SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            text_config=SimpleNamespace(vocab_size=10),
        )
        full_model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=base_language,
                visual=base_visual,
                config=full_config,
            ),
            lm_head=base_head,
            config=full_config,
            vocab_size=10,
        )
        router_config = SimpleNamespace(vocab_size=12)
        router_language = SimpleNamespace(config=router_config)
        router_head = object()
        router_generation = object()
        router_model = SimpleNamespace(
            model=router_language,
            lm_head=router_head,
            config=router_config,
            generation_config=router_generation,
        )

        detached = rebuild_script._detach_base_language_modules(full_model)
        self.assertEqual(detached, (base_language, base_head))
        self.assertIsNone(full_model.model.language_model)
        self.assertIsNone(full_model.lm_head)

        rebuild_script._attach_router_language_modules(
            full_model, router_model
        )
        self.assertIs(full_model.model.visual, base_visual)
        self.assertIs(full_model.model.language_model, router_language)
        self.assertIs(full_model.lm_head, router_head)
        self.assertIs(full_model.config.text_config, router_config)
        self.assertEqual(
            full_model.config.architectures,
            ["Qwen3_5ForConditionalGeneration"],
        )
        self.assertEqual(full_model.vocab_size, 12)
        self.assertIs(full_model.generation_config, router_generation)

    def test_router_artifacts_are_preserved_without_overwriting_model_config(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            router = root / "router"
            output = root / "output"
            router.mkdir()
            output.mkdir()
            for name in (
                "router_manifest.json",
                "skill_decode_map.json",
                "virtual_tokens.txt",
                "tokenizer.json",
                "config.json",
                "model.safetensors",
            ):
                (router / name).write_text(name, encoding="utf-8")
            (output / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
            )

            copied = rebuild_script._copy_router_files(router, output)

            self.assertEqual(
                set(copied),
                {
                    "router_manifest.json",
                    "skill_decode_map.json",
                    "virtual_tokens.txt",
                    "tokenizer.json",
                },
            )
            self.assertEqual(
                json.loads((output / "config.json").read_text(encoding="utf-8")),
                {"model_type": "qwen3_5"},
            )
            self.assertFalse((output / "model.safetensors").exists())

    def test_base_processor_metadata_is_copied_without_base_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            base = root / "base"
            output = root / "output"
            base.mkdir()
            output.mkdir()
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "video_preprocessor_config.json",
                "tokenizer.json",
                "config.json",
                "model.safetensors",
            ):
                (base / name).write_text(name, encoding="utf-8")

            copied = rebuild_script._copy_base_processor_files(base, output)

            self.assertEqual(
                copied,
                [
                    "preprocessor_config.json",
                    "processor_config.json",
                    "video_preprocessor_config.json",
                ],
            )
            self.assertFalse((output / "tokenizer.json").exists())
            self.assertFalse((output / "config.json").exists())
            self.assertFalse((output / "model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
