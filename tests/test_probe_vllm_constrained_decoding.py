from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_vllm_constrained_decoding.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "probe_vllm_constrained_decoding", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def _write_probe_bundle(directory: Path) -> None:
    (directory / "config.json").write_text(
        json.dumps({"eos_token_id": 9}), encoding="utf-8"
    )
    (directory / "virtual_tokens.txt").write_text(
        "<SK_L1_0>\n", encoding="utf-8"
    )
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 7, "content": "<SK_L1_0>"}
                ],
                "model": {"vocab": {}},
            }
        ),
        encoding="utf-8",
    )


class ProbeVllmConstrainedDecodingTest(unittest.TestCase):
    def test_reports_exact_processor_constraint_as_llmgen_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_probe_bundle(directory)
            output = io.StringIO()
            with (
                patch.object(
                    probe,
                    "_post_json",
                    side_effect=[
                        {"outputs": [{"token_ids": [7, 7, 7]}]},
                        {"outputs": [{"token_ids": [7, 9]}]},
                    ],
                ),
                redirect_stdout(output),
            ):
                exit_code = probe.main(["--model-dir", str(directory)])

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["static_allowed_token_ids"]["enforced"])
        self.assertTrue(report["request_level_logits_processor"]["enforced"])
        self.assertTrue(report["llmgen_trie_compatible"])

    def test_static_constraint_does_not_imply_trie_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_probe_bundle(directory)
            output = io.StringIO()
            with (
                patch.object(
                    probe,
                    "_post_json",
                    side_effect=[
                        {"outputs": [{"token_ids": [7, 7, 7]}]},
                        probe.ProbeError(
                            "HTTP 500: logits_processors is unsupported"
                        ),
                    ],
                ),
                redirect_stdout(output),
            ):
                exit_code = probe.main(["--model-dir", str(directory)])

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertTrue(report["static_allowed_token_ids"]["enforced"])
        self.assertFalse(report["request_level_logits_processor"]["supported"])
        self.assertFalse(report["llmgen_trie_compatible"])

    def test_resolves_router_virtual_token_without_loading_model_libraries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _write_probe_bundle(directory)
            args = probe._parse_args(["--model-dir", str(directory)])
            resolved = probe._resolve_probe_tokens(args)

        self.assertEqual(resolved, (7, "<SK_L1_0>", 9))


if __name__ == "__main__":
    unittest.main()
