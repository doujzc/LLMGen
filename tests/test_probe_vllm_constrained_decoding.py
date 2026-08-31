from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
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


class ProbeVllmConstrainedDecodingTest(unittest.TestCase):
    def test_reports_exact_processor_constraint_as_llmgen_compatible(self) -> None:
        output = io.StringIO()
        with (
            patch.object(
                probe,
                "_post_json",
                side_effect=[
                    {"outputs": [{"token_ids": [0, 0, 0]}]},
                    {"text": "例如，", "output_tokens": [0, 1]},
                ],
            ),
            redirect_stdout(output),
        ):
            exit_code = probe.main([])

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["static_allowed_token_ids"]["enforced"])
        self.assertTrue(report["request_level_logits_processor"]["enforced"])
        self.assertTrue(report["llmgen_trie_compatible"])

    def test_static_constraint_does_not_imply_trie_compatibility(self) -> None:
        output = io.StringIO()
        with (
            patch.object(
                probe,
                "_post_json",
                side_effect=[
                    {"outputs": [{"token_ids": [0, 0, 0]}]},
                    probe.ProbeError(
                        "HTTP 500: logits_processors is unsupported"
                    ),
                ],
            ),
            redirect_stdout(output),
        ):
            exit_code = probe.main([])

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertTrue(report["static_allowed_token_ids"]["enforced"])
        self.assertFalse(report["request_level_logits_processor"]["supported"])
        self.assertFalse(report["llmgen_trie_compatible"])

    def test_defaults_require_no_model_or_tokenizer_files(self) -> None:
        args = probe._parse_args([])

        self.assertEqual(args.force_token_id, 0)
        self.assertEqual(args.alternate_token_id, 1)
        self.assertIsNone(args.force_token_text)
        self.assertIsNone(args.alternate_token_text)


if __name__ == "__main__":
    unittest.main()
