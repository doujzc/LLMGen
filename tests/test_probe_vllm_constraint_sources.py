from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_vllm_constraint_sources.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "probe_vllm_constraint_sources", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


class ProbeVllmConstraintSourcesTest(unittest.TestCase):
    def test_reports_context_and_enclosing_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_dir = Path(temporary_directory) / "vllm"
            package_dir.mkdir()
            (package_dir / "sampling_params.py").write_text(
                "class SamplingParams:\n"
                "    def __init__(self, logits_processors=None):\n"
                "        self.logits_processors = logits_processors\n",
                encoding="utf-8",
            )

            report, exit_code = probe.run(
                [
                    "--package-dir",
                    str(package_dir),
                    "--term",
                    "logits_processors",
                ]
            )

        self.assertEqual(exit_code, 0)
        scan = report["scan"]
        self.assertEqual(scan["term_totals"]["logits_processors"], 2)
        self.assertEqual(scan["matching_files"], ["sampling_params.py"])
        self.assertEqual(
            scan["matches"][0]["scope"],
            "SamplingParams.function __init__",
        )
        self.assertEqual(
            report["classification"]["sampling_params_definition"]["count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
