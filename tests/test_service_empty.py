from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import service_empty


class EmptyServiceTest(unittest.TestCase):
    def test_lifecycle_always_returns_an_empty_json_list(self) -> None:
        service = service_empty.RetriverTest()

        service.load()
        for request in (
            None,
            {},
            {"data": {"query": "任意请求", "top_k": 2}},
            object(),
        ):
            self.assertEqual(json.loads(service.calc(request)), [])
        service.close()
        service.close()

    def test_single_copied_file_runs_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            copied_service = directory / "service_empty.py"
            shutil.copy2(Path(service_empty.__file__).resolve(), copied_service)
            completed = subprocess.run(
                [sys.executable, "-I", str(copied_service)],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(json.loads(completed.stdout), [])


if __name__ == "__main__":
    unittest.main()
