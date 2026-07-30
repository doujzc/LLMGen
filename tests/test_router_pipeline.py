from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "scripts" / "router_pipeline.sh"
OVERRIDE_NAMES = (
    "SKILLRET_ROOT",
    "SKILLRET_CONFIG",
    "DATASET_NAME",
    "DATASET_DIR",
    "RUN_DIR",
    "PROCESSED_DIR",
    "EMBEDDING_DIR",
    "STAGE1_DIR",
    "INDEX_DIR",
    "ROUTER_DATA_DIR",
    "ROUTER_OUTPUT_DIR",
    "BRANCHING_FACTORS",
    "ROUTER_FINETUNE_MODE",
    "ROUTER_NUM_GPUS",
    "ROUTER_DEEPSPEED_CONFIG",
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in OVERRIDE_NAMES:
        environment.pop(name, None)
    environment["PYTHON"] = sys.executable
    return environment


def _run(*arguments: str, environment: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(PIPELINE), *arguments],
        cwd=REPO_ROOT,
        env=environment or _environment(),
        text=True,
        capture_output=True,
        check=False,
    )


def _paths(dataset: str) -> dict[str, str]:
    result = _run(dataset, "paths")
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_pipeline_selects_only_dataset_specific_config_and_paths() -> None:
    clawhub = _paths("clawhub")
    light = _paths("light")
    assert clawhub["dataset_dir"] == "data/clawhub_training/final"
    assert light["dataset_dir"] == "data_light/final"
    assert clawhub["dataset_name"] == "clawhub-top1000-router-v1"
    assert light["dataset_name"] == "light301-router-v3"
    assert clawhub["run_dir"].startswith("runs/clawhub-top1000-")
    assert light["run_dir"].startswith("runs/light301-")
    assert clawhub["branching_factors"] == "128 128"
    assert light["branching_factors"] == "32 16"
    for paths in (clawhub, light):
        assert paths["router_finetune_mode"] == "full"
        assert paths["router_num_gpus"] == "4"
        assert paths["router_deepspeed_config"] == "configs/deepspeed_zero3.json"


def test_dataset_argument_overrides_inherited_config_selection() -> None:
    environment = _environment()
    environment["SKILLRET_CONFIG"] = "configs/clawhub.env"
    result = _run("light", "paths", environment=environment)
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert values["dataset"] == "light"
    assert values["config"].endswith("/configs/light.env")


def test_pipeline_rejects_unknown_dataset() -> None:
    result = _run("unknown", "paths")
    assert result.returncode == 2
    assert "Unknown dataset" in result.stderr


def test_web_command_uses_selected_run_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "router" / "retrieval"
    model_dir.mkdir(parents=True)
    environment = _environment()
    environment.update(
        {
            "RUN_DIR": str(tmp_path),
            "PYTHON": "/bin/echo",
            "DEVICE": "cpu",
        }
    )
    result = _run("light", "web", "--port", "9090", environment=environment)
    assert result.returncode == 0, result.stderr
    assert "-m web_server.server" in result.stdout
    assert f"--model-dir {model_dir}" in result.stdout
    assert "--device cpu" in result.stdout
    assert "--port 9090" in result.stdout
