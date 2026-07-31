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
    "ROUTER_EXPORT_MODEL_DIR",
    "ROUTER_CHECKPOINT_EXPORT_DIR",
    "ROUTER_CHECKPOINT_TOKENIZER_SOURCE",
    "ROUTER_CHECKPOINT_TEMPLATE_MANIFEST",
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


def test_export_command_uses_environment_model_source(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    index = tmp_path / "index"
    router_data = tmp_path / "router_data"
    model_dir = tmp_path / "custom-model"
    for directory in (processed, index, router_data, model_dir):
        directory.mkdir(parents=True)
    (processed / "catalog_train.jsonl").write_text(
        '{"skill_id":"s1","name":"Skill One"}\n',
        encoding="utf-8",
    )
    (index / "train_codes.jsonl").write_text(
        (
            '{"skill_id":"s1","indices":[0,0],'
            '"tokens":["<L1_0>","<L2_0>"]}\n'
        ),
        encoding="utf-8",
    )
    (index / "train_registry.json").write_text(
        '{"num_levels":2,"buckets":{"0/0":["s1"]}}\n',
        encoding="utf-8",
    )
    (index / "virtual_tokens.txt").write_text(
        "<L1_0>\n<L2_0>\n",
        encoding="utf-8",
    )
    (router_data / "retrieval_train.jsonl").write_text(
        '{"phase":"retrieval","target_skill_ids":["s1"]}\n',
        encoding="utf-8",
    )
    (model_dir / "router_manifest.json").write_text(
        '{"phase":"retrieval"}\n',
        encoding="utf-8",
    )
    environment = _environment()
    environment.update(
        {
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(processed),
            "INDEX_DIR": str(index),
            "ROUTER_DATA_DIR": str(router_data),
            "ROUTER_OUTPUT_DIR": str(tmp_path / "router"),
            "ROUTER_EXPORT_MODEL_DIR": str(model_dir),
        }
    )

    result = _run("light", "export-web", environment=environment)

    assert result.returncode == 0, result.stderr
    assert f"[10] export Web bundle from {model_dir}" in result.stdout
    assert (model_dir / "skill_decode_map.json").is_file()
    assert (model_dir / "virtual_tokens.txt").is_file()


def test_export_command_materializes_environment_checkpoint(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    index = tmp_path / "index"
    router_data = tmp_path / "router_data"
    router = tmp_path / "router"
    checkpoint = router / "retrieval" / "checkpoint-50"
    tokenizer_source = router / "retrieval_alignment"
    output = tmp_path / "exports" / "checkpoint-50"
    for directory in (
        processed,
        index,
        router_data,
        checkpoint,
        tokenizer_source,
    ):
        directory.mkdir(parents=True)
    required_files = (
        processed / "catalog_train.jsonl",
        index / "train_codes.jsonl",
        index / "train_registry.json",
        index / "virtual_tokens.txt",
        router_data / "retrieval_train.jsonl",
        router_data / "retrieval_validation.jsonl",
        router_data / "retrieval_alignment_train.jsonl",
        router_data / "memorization_train.jsonl",
        checkpoint / "trainer_state.json",
        tokenizer_source / "tokenizer_config.json",
        tokenizer_source / "router_manifest.json",
    )
    for path in required_files:
        path.write_text("{}\n", encoding="utf-8")
    invocation = tmp_path / "export-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_EXPORT_ARGS:?}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment()
    environment.update(
        {
            "FAKE_EXPORT_ARGS": str(invocation),
            "PYTHON": str(fake_python),
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(processed),
            "INDEX_DIR": str(index),
            "ROUTER_DATA_DIR": str(router_data),
            "ROUTER_OUTPUT_DIR": str(router),
            "ROUTER_EXPORT_MODEL_DIR": str(checkpoint),
            "ROUTER_CHECKPOINT_EXPORT_DIR": str(output),
            "ROUTER_CHECKPOINT_TOKENIZER_SOURCE": str(tokenizer_source),
            "ROUTER_CHECKPOINT_TEMPLATE_MANIFEST": str(
                tokenizer_source / "router_manifest.json"
            ),
        }
    )

    result = _run("light", "export-web", environment=environment)

    assert result.returncode == 0, result.stderr
    assert f"[10] export Web bundle from {checkpoint}" in result.stdout
    arguments = invocation.read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "scripts/export_router_bundle.py"
    assert arguments[arguments.index("--model-dir") + 1] == str(checkpoint)
    assert arguments[arguments.index("--output-dir") + 1] == str(output)
    assert arguments[
        arguments.index("--alignment-replay-fraction") + 1
    ] == "0.15"
    assert arguments[
        arguments.index("--memorization-replay-fraction") + 1
    ] == "0.05"
