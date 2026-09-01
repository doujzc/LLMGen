from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "scripts" / "router_pipeline.sh"
SKILLRET_SCRIPTS = REPO_ROOT / "scripts" / "skillret"
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


def _run_skillret_stage(
    script: str,
    *arguments: str,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SKILLRET_SCRIPTS / script), *arguments],
        cwd=REPO_ROOT,
        env=environment,
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


def _write_retrieval_stage_inputs(tmp_path: Path) -> dict[str, Path]:
    processed = tmp_path / "processed"
    index = tmp_path / "index"
    router_data = tmp_path / "router_data"
    router = tmp_path / "router"
    for directory in (processed, index, router_data, router / "memorization"):
        directory.mkdir(parents=True, exist_ok=True)

    for path in (
        processed / "catalog_train.jsonl",
        index / "virtual_tokens.txt",
        index / "train_codes.jsonl",
        index / "train_registry.json",
        router_data / "retrieval_train.jsonl",
        router_data / "retrieval_validation.jsonl",
        router_data / "retrieval_alignment_train.jsonl",
        router_data / "memorization_train.jsonl",
    ):
        path.write_text("{}\n", encoding="utf-8")
    return {
        "processed": processed,
        "index": index,
        "router_data": router_data,
        "router": router,
    }


def _read_invocations(path: Path) -> list[list[str]]:
    invocations: list[list[str]] = []
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "---":
            if current:
                invocations.append(current)
                current = []
        else:
            current.append(line)
    if current:
        invocations.append(current)
    return invocations


def test_split_retrieval_scripts_preserve_alignment_handoff(tmp_path: Path) -> None:
    paths = _write_retrieval_stage_inputs(tmp_path)
    (paths["router"] / "retrieval_alignment").mkdir()
    invocation = tmp_path / "router-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' '---' >> \"${FAKE_ROUTER_ARGS:?}\"\n"
        "printf '%s\\n' \"$@\" >> \"${FAKE_ROUTER_ARGS:?}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment()
    environment.update(
        {
            "FAKE_ROUTER_ARGS": str(invocation),
            "PYTHON": str(fake_python),
            "DEVICE": "cpu",
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(paths["processed"]),
            "INDEX_DIR": str(paths["index"]),
            "ROUTER_DATA_DIR": str(paths["router_data"]),
            "ROUTER_OUTPUT_DIR": str(paths["router"]),
            "ROUTER_FINETUNE_MODE": "full",
            "ROUTER_NUM_GPUS": "1",
            "ROUTER_DEEPSPEED_CONFIG": "configs/deepspeed_zero3.json",
            "ROUTER_PRECISION": "bf16",
            "ROUTER_GRADIENT_CHECKPOINTING": "1",
            "ROUTER_TRUST_REMOTE_CODE": "1",
            "ROUTER_ALIGNMENT_EPOCHS": "2",
            "ROUTER_ALIGNMENT_LR": "3e-5",
            "ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION": "0.15",
            "ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION": "0.05",
            "ROUTER_RESUME_ALIGNMENT": "alignment-checkpoint",
            "ROUTER_RESUME_RETRIEVAL": "retrieval-checkpoint",
            "ROUTER_EXTRA_ARGS": "--compat-test-flag",
        }
    )

    result = _run_skillret_stage(
        "06_train_retrieval.sh",
        "--retrieval-test-flag",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    alignment, retrieval = _read_invocations(invocation)
    assert alignment[0] == "scripts/train_router.py"
    assert alignment[alignment.index("--model-name-or-path") + 1] == str(
        paths["router"] / "memorization"
    )
    assert alignment[alignment.index("--phase-output-subdir") + 1] == (
        "retrieval_alignment"
    )
    assert "--retrieval-test-flag" not in alignment

    assert retrieval[0] == "scripts/train_router.py"
    assert retrieval[retrieval.index("--model-name-or-path") + 1] == str(
        paths["router"] / "retrieval_alignment"
    )
    assert retrieval[retrieval.index("--retrieval-alignment-replay-fraction") + 1] == (
        "0.15"
    )
    assert retrieval[retrieval.index("--retrieval-memorization-replay-fraction") + 1] == (
        "0.05"
    )
    assert retrieval[-1] == "--retrieval-test-flag"


def test_alignment_stage_runs_without_retrieval_dataset_inputs(tmp_path: Path) -> None:
    paths = _write_retrieval_stage_inputs(tmp_path)
    for name in ("retrieval_train.jsonl", "retrieval_validation.jsonl"):
        (paths["router_data"] / name).unlink()
    invocation = tmp_path / "alignment-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_ROUTER_ARGS:?}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment()
    environment.update(
        {
            "FAKE_ROUTER_ARGS": str(invocation),
            "PYTHON": str(fake_python),
            "DEVICE": "cpu",
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(paths["processed"]),
            "INDEX_DIR": str(paths["index"]),
            "ROUTER_DATA_DIR": str(paths["router_data"]),
            "ROUTER_OUTPUT_DIR": str(paths["router"]),
            "ROUTER_FINETUNE_MODE": "full",
            "ROUTER_NUM_GPUS": "1",
            "ROUTER_DEEPSPEED_CONFIG": "configs/deepspeed_zero3.json",
            "ROUTER_PRECISION": "bf16",
            "ROUTER_GRADIENT_CHECKPOINTING": "1",
            "ROUTER_TRUST_REMOTE_CODE": "1",
            "ROUTER_ALIGNMENT_EPOCHS": "2",
            "ROUTER_ALIGNMENT_LR": "3e-5",
            "ROUTER_RESUME_ALIGNMENT": "alignment-checkpoint",
            "ROUTER_EXTRA_ARGS": "--compat-test-flag",
        }
    )

    result = _run_skillret_stage(
        "06a_train_alignment.sh",
        "--alignment-test-flag",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    arguments = invocation.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--phase-output-subdir") + 1] == (
        "retrieval_alignment"
    )
    assert arguments[-1] == "--alignment-test-flag"


def test_single_device_full_training_accepts_all_optional_arrays_empty(
    tmp_path: Path,
) -> None:
    paths = _write_retrieval_stage_inputs(tmp_path)
    invocation = tmp_path / "retrieval-empty-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_ROUTER_ARGS:?}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment()
    environment.update(
        {
            "FAKE_ROUTER_ARGS": str(invocation),
            "PYTHON": str(fake_python),
            "DEVICE": "cpu",
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(paths["processed"]),
            "INDEX_DIR": str(paths["index"]),
            "ROUTER_DATA_DIR": str(paths["router_data"]),
            "ROUTER_OUTPUT_DIR": str(paths["router"]),
            "ROUTER_FINETUNE_MODE": "full",
            "ROUTER_NUM_GPUS": "1",
            "ROUTER_DEEPSPEED_CONFIG": "none",
            "ROUTER_PRECISION": "fp32",
            "ROUTER_GRADIENT_CHECKPOINTING": "0",
            "ROUTER_TRUST_REMOTE_CODE": "0",
            "ROUTER_ALIGNMENT_EPOCHS": "0",
            "ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION": "0",
            "ROUTER_RETRIEVAL_MEMORIZATION_REPLAY_FRACTION": "0",
            "ROUTER_RESUME_RETRIEVAL": "",
            "ROUTER_EXTRA_ARGS": "",
        }
    )

    result = _run_skillret_stage(
        "06_train_retrieval.sh",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    arguments = invocation.read_text(encoding="utf-8").splitlines()
    assert "--deepspeed" not in arguments
    assert "--bf16" not in arguments
    assert "--fp16" not in arguments
    assert "--resume-retrieval-from-checkpoint" not in arguments
    assert arguments[arguments.index("--model-name-or-path") + 1] == str(
        paths["router"] / "memorization"
    )


def test_alignment_training_accepts_all_optional_arrays_empty(
    tmp_path: Path,
) -> None:
    paths = _write_retrieval_stage_inputs(tmp_path)
    invocation = tmp_path / "alignment-empty-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_ROUTER_ARGS:?}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment()
    environment.update(
        {
            "FAKE_ROUTER_ARGS": str(invocation),
            "PYTHON": str(fake_python),
            "DEVICE": "cpu",
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(paths["processed"]),
            "INDEX_DIR": str(paths["index"]),
            "ROUTER_DATA_DIR": str(paths["router_data"]),
            "ROUTER_OUTPUT_DIR": str(paths["router"]),
            "ROUTER_FINETUNE_MODE": "full",
            "ROUTER_NUM_GPUS": "1",
            "ROUTER_DEEPSPEED_CONFIG": "none",
            "ROUTER_PRECISION": "fp32",
            "ROUTER_GRADIENT_CHECKPOINTING": "0",
            "ROUTER_TRUST_REMOTE_CODE": "0",
            "ROUTER_ALIGNMENT_EPOCHS": "2",
            "ROUTER_ALIGNMENT_LR": "3e-5",
            "ROUTER_RESUME_ALIGNMENT": "",
            "ROUTER_EXTRA_ARGS": "",
        }
    )

    result = _run_skillret_stage(
        "06a_train_alignment.sh",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    arguments = invocation.read_text(encoding="utf-8").splitlines()
    assert "--deepspeed" not in arguments
    assert "--bf16" not in arguments
    assert "--fp16" not in arguments
    assert "--resume-retrieval-from-checkpoint" not in arguments
    assert arguments[arguments.index("--model-name-or-path") + 1] == str(
        paths["router"] / "memorization"
    )


def test_alignment_only_router_data_uses_alignment_as_retrieval_passthrough(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    alignment_queries = tmp_path / "queries_alignment.jsonl"
    alignment_qrels = tmp_path / "qrels_alignment.jsonl"
    codes = tmp_path / "train_codes.jsonl"
    virtual_tokens = tmp_path / "virtual_tokens.txt"
    output_dir = tmp_path / "router_data"
    catalog.write_text(
        '{"skill_id": "only", "name": "Only", "description": "唯一能力"}\n',
        encoding="utf-8",
    )
    queries.write_text("", encoding="utf-8")
    qrels.write_text("", encoding="utf-8")
    alignment_queries.write_text(
        '{"id": "a1", "query": "使用唯一能力", "skill_ids": ["only"]}\n',
        encoding="utf-8",
    )
    alignment_qrels.write_text(
        '{"query_id": "a1", "skill_id": "only", "relevance": 1}\n',
        encoding="utf-8",
    )
    codes.write_text(
        '{"skill_id": "only", "tokens": ["<S0>"]}\n',
        encoding="utf-8",
    )
    virtual_tokens.write_text("<S0>\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_router_data.py"),
            "--catalog",
            str(catalog),
            "--queries",
            str(queries),
            "--qrels",
            str(qrels),
            "--alignment-queries",
            str(alignment_queries),
            "--alignment-qrels",
            str(alignment_qrels),
            "--codes",
            str(codes),
            "--virtual-tokens",
            str(virtual_tokens),
            "--output-dir",
            str(output_dir),
            "--skip-multiskill-retrieval",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    retrieval_rows = [
        json.loads(line)
        for line in (output_dir / "retrieval_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["alignment_only"] is True
    assert manifest["counts"]["retrieval"]["alignment_only_passthrough"] == 1
    assert retrieval_rows[0]["query_id"] == "a1"
    assert (output_dir / "retrieval_validation.jsonl").read_text(
        encoding="utf-8"
    ) == ""


def test_alignment_only_stage04_forwards_alignment_passthrough_flag(
    tmp_path: Path,
) -> None:
    paths = _write_retrieval_stage_inputs(tmp_path)
    for path in (
        paths["processed"] / "queries_train.jsonl",
        paths["processed"] / "qrels_train.jsonl",
        paths["processed"] / "queries_alignment.jsonl",
        paths["processed"] / "qrels_alignment.jsonl",
    ):
        path.write_text("{}\n", encoding="utf-8")
    invocation = tmp_path / "build-router-data-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_ROUTER_ARGS:?}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment()
    environment.update(
        {
            "FAKE_ROUTER_ARGS": str(invocation),
            "PYTHON": str(fake_python),
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(paths["processed"]),
            "INDEX_DIR": str(paths["index"]),
            "ROUTER_DATA_DIR": str(paths["router_data"]),
            "ROUTER_ALIGNMENT_ONLY": "1",
            "ROUTER_ALIGNMENT_EPOCHS": "0",
            "ROUTER_RETRIEVAL_ALIGNMENT_REPLAY_FRACTION": "0",
            "MEMORIZATION_VALIDATION_FRACTION": "0",
            "ROUTER_VALIDATION_FRACTION": "0",
            "ROUTER_DATA_SEED": "42",
        }
    )

    result = _run_skillret_stage(
        "04_build_router_data.sh",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    arguments = invocation.read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "scripts/build_router_data.py"
    assert "--skip-multiskill-retrieval" in arguments
    assert arguments[arguments.index("--alignment-queries") + 1] == str(
        paths["processed"] / "queries_alignment.jsonl"
    )


def test_alignment_only_evaluation_uses_explicit_model_and_alignment_queries(
    tmp_path: Path,
) -> None:
    paths = _write_retrieval_stage_inputs(tmp_path)
    eval_model = tmp_path / "attempt" / "retrieval"
    eval_model.mkdir(parents=True)
    for path in (
        paths["processed"] / "queries_alignment.jsonl",
        paths["processed"] / "qrels_alignment.jsonl",
    ):
        path.write_text("{}\n", encoding="utf-8")
    invocation = tmp_path / "evaluate-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_ROUTER_ARGS:?}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment()
    environment.update(
        {
            "FAKE_ROUTER_ARGS": str(invocation),
            "PYTHON": str(fake_python),
            "DEVICE": "cpu",
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(paths["processed"]),
            "INDEX_DIR": str(paths["index"]),
            "ROUTER_DATA_DIR": str(paths["router_data"]),
            "ROUTER_OUTPUT_DIR": str(paths["router"]),
            "ROUTER_EVAL_MODEL_DIR": str(eval_model),
            "ROUTER_MODEL": "base-model",
            "ROUTER_ALIGNMENT_ONLY": "1",
            "EVAL_PROTOCOL": "closedset",
            "QUERY_SET": "test",
            "EVAL_CUTOFFS": "1 5",
            "EVAL_BATCH_SIZE": "1",
            "EVAL_MAX_CODE_PATHS": "1",
            "EVAL_TOP_K": "1",
            "EVAL_DTYPE": "float32",
        }
    )

    result = _run_skillret_stage(
        "07_evaluate.sh",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    arguments = invocation.read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "scripts/infer_router.py"
    assert arguments[arguments.index("--model-name-or-path") + 1] == str(
        eval_model
    )
    assert arguments[arguments.index("--queries") + 1] == str(
        paths["processed"] / "queries_alignment.jsonl"
    )
    assert arguments[arguments.index("--qrels") + 1] == str(
        paths["processed"] / "qrels_alignment.jsonl"
    )


def test_evaluation_defaults_to_retrieval_model_for_multiskill_runs(
    tmp_path: Path,
) -> None:
    paths = _write_retrieval_stage_inputs(tmp_path)
    (paths["router"] / "retrieval").mkdir()
    for path in (
        paths["processed"] / "queries_test.jsonl",
        paths["processed"] / "qrels_test.jsonl",
    ):
        path.write_text("{}\n", encoding="utf-8")
    invocation = tmp_path / "evaluate-default-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_ROUTER_ARGS:?}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment()
    environment.update(
        {
            "FAKE_ROUTER_ARGS": str(invocation),
            "PYTHON": str(fake_python),
            "DEVICE": "cpu",
            "RUN_DIR": str(tmp_path),
            "PROCESSED_DIR": str(paths["processed"]),
            "INDEX_DIR": str(paths["index"]),
            "ROUTER_DATA_DIR": str(paths["router_data"]),
            "ROUTER_OUTPUT_DIR": str(paths["router"]),
            "ROUTER_EVAL_MODEL_DIR": "",
            "ROUTER_MODEL": "base-model",
            "ROUTER_ALIGNMENT_ONLY": "0",
            "EVAL_PROTOCOL": "closedset",
            "QUERY_SET": "test",
            "EVAL_CUTOFFS": "1",
            "EVAL_BATCH_SIZE": "1",
            "EVAL_MAX_CODE_PATHS": "1",
            "EVAL_TOP_K": "1",
            "EVAL_DTYPE": "float32",
        }
    )

    result = _run_skillret_stage(
        "07_evaluate.sh",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    arguments = invocation.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--model-name-or-path") + 1] == str(
        paths["router"] / "retrieval"
    )
