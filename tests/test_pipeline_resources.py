from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmgen.pipeline.resources import (
    ResourceResolutionError,
    base_model_fingerprint,
    collect_run_provenance,
    deepspeed_config_fingerprint,
    probe_runtime_environment,
    resolve_runtime_resources,
    validate_runtime_device_request,
    verify_run_provenance,
    visible_devices_environment,
)


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def get_device_properties(self, index: int):
        return SimpleNamespace(name=f"GPU-{index}", total_memory=16_000 + index)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return (9, index)


class _FakeNpu:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int):
        return SimpleNamespace(name=f"NPU-{index}", total_memory=32_000)


def _runtime_probe_payload(
    executable: Path,
    *,
    python_version: str = "9.8.7",
    cuda_count: int = 0,
    npu_count: int = 0,
) -> dict[str, object]:
    def accelerator(kind: str, count: int) -> dict[str, object]:
        return {
            "available": count > 0,
            "device_count": count,
            "topology": [
                {"index": index, "name": f"{kind.upper()}-{index}"}
                for index in range(count)
            ],
        }

    return {
        "schema_version": 1,
        "python": {
            "version": python_version,
            "implementation": "FixturePython",
            "executable": str(executable),
            "platform": "fixture-platform",
        },
        "packages": {
            "torch": "fixture-torch",
            "torch_npu": None,
            "transformers": "fixture-transformers",
            "peft": None,
            "accelerate": None,
            "deepspeed": None,
            "vllm": None,
        },
        "torch": {
            "import_available": True,
            "import_error_type": None,
            "accelerators": {
                "cuda": accelerator("cuda", cuda_count),
                "npu": accelerator("npu", npu_count),
            },
        },
    }


def _write_probe_executable(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '__LLMGEN_RUNTIME_PROBE__={serialized}'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _local_model(root: Path) -> Path:
    path = root / "base-model"
    path.mkdir(parents=True, exist_ok=True)
    config = path / "config.json"
    if not config.exists():
        config.write_text('{"model_type":"fixture"}', encoding="utf-8")
    return path


def test_auto_devices_resolve_visible_cuda_topology() -> None:
    resolved = resolve_runtime_resources(
        {
            "device": "cuda",
            "devices": "auto",
            "num_devices": "auto",
            "distributed": "auto",
            "deepspeed": "auto",
        },
        environment={"CUDA_VISIBLE_DEVICES": "3,5"},
        torch_module=SimpleNamespace(cuda=_FakeCuda()),
    )

    assert resolved["resolved"] == {
        "accelerator": "cuda",
        "device_ids": [0, 1],
        "num_devices": 2,
        "accelerator_available": True,
        "detected_device_count": 2,
        "visible_devices": "3,5",
        "visible_devices_variable": "CUDA_VISIBLE_DEVICES",
    }
    assert resolved["topology"][1]["name"] == "GPU-1"


def test_npu_visibility_recognizes_ascend_runtime_variable() -> None:
    resolved = resolve_runtime_resources(
        {
            "device": "npu",
            "devices": "auto",
            "num_devices": "auto",
            "distributed": "auto",
            "deepspeed": "auto",
        },
        environment={"ASCEND_RT_VISIBLE_DEVICES": "6"},
        torch_module=SimpleNamespace(npu=_FakeNpu()),
    )

    assert resolved["resolved"]["num_devices"] == 1
    assert resolved["resolved"]["visible_devices"] == "6"
    assert (
        resolved["resolved"]["visible_devices_variable"]
        == "ASCEND_RT_VISIBLE_DEVICES"
    )
    assert resolved["topology"][0]["name"] == "NPU-0"


def test_explicit_devices_environment_matches_accelerator_kind() -> None:
    assert visible_devices_environment(
        {"device": "cuda", "devices": ["2", "4"]}
    ) == {"CUDA_VISIBLE_DEVICES": "2,4"}
    assert visible_devices_environment(
        {"device": "npu", "devices": "1,3"}
    ) == {"ASCEND_RT_VISIBLE_DEVICES": "1,3"}
    assert visible_devices_environment(
        {"device": "cpu", "devices": "auto"}
    ) == {}


def test_configured_python_probe_drives_provenance_and_auto_resources(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "fixture-python"
    payload = _runtime_probe_payload(executable, cuda_count=3)
    _write_probe_executable(executable, payload)
    runtime = {
        "python": str(executable),
        "device": "cuda",
        "devices": "auto",
        "num_devices": "auto",
        "distributed": "auto",
        "deepspeed": "none",
    }

    direct_probe = probe_runtime_environment(runtime, environment={})
    provenance = collect_run_provenance(
        repo_root=tmp_path,
        runtime=runtime,
        base_model=str(_local_model(tmp_path)),
        environment={"SECRET": "must-not-appear"},
    )

    assert direct_probe == payload
    assert provenance["schema_version"] == 2
    assert provenance["python"] == payload["python"]
    assert provenance["packages"] == payload["packages"]
    assert provenance["resources"]["resolved"]["num_devices"] == 3
    assert provenance["resources"]["topology"][2]["name"] == "CUDA-2"
    assert provenance["runtime_probe"]["torch"] == payload["torch"]
    assert "must-not-appear" not in repr(provenance)

    assert verify_run_provenance(
        provenance,
        repo_root=tmp_path,
        runtime=runtime,
        base_model=str(_local_model(tmp_path)),
        environment={"SECRET": "must-not-appear"},
    ) == provenance


def test_verify_uses_configured_python_probe_and_rejects_runtime_drift(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "fixture-python"
    runtime = {
        "python": str(executable),
        "device": "cpu",
        "devices": "auto",
        "num_devices": "auto",
        "distributed": "auto",
        "deepspeed": "none",
    }
    _write_probe_executable(executable, _runtime_probe_payload(executable))
    frozen = collect_run_provenance(
        repo_root=tmp_path,
        runtime=runtime,
        base_model=str(_local_model(tmp_path)),
        environment={},
    )
    _write_probe_executable(
        executable,
        _runtime_probe_payload(executable, python_version="9.8.8"),
    )

    with pytest.raises(ResourceResolutionError, match="python"):
        verify_run_provenance(
            frozen,
            repo_root=tmp_path,
            runtime=runtime,
            base_model=str(_local_model(tmp_path)),
            environment={},
        )


def test_explicit_device_count_conflict_is_rejected() -> None:
    runtime = {"devices": ["0", "1"], "num_devices": 1}
    with pytest.raises(ResourceResolutionError, match="must equal"):
        validate_runtime_device_request(runtime)


def test_local_base_model_fingerprint_records_metadata_and_config_hash(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"weights")

    fingerprint = base_model_fingerprint(str(model), environment={})

    assert fingerprint["kind"] == "local_directory"
    assert fingerprint["file_count"] == 2
    assert fingerprint["config_sha256"]
    assert fingerprint["tree_content_sha256"]
    assert all(item["sha256"] for item in fingerprint["files"])
    assert fingerprint["content_hash_scope"] == (
        "every regular file under the local model path"
    )


def test_collect_rejects_unavailable_local_base_model_paths(tmp_path: Path) -> None:
    runtime = {
        "device": "cpu",
        "devices": "auto",
        "num_devices": "auto",
        "distributed": "auto",
        "deepspeed": "none",
    }

    for base_model in ("missing-model", str(tmp_path / "absolute-missing-model")):
        with pytest.raises(ResourceResolutionError, match="does not exist"):
            collect_run_provenance(
                repo_root=tmp_path,
                runtime=runtime,
                base_model=base_model,
                environment={},
            )


def test_collect_resolves_relative_base_model_against_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    model = repo_root / "models" / "router"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    executable = tmp_path / "fixture-python"
    _write_probe_executable(executable, _runtime_probe_payload(executable))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    provenance = collect_run_provenance(
        repo_root=repo_root,
        runtime={
            "python": str(executable),
            "device": "cpu",
            "devices": "auto",
            "num_devices": "auto",
            "distributed": "auto",
            "deepspeed": "none",
        },
        base_model="models/router",
        environment={},
    )

    assert provenance["base_model"]["kind"] == "local_directory"
    assert provenance["base_model"]["path"] == str(model)


def test_offline_hf_reference_uses_cached_revision(tmp_path: Path) -> None:
    cache = tmp_path / "hf"
    ref = cache / "hub" / "models--org--model" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("abcdef\n", encoding="utf-8")

    fingerprint = base_model_fingerprint("org/model", environment={"HF_HOME": str(cache)})

    assert fingerprint["revision"] == "abcdef"
    assert fingerprint["revision_source"] == "local_hf_cache"
    assert fingerprint["pinned"] is True


@pytest.mark.parametrize("base_model", ["org/model", "org/model@abcdef"])
def test_run_provenance_rejects_remote_base_model_ids(
    tmp_path: Path,
    base_model: str,
) -> None:
    with pytest.raises(ResourceResolutionError, match="local model directory"):
        collect_run_provenance(
            repo_root=tmp_path,
            runtime={
                "device": "cpu",
                "devices": "auto",
                "num_devices": "auto",
                "distributed": "auto",
                "deepspeed": "none",
            },
            base_model=base_model,
            environment={"HF_HOME": str(tmp_path / "empty-cache")},
        )


def test_explicit_deepspeed_config_is_content_fingerprinted_and_verified(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "fixture-python"
    _write_probe_executable(executable, _runtime_probe_payload(executable))
    config = tmp_path / "deepspeed.json"
    config.write_text('{"zero_optimization":{"stage":2}}', encoding="utf-8")
    runtime = {
        "python": str(executable),
        "device": "cpu",
        "devices": "auto",
        "num_devices": "auto",
        "distributed": "auto",
        "deepspeed": "deepspeed.json",
    }

    fingerprint = deepspeed_config_fingerprint(runtime, repo_root=tmp_path)
    frozen = collect_run_provenance(
        repo_root=tmp_path,
        runtime=runtime,
        base_model=str(_local_model(tmp_path)),
        environment={},
    )

    assert fingerprint["path"] == str(config)
    assert fingerprint["sha256"]
    assert frozen["runtime_artifacts"]["deepspeed"] == fingerprint

    config.write_text('{"zero_optimization":{"stage":3}}', encoding="utf-8")
    with pytest.raises(ResourceResolutionError, match="runtime_artifacts"):
        verify_run_provenance(
            frozen,
            repo_root=tmp_path,
            runtime=runtime,
            base_model=str(_local_model(tmp_path)),
            environment={},
        )


def test_missing_explicit_deepspeed_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ResourceResolutionError, match="does not exist"):
        deepspeed_config_fingerprint(
            {"deepspeed": "missing.json"},
            repo_root=tmp_path,
        )


def test_run_provenance_is_secret_free_and_includes_resources(tmp_path: Path) -> None:
    provenance = collect_run_provenance(
        repo_root=tmp_path,
        runtime={
            "device": "cpu",
            "devices": "auto",
            "num_devices": "auto",
            "distributed": "auto",
            "deepspeed": "none",
        },
        base_model=str(_local_model(tmp_path)),
        environment={"SECRET": "must-not-appear"},
    )

    assert provenance["resources"]["resolved"]["num_devices"] == 1
    assert provenance["base_model"]["kind"] == "local_directory"
    assert "must-not-appear" not in repr(provenance)

    verified = verify_run_provenance(
        provenance,
        repo_root=tmp_path,
        runtime={
            "device": "cpu",
            "devices": "auto",
            "num_devices": "auto",
            "distributed": "auto",
            "deepspeed": "none",
        },
        base_model=str(_local_model(tmp_path)),
        environment={"SECRET": "must-not-appear"},
    )
    assert verified == provenance


def test_run_provenance_rejects_base_model_drift(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    runtime = {
        "device": "cpu",
        "devices": "auto",
        "num_devices": "auto",
        "distributed": "auto",
        "deepspeed": "none",
    }
    frozen = collect_run_provenance(
        repo_root=tmp_path,
        runtime=runtime,
        base_model=str(model),
        environment={},
    )
    (model / "config.json").write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(ResourceResolutionError, match="base_model"):
        verify_run_provenance(
            frozen,
            repo_root=tmp_path,
            runtime=runtime,
            base_model=str(model),
            environment={},
        )
