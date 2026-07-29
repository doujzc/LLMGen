from __future__ import annotations

import subprocess

import pytest

from training_console import gpu


GPU_ROWS = [
    {
        "index": "0",
        "uuid": "GPU-alpha",
        "pci_bus_id": "00000000:01:00.0",
        "name": "Test GPU A",
    },
    {
        "index": "2",
        "uuid": "GPU-charlie",
        "pci_bus_id": "00000000:03:00.0",
        "name": "Test GPU C",
    },
]


def test_numeric_visible_devices_bind_to_stable_gpu_uuids() -> None:
    resolved = gpu.resolve_cuda_visible_devices("2,0", GPU_ROWS)

    assert resolved == {
        "requested": ["2", "0"],
        "runtime_value": "GPU-charlie,GPU-alpha",
        "bindings": [
            {
                "requested": "2",
                "index": "2",
                "uuid": "GPU-charlie",
                "name": "Test GPU C",
            },
            {
                "requested": "0",
                "index": "0",
                "uuid": "GPU-alpha",
                "name": "Test GPU A",
            },
        ],
        "verified": True,
    }


def test_missing_numeric_gpu_fails_before_training_launch() -> None:
    with pytest.raises(ValueError, match="available indices: 0, 2"):
        gpu.resolve_cuda_visible_devices("1", GPU_ROWS)


def test_gpu_probe_attaches_compute_process_groups(monkeypatch) -> None:
    def fake_query(query: str, timeout_seconds: float):
        assert timeout_seconds == 2.0
        if query.startswith("gpu="):
            stdout = (
                "0, GPU-alpha, 00000000:01:00.0, Test GPU A, "
                "72, 2048, 24576, 61\n"
            )
        else:
            stdout = "GPU-alpha, 4321, 1536\n"
        return subprocess.CompletedProcess([], 0, stdout=stdout)

    monkeypatch.setattr(gpu, "_run_nvidia_smi", fake_query)
    monkeypatch.setattr(gpu, "_process_group_id", lambda pid: 4000)

    assert gpu.probe_gpu_metrics() == [
        {
            "index": "0",
            "uuid": "GPU-alpha",
            "pci_bus_id": "00000000:01:00.0",
            "name": "Test GPU A",
            "utilization": 72,
            "memory_used_mib": 2048,
            "memory_total_mib": 24576,
            "temperature_c": 61,
            "processes": [
                {
                    "pid": 4321,
                    "process_group_id": 4000,
                    "used_memory_mib": 1536,
                }
            ],
        }
    ]
