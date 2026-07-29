"""GPU inventory and deterministic CUDA device binding helpers."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Mapping, Sequence


GPU_QUERY = (
    "index,uuid,pci.bus_id,name,utilization.gpu,memory.used,"
    "memory.total,temperature.gpu"
)
COMPUTE_PROCESS_QUERY = "gpu_uuid,pid,used_gpu_memory"


def _run_nvidia_smi(
    query: str,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-{query}",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed if completed.returncode == 0 else None


def _process_group_id(pid: int) -> int | None:
    try:
        return int(os.getpgid(pid))
    except (OSError, ValueError):
        return None


def probe_gpu_metrics(
    *,
    include_processes: bool = True,
    timeout_seconds: float = 2.0,
) -> list[dict[str, Any]] | None:
    """Return host-visible NVIDIA GPUs and optional compute-process telemetry."""

    completed = _run_nvidia_smi(f"gpu={GPU_QUERY}", timeout_seconds)
    if completed is None:
        return None

    rows: list[dict[str, Any]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", 7)]
        if len(parts) != 8:
            continue
        try:
            row: dict[str, Any] = {
                "index": parts[0],
                "uuid": parts[1],
                "pci_bus_id": parts[2],
                "name": parts[3],
                "utilization": int(parts[4]),
                "memory_used_mib": int(parts[5]),
                "memory_total_mib": int(parts[6]),
                "temperature_c": int(parts[7]),
                "processes": [],
            }
        except ValueError:
            continue
        rows.append(row)
        by_uuid[row["uuid"]] = row

    if not include_processes or not rows:
        return rows

    process_result = _run_nvidia_smi(
        f"compute-apps={COMPUTE_PROCESS_QUERY}",
        timeout_seconds,
    )
    if process_result is None:
        return rows
    for raw_line in process_result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", 2)]
        if len(parts) != 3:
            continue
        gpu = by_uuid.get(parts[0])
        if gpu is None:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        try:
            used_memory_mib: int | None = int(parts[2])
        except ValueError:
            used_memory_mib = None
        gpu["processes"].append(
            {
                "pid": pid,
                "process_group_id": _process_group_id(pid),
                "used_memory_mib": used_memory_mib,
            }
        )
    return rows


def resolve_cuda_visible_devices(
    requested: str,
    gpus: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Resolve numeric nvidia-smi indices to stable GPU UUID runtime bindings."""

    tokens = [
        token.strip()
        for token in str(requested).split(",")
        if token.strip()
    ]
    if not tokens:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain at least one GPU")

    if gpus is None:
        return {
            "requested": tokens,
            "runtime_value": ",".join(tokens),
            "bindings": [
                {
                    "requested": token,
                    "index": token if token.isdecimal() else "",
                    "uuid": token if token.startswith(("GPU-", "MIG-")) else "",
                    "name": "",
                }
                for token in tokens
            ],
            "verified": False,
        }

    by_index = {str(gpu.get("index", "")): gpu for gpu in gpus}
    bindings: list[dict[str, str]] = []
    runtime_tokens: list[str] = []
    missing_indices: list[str] = []
    for token in tokens:
        matched: Mapping[str, Any] | None = None
        if token.isdecimal():
            matched = by_index.get(token)
            if matched is None:
                missing_indices.append(token)
                continue
        else:
            uuid_matches = [
                gpu
                for gpu in gpus
                if str(gpu.get("uuid", "")).startswith(token)
                or token.startswith(str(gpu.get("uuid", "")))
            ]
            if len(uuid_matches) == 1:
                matched = uuid_matches[0]

        runtime_token = (
            str(matched.get("uuid", ""))
            if matched is not None and matched.get("uuid")
            else token
        )
        runtime_tokens.append(runtime_token)
        bindings.append(
            {
                "requested": token,
                "index": str(matched.get("index", "")) if matched else "",
                "uuid": str(matched.get("uuid", "")) if matched else runtime_token,
                "name": str(matched.get("name", "")) if matched else "",
            }
        )

    if missing_indices:
        available = ", ".join(sorted(by_index, key=lambda value: int(value)))
        raise ValueError(
            "configured GPU index is not present in nvidia-smi: "
            f"{', '.join(missing_indices)}; available indices: {available or 'none'}"
        )
    return {
        "requested": tokens,
        "runtime_value": ",".join(runtime_tokens),
        "bindings": bindings,
        "verified": True,
    }
