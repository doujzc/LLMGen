"""Local-only runtime-resource and reproducibility provenance snapshots.

The pipeline must be able to create a Run on machines without an accelerator or
network access.  Consequently this module never downloads a model and treats
optional training packages as optional when collecting provenance.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from .io import sha256_file


class ResourceResolutionError(ValueError):
    """Raised for an impossible declared runtime resource request."""


_RUNTIME_PROBE_MARKER = "__LLMGEN_RUNTIME_PROBE__="
_RUNTIME_PROBE_TIMEOUT_SECONDS = 30.0
_RUNTIME_PROBE_SCRIPT = r'''
import importlib.metadata
import json
import platform
import sys


def optional_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def accelerator_snapshot(module, kind):
    if module is None:
        return {"available": False, "device_count": 0, "topology": []}
    try:
        available = bool(module.is_available())
    except Exception as error:
        return {
            "available": False,
            "device_count": 0,
            "topology": [],
            "probe_error_type": type(error).__name__,
        }
    try:
        count = int(module.device_count()) if available else 0
    except Exception as error:
        return {
            "available": available,
            "device_count": 0,
            "topology": [],
            "probe_error_type": type(error).__name__,
        }
    topology = []
    for index in range(count):
        details = {"index": index}
        try:
            properties = module.get_device_properties(index)
            name = getattr(properties, "name", None)
            if name is not None:
                details["name"] = str(name)
            memory = getattr(properties, "total_memory", None)
            if isinstance(memory, int):
                details["total_memory"] = memory
            if kind == "cuda" and hasattr(module, "get_device_capability"):
                capability = module.get_device_capability(index)
                details["capability"] = [int(capability[0]), int(capability[1])]
        except Exception as error:
            details["probe_error_type"] = type(error).__name__
        topology.append(details)
    return {"available": available, "device_count": count, "topology": topology}


packages = {
    name: optional_version(distribution)
    for name, distribution in (
        ("torch", "torch"),
        ("torch_npu", "torch-npu"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("deepspeed", "deepspeed"),
        ("vllm", "vllm"),
    )
}
torch = None
torch_error = None
try:
    import torch
    try:
        import torch_npu  # noqa: F401 - registers torch.npu on Ascend hosts
    except ImportError:
        pass
except Exception as error:
    torch_error = type(error).__name__

payload = {
    "schema_version": 1,
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
    },
    "packages": packages,
    "torch": {
        "import_available": torch is not None,
        "import_error_type": torch_error,
        "accelerators": {
            "cuda": accelerator_snapshot(getattr(torch, "cuda", None), "cuda"),
            "npu": accelerator_snapshot(getattr(torch, "npu", None), "npu"),
        },
    },
}
print("__LLMGEN_RUNTIME_PROBE__=" + json.dumps(payload, sort_keys=True))
'''


def _device_ids(value: Any) -> tuple[str, ...] | None:
    """Return explicit configured IDs, or ``None`` for ``auto``."""

    if value == "auto":
        return None
    if isinstance(value, str):
        ids = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, list):
        ids = tuple(str(part).strip() for part in value if str(part).strip())
    else:  # Config validation owns the user-facing type error.
        return None
    if not ids:
        raise ResourceResolutionError("runtime.devices must name at least one device")
    if len(set(ids)) != len(ids):
        raise ResourceResolutionError("runtime.devices contains duplicate device IDs")
    return ids


def validate_runtime_device_request(runtime: Mapping[str, Any]) -> None:
    """Reject ambiguous explicit ``devices`` / ``num_devices`` combinations."""

    ids = _device_ids(runtime.get("devices"))
    configured_count = runtime.get("num_devices")
    if ids is not None and configured_count != "auto" and int(configured_count) != len(ids):
        raise ResourceResolutionError(
            "runtime.num_devices must equal the number of explicit runtime.devices"
        )


def _torch_module() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def visible_devices_environment(runtime: Mapping[str, Any]) -> dict[str, str]:
    """Return the launcher visibility variable for explicit runtime devices.

    An empty mapping means device selection is automatic. Ascend launchers use
    ``ASCEND_RT_VISIBLE_DEVICES``; legacy aliases can be supplied explicitly in
    ``runtime.environment`` when an older runtime requires them.
    """

    ids = _device_ids(runtime.get("devices"))
    if ids is None:
        return {}
    kind = str(runtime.get("device") or "cpu").split(":", 1)[0].casefold()
    joined = ",".join(ids)
    if kind == "cuda":
        return {"CUDA_VISIBLE_DEVICES": joined}
    if kind == "npu":
        return {"ASCEND_RT_VISIBLE_DEVICES": joined}
    return {}


def _effective_runtime_environment(
    runtime: Mapping[str, Any],
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    effective = dict(os.environ)
    if environment is not None:
        effective.update({str(key): str(value) for key, value in environment.items()})
    effective.update(visible_devices_environment(runtime))
    configured = runtime.get("environment")
    if isinstance(configured, Mapping):
        effective.update({str(key): str(value) for key, value in configured.items()})
    return effective


def probe_runtime_environment(
    runtime: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
    timeout_seconds: float = _RUNTIME_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Probe the exact configured training interpreter with bounded execution."""

    configured = str(runtime.get("python") or "").strip()
    executable = configured or sys.executable
    working_directory = Path(repo_root).resolve() if repo_root is not None else None
    if configured and working_directory is not None:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute() and len(candidate.parts) > 1:
            executable = str((working_directory / candidate).resolve())
    effective_environment = _effective_runtime_environment(runtime, environment)
    try:
        result = subprocess.run(
            [executable, "-c", _RUNTIME_PROBE_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            env=effective_environment,
            cwd=working_directory,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ResourceResolutionError(
            f"runtime.python probe timed out after {timeout_seconds:g}s: {executable}"
        ) from error
    except OSError as error:
        raise ResourceResolutionError(
            f"runtime.python cannot be executed for provenance probe: {executable}"
        ) from error
    if result.returncode != 0:
        raise ResourceResolutionError(
            "runtime.python provenance probe failed with exit code "
            f"{result.returncode}: {executable}"
        )
    payload_text = next(
        (
            line[len(_RUNTIME_PROBE_MARKER) :]
            for line in reversed(result.stdout.splitlines())
            if line.startswith(_RUNTIME_PROBE_MARKER)
        ),
        None,
    )
    if payload_text is None:
        raise ResourceResolutionError(
            f"runtime.python provenance probe returned no JSON marker: {executable}"
        )
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ResourceResolutionError(
            f"runtime.python provenance probe returned invalid JSON: {executable}"
        ) from error
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(key), dict) for key in ("python", "packages", "torch")
    ):
        raise ResourceResolutionError(
            f"runtime.python provenance probe returned an invalid payload: {executable}"
        )
    return payload


def _cuda_topology(torch: Any, count: int) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for index in range(count):
        details: dict[str, Any] = {"index": index}
        try:
            properties = torch.cuda.get_device_properties(index)
            details.update(
                {
                    "name": str(properties.name),
                    "total_memory": int(properties.total_memory),
                }
            )
            capability = torch.cuda.get_device_capability(index)
            details["capability"] = [int(capability[0]), int(capability[1])]
        except Exception as error:  # Runtime visibility can change during startup.
            details["probe_error"] = f"{type(error).__name__}: {error}"
        devices.append(details)
    return devices


def _npu_topology(torch: Any, count: int) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    npu = getattr(torch, "npu", None)
    for index in range(count):
        details: dict[str, Any] = {"index": index}
        try:
            properties = npu.get_device_properties(index) if npu is not None else None
            if properties is not None:
                details["name"] = str(getattr(properties, "name", "npu"))
                memory = getattr(properties, "total_memory", None)
                if isinstance(memory, int):
                    details["total_memory"] = memory
        except Exception as error:
            details["probe_error"] = f"{type(error).__name__}: {error}"
        devices.append(details)
    return devices


def resolve_runtime_resources(
    runtime: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
    runtime_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the current local accelerator topology without launching work.

    ``num_devices=auto`` resolves against the accelerator requested by
    ``runtime.device``.  An unavailable accelerator is recorded as zero rather
    than silently converted to CPU; the actual training entrypoint can then
    produce its normal actionable availability error.
    """

    validate_runtime_device_request(runtime)
    env = dict(os.environ if environment is None else environment)
    requested_device = str(runtime.get("device") or "cpu")
    kind = requested_device.split(":", 1)[0].casefold()
    configured_ids = _device_ids(runtime.get("devices"))
    torch = torch_module if torch_module is not None else (
        None if runtime_probe is not None else _torch_module()
    )
    probe_accelerators: Mapping[str, Any] = {}
    if runtime_probe is not None and torch_module is None:
        torch_probe = runtime_probe.get("torch")
        if isinstance(torch_probe, Mapping):
            accelerators = torch_probe.get("accelerators")
            if isinstance(accelerators, Mapping):
                probe_accelerators = accelerators

    visible_variable: str | None = None
    detected_count = 0
    topology: list[dict[str, Any]] = []
    if kind == "cuda":
        visible_variable = "CUDA_VISIBLE_DEVICES"
        cuda_probe = probe_accelerators.get("cuda")
        if isinstance(cuda_probe, Mapping):
            detected_count = int(cuda_probe.get("device_count") or 0)
            raw_topology = cuda_probe.get("topology")
            topology = list(raw_topology) if isinstance(raw_topology, list) else []
        else:
            try:
                available = bool(torch is not None and torch.cuda.is_available())
                detected_count = int(torch.cuda.device_count()) if available else 0
            except Exception:
                detected_count = 0
            topology = _cuda_topology(torch, detected_count) if torch is not None else []
    elif kind == "npu":
        visible_variable = next(
            (
                name
                for name in (
                    "ASCEND_RT_VISIBLE_DEVICES",
                    "ASCEND_VISIBLE_DEVICES",
                    "NPU_VISIBLE_DEVICES",
                )
                if name in env
            ),
            "ASCEND_RT_VISIBLE_DEVICES",
        )
        npu_probe = probe_accelerators.get("npu")
        if isinstance(npu_probe, Mapping):
            detected_count = int(npu_probe.get("device_count") or 0)
            raw_topology = npu_probe.get("topology")
            topology = list(raw_topology) if isinstance(raw_topology, list) else []
        else:
            npu = getattr(torch, "npu", None) if torch is not None else None
            try:
                available = bool(npu is not None and npu.is_available())
                detected_count = int(npu.device_count()) if available else 0
            except Exception:
                detected_count = 0
            topology = _npu_topology(torch, detected_count) if torch is not None else []
    else:
        detected_count = 1
        topology = [{"index": 0, "name": "cpu"}]

    configured_count = runtime.get("num_devices")
    if configured_ids is not None:
        resolved_count = len(configured_ids)
        resolved_ids = list(configured_ids)
    elif configured_count == "auto":
        resolved_count = detected_count
        resolved_ids = list(range(detected_count))
    else:
        resolved_count = int(configured_count)
        resolved_ids = list(range(resolved_count))

    return {
        "schema_version": 1,
        "requested": {
            "device": requested_device,
            "devices": runtime.get("devices"),
            "num_devices": configured_count,
            "distributed": runtime.get("distributed"),
            "deepspeed": runtime.get("deepspeed"),
        },
        "resolved": {
            "accelerator": kind,
            "device_ids": resolved_ids,
            "num_devices": resolved_count,
            "accelerator_available": detected_count > 0,
            "detected_device_count": detected_count,
            "visible_devices": env.get(visible_variable) if visible_variable else None,
            "visible_devices_variable": visible_variable,
        },
        "topology": topology,
    }


def _run_git(repo_root: Path, arguments: list[str]) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def git_provenance(repo_root: str | Path) -> dict[str, Any]:
    """Return content hashes for committed, modified, and untracked source."""

    root = Path(repo_root).resolve()
    status = _run_git(root, ["status", "--porcelain=v1", "-z"])
    unstaged = _run_git(root, ["diff", "--binary", "HEAD"])
    staged = _run_git(root, ["diff", "--cached", "--binary"])
    head = _run_git(root, ["rev-parse", "HEAD"])
    untracked = _run_git(root, ["ls-files", "--others", "--exclude-standard", "-z"])
    if any(value is None for value in (status, unstaged, staged, head, untracked)):
        return {"repository": str(root), "available": False}
    assert status is not None and unstaged is not None and staged is not None
    assert head is not None and untracked is not None
    diff = unstaged + b"\0--STAGED--\0" + staged
    source_roots = {"src", "scripts", "configs", "vendor", "tests", "docs"}
    root_files = {"pyproject.toml", "uv.lock", "requirements.txt", "AGENTS.md"}
    untracked_files: list[dict[str, Any]] = []
    for raw_name in untracked.split(b"\0"):
        if not raw_name:
            continue
        name = raw_name.decode("utf-8", errors="surrogateescape")
        relative = Path(name)
        if relative.parts[0] not in source_roots and name not in root_files:
            continue
        path = root / relative
        if path.is_file():
            untracked_files.append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    serialized_untracked = json.dumps(
        sorted(untracked_files, key=lambda item: item["path"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "repository": str(root),
        "available": True,
        "commit": head.decode("ascii", errors="replace").strip(),
        "dirty": bool(status),
        "status_sha256": sha256(status).hexdigest(),
        "working_tree_diff_sha256": sha256(diff).hexdigest(),
        "untracked_source_sha256": sha256(serialized_untracked).hexdigest(),
        "untracked_source_files": untracked_files,
    }


def base_model_fingerprint(
    base_model: str,
    *,
    environment: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint a local model or a locally-known Hugging Face revision.

    Local model bytes are part of the algorithm input, so every regular file is
    content-hashed. Run creation can therefore take noticeable time for a large
    checkpoint, but a model file changed in place can never be silently reused
    under the same pipeline lineage.
    """

    candidate = Path(base_model).expanduser()
    if not candidate.is_absolute() and repo_root is not None:
        candidate = Path(repo_root).resolve() / candidate
    if candidate.exists():
        path = candidate.resolve()
        files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        entries = []
        for item in files:
            entries.append(
                {
                    "path": (
                        item.name
                        if path.is_file()
                        else item.relative_to(path).as_posix()
                    ),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
        config_path = path if path.name == "config.json" else path / "config.json"
        serialized_entries = json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "kind": "local_file" if path.is_file() else "local_directory",
            "path": str(path),
            "file_count": len(entries),
            "tree_content_sha256": sha256(serialized_entries).hexdigest(),
            "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
            "content_hash_scope": "every regular file under the local model path",
            "files": entries,
        }

    # A missing absolute/relative filesystem path must not be misreported as a
    # Hugging Face model ID merely because the training host has not mounted it
    # yet. Recording it explicitly makes a later training failure diagnosable.
    if candidate.is_absolute() or base_model.startswith(("./", "../", "~")):
        return {
            "kind": "local_path_unavailable",
            "path": str(candidate.resolve()),
            "exists": False,
            "content_hash_scope": "unavailable at Run creation",
        }

    env = dict(os.environ if environment is None else environment)
    reference = base_model
    cache_root = Path(env.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    cache_model = cache_root / ("models--" + reference.replace("/", "--"))
    cached_revision: str | None = None
    ref_file = cache_model / "refs" / "main"
    if ref_file.is_file():
        cached_revision = ref_file.read_text(encoding="utf-8").strip() or None
    if cached_revision is None:
        snapshots = sorted((cache_model / "snapshots").glob("*")) if (cache_model / "snapshots").is_dir() else []
        if len(snapshots) == 1:
            cached_revision = snapshots[0].name
    return {
        "kind": "huggingface_reference",
        "reference": reference,
        "reference_sha256": sha256(reference.encode("utf-8")).hexdigest(),
        "revision": cached_revision,
        "revision_sha256": (
            sha256(cached_revision.encode("utf-8")).hexdigest()
            if cached_revision
            else None
        ),
        "revision_source": (
            "local_hf_cache" if cached_revision else "unresolved_offline"
        ),
        "pinned": bool(cached_revision),
    }


def deepspeed_config_fingerprint(
    runtime: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Fingerprint an explicit external DeepSpeed configuration file."""

    configured = str(runtime.get("deepspeed") or "").strip()
    if configured in {"auto", "none"}:
        return {"kind": configured, "configured": configured}
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = Path(repo_root) / candidate
    path = candidate.resolve()
    if not path.is_file():
        raise ResourceResolutionError(
            f"runtime.deepspeed config file does not exist: {path}"
        )
    return {
        "kind": "local_file",
        "configured": configured,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def collect_run_provenance(
    *,
    repo_root: str | Path,
    runtime: Mapping[str, Any],
    base_model: str,
    environment: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Collect an immutable, secret-free provenance snapshot before training."""

    model_fingerprint = base_model_fingerprint(
        base_model,
        environment=environment,
        repo_root=repo_root,
    )
    if model_fingerprint.get("kind") == "local_path_unavailable":
        raise ResourceResolutionError(
            "router.base_model must reference an existing local model directory; "
            "resolved path does not exist: "
            f"{model_fingerprint.get('path')}"
        )
    if model_fingerprint.get("kind") == "huggingface_reference":
        raise ResourceResolutionError(
            "router.base_model must be a local model directory path; remote "
            f"Hugging Face model IDs are not supported: {base_model}. Materialize "
            "the exact model snapshot locally before creating the Run"
        )
    if model_fingerprint.get("kind") == "local_file":
        raise ResourceResolutionError(
            "router.base_model must be a local model directory, not a file: "
            f"{model_fingerprint.get('path')}"
        )

    effective_environment = _effective_runtime_environment(runtime, environment)
    runtime_probe = probe_runtime_environment(
        runtime,
        environment=effective_environment,
        repo_root=repo_root,
    )

    return {
        "schema_version": 2,
        "python": runtime_probe["python"],
        "packages": runtime_probe["packages"],
        "runtime_probe": {
            "schema_version": runtime_probe.get("schema_version"),
            "torch": runtime_probe["torch"],
        },
        "runtime_artifacts": {
            "deepspeed": deepspeed_config_fingerprint(
                runtime,
                repo_root=repo_root,
            ),
        },
        "git": git_provenance(repo_root),
        "resources": resolve_runtime_resources(
            runtime,
            environment=effective_environment,
            torch_module=torch_module,
            runtime_probe=runtime_probe,
        ),
        "base_model": model_fingerprint,
    }


def verify_run_provenance(
    frozen: Mapping[str, Any],
    *,
    repo_root: str | Path,
    runtime: Mapping[str, Any],
    base_model: str,
    environment: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Recompute immutable training inputs and reject silent environment drift."""

    current = collect_run_provenance(
        repo_root=repo_root,
        runtime=runtime,
        base_model=base_model,
        environment=environment,
        torch_module=torch_module,
    )
    comparisons = {
        "python": (frozen.get("python"), current.get("python")),
        "packages": (frozen.get("packages"), current.get("packages")),
        "runtime_probe": (
            frozen.get("runtime_probe"),
            current.get("runtime_probe"),
        ),
        "runtime_artifacts": (
            frozen.get("runtime_artifacts"),
            current.get("runtime_artifacts"),
        ),
        "git": (frozen.get("git"), current.get("git")),
        "resources": (frozen.get("resources"), current.get("resources")),
        "base_model": (frozen.get("base_model"), current.get("base_model")),
    }
    changed = [name for name, (expected, actual) in comparisons.items() if expected != actual]
    if changed:
        raise ResourceResolutionError(
            "training provenance changed after Run creation: "
            + ", ".join(changed)
            + "; fork or create a new Run with the current environment"
        )
    return current
