"""Atomic file-backed profile and run registry for the training console."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import tempfile
from typing import Any, BinaryIO, Iterable, Iterator, Mapping

from .config import ConfigValidationError, validate_profile_id


RUN_ID_PREFIX = "run_"
LOG_TAIL_MAX_BYTES = 1024 * 1024
LOG_TAIL_READ_CHUNK_BYTES = 64 * 1024
LOG_TAIL_MAX_LINE_BYTES = 16 * 1024
_LOG_LINE_TRUNCATION_MARKER = b"[... log line truncated ...] "

# Only these non-secret parent-process variables may cross the console boundary.
# They are captured in config.json when a run is submitted so a later runner sees
# the same runtime environment. Training controls belong in ConfigResolver's
# ALLOWED_KEYS instead of this list.
RUNTIME_ENV_KEYS = frozenset(
    {
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "CONDA_SHLVL",
        "CURL_CA_BUNDLE",
        "HF_HOME",
        "HOME",
        "HUGGINGFACE_HUB_CACHE",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TOKENIZERS_PARALLELISM",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TRANSFORMERS_CACHE",
        "TRITON_CACHE_DIR",
        "TZ",
        "USER",
        "VIRTUAL_ENV",
        "WANDB_DISABLED",
        "WANDB_ENTITY",
        "WANDB_MODE",
        "WANDB_PROJECT",
        "XDG_CACHE_HOME",
        "no_proxy",
    }
)
RUNTIME_ENV_PREFIXES = (
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "KMP_",
    "MKL_",
    "NCCL_",
    "NVIDIA_",
    "NUMEXPR_",
    "OMP_",
    "OPENBLAS_",
    "PYTORCH_CUDA_",
    "ROCR_",
    "TORCH_NCCL_",
    "VECLIB_",
)

# Secret names are explicit, never included in a persisted snapshot, and only
# forwarded from the Web process to runner to training at process launch.
SECRET_ENV_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ALL_PROXY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
    }
)

# These variables can inject code or hidden training arguments and must not cross
# the boundary even if a future runtime prefix becomes broader.
BLOCKED_RUNTIME_ENV_KEYS = frozenset(
    {
        "BASH_ENV",
        "BASHOPTS",
        "CDPATH",
        "ENV",
        "GLOBIGNORE",
        "LD_PRELOAD",
        "PREPARE_SCRIPT",
        "PROMPT_COMMAND",
        "PS4",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "ROUTER_EXTRA_ARGS",
        "SHELLOPTS",
        "SKILLRET_CONFIG",
        "SKILLRET_ROOT",
    }
)

_BEARER_RE = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+",
)
_AUTHORIZATION_RE = re.compile(
    r"(?im)(?P<prefix>\bAuthorization\s*[:=]\s*)(?P<value>[^\r\n]+)",
)
_URL_USERINFO_RE = re.compile(
    r"(?i)(?P<scheme>\b[a-z][a-z0-9+.-]*://)(?P<userinfo>[^/\s]+@)",
)
_SECRET_NAME_PATTERN = (
    r"(?:"
    r"ALL_PROXY|HTTP_PROXY|HTTPS_PROXY|"
    r"ANTHROPIC_API_KEY|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|"
    r"AWS_SESSION_TOKEN|AZURE_OPENAI_API_KEY|GOOGLE_API_KEY|HF_TOKEN|"
    r"HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY|WANDB_API_KEY|"
    r"API[_-]?KEY|ACCESS[_-]?KEY|SECRET[_-]?KEY|"
    r"ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN|BEARER[_-]?TOKEN|"
    r"TOKEN|SECRET|PASSWORD"
    r")"
)
_QUOTED_SECRET_RE = re.compile(
    rf"(?i)(?P<prefix>[\"']?{_SECRET_NAME_PATTERN}[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)"
)
_UNQUOTED_SECRET_RE = re.compile(
    rf"(?i)(?P<prefix>\b{_SECRET_NAME_PATTERN}\b\s*[:=]\s*)"
    r"(?![\"']|\[REDACTED\])(?P<value>[^\s,;}\]]+)"
)


def utc_now() -> str:
    """Return a stable UTC timestamp suitable for persisted metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SENSITIVE_RUNTIME_ENV_COMPONENTS = frozenset(
    {
        "APIKEY",
        "AUTH",
        "AUTHORIZATION",
        "COOKIE",
        "CREDENTIAL",
        "CREDENTIALS",
        "KEY",
        "PASSWORD",
        "PASSWD",
        "PWD",
        "SECRET",
        "SIG",
        "SIGNATURE",
        "TOKEN",
    }
)
_SENSITIVE_RUNTIME_ENV_COMPACT_FRAGMENTS = frozenset(
    {
        "APIKEY",
        "ACCESSKEY",
        "SECRETKEY",
        "SIGNINGKEY",
        "PRIVATEKEY",
        "ACCESSTOKEN",
        "REFRESHTOKEN",
        "BEARERTOKEN",
        "AUTHTOKEN",
        "CLIENTSECRET",
        "TOKEN",
        "SECRET",
        "KEY",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "BEARER",
        "COOKIE",
        "SESSION",
        "JWT",
        "SIGNATURE",
    }
)


def _has_sensitive_runtime_env_name(name: str) -> bool:
    camel_separated = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        str(name),
    )
    components = {
        component
        for component in re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            camel_separated,
        ).upper().split("_")
        if component
    }
    compact = re.sub(r"[^A-Za-z0-9]+", "", camel_separated).upper()
    return bool(
        components & _SENSITIVE_RUNTIME_ENV_COMPONENTS
        or any(
            fragment in compact
            for fragment in _SENSITIVE_RUNTIME_ENV_COMPACT_FRAGMENTS
        )
    )


def snapshot_runtime_environment(
    environment: Mapping[str, Any],
) -> dict[str, str]:
    """Return the allowlisted, non-secret runtime environment to persist."""

    captured: dict[str, str] = {}
    for raw_key, raw_value in environment.items():
        key = str(raw_key)
        if key in SECRET_ENV_KEYS or key in BLOCKED_RUNTIME_ENV_KEYS:
            continue
        explicitly_allowed = key in RUNTIME_ENV_KEYS
        if not explicitly_allowed and not key.startswith(RUNTIME_ENV_PREFIXES):
            continue
        if not explicitly_allowed and _has_sensitive_runtime_env_name(key):
            continue
        value = str(raw_value)
        if "\x00" in value or "\n" in value or "\r" in value:
            continue
        if len(value) > 16_384:
            continue
        captured[key] = value
    return dict(sorted(captured.items()))


def secret_runtime_environment(
    environment: Mapping[str, Any],
) -> dict[str, str]:
    """Select explicit secret variables without persisting their values."""

    selected: dict[str, str] = {}
    for key in SECRET_ENV_KEYS:
        value = environment.get(key)
        if value is None:
            continue
        normalized = str(value)
        if normalized and "\x00" not in normalized:
            selected[key] = normalized
    return selected


def child_process_environment(
    runtime_environment: Mapping[str, Any],
    secret_source: Mapping[str, Any],
) -> dict[str, str]:
    """Build a sanitized child environment from snapshot plus explicit secrets."""

    environment = snapshot_runtime_environment(runtime_environment)
    environment.update(secret_runtime_environment(secret_source))
    return environment


@contextmanager
def secure_binary_append(path: Path) -> Iterator[BinaryIO]:
    """Open an append-only log with owner-only permissions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "ab", buffering=0) as stream:
            descriptor = -1
            yield stream
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def redact_sensitive_text(
    text: str,
    secret_values: Iterable[str] = (),
) -> str:
    """Redact known values and common credential forms from displayed logs."""

    redacted = text
    known = {
        str(value)
        for value in secret_values
        if value is not None and str(value)
    }
    for value in sorted(known, key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED]")
    redacted = _BEARER_RE.sub("Bearer [REDACTED]", redacted)
    redacted = _AUTHORIZATION_RE.sub(
        lambda match: (
            match.group(0)
            if match.group("value").lstrip().startswith("Bearer [REDACTED]")
            else f"{match.group('prefix')}[REDACTED]"
        ),
        redacted,
    )
    redacted = _URL_USERINFO_RE.sub(
        lambda match: f"{match.group('scheme')}[REDACTED]@",
        redacted,
    )
    redacted = _QUOTED_SECRET_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}"
            f"[REDACTED]{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _UNQUOTED_SECRET_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]",
        redacted,
    )
    return redacted


def _bounded_log_tail(path: Path, maximum_lines: int) -> str:
    """Read a bounded tail without materializing an unbounded training log."""

    chunks: list[bytes] = []
    newline_count = 0
    bytes_read = 0
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        while (
            position > 0
            and bytes_read < LOG_TAIL_MAX_BYTES
            and newline_count <= maximum_lines
        ):
            size = min(
                LOG_TAIL_READ_CHUNK_BYTES,
                position,
                LOG_TAIL_MAX_BYTES - bytes_read,
            )
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            newline_count += chunk.count(b"\n")

    if not chunks:
        return ""
    lines = b"".join(reversed(chunks)).splitlines(keepends=True)
    first_line_is_partial = position > 0
    if len(lines) > maximum_lines:
        lines = lines[-maximum_lines:]
        first_line_is_partial = False

    bounded_lines: list[bytes] = []
    for index, line in enumerate(lines):
        newline = b""
        body = line
        if body.endswith(b"\r\n"):
            body, newline = body[:-2], b"\r\n"
        elif body.endswith((b"\n", b"\r")):
            body, newline = body[:-1], body[-1:]
        content_limit = max(0, LOG_TAIL_MAX_LINE_BYTES - len(newline))
        truncated = len(body) > content_limit
        partial = index == 0 and first_line_is_partial
        if truncated or partial:
            marker = _LOG_LINE_TRUNCATION_MARKER[:content_limit]
            keep = max(
                0,
                content_limit - len(marker),
            )
            body = marker + (body[-keep:] if keep else b"")
        bounded_lines.append(body + newline)
    return b"".join(bounded_lines).decode("utf-8", errors="replace")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def env_text(resolved: Mapping[str, Any]) -> str:
    """Render a deterministic, review-only shell environment snapshot."""

    lines = [
        "# Generated by the LLMGen training console.",
        "# config.json is the authoritative runner input.",
    ]
    for key in sorted(resolved):
        if key in {"DATASET", "PIPELINE_COMMAND"}:
            continue
        value = str(resolved[key])
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def gpu_contract(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Return the non-secret GPU assignment recorded with each run."""

    configured_gpus = [
        token.strip()
        for token in str(resolved.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if token.strip()
    ]
    return {
        "configured_gpus": configured_gpus,
        "configured_num_gpus": str(resolved.get("ROUTER_NUM_GPUS", "")),
        "cuda_device_order": str(
            resolved.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        ),
    }


def pid_alive(pid: Any) -> bool:
    """Return whether a local PID exists without changing its state."""

    try:
        parsed = int(pid)
        if parsed < 1:
            return False
        os.kill(parsed, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


class StateStore:
    """Persist mutable profiles and independent-run metadata as JSON files."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.profiles_dir = self.root / "profiles"
        self.runs_dir = self.root / "runs"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / ".registry.lock"

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _profile_dir(self, profile_id: str) -> Path:
        return self.profiles_dir / validate_profile_id(profile_id)

    @staticmethod
    def _profile_path(profile_dir: Path, version: int) -> Path:
        if version < 1:
            raise ConfigValidationError(
                [{"field": "version", "message": "配置版本必须大于 0"}]
            )
        return profile_dir / f"v{version:04d}.json"

    def save_profile(
        self,
        *,
        profile_id: str,
        dataset: str,
        command: str,
        overrides: Mapping[str, str],
        resolved: Mapping[str, str],
        notes: str = "",
        version: int | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        profile_id = validate_profile_id(profile_id)
        if "\x00" in notes or len(notes) > 2000:
            raise ConfigValidationError(
                [{"field": "notes", "message": "备注长度不能超过 2000"}]
            )
        profile_dir = self._profile_dir(profile_id)
        with self.lock():
            profile_dir.mkdir(parents=True, exist_ok=True)
            versions = self._version_numbers(profile_dir)
            now = utc_now()
            if version is None:
                if versions:
                    raise ConfigValidationError(
                        [
                            {
                                "field": "profile_id",
                                "message": (
                                    f"配置已存在：{profile_id}；"
                                    "请先加载需要修改的版本"
                                ),
                            }
                        ]
                    )
                version = 1
                revision = 1
                created_at = now
                parent_version = None
            elif version not in versions:
                raise ConfigValidationError(
                    [
                        {
                            "field": "version",
                            "message": f"配置版本不存在：v{version}",
                        }
                    ]
                )
            else:
                path = self._profile_path(profile_dir, version)
                existing = _read_json(path)
                current_revision = max(
                    1,
                    int(existing.get("revision", 1)),
                )
                if (
                    expected_revision is not None
                    and int(expected_revision) != current_revision
                ):
                    raise ConfigValidationError(
                        [
                            {
                                "field": "revision",
                                "message": (
                                    "配置已在其他页面更新："
                                    f"当前为 r{current_revision}，"
                                    f"本页面基于 r{expected_revision}"
                                ),
                            }
                        ]
                    )
                revision = current_revision + 1
                created_at = existing.get("created_at") or now
                parent_version = existing.get("parent_version")
            payload: dict[str, Any] = {
                "schema_version": 2,
                "profile_id": profile_id,
                "name": profile_id,
                "version": version,
                "revision": revision,
                "dataset": dataset,
                "command": command,
                "parent_version": parent_version,
                "notes": notes.strip(),
                "created_at": created_at,
                "updated_at": now,
                "overrides": dict(sorted(overrides.items())),
                "resolved": dict(sorted(resolved.items())),
            }
            path = self._profile_path(profile_dir, version)
            _atomic_write_json(path, payload)
        return payload

    @staticmethod
    def _version_numbers(profile_dir: Path) -> list[int]:
        versions: list[int] = []
        for path in profile_dir.glob("v[0-9][0-9][0-9][0-9].json"):
            try:
                versions.append(int(path.stem[1:]))
            except ValueError:
                continue
        return sorted(versions)

    def get_profile(self, profile_id: str, version: int | None = None) -> dict[str, Any]:
        profile_dir = self._profile_dir(profile_id)
        versions = self._version_numbers(profile_dir)
        if not versions:
            raise FileNotFoundError(f"配置不存在：{profile_id}")
        selected = versions[-1] if version is None else int(version)
        path = self._profile_path(profile_dir, selected)
        if not path.is_file():
            raise FileNotFoundError(f"配置版本不存在：{profile_id} v{selected}")
        profile = _read_json(path)
        profile.setdefault("revision", 1)
        profile.setdefault("updated_at", profile.get("created_at"))
        return profile

    def list_profiles(self) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for profile_dir in sorted(self.profiles_dir.iterdir()):
            if not profile_dir.is_dir():
                continue
            try:
                profile_id = validate_profile_id(profile_dir.name)
            except ConfigValidationError:
                continue
            versions = self._version_numbers(profile_dir)
            if not versions:
                continue
            rows: list[dict[str, Any]] = []
            for version in reversed(versions):
                profile = _read_json(self._profile_path(profile_dir, version))
                rows.append(
                    {
                        "version": version,
                        "revision": max(1, int(profile.get("revision", 1))),
                        "dataset": profile.get("dataset"),
                        "command": profile.get("command"),
                        "created_at": profile.get("created_at"),
                        "updated_at": (
                            profile.get("updated_at")
                            or profile.get("created_at")
                        ),
                        "parent_version": profile.get("parent_version"),
                        "override_count": len(profile.get("overrides", {})),
                    }
                )
            profiles.append(
                {
                    "profile_id": profile_id,
                    "latest_version": versions[-1],
                    "versions": rows,
                }
            )
        return profiles

    def profile_env(self, profile_id: str, version: int) -> str:
        profile = self.get_profile(profile_id, version)
        return env_text(profile["resolved"])

    def create_run(
        self,
        profile_id: str,
        version: int,
        *,
        runtime_environment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self.get_profile(profile_id, version)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{RUN_ID_PREFIX}{timestamp}_{secrets.token_hex(3)}"
        run_dir = self.runs_dir / run_id
        resolved = profile["resolved"]
        frozen_runtime_environment = snapshot_runtime_environment(
            os.environ if runtime_environment is None else runtime_environment
        )
        command_argv = [
            "bash",
            "scripts/router_pipeline.sh",
            profile["dataset"],
            profile["command"],
        ]
        with self.lock():
            run_dir.mkdir(parents=False, exist_ok=False)
            config_payload = {
                "schema_version": 1,
                "run_id": run_id,
                "profile_id": profile["profile_id"],
                "profile_version": profile["version"],
                "profile_revision": profile.get("revision", 1),
                "dataset": profile["dataset"],
                "command": profile["command"],
                "created_at": utc_now(),
                "overrides": profile["overrides"],
                "resolved": resolved,
                "runtime_env": frozen_runtime_environment,
            }
            config_path = run_dir / "config.json"
            env_path = run_dir / "config.env"
            log_path = run_dir / "train.log"
            runner_log_path = run_dir / "runner.log"
            _atomic_write_json(config_path, config_payload)
            _atomic_write_text(env_path, env_text(resolved))
            run_payload: dict[str, Any] = {
                "schema_version": 1,
                "run_id": run_id,
                "profile_id": profile["profile_id"],
                "profile_version": profile["version"],
                "profile_revision": profile.get("revision", 1),
                "dataset": profile["dataset"],
                "command": profile["command"],
                "status": "queued",
                "stage": "等待独立运行器",
                "runner_pid": None,
                "training_pid": None,
                "training_pgid": None,
                "created_at": utc_now(),
                "started_at": None,
                "finished_at": None,
                "updated_at": utc_now(),
                "exit_code": None,
                "latest_checkpoint": "",
                "progress_text": "已保存不可变运行快照",
                "command_argv": command_argv,
                "config_path": str(config_path),
                "env_path": str(env_path),
                "log_path": str(log_path),
                "runner_log_path": str(runner_log_path),
                "run_dir": str(run_dir),
                "artifact_run_dir": str(resolved.get("RUN_DIR", "")),
                **gpu_contract(resolved),
                "gpu_bindings": [],
                "runtime_visible_devices": "",
                "gpu_binding_verified": False,
            }
            _atomic_write_json(run_dir / "run.json", run_payload)
        return run_payload

    def _run_dir(self, run_id: str) -> Path:
        if (
            not run_id.startswith(RUN_ID_PREFIX)
            or "/" in run_id
            or "\\" in run_id
            or ".." in run_id
        ):
            raise ValueError("invalid run id")
        return self.runs_dir / run_id

    def get_run(self, run_id: str, *, observe: bool = True) -> dict[str, Any]:
        path = self._run_dir(run_id) / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"运行不存在：{run_id}")
        payload = _read_json(path)
        if "configured_gpus" not in payload:
            try:
                config = _read_json(Path(payload["config_path"]))
                resolved = config.get("resolved", {})
                if isinstance(resolved, dict):
                    payload.update(gpu_contract(resolved))
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                pass
        if observe:
            payload = dict(payload)
            payload["runner_alive"] = pid_alive(payload.get("runner_pid"))
            payload["training_alive"] = pid_alive(payload.get("training_pid"))
            if (
                payload.get("status") in {"starting", "running"}
                and not payload["runner_alive"]
                and not payload["training_alive"]
            ):
                payload["stored_status"] = payload["status"]
                payload["status"] = "unknown"
        return payload

    def update_run(self, run_id: str, **changes: Any) -> dict[str, Any]:
        path = self._run_dir(run_id) / "run.json"
        with self.lock():
            if not path.is_file():
                raise FileNotFoundError(f"运行不存在：{run_id}")
            payload = _read_json(path)
            payload.update(changes)
            payload["updated_at"] = utc_now()
            _atomic_write_json(path, payload)
        return payload

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        candidates = sorted(
            (
                path
                for path in self.runs_dir.iterdir()
                if path.is_dir() and path.name.startswith(RUN_ID_PREFIX)
            ),
            reverse=True,
        )
        for path in candidates[: max(0, min(int(limit), 200))]:
            try:
                rows.append(self.get_run(path.name))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return rows

    def tail_log(
        self,
        run_id: str,
        lines: int = 200,
        *,
        secret_values: Iterable[str] = (),
    ) -> str:
        run = self.get_run(run_id, observe=False)
        path = Path(run["log_path"])
        if not path.is_file():
            return ""
        maximum = max(1, min(int(lines), 2000))
        return redact_sensitive_text(
            _bounded_log_tail(path, maximum),
            secret_values,
        )
