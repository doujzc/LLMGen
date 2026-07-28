#!/usr/bin/env python3
"""Serve the independent LLMGen training configuration console."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .config import ALLOWED_KEYS, ConfigResolver, ConfigValidationError, DATASETS
from .store import (
    StateStore,
    child_process_environment,
    env_text,
    secret_runtime_environment,
    secure_binary_append,
    snapshot_runtime_environment,
    utc_now,
)


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
REPO_ROOT = PACKAGE_DIR.parent
STATIC_ROUTES = {
    "/": STATIC_DIR / "index.html",
    "/static/app.js": STATIC_DIR / "app.js",
    "/static/styles.css": STATIC_DIR / "styles.css",
    "/static/skill-router-mark.svg": STATIC_DIR / "skill-router-mark.svg",
}


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host is restricted to the local machine."""

    normalized = str(host).strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if "%" in normalized:
        normalized = normalized.split("%", 1)[0]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def request_host_is_loopback(host: str) -> bool:
    """Return whether an HTTP Host header names a loopback authority."""

    try:
        parsed = urlparse(f"//{host.strip()}")
        hostname = (parsed.hostname or "").rstrip(".").lower()
        parsed.port
    except ValueError:
        return False
    if (
        parsed.username
        or parsed.password
        or not parsed.netloc
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return False
    return is_loopback_host(hostname)


def origin_matches_host(origin: str, host: str) -> bool:
    """Validate a browser Origin and Host as the same loopback authority."""

    if not origin:
        return True
    try:
        parsed = urlparse(origin)
        request_authority = urlparse(f"//{host.strip()}")
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if (
        parsed.username
        or parsed.password
        or not parsed.netloc
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return False
    if (
        request_authority.username
        or request_authority.password
        or not request_authority.netloc
        or request_authority.path
        or request_authority.params
        or request_authority.query
        or request_authority.fragment
    ):
        return False
    origin_host = (parsed.hostname or "").rstrip(".").lower()
    request_host = (request_authority.hostname or "").rstrip(".").lower()
    if not (
        is_loopback_host(origin_host)
        and is_loopback_host(request_host)
        and origin_host == request_host
    ):
        return False
    try:
        default_port = 80 if parsed.scheme == "http" else 443
        origin_port = parsed.port or default_port
        request_port = request_authority.port or default_port
    except ValueError:
        return False
    return origin_port == request_port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Loopback bind only; use SSH remotely, or an authenticated proxy "
            "that rewrites Host and Origin to loopback."
        ),
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument(
        "--inference-url",
        default="http://127.0.0.1:8080/",
        help="Link shown for the separate inference console.",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Save run snapshots without launching processes (UI/QA use only).",
    )
    return parser.parse_args()


def probe_gpu_metrics() -> list[dict[str, Any]] | None:
    """Return a lightweight NVIDIA GPU snapshot without importing torch."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                (
                    "--query-gpu=index,name,utilization.gpu,memory.used,"
                    "memory.total,temperature.gpu"
                ),
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", 5)]
        if len(parts) != 6:
            continue
        try:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "utilization": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "memory_total_mib": int(parts[4]),
                    "temperature_c": int(parts[5]),
                }
            )
        except ValueError:
            continue
    return rows


def probe_gpu_count() -> int | None:
    """Return the current NVIDIA GPU count when nvidia-smi is available."""

    metrics = probe_gpu_metrics()
    return None if metrics is None else len(metrics)


class TrainingConsoleService:
    """Application service that never imports or owns the training implementation."""

    def __init__(
        self,
        *,
        repo_root: Path,
        state_root: Path,
        inference_url: str,
        launch_enabled: bool = True,
        launcher: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        self.repo_root = repo_root.expanduser().resolve()
        self.store = StateStore(state_root)
        resolver_environment = snapshot_runtime_environment(os.environ)
        resolver_environment.update(
            {
                key: value
                for key, value in os.environ.items()
                if key in ALLOWED_KEYS
            }
        )
        self.resolver = ConfigResolver(
            self.repo_root,
            inherited_env=resolver_environment,
        )
        self.inference_url = inference_url
        self.launch_enabled = launch_enabled
        self._launcher = launcher

    def health(self) -> dict[str, Any]:
        profiles = self.store.list_profiles()
        runs = self.store.list_runs(limit=200)
        active = sum(
            run.get("status") in {"queued", "starting", "running", "stopping"}
            for run in runs
        )
        gpus = probe_gpu_metrics()
        return {
            "ready": True,
            "service": "training-console",
            "repo_root": str(self.repo_root),
            "state_root": str(self.store.root),
            "launch_enabled": self.launch_enabled,
            "inference_url": self.inference_url,
            "profile_count": len(profiles),
            "run_count": len(runs),
            "active_runs": active,
            "gpu_count": None if gpus is None else len(gpus),
            "gpus": gpus or [],
        }

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        overrides = payload.get("overrides", {})
        if not isinstance(overrides, dict) or len(overrides) > 200:
            raise ConfigValidationError(
                [{"field": "overrides", "message": "配置覆盖必须是对象"}]
            )
        dataset = str(payload.get("dataset", "clawhub"))
        command = str(payload.get("command", "full"))
        result = self.resolver.validate(dataset, command, overrides)
        result["env_text"] = env_text(result["resolved"])
        return result

    def save_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate(payload)
        selected = payload.get("version", payload.get("parent_version"))
        expected = payload.get("expected_revision")
        try:
            version = None if selected in (None, "") else int(selected)
            expected_revision = (
                None if expected in (None, "") else int(expected)
            )
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                [
                    {
                        "field": "version",
                        "message": "配置版本和修订号必须是整数",
                    }
                ]
            ) from exc
        profile = self.store.save_profile(
            profile_id=str(payload.get("profile_id", "")),
            dataset=validation["dataset"],
            command=validation["command"],
            overrides=validation["overrides"],
            resolved=validation["resolved"],
            notes=str(payload.get("notes", "")),
            version=version,
            expected_revision=expected_revision,
        )
        return {
            "profile": profile,
            "validation": validation,
        }

    def _launch(self, run: dict[str, Any]) -> int:
        if self._launcher is not None:
            return int(self._launcher(run))
        with Path(run["config_path"]).open("r", encoding="utf-8") as stream:
            config_snapshot = json.load(stream)
        runtime_environment = config_snapshot.get("runtime_env", {})
        if not isinstance(runtime_environment, dict):
            raise ValueError("run snapshot has an invalid runtime environment")
        runner_environment = child_process_environment(
            runtime_environment,
            os.environ,
        )
        runner_log_path = Path(run["runner_log_path"])
        runner_log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            "-m",
            "training_console.runner",
            "--repo-root",
            str(self.repo_root),
            "--state-root",
            str(self.store.root),
            "--run-id",
            run["run_id"],
        ]
        with secure_binary_append(runner_log_path) as runner_log:
            process = subprocess.Popen(
                argv,
                cwd=self.repo_root,
                env=runner_environment,
                stdin=subprocess.DEVNULL,
                stdout=runner_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        return int(process.pid)

    def submit_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(payload.get("profile_id", ""))
        try:
            version = int(payload.get("version"))
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                [{"field": "version", "message": "请选择已保存的配置版本"}]
            ) from exc
        run = self.store.create_run(
            profile_id,
            version,
            runtime_environment=snapshot_runtime_environment(os.environ),
        )
        if not self.launch_enabled:
            return self.store.update_run(
                run["run_id"],
                status="saved",
                stage="未启动",
                progress_text="服务使用 --no-launch，仅保存了运行快照",
            )
        self.store.update_run(
            run["run_id"],
            status="starting",
            stage="启动独立运行器",
            started_at=utc_now(),
        )
        try:
            runner_pid = self._launch(run)
        except Exception as exc:
            return self.store.update_run(
                run["run_id"],
                status="failed_to_start",
                stage="独立运行器启动失败",
                progress_text=str(exc),
                finished_at=utc_now(),
            )
        self.store.update_run(run["run_id"], runner_pid=runner_pid)
        return self.store.get_run(run["run_id"])

    def request_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist a cooperative stop request for the detached runner."""

        run_id = str(payload.get("run_id", "")).strip()
        if not run_id:
            raise ConfigValidationError(
                [{"field": "run_id", "message": "请选择要停止的运行"}]
            )
        run = self.store.get_run(run_id)
        status = str(run.get("status", ""))
        if status == "stopping":
            return run
        if status not in {"queued", "starting", "running", "unknown"}:
            raise ConfigValidationError(
                [
                    {
                        "field": "run_id",
                        "message": f"当前状态不可停止：{status or 'unknown'}",
                    }
                ]
            )
        requested_at = utc_now()
        if not run.get("runner_alive") and not run.get("training_alive"):
            return self.store.update_run(
                run_id,
                status="stopped",
                stage="用户停止",
                progress_text="停止请求已记录；运行进程已不存在",
                stop_requested_at=requested_at,
                stop_requested_stage=run.get("stage", ""),
                stopped_at=requested_at,
                finished_at=requested_at,
            )
        return self.store.update_run(
            run_id,
            status="stopping",
            stage="正在停止",
            progress_text="停止请求已写入磁盘，等待独立运行器安全退出",
            stop_requested_at=requested_at,
            stop_requested_stage=run.get("stage", ""),
        )

    def tail_log(self, run_id: str, lines: int) -> str:
        """Return a redacted training-log tail for the browser."""

        secrets = secret_runtime_environment(os.environ)
        return self.store.tail_log(
            run_id,
            lines,
            secret_values=secrets.values(),
        )


def handler_class(service: TrainingConsoleService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LLMGenTrainingConsole/1.0"

        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._headers(
                status,
                "application/json; charset=utf-8",
                len(body),
            )
            self.wfile.write(body)

        def _text(
            self,
            status: int,
            text: str,
            *,
            content_type: str = "text/plain; charset=utf-8",
            filename: str | None = None,
        ) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{filename}"',
                )
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, **extra: Any) -> None:
            self._json(status, {"error": message, **extra})

        def _query(self, parsed, key: str, default: str = "") -> str:
            return parse_qs(parsed.query).get(key, [default])[0]

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if not request_host_is_loopback(self.headers.get("Host", "")):
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "requests require a loopback Host",
                )
                return
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    self._json(HTTPStatus.OK, service.health())
                    return
                if parsed.path == "/api/schema":
                    dataset = self._query(parsed, "dataset", "clawhub")
                    if dataset not in DATASETS:
                        raise ConfigValidationError(
                            [{"field": "dataset", "message": "未知数据集"}]
                        )
                    self._json(HTTPStatus.OK, service.resolver.schema(dataset))
                    return
                if parsed.path == "/api/profiles":
                    self._json(
                        HTTPStatus.OK,
                        {"profiles": service.store.list_profiles()},
                    )
                    return
                if parsed.path == "/api/profile":
                    profile_id = self._query(parsed, "id")
                    version_text = self._query(parsed, "version")
                    version = int(version_text) if version_text else None
                    self._json(
                        HTTPStatus.OK,
                        service.store.get_profile(profile_id, version),
                    )
                    return
                if parsed.path == "/api/profile-env":
                    profile_id = self._query(parsed, "id")
                    version = int(self._query(parsed, "version"))
                    self._text(
                        HTTPStatus.OK,
                        service.store.profile_env(profile_id, version),
                        filename=f"{profile_id}-v{version}.env",
                    )
                    return
                if parsed.path == "/api/runs":
                    limit = int(self._query(parsed, "limit", "20"))
                    self._json(
                        HTTPStatus.OK,
                        {"runs": service.store.list_runs(limit)},
                    )
                    return
                if parsed.path == "/api/run":
                    self._json(
                        HTTPStatus.OK,
                        service.store.get_run(self._query(parsed, "id")),
                    )
                    return
                if parsed.path == "/api/run-log":
                    lines = int(self._query(parsed, "tail", "200"))
                    self._json(
                        HTTPStatus.OK,
                        {
                            "run_id": self._query(parsed, "id"),
                            "text": service.tail_log(
                                self._query(parsed, "id"),
                                lines,
                            ),
                        },
                    )
                    return
            except ConfigValidationError as exc:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                    errors=exc.errors,
                )
                return
            except (FileNotFoundError, ValueError) as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
                return
            except (OSError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return

            static_path = STATIC_ROUTES.get(parsed.path)
            if static_path is None or not static_path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            body = static_path.read_bytes()
            content_type, _ = mimetypes.guess_type(static_path.name)
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type",
                content_type or "application/octet-stream",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            request_path = urlparse(self.path).path
            if request_path not in {
                "/api/validate",
                "/api/profiles",
                "/api/runs",
                "/api/runs/stop",
            }:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            if self.headers.get_content_type() != "application/json":
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "Content-Type must be application/json",
                )
                return
            origin = self.headers.get("Origin", "")
            host = self.headers.get("Host", "")
            if not request_host_is_loopback(host) or not origin_matches_host(
                origin,
                host,
            ):
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "POST requests require a same-origin loopback Host",
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 1_000_000:
                    raise ValueError("invalid request body size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                if request_path == "/api/validate":
                    result = service.validate(payload)
                elif request_path == "/api/profiles":
                    result = service.save_profile(payload)
                elif request_path == "/api/runs":
                    result = service.submit_run(payload)
                else:
                    result = service.request_stop(payload)
                self._json(HTTPStatus.OK, result)
            except ConfigValidationError as exc:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                    errors=exc.errors,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except FileNotFoundError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
            except Exception as exc:
                print(f"training console request failed: {exc}", file=sys.stderr)
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "request failed")

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    return Handler


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    if not (repo_root / "scripts/router_pipeline.sh").is_file():
        raise SystemExit(f"not an LLMGen repository: {repo_root}")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if not is_loopback_host(args.host):
        raise SystemExit(
            "--host must be a loopback address; use an SSH tunnel or "
            "authenticated reverse proxy for remote access"
        )
    state_root = args.state_root
    if state_root is None:
        state_root = Path(
            os.environ.get(
                "LLMGEN_TRAINING_CONSOLE_STATE",
                repo_root / ".llmgen/training-console",
            )
        )
    service = TrainingConsoleService(
        repo_root=repo_root,
        state_root=state_root,
        inference_url=args.inference_url,
        launch_enabled=not args.no_launch,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class(service))
    print(
        f"LLMGen training console: http://{args.host}:{args.port} "
        f"(state: {service.store.root})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping training console Web service.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
