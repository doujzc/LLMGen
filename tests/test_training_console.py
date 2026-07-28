from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from training_console.config import ConfigResolver, ConfigValidationError
from training_console.server import (
    TrainingConsoleService,
    handler_class,
    is_loopback_host,
    origin_matches_host,
    request_host_is_loopback,
)
from training_console.store import (
    LOG_TAIL_MAX_BYTES,
    LOG_TAIL_MAX_LINE_BYTES,
    LOG_TAIL_READ_CHUNK_BYTES,
    StateStore,
    child_process_environment,
    redact_sensitive_text,
    snapshot_runtime_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_env() -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
    }


def _request(
    url: str,
    payload: dict | None = None,
    *,
    headers: dict[str, str] | None = None,
):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"} if body else {}
    request_headers.update(headers or {})
    request = Request(
        url,
        data=body,
        headers=request_headers,
    )
    with urlopen(request, timeout=5) as response:
        content_type = response.headers.get_content_type()
        data = response.read()
        if content_type == "application/json":
            return response.status, json.loads(data)
        return response.status, data.decode("utf-8")


def _error_status(request: Request) -> int:
    with pytest.raises(HTTPError) as caught:
        urlopen(request, timeout=5)
    return caught.value.code


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _link_console_package(fake_repo: Path) -> None:
    (fake_repo / "training_console").symlink_to(
        REPO_ROOT / "training_console",
        target_is_directory=True,
    )


def test_schema_covers_every_training_stage_and_resolves_dataset_defaults() -> None:
    resolver = ConfigResolver(REPO_ROOT, inherited_env=_clean_env())

    clawhub = resolver.schema("clawhub")
    light = resolver.schema("light")

    assert {stage["id"] for stage in clawhub["stages"]} == {
        "base",
        "embedding",
        "tokenizer",
        "code",
        "router_data",
        "memorization",
        "alignment",
        "retrieval",
        "evaluation",
    }
    assert len(clawhub["fields"]) >= 90
    assert clawhub["defaults"]["BRANCHING_FACTORS"] == "128 128"
    assert light["defaults"]["BRANCHING_FACTORS"] == "32 16"
    assert clawhub["defaults"]["ROUTER_FINETUNE_MODE"] == "full"
    assert clawhub["secrets"]["OPENAI_API_KEY"]["persisted"] is False


def test_config_resolution_is_anchored_to_repo_and_ignores_bash_env(
    tmp_path,
) -> None:
    injection = tmp_path / "bash-env.sh"
    injection.write_text(
        "export ROUTER_MODEL=unexpected-injected-model\n",
        encoding="utf-8",
    )
    inherited = {
        **_clean_env(),
        "SKILLRET_ROOT": "/missing/stale-clone",
        "BASH_ENV": str(injection),
    }

    schema = ConfigResolver(REPO_ROOT, inherited_env=inherited).schema("clawhub")

    assert schema["defaults"]["ROUTER_MODEL"] == "Qwen/Qwen3-1.7B"
    assert schema["defaults"]["NUM_LEVELS"] == "2"
    assert schema["defaults"]["ROUTER_NUM_GPUS"] == "4"


def test_config_resolution_propagates_nested_source_failures(tmp_path) -> None:
    fake_repo = tmp_path / "repo"
    configs = fake_repo / "configs"
    configs.mkdir(parents=True)
    (configs / "clawhub.env").write_text(
        'source "$SKILLRET_ROOT/configs/missing.env"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError) as caught:
        ConfigResolver(fake_repo, inherited_env=_clean_env()).schema("clawhub")

    assert caught.value.errors[0]["field"] == "DATASET"
    assert "missing.env" in caught.value.errors[0]["message"]


def test_validation_rejects_unknown_values_newlines_and_gpu_mismatch() -> None:
    resolver = ConfigResolver(REPO_ROOT, inherited_env=_clean_env())

    with pytest.raises(ConfigValidationError) as unknown:
        resolver.validate("clawhub", "full", {"SHELL_COMMAND": "rm -rf /"})
    assert unknown.value.errors[0]["field"] == "SHELL_COMMAND"

    with pytest.raises(ConfigValidationError) as newline:
        resolver.validate("clawhub", "full", {"RUN_DIR": "runs/a\nBAD=1"})
    assert newline.value.errors[0]["field"] == "RUN_DIR"

    with pytest.raises(ConfigValidationError) as mismatch:
        resolver.validate(
            "clawhub",
            "full",
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "ROUTER_NUM_GPUS": "4",
            },
        )
    assert any(
        error["field"] == "ROUTER_NUM_GPUS"
        for error in mismatch.value.errors
    )


def test_runtime_environment_is_allowlisted_frozen_and_excludes_secrets() -> None:
    captured = snapshot_runtime_environment(
        {
            "PATH": "/trusted/bin",
            "HOME": "/home/tester",
            "NCCL_DEBUG": "INFO",
            "OMP_NUM_THREADS": "4",
            "OPENAI_API_KEY": "must-not-persist",
            "HTTPS_PROXY": "https://user:proxy-secret@proxy.example",
            "BASH_ENV": "/tmp/inject.sh",
            "PYTHONPATH": "/tmp/import-inject",
            "PREPARE_SCRIPT": "/tmp/prepare.py",
            "ROUTER_EXTRA_ARGS": "--unexpected-option",
            "UNRELATED_VALUE": "ignored",
        }
    )

    assert captured == {
        "HOME": "/home/tester",
        "NCCL_DEBUG": "INFO",
        "OMP_NUM_THREADS": "4",
        "PATH": "/trusted/bin",
    }
    child = child_process_environment(
        captured,
        {
            "OPENAI_API_KEY": "runtime-secret",
            "HTTPS_PROXY": "https://user:proxy-secret@proxy.example",
            "BASH_ENV": "/tmp/inject.sh",
        },
    )
    assert child["OPENAI_API_KEY"] == "runtime-secret"
    assert child["HTTPS_PROXY"].endswith("@proxy.example")
    assert "BASH_ENV" not in child


def test_loopback_bind_policy() -> None:
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("[::1]")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.10")
    assert not is_loopback_host("training.example.com")
    assert origin_matches_host(
        "http://127.0.0.1:8090",
        "127.0.0.1:8090",
    )
    assert origin_matches_host("https://localhost", "localhost")
    assert request_host_is_loopback("localhost:8090")
    assert request_host_is_loopback("[::1]:8090")
    assert not request_host_is_loopback("attacker.example:8090")
    assert not request_host_is_loopback("[malformed")
    assert not origin_matches_host(
        "http://attacker.example:8090",
        "attacker.example:8090",
    )
    assert not origin_matches_host(
        "http://localhost:8090",
        "localhost:8091",
    )


def test_log_redaction_covers_known_values_keys_and_bearer_tokens() -> None:
    redacted = redact_sensitive_text(
        "raw-value\n"
        'api_key="second-value"\n'
        "Authorization: Bearer bearer-token\n"
        "password=hunter2\n"
        "HTTPS_PROXY=https://proxy-user:proxy-pass@proxy.example\n"
        "mirror=socks5://mirror-user:mirror-pass@mirror.example\n",
        ["raw-value"],
    )

    assert "raw-value" not in redacted
    assert "second-value" not in redacted
    assert "bearer-token" not in redacted
    assert "hunter2" not in redacted
    assert "proxy-user" not in redacted
    assert "proxy-pass" not in redacted
    assert "mirror-user" not in redacted
    assert "mirror-pass" not in redacted
    assert "HTTPS_PROXY=[REDACTED]" in redacted
    assert "socks5://[REDACTED]@mirror.example" in redacted
    assert redacted.count("[REDACTED]") == 6


def test_log_tail_bounds_bytes_lines_and_single_line_length(
    tmp_path,
    monkeypatch,
) -> None:
    store = StateStore(tmp_path / "state")
    profile = store.save_profile(
        profile_id="bounded-log-test",
        dataset="clawhub",
        command="full",
        overrides={},
        resolved={},
    )
    run = store.create_run(
        profile["profile_id"],
        profile["version"],
        runtime_environment={},
    )
    log_path = Path(run["log_path"])
    log_path.write_bytes(
        b"EARLY-MARKER"
        + (b"x" * (LOG_TAIL_MAX_BYTES + 4096))
        + b"TAIL-MARKER"
    )

    original_open = Path.open
    read_sizes: list[int] = []

    class GuardedLogReader:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.stream.close()

        def seek(self, *args):
            return self.stream.seek(*args)

        def tell(self):
            return self.stream.tell()

        def read(self, size: int = -1):
            assert size >= 0, "log tail must never issue an unbounded read"
            read_sizes.append(size)
            return self.stream.read(size)

    def guarded_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path == log_path and mode == "rb":
            return GuardedLogReader(stream)
        return stream

    monkeypatch.setattr(Path, "open", guarded_open)
    bounded = store.tail_log(run["run_id"], lines=1)

    assert read_sizes
    assert sum(read_sizes) <= LOG_TAIL_MAX_BYTES
    assert max(read_sizes) <= LOG_TAIL_READ_CHUNK_BYTES
    assert "EARLY-MARKER" not in bounded
    assert bounded.endswith("TAIL-MARKER")
    assert "log line truncated" in bounded
    assert len(bounded.encode("utf-8")) <= LOG_TAIL_MAX_LINE_BYTES

    log_path.write_text(
        "".join(f"line-{index}\n" for index in range(20)),
        encoding="utf-8",
    )
    assert store.tail_log(run["run_id"], lines=3).splitlines() == [
        "line-17",
        "line-18",
        "line-19",
    ]


def test_profile_versions_are_immutable_and_export_review_only_env(tmp_path) -> None:
    resolver = ConfigResolver(REPO_ROOT, inherited_env=_clean_env())
    store = StateStore(tmp_path / "state")
    first = resolver.validate("clawhub", "full", {"ROUTER_RETRIEVAL_EPOCHS": "7"})
    v1 = store.save_profile(
        profile_id="clawhub-full-4gpu",
        dataset="clawhub",
        command="full",
        overrides=first["overrides"],
        resolved=first["resolved"],
    )
    second = resolver.validate("clawhub", "full", {"ROUTER_RETRIEVAL_EPOCHS": "9"})
    v2 = store.save_profile(
        profile_id="clawhub-full-4gpu",
        dataset="clawhub",
        command="full",
        overrides=second["overrides"],
        resolved=second["resolved"],
        parent_version=1,
    )

    assert (v1["version"], v2["version"]) == (1, 2)
    assert store.get_profile("clawhub-full-4gpu", 1)["resolved"][
        "ROUTER_RETRIEVAL_EPOCHS"
    ] == "7"
    assert store.get_profile("clawhub-full-4gpu", 2)["resolved"][
        "ROUTER_RETRIEVAL_EPOCHS"
    ] == "9"
    exported = store.profile_env("clawhub-full-4gpu", 2)
    assert "ROUTER_RETRIEVAL_EPOCHS=9" in exported
    assert "OPENAI_API_KEY" not in exported


def test_http_api_saves_profiles_and_run_snapshots_without_launching(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("OPENAI_API_KEY", "snapshot-secret")
    monkeypatch.setenv("BASH_ENV", "/tmp/should-not-run")
    monkeypatch.setenv("PYTHONPATH", "/tmp/should-not-import")
    monkeypatch.setenv("PREPARE_SCRIPT", "/tmp/should-not-prepare")
    monkeypatch.setenv("ROUTER_EXTRA_ARGS", "--should-not-pass")
    service = TrainingConsoleService(
        repo_root=REPO_ROOT,
        state_root=tmp_path / "state",
        inference_url="http://127.0.0.1:8080/",
        launch_enabled=False,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, schema = _request(base + "/api/schema?dataset=clawhub")
        assert status == 200
        assert schema["default_profile_id"] == "clawhub-full-4gpu"

        _, saved = _request(
            base + "/api/profiles",
            {
                "profile_id": "clawhub-full-4gpu",
                "dataset": "clawhub",
                "command": "full",
                "overrides": {"ROUTER_RETRIEVAL_EPOCHS": "2"},
            },
        )
        assert saved["profile"]["version"] == 1

        _, run = _request(
            base + "/api/runs",
            {
                "profile_id": "clawhub-full-4gpu",
                "version": 1,
            },
        )
        assert run["status"] == "saved"
        assert Path(run["config_path"]).is_file()
        assert Path(run["env_path"]).is_file()
        assert Path(run["log_path"]).exists() is False
        snapshot_text = Path(run["config_path"]).read_text(encoding="utf-8")
        snapshot = json.loads(snapshot_text)
        assert snapshot["runtime_env"]["NCCL_DEBUG"] == "INFO"
        assert "OPENAI_API_KEY" not in snapshot_text
        assert "BASH_ENV" not in snapshot["runtime_env"]
        assert "PYTHONPATH" not in snapshot["runtime_env"]
        assert "PREPARE_SCRIPT" not in snapshot["runtime_env"]
        assert "ROUTER_EXTRA_ARGS" not in snapshot["runtime_env"]

        _, runs = _request(base + "/api/runs?limit=10")
        assert runs["runs"][0]["run_id"] == run["run_id"]

        with urlopen(base + "/", timeout=5) as response:
            page = response.read().decode("utf-8")
        assert "独立任务" in page
        assert "训练控制台" in page
        assert 'id="run-runner-pid"' in page
        assert 'id="run-training-pid"' in page
        assert 'id="run-exit-code"' in page

        with urlopen(base + "/static/styles.css", timeout=5) as response:
            styles = response.read().decode("utf-8")
        assert "@media (max-width: 1320px)" in styles

        _, validated = _request(
            base + "/api/validate",
            {
                "dataset": "clawhub",
                "command": "full",
                "overrides": {},
            },
        )
        assert validated["contract"]["log_dir"] == ""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_post_requires_json_and_rejects_cross_origin(tmp_path) -> None:
    service = TrainingConsoleService(
        repo_root=REPO_ROOT,
        state_root=tmp_path / "state",
        inference_url="",
        launch_enabled=False,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    body = json.dumps(
        {
            "dataset": "clawhub",
            "command": "full",
            "overrides": {},
        }
    ).encode("utf-8")
    try:
        wrong_type = Request(
            base + "/api/validate",
            data=body,
            headers={"Content-Type": "text/plain"},
        )
        assert _error_status(wrong_type) == 415

        cross_origin = Request(
            base + "/api/validate",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://attacker.example",
            },
        )
        assert _error_status(cross_origin) == 403

        rebound_origin = Request(
            base + "/api/validate",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Host": f"attacker.example:{server.server_port}",
                "Origin": f"http://attacker.example:{server.server_port}",
            },
        )
        assert _error_status(rebound_origin) == 403

        rebound_get = Request(
            base + "/api/profiles",
            headers={"Host": f"attacker.example:{server.server_port}"},
        )
        assert _error_status(rebound_get) == 403

        status, result = _request(
            base + "/api/validate",
            {
                "dataset": "clawhub",
                "command": "full",
                "overrides": {},
            },
            headers={"Origin": base},
        )
        assert status == 200
        assert result["dataset"] == "clawhub"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_detached_runner_survives_without_a_web_request_owner(
    tmp_path,
    monkeypatch,
) -> None:
    fake_repo = tmp_path / "repo"
    scripts = fake_repo / "scripts"
    scripts.mkdir(parents=True)
    _link_console_package(fake_repo)
    pipeline = scripts / "router_pipeline.sh"
    pipeline.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$RUN_DIR\"\n"
        "printf '[06b] detached retrieval\\n'\n"
        "printf 'OPENAI_API_KEY=%s\\n' \"${OPENAI_API_KEY:-missing}\"\n"
        "printf 'Authorization: Bearer direct-bearer-token\\n'\n"
        "{\n"
        "  printf 'safe=%s,%s\\n' \"${NCCL_DEBUG:-}\" \"${OMP_NUM_THREADS:-}\"\n"
        "  printf 'blocked=%s,%s,%s,%s\\n' "
        "\"${BASH_ENV:-}\" \"${PYTHONPATH:-}\" "
        "\"${PREPARE_SCRIPT:-}\" \"${ROUTER_EXTRA_ARGS:-}\"\n"
        "  printf 'secret=%s\\n' \"${OPENAI_API_KEY:-}\"\n"
        "  printf 'started\\n'\n"
        "} > \"$RUN_DIR/marker.txt\"\n"
        "sleep 0.6\n"
        "printf 'finished\\n' >> \"$RUN_DIR/marker.txt\"\n",
        encoding="utf-8",
    )
    pipeline.chmod(0o755)
    state_root = tmp_path / "state"
    artifact_dir = tmp_path / "artifacts"
    store = StateStore(state_root)
    profile = store.save_profile(
        profile_id="detached-test",
        dataset="clawhub",
        command="full",
        overrides={"RUN_DIR": str(artifact_dir)},
        resolved={"RUN_DIR": str(artifact_dir)},
    )
    injection_marker = tmp_path / "bash-env-was-sourced"
    injection_script = tmp_path / "inject.sh"
    injection_script.write_text(
        f"touch {injection_marker}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BASH_ENV", str(injection_script))
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-pythonpath")
    monkeypatch.setenv("PREPARE_SCRIPT", "/tmp/untrusted-prepare.py")
    monkeypatch.setenv("ROUTER_EXTRA_ARGS", "--untrusted-extra")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setenv("OPENAI_API_KEY", "known-openai-secret")
    service = TrainingConsoleService(
        repo_root=fake_repo,
        state_root=state_root,
        inference_url="",
        launch_enabled=True,
    )

    run = service.submit_run(
        {
            "profile_id": profile["profile_id"],
            "version": profile["version"],
        }
    )
    deadline = time.time() + 8
    observed = run
    while time.time() < deadline:
        observed = store.get_run(run["run_id"])
        if observed.get("training_pid"):
            try:
                training_session = os.getsid(observed["training_pid"])
            except ProcessLookupError:
                training_session = None
            if training_session is not None:
                assert training_session != os.getsid(os.getpid())
        if observed["status"] in {"succeeded", "failed", "failed_to_start"}:
            break
        time.sleep(0.1)

    assert observed["status"] == "succeeded"
    assert (artifact_dir / "marker.txt").read_text(encoding="utf-8") == (
        "safe=INFO,7\n"
        "blocked=,,,\n"
        "secret=known-openai-secret\n"
        "started\n"
        "finished\n"
    )
    assert not injection_marker.exists()
    raw_log = Path(observed["log_path"]).read_text(
        encoding="utf-8"
    )
    assert "detached retrieval" in raw_log
    assert "known-openai-secret" in raw_log
    redacted = service.tail_log(observed["run_id"], 200)
    assert "known-openai-secret" not in redacted
    assert "direct-bearer-token" not in redacted
    assert "OPENAI_API_KEY=[REDACTED]" in redacted
    assert "Authorization: Bearer [REDACTED]" in redacted
    assert stat.S_IMODE(Path(observed["log_path"]).stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(observed["runner_log_path"]).stat().st_mode) == 0o600


def test_training_completes_after_web_process_is_killed(tmp_path) -> None:
    fake_repo = tmp_path / "repo"
    scripts = fake_repo / "scripts"
    scripts.mkdir(parents=True)
    _link_console_package(fake_repo)
    pipeline = scripts / "router_pipeline.sh"
    pipeline.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$RUN_DIR\"\n"
        "printf 'started\\n' > \"$RUN_DIR/web-crash-marker.txt\"\n"
        "sleep 1\n"
        "printf 'finished\\n' >> \"$RUN_DIR/web-crash-marker.txt\"\n",
        encoding="utf-8",
    )
    pipeline.chmod(0o755)

    state_root = tmp_path / "state"
    artifact_dir = tmp_path / "artifacts"
    store = StateStore(state_root)
    profile = store.save_profile(
        profile_id="web-crash-test",
        dataset="clawhub",
        command="full",
        overrides={"RUN_DIR": str(artifact_dir)},
        resolved={"RUN_DIR": str(artifact_dir)},
    )
    port = _free_loopback_port()
    base = f"http://127.0.0.1:{port}"
    web_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "training_console.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--repo-root",
            str(fake_repo),
            "--state-root",
            str(state_root),
        ],
        cwd=REPO_ROOT,
        env=dict(os.environ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    run: dict = {}
    try:
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                with urlopen(base + "/api/health", timeout=0.5) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("training console did not become ready")

        _, run = _request(
            base + "/api/runs",
            {
                "profile_id": profile["profile_id"],
                "version": profile["version"],
            },
            headers={"Origin": base},
        )
        assert run["runner_pid"]
        web_process.kill()
        web_process.wait(timeout=5)

        deadline = time.time() + 10
        observed = run
        while time.time() < deadline:
            observed = store.get_run(run["run_id"])
            if observed["status"] in {"succeeded", "failed", "failed_to_start"}:
                break
            time.sleep(0.1)

        assert observed["status"] == "succeeded"
        assert (artifact_dir / "web-crash-marker.txt").read_text(
            encoding="utf-8"
        ) == "started\nfinished\n"
        assert stat.S_IMODE(Path(observed["log_path"]).stat().st_mode) == 0o600
        assert stat.S_IMODE(Path(observed["runner_log_path"]).stat().st_mode) == 0o600
    finally:
        if web_process.poll() is None:
            web_process.kill()
            web_process.wait(timeout=5)
        if run:
            try:
                observed = store.get_run(run["run_id"], observe=False)
                for key in ("training_pid", "runner_pid"):
                    pid = observed.get(key)
                    if pid and os.getsid(pid) != os.getsid(os.getpid()):
                        os.kill(pid, 15)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
