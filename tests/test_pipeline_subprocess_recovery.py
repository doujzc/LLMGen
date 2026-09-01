from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

from llmgen.pipeline.config import load_pipeline_config
from llmgen.pipeline.io import atomic_write_json, read_json
from llmgen.pipeline.logging import PipelineLogger
from llmgen.pipeline.runner import PipelineRunnerError, create_pipeline_run
from llmgen.pipeline.stages import StageResult, StageSpec
from llmgen.pipeline.stages import base as stage_base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "router_pipeline.yaml"


def _runner(tmp_path: Path, handler) -> object:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test")
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps({"id": "one", "name": "One", "description": "one"}) + "\n",
        encoding="utf-8",
    )
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "config.json").write_text("{}", encoding="utf-8")
    config = load_pipeline_config(
        CONFIG,
        candidates=candidates,
        output=tmp_path / "run",
        overrides=(
            f"router.base_model={base_model}",
            "runtime.device=cpu",
            "runtime.num_devices=1",
            "runtime.deepspeed=none",
            f"router.base_model={model}",
        ),
        environment={},
    )
    spec = StageSpec("ingest", "00_ingest", (), (), handler, "subprocess test")
    return create_pipeline_run(config, stage_specs=(spec,), repo_root=ROOT)


def _command_state(runner, attempt: int = 1) -> Path:
    return (
        runner.state.stage_dir("ingest")
        / "attempts"
        / f"{attempt:04d}"
        / "command_state"
        / "0001.json"
    )


def _record_running_command(runner, record: dict) -> Path:
    runner.state.update_stage("ingest", status="running", attempt=1)
    path = _command_state(runner)
    atomic_write_json(path, record, mode=0o600)
    return path


def _pid_is_executing(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return not stage_base._process_is_zombie(pid)


def test_run_command_persists_success_and_failure_terminal_states(
    tmp_path: Path,
) -> None:
    def succeed(context) -> StageResult:
        context.run_command(
            [sys.executable, "-c", "print('ready')"],
            label="quick-success",
        )
        return StageResult()

    success = _runner(tmp_path / "success", succeed)
    success.stage("ingest")
    record = read_json(_command_state(success))
    assert record["status"] == "completed"
    assert record["return_code"] == 0
    assert record["command_index"] == 1
    assert record["command_id"]
    assert record["host"] == socket.gethostname()
    assert record["pid"] > 0
    assert record["pgid"] > 0
    assert record["finished_at"]
    assert record["argv"][-1] == "print('ready')"

    def fail(context) -> StageResult:
        context.run_command(
            [sys.executable, "-c", "raise SystemExit(7)"],
            label="quick-failure",
        )
        return StageResult()

    failed = _runner(tmp_path / "failure", fail)
    with pytest.raises(PipelineRunnerError, match="exit code 7"):
        failed.stage("ingest")
    failed_record = read_json(_command_state(failed))
    assert failed_record["status"] == "failed"
    assert failed_record["return_code"] == 7
    assert failed_record["finished_at"]


def test_run_command_interruption_terminates_and_reaps_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_event = PipelineLogger.event

    def interrupt_on_output(self, event: str, *, level: str = "INFO", **fields) -> None:
        if event == "subprocess.output":
            raise RuntimeError("simulated runner interruption")
        original_event(self, event, level=level, **fields)

    monkeypatch.setattr(PipelineLogger, "event", interrupt_on_output)

    def handler(context) -> StageResult:
        script = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "print(child.pid, flush=True); time.sleep(30)"
        )
        context.run_command([sys.executable, "-c", script], label="interrupt-group")
        return StageResult()

    runner = _runner(tmp_path, handler)
    with pytest.raises(PipelineRunnerError, match="simulated runner interruption"):
        runner.stage("ingest")

    record = read_json(_command_state(runner))
    descendant_pid = int(
        (
            runner.state.stage_dir("ingest")
            / "attempts"
            / "0001"
            / "subprocess.log"
        ).read_text(encoding="utf-8").strip().splitlines()[0]
    )
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and (
        _pid_is_executing(record["pid"]) or _pid_is_executing(descendant_pid)
    ):
        time.sleep(0.02)
    assert not _pid_is_executing(record["pid"])
    assert not _pid_is_executing(descendant_pid)
    assert record["status"] == "interrupted"
    assert record["termination"]["term_sent"] is True
    assert record["termination"]["return_code"] < 0


def test_termination_escalates_from_sigterm_to_sigkill_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []

    class Process:
        def __init__(self) -> None:
            self.waits: list[float | None] = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise subprocess.TimeoutExpired("test", timeout)
            return -signal.SIGKILL

        def poll(self):
            return None

        def terminate(self) -> None:
            raise AssertionError("killpg is available")

        def kill(self) -> None:
            raise AssertionError("killpg is available")

    process = Process()
    monkeypatch.setattr(stage_base.os, "killpg", lambda pgid, value: signals.append(value))
    result = stage_base._terminate_process_group(process, pgid=123, grace_seconds=0.01)

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.waits == [0.01, 0.01]
    assert result["term_sent"] is True
    assert result["kill_sent"] is True
    assert result["return_code"] == -signal.SIGKILL


def test_termination_reports_unconfirmed_after_two_bounded_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self) -> None:
            self.waits: list[float | None] = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            raise subprocess.TimeoutExpired("test", timeout)

        def poll(self):
            return None

    process = Process()
    signals: list[int] = []
    monkeypatch.setattr(stage_base.os, "killpg", lambda pgid, value: signals.append(value))

    with pytest.raises(stage_base.ProcessTerminationError) as raised:
        stage_base._terminate_process_group(process, pgid=123, grace_seconds=0.01)

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.waits == [0.01, 0.01]
    assert raised.value.details["termination_confirmed"] is False
    assert "did not exit after SIGKILL" in str(raised.value)


def test_runner_recovers_exited_command_before_starting_next_attempt(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, lambda context: StageResult())
    state_path = _record_running_command(
        runner,
        {
            "schema_version": 1,
            "status": "running",
            "command_index": 1,
            "command_id": "stale-command",
            "host": socket.gethostname(),
            "pid": 999_999_999,
            "pgid": 999_999_999,
            "process_identity": None,
            "started_at": "2026-01-01T00:00:00Z",
        },
    )

    execution = runner.stage("ingest")

    assert execution.attempt == 2
    recovered = read_json(state_path)
    assert recovered["status"] == "stale"
    assert recovered["previous_status"] == "running"
    assert recovered["recovery"]["status"] == "recovered"
    assert recovered["recovered_at"]


def test_runner_recovers_starting_record_only_after_launch_owner_exits(
    tmp_path: Path,
) -> None:
    exited_owner = _runner(tmp_path / "exited", lambda context: StageResult())
    state_path = _record_running_command(
        exited_owner,
        {
            "schema_version": 1,
            "status": "starting",
            "command_index": 1,
            "command_id": "starting-stale",
            "host": socket.gethostname(),
            "runner_pid": 999_999_999,
            "runner_pgid": 999_999_999,
            "runner_process_identity": None,
            "pid": None,
            "pgid": None,
            "process_identity": None,
            "started_at": "2026-01-01T00:00:00Z",
        },
    )
    assert exited_owner.stage("ingest").attempt == 2
    recovered = read_json(state_path)
    assert recovered["status"] == "stale"
    assert recovered["recovery"]["status"] == "recovered"
    assert "launch owner" in recovered["recovery"]["reason"]

    live_owner = _runner(tmp_path / "live", lambda context: StageResult())
    runner_pid = os.getpid()
    _record_running_command(
        live_owner,
        {
            "schema_version": 1,
            "status": "starting",
            "command_index": 1,
            "command_id": "starting-live",
            "host": socket.gethostname(),
            "runner_pid": runner_pid,
            "runner_pgid": os.getpgid(runner_pid),
            "runner_process_identity": stage_base._process_identity(runner_pid),
            "pid": None,
            "pgid": None,
            "process_identity": None,
            "started_at": "2026-01-01T00:00:00Z",
        },
    )
    with pytest.raises(PipelineRunnerError, match="live Runner"):
        live_owner.stage("ingest")
    assert live_owner.state.read_stage("ingest")["attempt"] == 1


def test_runner_refuses_live_same_host_or_unverifiable_remote_command(
    tmp_path: Path,
) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        local = _runner(tmp_path / "local", lambda context: StageResult())
        _record_running_command(
            local,
            {
                "schema_version": 1,
                "status": "running",
                "command_index": 1,
                "command_id": "live-command",
                "host": socket.gethostname(),
                "pid": child.pid,
                "pgid": os.getpgid(child.pid),
                "process_identity": stage_base._process_identity(child.pid),
                "started_at": "2026-01-01T00:00:00Z",
            },
        )
        with pytest.raises(PipelineRunnerError, match=r"still alive.*pid="):
            local.stage("ingest")
        assert local.state.read_stage("ingest")["attempt"] == 1
    finally:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=5)

    remote = _runner(tmp_path / "remote", lambda context: StageResult())
    _record_running_command(
        remote,
        {
            "schema_version": 1,
            "status": "running",
            "command_index": 1,
            "command_id": "remote-command",
            "host": "different-training-host",
            "pid": 999_999_999,
            "pgid": 999_999_999,
            "process_identity": None,
            "started_at": "2026-01-01T00:00:00Z",
        },
    )
    with pytest.raises(PipelineRunnerError, match="cannot verify or terminate"):
        remote.stage("ingest")
    assert remote.state.read_stage("ingest")["attempt"] == 1


@pytest.mark.parametrize("reap_leader", [True, False], ids=["leader-gone", "leader-zombie"])
def test_runner_refuses_live_process_group_after_leader_exits(
    tmp_path: Path,
    reap_leader: bool,
) -> None:
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                "print(child.pid, flush=True)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert leader.stdout is not None
    pgid = os.getpgid(leader.pid)
    identity = stage_base._process_identity(leader.pid)
    descendant_pid = int(leader.stdout.readline().strip())
    if reap_leader:
        leader.wait(timeout=5)
    else:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not stage_base._process_is_zombie(
            leader.pid
        ):
            time.sleep(0.02)
        assert stage_base._process_is_zombie(leader.pid)
    try:
        runner = _runner(tmp_path, lambda context: StageResult())
        _record_running_command(
            runner,
            {
                "schema_version": 1,
                "status": "running",
                "command_index": 1,
                "command_id": "orphaned-group",
                "host": socket.gethostname(),
                "pid": leader.pid,
                "pgid": pgid,
                "process_identity": identity,
                "started_at": "2026-01-01T00:00:00Z",
            },
        )
        with pytest.raises(
            PipelineRunnerError,
            match="process group still has live members",
        ):
            runner.stage("ingest")
    finally:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        leader.wait(timeout=5)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and _pid_is_executing(descendant_pid):
        time.sleep(0.02)
    assert not _pid_is_executing(descendant_pid)


def test_group_probe_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path, lambda context: StageResult())
    _record_running_command(
        runner,
        {
            "schema_version": 1,
            "status": "running",
            "command_index": 1,
            "command_id": "unverifiable-group",
            "host": socket.gethostname(),
            "pid": 999_999_999,
            "pgid": 12345,
            "process_identity": None,
            "started_at": "2026-01-01T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        stage_base.os,
        "killpg",
        lambda pgid, value: (_ for _ in ()).throw(OSError("probe unavailable")),
    )
    with pytest.raises(PipelineRunnerError, match="could not be verified"):
        runner.stage("ingest")


def test_launch_guard_prevents_work_in_spawn_to_persist_sigkill_window(
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "run" / "stages" / "00_ingest"
    window = tmp_path / "spawned-window.json"
    target_started = tmp_path / "target-started"
    helper = r'''
import json
from pathlib import Path
import sys
import time

from llmgen.pipeline.logging import PipelineLogger
from llmgen.pipeline.stages import StageResult, StageSpec
from llmgen.pipeline.stages import base
from llmgen.pipeline.stages.base import StageContext

repo = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
window = Path(sys.argv[3])
target = Path(sys.argv[4])

class Config:
    def get(self, key, default=None):
        return default

class State:
    def stage_dir(self, name):
        return run_dir / "stages" / "00_ingest"

original = base.atomic_write_json
def block_running_record(path, value, **kwargs):
    if Path(path).parent.name == "command_state" and value.get("status") == "running":
        window.write_text(json.dumps(value), encoding="utf-8")
        time.sleep(30)
    return original(path, value, **kwargs)
base.atomic_write_json = block_running_record

spec = StageSpec("ingest", "00_ingest", (), (), lambda context: StageResult(), "test")
context = StageContext(
    repo_root=repo,
    run_dir=run_dir,
    config=Config(),
    registry=None,
    state=State(),
    spec=spec,
    attempt=1,
    logger=PipelineLogger(run_dir, run_id="guard-test"),
)
target_code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('started'); import time; time.sleep(30)"
context.run_command([sys.executable, "-c", target_code, str(target)], label="guard-window")
'''
    runner_process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            helper,
            str(ROOT),
            str(tmp_path / "run"),
            str(window),
            str(target_started),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not window.is_file():
        if runner_process.poll() is not None:
            pytest.fail(f"window helper exited early: {runner_process.returncode}")
        time.sleep(0.02)
    assert window.is_file()
    launching = json.loads(window.read_text(encoding="utf-8"))
    launcher_pid = int(launching["pid"])

    os.kill(runner_process.pid, signal.SIGKILL)
    runner_process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _pid_is_executing(launcher_pid):
        time.sleep(0.02)

    assert not _pid_is_executing(launcher_pid)
    assert not target_started.exists()
    state_path = stage_dir / "attempts" / "0001" / "command_state" / "0001.json"
    assert read_json(state_path)["status"] == "starting"
    recovered = stage_base.recover_stale_stage_commands(
        stage_dir,
        stage_name="ingest",
    )
    assert len(recovered) == 1
    assert read_json(state_path)["status"] == "stale"
    time.sleep(0.1)
    assert not target_started.exists()
