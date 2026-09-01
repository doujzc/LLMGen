"""Stage protocol shared by the Runner and concrete adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from ..artifacts import ArtifactRegistry
from ..checkpoints import CheckpointError, select_checkpoint
from ..config import PipelineConfig
from ..io import atomic_write_json, atomic_write_text, read_json, sha256_file, utc_now
from ..logging import PipelineLogger
from ..state import RunStateStore


class StageExecutionError(RuntimeError):
    """Raised when a Stage adapter or child process fails."""


class ProcessTerminationError(StageExecutionError):
    """Raised when a child cannot be confirmed dead after bounded escalation."""

    def __init__(self, message: str, *, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


_ACTIVE_COMMAND_STATUSES = frozenset(
    {"starting", "running", "termination_unconfirmed"}
)
_COMMAND_STATE_DIRECTORY = "command_state"
_COMMAND_TERMINATION_GRACE_SECONDS = 5.0
_LAUNCH_GUARD = """\
import os
import sys

release_fd = int(sys.argv[1])
try:
    release = os.read(release_fd, 1)
finally:
    os.close(release_fd)
if release != b"1":
    raise SystemExit(125)
command = sys.argv[2:]
if not command:
    raise SystemExit(126)
os.execvpe(command[0], command, os.environ)
"""


def _host_identity() -> str:
    """Return the stable host label persisted with a child-process record."""

    return socket.gethostname()


def _process_identity(pid: int) -> dict[str, Any] | None:
    """Return a process birth identity so PID reuse is not mistaken for liveness."""

    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            value = proc_stat.read_text(encoding="utf-8")
            # ``comm`` is parenthesized and may itself contain spaces.  Fields
            # after its final ')' start at proc-stat field 3; starttime is 22.
            fields = value[value.rfind(")") + 2 :].split()
            start_ticks = fields[19]
            boot_id_path = Path("/proc/sys/kernel/random/boot_id")
            boot_id = (
                boot_id_path.read_text(encoding="utf-8").strip()
                if boot_id_path.is_file()
                else None
            )
            return {
                "kind": "linux_proc_starttime",
                "boot_id": boot_id,
                "start_ticks": start_ticks,
            }
        except (IndexError, OSError, ValueError):
            return None
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    started = " ".join(result.stdout.split())
    if result.returncode != 0 or not started:
        return None
    return {"kind": "ps_lstart", "started_at": started}


def _process_is_zombie(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            value = proc_stat.read_text(encoding="utf-8")
            fields = value[value.rfind(")") + 2 :].split()
            return bool(fields and fields[0] == "Z")
        except OSError:
            return False
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return False
    status = result.stdout.strip()
    return result.returncode == 0 and status.startswith("Z")


def _probe_process_leader(record: Mapping[str, Any]) -> tuple[str, str]:
    """Probe only a recorded leader PID, including its birth identity."""

    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        return "unknown", "the running record has no valid pid"
    if pid <= 0:
        return "unknown", "the running record has no valid pid"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited", "the recorded pid no longer exists"
    except PermissionError:
        return "unknown", "permission denied while probing the recorded pid"
    except OSError as error:
        return "unknown", f"the recorded pid could not be probed: {error}"
    if _process_is_zombie(pid):
        return "exited", "the recorded pid is a zombie"

    expected_identity = record.get("process_identity")
    if isinstance(expected_identity, Mapping):
        current_identity = _process_identity(pid)
        if current_identity is None:
            return "unknown", "the process birth identity could not be verified"
        if dict(current_identity) != dict(expected_identity):
            return "exited", "the pid was reused by a different process"

    try:
        expected_pgid = int(record.get("pgid"))
    except (TypeError, ValueError):
        return "unknown", "the running record has no valid process-group id"
    try:
        current_pgid = os.getpgid(pid)
    except ProcessLookupError:
        return "exited", "the recorded pid exited during recovery"
    except (AttributeError, PermissionError, OSError) as error:
        return "unknown", f"the process group could not be verified: {error}"
    if current_pgid != expected_pgid:
        return "exited", "the pid now belongs to a different process group"
    return "alive", "the recorded process identity is still alive"


def _probe_process_group(pgid_value: Any) -> tuple[str, str]:
    """Find live members of a PGID after its recorded leader disappeared."""

    try:
        pgid = int(pgid_value)
    except (TypeError, ValueError):
        return "unknown", "the running record has no valid process-group id"
    if pgid <= 0:
        return "unknown", "the running record has no valid process-group id"
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return "exited", "the recorded process group no longer exists"
    except PermissionError:
        group_exists = True
    except (AttributeError, OSError) as error:
        return "unknown", f"the process group could not be probed: {error}"
    else:
        group_exists = True

    live_members: list[int] = []
    matched_members = 0
    proc_root = Path("/proc")
    if proc_root.is_dir():
        scan_failed = False
        for child in proc_root.iterdir():
            if not child.name.isdigit():
                continue
            try:
                value = (child / "stat").read_text(encoding="utf-8")
                fields = value[value.rfind(")") + 2 :].split()
                member_pgid = int(fields[2])
                state = fields[0]
            except (IndexError, OSError, ValueError):
                scan_failed = True
                continue
            if member_pgid != pgid:
                continue
            matched_members += 1
            if state != "Z":
                live_members.append(int(child.name))
        if live_members:
            return (
                "alive",
                "the recorded process group still has live members "
                f"pids={live_members[:16]}",
            )
        if matched_members:
            return "exited", "the recorded process group contains only zombies"
        if scan_failed and group_exists:
            return "unknown", "the process group exists but /proc scanning was incomplete"
    else:
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,pgid=,stat="],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as error:
            return "unknown", f"the process group exists but ps failed: {error}"
        if result.returncode != 0:
            return "unknown", "the process group exists but ps could not enumerate it"
        for line in result.stdout.splitlines():
            fields = line.split(None, 2)
            if len(fields) != 3:
                continue
            try:
                member_pid = int(fields[0])
                member_pgid = int(fields[1])
            except ValueError:
                continue
            if member_pgid != pgid:
                continue
            matched_members += 1
            if not fields[2].startswith("Z"):
                live_members.append(member_pid)
        if live_members:
            return (
                "alive",
                "the recorded process group still has live members "
                f"pids={live_members[:16]}",
            )
        if matched_members:
            return "exited", "the recorded process group contains only zombies"

    # A process can exit between killpg(0) and enumeration.  Re-probe before
    # failing closed on an apparently empty but previously existing group.
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return "exited", "the recorded process group exited during recovery"
    except PermissionError:
        pass
    except (AttributeError, OSError) as error:
        return "unknown", f"the process group could not be re-probed: {error}"
    return "unknown", "the process group exists but its members could not be verified"


def _probe_recorded_process(record: Mapping[str, Any]) -> tuple[str, str]:
    """Probe a recorded leader and, if needed, every member of its PGID."""

    leader_status, leader_reason = _probe_process_leader(record)
    if leader_status == "alive":
        return leader_status, leader_reason
    group_status, group_reason = _probe_process_group(record.get("pgid"))
    if group_status == "alive":
        return "alive", f"{leader_reason}; {group_reason}"
    if group_status == "unknown":
        return "unknown", f"{leader_reason}; {group_reason}"
    return "exited", f"{leader_reason}; {group_reason}"


def recover_stale_stage_commands(
    stage_dir: str | Path,
    *,
    stage_name: str,
) -> tuple[dict[str, Any], ...]:
    """Recover exited commands or reject an unsafe new Stage attempt.

    A Run lock prevents two healthy Runner processes from entering the same
    Stage concurrently.  This check covers the harder crash case where the
    Runner disappeared but a training process survived and kept using the
    accelerator.  Commands recorded on another host are deliberately not
    guessed about.
    """

    root = Path(stage_dir)
    local_host = _host_identity()
    recovered: list[dict[str, Any]] = []
    state_paths = sorted(
        path
        for path in (root / "attempts").glob(
            f"*/{_COMMAND_STATE_DIRECTORY}/*.json"
        )
        if path.parents[1].name.isdigit()
    )
    for state_path in state_paths:
        try:
            record = read_json(state_path)
        except (OSError, ValueError) as error:
            raise StageExecutionError(
                f"cannot safely start stage {stage_name}: command state is unreadable: "
                f"{state_path} ({error})"
            ) from error
        if not isinstance(record, dict):
            raise StageExecutionError(
                f"cannot safely start stage {stage_name}: command state is invalid: "
                f"{state_path}"
            )
        if record.get("status") not in _ACTIVE_COMMAND_STATUSES:
            continue
        attempt = state_path.parents[1].name
        command_index = record.get("command_index", record.get("index"))
        command_id = record.get("command_id") or "<unknown>"
        recorded_host = str(record.get("host") or "")
        identity = (
            f"attempt={attempt} command_index={command_index} "
            f"command_id={command_id}"
        )
        if not recorded_host:
            raise StageExecutionError(
                f"cannot safely start stage {stage_name}: {identity} is recorded as "
                "running without a host identity"
            )
        if recorded_host != local_host:
            raise StageExecutionError(
                f"cannot safely start stage {stage_name}: {identity} is recorded as "
                f"running on host {recorded_host}; current host {local_host} cannot "
                "verify or terminate that process"
            )
        recorded_status = str(record.get("status"))
        if recorded_status == "starting" and record.get("pid") is None:
            owner_record = {
                "pid": record.get("runner_pid"),
                "pgid": record.get("runner_pgid"),
                "process_identity": record.get("runner_process_identity"),
            }
            # The Runner is not started in a dedicated process group, so its
            # surrounding shell/test group is not evidence that this specific
            # launch owner survived.
            process_status, owner_reason = _probe_process_leader(owner_record)
            reason = f"launch owner: {owner_reason}"
        else:
            process_status, reason = _probe_recorded_process(record)
        if process_status == "alive":
            if recorded_status == "starting":
                raise StageExecutionError(
                    f"cannot safely start stage {stage_name}: {identity} is still "
                    f"being launched by a live Runner on host {local_host} "
                    f"(runner_pid={record.get('runner_pid')})"
                )
            raise StageExecutionError(
                f"cannot safely start stage {stage_name}: {identity} is still alive "
                f"on host {local_host} (pid={record.get('pid')}, "
                f"pgid={record.get('pgid')}); {reason}"
            )
        if process_status == "unknown":
            raise StageExecutionError(
                f"cannot safely start stage {stage_name}: {identity} on host "
                f"{local_host} could not be verified ({reason})"
            )
        recovered_at = utc_now()
        record.update(
            {
                "status": "stale",
                "previous_status": record.get("status"),
                "finished_at": record.get("finished_at") or recovered_at,
                "recovered_at": recovered_at,
                "recovery": {"status": "recovered", "reason": reason},
            }
        )
        atomic_write_json(state_path, record, mode=0o600)
        recovered.append(record)
    return tuple(recovered)


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    pgid: int,
    grace_seconds: float = _COMMAND_TERMINATION_GRACE_SECONDS,
) -> dict[str, Any]:
    """Terminate a Stage child group, escalating and always reaping its leader."""

    result: dict[str, Any] = {
        "term_sent": False,
        "kill_sent": False,
        "grace_seconds": grace_seconds,
    }
    try:
        os.killpg(pgid, signal.SIGTERM)
        result["term_sent"] = True
    except ProcessLookupError:
        pass
    except (AttributeError, OSError) as error:
        result["term_error"] = f"{type(error).__name__}: {error}"
        if process.poll() is None:
            try:
                process.terminate()
                result["term_sent"] = True
            except OSError as fallback_error:
                result["term_error"] = (
                    f"{result.get('term_error')}; fallback: "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                )
    try:
        return_code = process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
            result["kill_sent"] = True
        except ProcessLookupError:
            pass
        except (AttributeError, OSError) as error:
            result["kill_error"] = f"{type(error).__name__}: {error}"
            if process.poll() is None:
                try:
                    process.kill()
                    result["kill_sent"] = True
                except OSError as fallback_error:
                    result["kill_error"] = (
                        f"{result.get('kill_error')}; fallback: "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )
        try:
            return_code = process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired as error:
            result.update(
                {
                    "termination_confirmed": False,
                    "cleanup_error_type": type(error).__name__,
                    "cleanup_error": (
                        "child did not exit after SIGKILL within "
                        f"{grace_seconds} seconds"
                    ),
                }
            )
            raise ProcessTerminationError(
                result["cleanup_error"],
                details=result,
            ) from error
    result["return_code"] = return_code
    result["termination_confirmed"] = True
    return result


@dataclass(frozen=True)
class ArtifactOutput:
    logical_name: str
    path: Path
    artifact_schema: str
    format: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    artifacts: tuple[ArtifactOutput, ...] = ()
    progress: Mapping[str, Any] = field(default_factory=dict)


StageHandler = Callable[["StageContext"], StageResult]


@dataclass(frozen=True)
class StageSpec:
    name: str
    directory: str
    dependencies: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    handler: StageHandler
    description: str
    implementation_paths: tuple[str, ...] = ()


@dataclass
class StageContext:
    repo_root: Path
    run_dir: Path
    config: PipelineConfig
    registry: ArtifactRegistry
    state: RunStateStore
    spec: StageSpec
    attempt: int
    logger: PipelineLogger
    resume_checkpoint: str | None = None
    checkpoint_lineage: dict[str, Any] = field(default_factory=dict)
    checkpoint_lineage_path: Path | None = None
    allow_legacy_checkpoint: bool = False

    @property
    def stage_dir(self) -> Path:
        return self.state.stage_dir(self.spec.name)

    @property
    def attempt_dir(self) -> Path:
        return self.stage_dir / "attempts" / f"{self.attempt:04d}"

    @property
    def output_dir(self) -> Path:
        """Attempt-owned output, promoted only after the handler succeeds."""

        return self.attempt_dir / "output"

    @property
    def formal_output_dir(self) -> Path:
        """Stable output visible to downstream completed stages."""

        return self.stage_dir / "output"

    def artifact(self, logical_name: str) -> Path:
        return self.registry.resolve(logical_name)

    def set_checkpoint_code_plan(self, path: str | Path) -> None:
        """Bind a code-plan hash after the codebook stage materializes it."""

        plan = Path(path).resolve()
        if not plan.is_file():
            raise CheckpointError(f"code plan is missing for checkpoint lineage: {plan}")
        self.checkpoint_lineage["code_plan_sha256"] = sha256_file(plan)
        if self.checkpoint_lineage_path is None:
            raise CheckpointError("checkpoint lineage path is not initialized")
        atomic_write_json(self.checkpoint_lineage_path, self.checkpoint_lineage, mode=0o600)

    def select_resume_checkpoint(
        self,
        *,
        kind: str,
        root: str | Path,
    ) -> str | None:
        """Select and record a complete, lineage-compatible training checkpoint."""

        resolved_root = Path(root).expanduser().resolve()
        discovery_roots = [resolved_root]
        if self.resume_checkpoint is None:
            try:
                relative = resolved_root.relative_to(self.attempt_dir.resolve())
            except ValueError:
                pass
            else:
                previous = sorted(
                    (self.stage_dir / "attempts").glob("[0-9][0-9][0-9][0-9]"),
                    reverse=True,
                )
                discovery_roots.extend(
                    attempt / relative
                    for attempt in previous
                    if attempt.resolve() != self.attempt_dir.resolve()
                )
        try:
            if self.resume_checkpoint is not None:
                selection = select_checkpoint(
                    resolved_root,
                    kind=kind,
                    expected_lineage=self.checkpoint_lineage,
                    explicit=self.resume_checkpoint,
                    allow_legacy=self.allow_legacy_checkpoint,
                )
            else:
                selections = [
                    candidate
                    for candidate in (
                        select_checkpoint(
                            candidate_root,
                            kind=kind,
                            expected_lineage=self.checkpoint_lineage,
                            allow_legacy=self.allow_legacy_checkpoint,
                        )
                        for candidate_root in discovery_roots
                    )
                    if candidate is not None
                ]
                selection = (
                    max(selections, key=lambda value: value.global_step)
                    if selections
                    else None
                )
        except CheckpointError as error:
            record = {
                "kind": kind,
                "root": str(resolved_root),
                "discovery_roots": [str(value) for value in discovery_roots],
                "requested": self.resume_checkpoint,
                "selected": None,
                "error": str(error),
                "allow_legacy_checkpoint": self.allow_legacy_checkpoint,
            }
            self.update_progress(checkpoint_resume=record)
            self.logger.event("checkpoint.selection_failed", level="ERROR", **record)
            raise
        record: dict[str, Any] = {
            "kind": kind,
            "root": str(resolved_root),
            "discovery_roots": [str(value) for value in discovery_roots],
            "requested": self.resume_checkpoint,
            "selected": selection.to_dict() if selection is not None else None,
            "allow_legacy_checkpoint": self.allow_legacy_checkpoint,
        }
        self.update_progress(checkpoint_resume=record)
        self.logger.event("checkpoint.selection", **record)
        return str(selection.path) if selection is not None else None

    def update_progress(self, **progress: Any) -> None:
        current = self.state.read_stage(self.spec.name).get("progress") or {}
        merged = {**dict(current), **progress}
        self.state.update_stage(self.spec.name, progress=merged)
        atomic_write_json(self.stage_dir / "progress.json", merged)
        self.logger.event("stage.progress", **merged)

    def run_command(
        self,
        argv: Sequence[str | os.PathLike[str]],
        *,
        environment: Mapping[str, str] | None = None,
        label: str | None = None,
    ) -> None:
        command = [str(value) for value in argv]
        if not command:
            raise StageExecutionError("empty subprocess command")
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        command_log = self.attempt_dir / "commands.jsonl"
        existing = command_log.read_text(encoding="utf-8") if command_log.exists() else ""
        index = sum(bool(line.strip()) for line in existing.splitlines()) + 1
        command_id = uuid.uuid4().hex
        runner_pid = os.getpid()
        try:
            runner_pgid = os.getpgid(runner_pid)
        except (AttributeError, ProcessLookupError, OSError):
            runner_pgid = runner_pid
        command_state_dir = self.attempt_dir / _COMMAND_STATE_DIRECTORY
        command_state_dir.mkdir(parents=True, exist_ok=True)
        command_state_path = command_state_dir / f"{index:04d}.json"
        command_record = {
            "schema_version": 1,
            "index": index,
            "command_index": index,
            "command_id": command_id,
            "label": label,
            "started_at": utc_now(),
            "argv": command,
            "cwd": str(self.repo_root),
            "host": _host_identity(),
            "runner_pid": runner_pid,
            "runner_pgid": runner_pgid,
            "runner_process_identity": _process_identity(runner_pid),
            "launch_protocol": "pipe_release_v1",
            "launch_released_at": None,
            "status": "starting",
            "pid": None,
            "pgid": None,
            "process_identity": None,
            "finished_at": None,
            "return_code": None,
        }

        def persist_command_record() -> None:
            atomic_write_json(command_state_path, command_record, mode=0o600)
            # Keep the long-standing one-line-per-command log useful while the
            # separate state file provides a cheap recovery scan.  Rewriting
            # only this command's final line keeps its lifecycle snapshot in
            # sync without changing command indexes.
            atomic_write_text(
                command_log,
                existing
                + json.dumps(
                    {
                        **command_record,
                        "state_path": str(
                            command_state_path.relative_to(self.attempt_dir)
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                mode=0o600,
            )

        persist_command_record()
        child_environment = os.environ.copy()
        if environment is not None:
            child_environment.update(
                {str(key): str(value) for key, value in environment.items()}
            )
        child_environment["LLMGEN_PIPELINE_COMMAND_ID"] = command_id
        raw_log = self.attempt_dir / "subprocess.log"
        descriptor = os.open(
            raw_log,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        process: subprocess.Popen[str] | None = None
        reader: threading.Thread | None = None
        pgid: int | None = None
        return_code: int | None = None
        release_read: int | None = None
        release_write: int | None = None
        try:
            os.fchmod(descriptor, 0o600)
            release_read, release_write = os.pipe()
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _LAUNCH_GUARD,
                    str(release_read),
                    *command,
                ],
                cwd=self.repo_root,
                env=child_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
                pass_fds=(release_read,),
            )
            os.close(release_read)
            release_read = None
            try:
                pgid = os.getpgid(process.pid)
            except (AttributeError, ProcessLookupError, OSError):
                # start_new_session makes the child its process-group leader on
                # supported platforms, so pid is the correct recovery fallback.
                pgid = process.pid
            command_record.update(
                {
                    "status": "running",
                    "pid": process.pid,
                    "pgid": pgid,
                    "process_identity": _process_identity(process.pid),
                }
            )
            persist_command_record()
            # The guard cannot exec the real training command until the child
            # identity above is durable.  If this Runner is SIGKILLed anywhere
            # before this write, the pipe closes and the guard exits with 125.
            released = os.write(release_write, b"1")
            if released != 1:
                raise StageExecutionError("failed to release subprocess launch guard")
            os.close(release_write)
            release_write = None
            command_record["launch_released_at"] = utc_now()
            persist_command_record()
            self.logger.event(
                "subprocess.begin",
                command_index=index,
                command_id=command_id,
                label=label,
                pid=process.pid,
                pgid=pgid,
                argv=shlex.join(command),
            )
            assert process.stdout is not None
            lines: queue.Queue[str | None] = queue.Queue()

            def read_output() -> None:
                try:
                    for value in process.stdout:
                        lines.put(value)
                finally:
                    lines.put(None)

            reader = threading.Thread(
                target=read_output,
                name=f"pipeline-{self.spec.name}-stdout",
                daemon=True,
            )
            reader.start()
            heartbeat_seconds = float(
                self.config.get("logging.progress_interval_seconds", 30)
            )
            command_started = time.monotonic()
            last_output = command_started
            while True:
                try:
                    raw_line = lines.get(timeout=heartbeat_seconds)
                except queue.Empty:
                    self.logger.event(
                        "subprocess.heartbeat",
                        command_index=index,
                        elapsed_ms=round(
                            (time.monotonic() - command_started) * 1000,
                            3,
                        ),
                        quiet_seconds=round(time.monotonic() - last_output, 3),
                    )
                    continue
                if raw_line is None:
                    break
                last_output = time.monotonic()
                os.write(descriptor, raw_line.encode("utf-8", errors="replace"))
                line = raw_line.rstrip("\r\n")
                if self.config.get("logging.console_text_preview", False):
                    preview_chars = int(
                        self.config.get("logging.file_text_preview_chars", 1000)
                    )
                    self.logger.event(
                        "subprocess.output",
                        level="INFO",
                        command_index=index,
                        line=line[:preview_chars],
                        truncated=len(line) > preview_chars,
                    )
                else:
                    self.logger.event(
                        "subprocess.output",
                        level="DEBUG",
                        command_index=index,
                        line_chars=len(line),
                    )
            reader.join()
            return_code = process.wait()
            command_record.update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "finished_at": utc_now(),
                    "return_code": return_code,
                }
            )
            persist_command_record()
        except BaseException as error:
            termination: dict[str, Any] | None = None
            if process is not None:
                try:
                    termination = _terminate_process_group(
                        process,
                        pgid=pgid if pgid is not None else process.pid,
                    )
                except ProcessTerminationError as cleanup_error:
                    termination = {
                        **cleanup_error.details,
                        "cleanup_error_type": type(cleanup_error).__name__,
                        "cleanup_error": str(cleanup_error),
                        "return_code": process.poll(),
                        "termination_confirmed": False,
                    }
                except BaseException as cleanup_error:
                    # Preserve the triggering exception, but retain cleanup
                    # diagnostics in the durable command record.
                    termination = {
                        "cleanup_error_type": type(cleanup_error).__name__,
                        "cleanup_error": str(cleanup_error),
                        "return_code": process.poll(),
                        "termination_confirmed": False,
                    }
                if reader is not None:
                    reader.join(timeout=1.0)
            command_record.update(
                {
                    "status": (
                        "termination_unconfirmed"
                        if process is not None
                        and termination is not None
                        and not termination.get("termination_confirmed", False)
                        else (
                            "interrupted"
                            if process is not None
                            else "failed_to_start"
                        )
                    ),
                    "finished_at": utc_now(),
                    "return_code": (
                        termination.get("return_code")
                        if termination is not None
                        else None
                    ),
                    "error_type": type(error).__name__,
                    "termination": termination,
                }
            )
            try:
                persist_command_record()
            except OSError:
                # The original failure remains more useful; an unwritable
                # attempt directory will also make the Stage fail closed.
                pass
            try:
                self.logger.event(
                    "subprocess.interrupted",
                    level="ERROR",
                    command_index=index,
                    command_id=command_id,
                    error_type=type(error).__name__,
                    termination=termination,
                )
            except BaseException:
                pass
            raise
        finally:
            if release_read is not None:
                os.close(release_read)
            if release_write is not None:
                os.close(release_write)
            os.close(descriptor)
        assert return_code is not None
        if return_code != 0:
            self.logger.event(
                "subprocess.failed",
                level="ERROR",
                command_index=index,
                command_id=command_id,
                return_code=return_code,
                log_path=str(raw_log),
            )
            raise StageExecutionError(
                f"subprocess failed with exit code {return_code}: {shlex.join(command)}"
            )
        self.logger.event(
            "subprocess.complete",
            command_index=index,
            command_id=command_id,
            return_code=return_code,
        )
