"""Stage protocol shared by the Runner and concrete adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from ..artifacts import ArtifactRegistry
from ..config import PipelineConfig
from ..io import atomic_write_json, atomic_write_text, utc_now
from ..logging import PipelineLogger
from ..state import RunStateStore


class StageExecutionError(RuntimeError):
    """Raised when a Stage adapter or child process fails."""


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

    @property
    def stage_dir(self) -> Path:
        return self.state.stage_dir(self.spec.name)

    @property
    def attempt_dir(self) -> Path:
        return self.stage_dir / "attempts" / f"{self.attempt:04d}"

    @property
    def output_dir(self) -> Path:
        return self.stage_dir / "output"

    def artifact(self, logical_name: str) -> Path:
        return self.registry.resolve(logical_name)

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
        command_record = {
            "index": index,
            "label": label,
            "started_at": utc_now(),
            "argv": command,
            "cwd": str(self.repo_root),
        }
        atomic_write_text(
            command_log,
            existing
            + json.dumps(
                command_record,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            mode=0o600,
        )
        self.logger.event(
            "subprocess.begin",
            command_index=index,
            label=label,
            argv=shlex.join(command),
        )
        child_environment = os.environ.copy()
        if environment is not None:
            child_environment.update(
                {str(key): str(value) for key, value in environment.items()}
            )
        raw_log = self.attempt_dir / "subprocess.log"
        descriptor = os.open(
            raw_log,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        process: subprocess.Popen[str] | None = None
        try:
            os.fchmod(descriptor, 0o600)
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=child_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
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
        except BaseException:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            raise
        finally:
            os.close(descriptor)
        if return_code != 0:
            self.logger.event(
                "subprocess.failed",
                level="ERROR",
                command_index=index,
                return_code=return_code,
                log_path=str(raw_log),
            )
            raise StageExecutionError(
                f"subprocess failed with exit code {return_code}: {shlex.join(command)}"
            )
        self.logger.event(
            "subprocess.complete",
            command_index=index,
            return_code=return_code,
        )
