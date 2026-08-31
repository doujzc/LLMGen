"""Marker-prefixed console logging plus structured Run logs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
from typing import Any, Iterable, Mapping

from .io import canonical_json, utc_now


_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization)(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def _safe_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        text = canonical_json(value)
    else:
        text = str(value)
    return text.replace("\n", "\\n").replace("\r", "\\r")


class PipelineLogger:
    """A deliberately small logger suitable for subprocess orchestration."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        marker: str = "[[LLMGEN-PIPELINE]]",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        secret_values: Iterable[str] = (),
        config_hash: str | None = None,
        stage: str | None = None,
        attempt: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = run_id
        self.marker = marker
        self.console_level = self._level(console_level)
        self.file_level = self._level(file_level)
        self.secret_values = tuple(
            sorted(
                {str(value) for value in secret_values if str(value)},
                key=len,
                reverse=True,
            )
        )
        self.config_hash = config_hash
        self.stage = stage
        self.attempt = attempt
        self._lock = threading.Lock()
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _level(value: str) -> int:
        normalized = str(value).upper()
        if normalized not in _LEVELS:
            raise ValueError(f"invalid log level: {value}")
        return _LEVELS[normalized]

    def child(self, *, stage: str, attempt: int) -> "PipelineLogger":
        return PipelineLogger(
            self.run_dir,
            run_id=self.run_id,
            marker=self.marker,
            console_level=next(
                name for name, number in _LEVELS.items() if number == self.console_level
            ),
            file_level=next(
                name for name, number in _LEVELS.items() if number == self.file_level
            ),
            secret_values=self.secret_values,
            config_hash=self.config_hash,
            stage=stage,
            attempt=attempt,
        )

    def redact(self, text: str) -> str:
        value = str(text)
        for secret in self.secret_values:
            value = value.replace(secret, "[REDACTED]")
        value = _BEARER.sub("Bearer [REDACTED]", value)
        value = _SECRET_ASSIGNMENT.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
            value,
        )
        return value

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, Mapping):
            return {
                str(key): self._redact_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._redact_value(item) for item in value]
        return value

    def event(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        normalized_level = str(level).upper()
        number = self._level(normalized_level)
        payload: dict[str, Any] = {
            "timestamp": utc_now(),
            "level": normalized_level,
            "run_id": self.run_id,
            "event": str(event),
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        if self.config_hash is not None:
            payload["config_hash"] = self.config_hash
        if self.stage is not None:
            payload["stage"] = self.stage
        if self.attempt is not None:
            payload["attempt"] = self.attempt
        payload.update(fields)
        redacted_payload = {
            key: self._redact_value(value)
            for key, value in payload.items()
        }
        console_fields = " ".join(
            f"{key}={_safe_value(value)}"
            for key, value in redacted_payload.items()
            if key not in {"timestamp", "level", "event"}
        )
        console_line = f"{self.marker} event={event}"
        if console_fields:
            console_line += " " + console_fields
        with self._lock:
            if number >= self.console_level:
                print(console_line, file=sys.stdout, flush=True)
            if number >= self.file_level:
                self._append(
                    self.run_dir / "logs" / "pipeline.jsonl",
                    json.dumps(redacted_payload, ensure_ascii=False, sort_keys=True) + "\n",
                )
                human = (
                    f"{redacted_payload['timestamp']} {normalized_level} "
                    f"{console_line}\n"
                )
                self._append(self.run_dir / "logs" / "pipeline.log", human)
                if self.stage is not None:
                    self._append(
                        self.run_dir / "logs" / "stages" / f"{self.stage}.log",
                        human,
                    )

    @staticmethod
    def _append(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, text.encode("utf-8", errors="replace"))
        finally:
            os.close(descriptor)
