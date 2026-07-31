"""Incrementally extract bounded loss series from persisted training logs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import threading
from typing import Any


METRIC_READ_CHUNK_BYTES = 256 * 1024
METRIC_MAX_PENDING_LINE_BYTES = 64 * 1024
METRIC_MAX_CACHED_POINTS = 50_000
METRIC_MAX_CACHED_RUNS = 16

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROGRESS_RE = re.compile(
    r"(?<![\d.])(?P<done>\d[\d,]*)\s*/\s*(?P<total>\d[\d,]*)(?![\d.])"
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_METRIC_FIELD_RE = re.compile(
    rf"""(?P<quote>["'])
        (?P<key>
            loss|eval_loss|epoch|epochs|learning_rate|grad_norm|global_step|step
        )
        (?P=quote)\s*:\s*(?P<value>{_NUMBER})
    """,
    re.VERBOSE,
)
_FLAT_MAPPING_RE = re.compile(r"\{[^{}\r\n]{1,16384}\}")
_PHASE_PATTERNS = (
    (
        "tokenizer",
        "02 Tokenizer",
        (
            re.compile(r"\[0?2\]", re.IGNORECASE),
            re.compile(
                r"""["']event["']\s*:\s*["']stage1_progress["']""",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "memorization",
        "05 Memorization",
        (
            re.compile(r"\[0?5\]", re.IGNORECASE),
            re.compile(
                r"\[memorization\]\s+training\s+mixture",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "alignment",
        "06a Alignment",
        (re.compile(r"\[06a\]", re.IGNORECASE),),
    ),
    (
        "retrieval",
        "06b Retrieval",
        (re.compile(r"\[06b\]", re.IGNORECASE),),
    ),
)


def _uniform_sample(
    points: list[dict[str, Any]],
    maximum: int,
) -> list[dict[str, Any]]:
    if maximum <= 0 or not points:
        return []
    if len(points) <= maximum:
        return list(points)
    if maximum == 1:
        return [points[-1]]
    scale = (len(points) - 1) / (maximum - 1)
    indices = {
        min(len(points) - 1, round(index * scale))
        for index in range(maximum)
    }
    return [points[index] for index in sorted(indices)]


def _finite_number(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


class LossLogParser:
    """Parse flat Hugging Face Trainer metric records one log line at a time."""

    def __init__(self) -> None:
        self.phase_id = "training"
        self.phase_label = "Training"
        self.step: int | None = None
        self.total_steps: int | None = None
        self.points: list[dict[str, Any]] = []
        self.total_points = 0
        self._last_signature: tuple[Any, ...] | None = None

    def _set_phase(self, line: str) -> None:
        for phase_id, phase_label, patterns in _PHASE_PATTERNS:
            if not any(pattern.search(line) for pattern in patterns):
                continue
            if phase_id != self.phase_id:
                self.step = None
                self.total_steps = None
            self.phase_id = phase_id
            self.phase_label = phase_label
            return
        if (
            self.phase_id == "training"
            and re.search(
                r"\[retrieval\]\s+training\s+mixture",
                line,
                re.IGNORECASE,
            )
        ):
            self.phase_id = "retrieval"
            self.phase_label = "Retrieval"

    def _update_progress(self, line: str) -> None:
        matches = list(_PROGRESS_RE.finditer(line))
        if not matches:
            return
        match = matches[-1]
        done = int(match.group("done").replace(",", ""))
        total = int(match.group("total").replace(",", ""))
        if total > 0 and 0 <= done <= total:
            self.step = done
            self.total_steps = total

    def _append_mapping(self, mapping: str) -> None:
        fields: dict[str, float] = {}
        for match in _METRIC_FIELD_RE.finditer(mapping):
            value = _finite_number(match.group("value"))
            if value is not None:
                fields[match.group("key")] = value
        if "eval_loss" in fields:
            kind = "eval"
            loss = fields["eval_loss"]
        elif "loss" in fields:
            kind = "train"
            loss = fields["loss"]
        else:
            return

        explicit_step = fields.get("step")
        if explicit_step is None:
            explicit_step = fields.get("global_step")
        step = int(explicit_step) if explicit_step is not None else self.step
        epoch = fields.get("epoch")
        total_epochs = fields.get("epochs")
        signature = (
            self.phase_id,
            kind,
            step,
            epoch,
            loss,
            fields.get("learning_rate"),
            fields.get("grad_norm"),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature
        self.total_points += 1
        self.points.append(
            {
                "sequence": self.total_points,
                "phase_id": self.phase_id,
                "phase": self.phase_label,
                "kind": kind,
                "loss": loss,
                "step": step,
                "total_steps": self.total_steps,
                "epoch": epoch,
                "total_epochs": (
                    int(total_epochs) if total_epochs is not None else None
                ),
                "learning_rate": fields.get("learning_rate"),
                "grad_norm": fields.get("grad_norm"),
            }
        )
        if len(self.points) > METRIC_MAX_CACHED_POINTS:
            self.points = _uniform_sample(
                self.points,
                METRIC_MAX_CACHED_POINTS // 2,
            )

    def feed_line(self, raw_line: str) -> None:
        line = _ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not line:
            return
        self._set_phase(line)
        self._update_progress(line)
        if "loss" not in line or "{" not in line or "}" not in line:
            return
        for match in _FLAT_MAPPING_RE.finditer(line):
            self._append_mapping(match.group(0))

    def snapshot(self, maximum_points: int) -> dict[str, Any]:
        points = _uniform_sample(self.points, maximum_points)
        phases: OrderedDict[str, dict[str, Any]] = OrderedDict()
        latest: dict[str, dict[str, Any]] = {}
        kind_counts = {"train": 0, "eval": 0}
        best_eval: dict[str, Any] | None = None
        for point in self.points:
            phase = phases.setdefault(
                str(point["phase_id"]),
                {
                    "id": point["phase_id"],
                    "label": point["phase"],
                    "points": 0,
                    "train_points": 0,
                    "eval_points": 0,
                    "latest": {},
                    "best_eval": None,
                },
            )
            phase["points"] += 1
            kind = str(point["kind"])
            phase[f"{kind}_points"] += 1
            phase["latest"][kind] = point
            kind_counts[kind] += 1
            latest[kind] = point
            if point["kind"] == "eval" and (
                best_eval is None or point["loss"] < best_eval["loss"]
            ):
                best_eval = point
            if point["kind"] == "eval" and (
                phase["best_eval"] is None
                or point["loss"] < phase["best_eval"]["loss"]
            ):
                phase["best_eval"] = point
        return {
            "points": points,
            "total_points": self.total_points,
            "available_points": len(self.points),
            "sampled": len(points) < len(self.points),
            "phases": list(phases.values()),
            "kind_counts": kind_counts,
            "latest": latest,
            "best_eval": best_eval,
        }


@dataclass
class _MetricFileState:
    identity: tuple[int, int]
    offset: int = 0
    pending: bytes = b""
    parser: LossLogParser = field(default_factory=LossLogParser)


class LossMetricReader:
    """Maintain small incremental parsers for recently inspected run logs."""

    def __init__(self, *, maximum_cached_runs: int = METRIC_MAX_CACHED_RUNS) -> None:
        self.maximum_cached_runs = max(1, int(maximum_cached_runs))
        self._states: OrderedDict[Path, _MetricFileState] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _empty_payload() -> dict[str, Any]:
        return {
            "points": [],
            "total_points": 0,
            "available_points": 0,
            "sampled": False,
            "phases": [],
            "kind_counts": {"train": 0, "eval": 0},
            "latest": {},
            "best_eval": None,
            "updated_at": None,
        }

    @staticmethod
    def _consume(state: _MetricFileState, chunk: bytes) -> None:
        data = state.pending + chunk
        state.pending = b""
        for raw_line in data.splitlines(keepends=True):
            if raw_line.endswith((b"\n", b"\r")):
                state.parser.feed_line(
                    raw_line.decode("utf-8", errors="replace")
                )
            else:
                state.pending = raw_line
        if len(state.pending) > METRIC_MAX_PENDING_LINE_BYTES:
            state.pending = state.pending[-METRIC_MAX_PENDING_LINE_BYTES:]

    def _state_for(self, path: Path, stat_result: Any) -> _MetricFileState:
        identity = (int(stat_result.st_dev), int(stat_result.st_ino))
        state = self._states.get(path)
        if (
            state is None
            or state.identity != identity
            or int(stat_result.st_size) < state.offset
        ):
            state = _MetricFileState(identity=identity)
            self._states[path] = state
        self._states.move_to_end(path)
        while len(self._states) > self.maximum_cached_runs:
            self._states.popitem(last=False)
        return state

    def read(
        self,
        log_path: str | Path,
        *,
        maximum_points: int = 2000,
    ) -> dict[str, Any]:
        path = Path(log_path).expanduser().resolve()
        maximum = max(10, min(int(maximum_points), 5000))
        with self._lock:
            try:
                stat_result = path.stat()
            except FileNotFoundError:
                return self._empty_payload()
            if not path.is_file():
                return self._empty_payload()
            state = self._state_for(path, stat_result)
            with path.open("rb") as stream:
                stream.seek(state.offset)
                while True:
                    chunk = stream.read(METRIC_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    self._consume(state, chunk)
                state.offset = stream.tell()
            payload = state.parser.snapshot(maximum)
            payload["updated_at"] = datetime.fromtimestamp(
                stat_result.st_mtime,
                tz=timezone.utc,
            ).isoformat(timespec="seconds")
            return payload
