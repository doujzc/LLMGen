#!/usr/bin/env python3
"""Detached supervisor that runs the existing LLMGen CLI from a saved snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

from .config import ALLOWED_KEYS, DATASETS, PIPELINE_COMMANDS
from .gpu import probe_gpu_metrics, resolve_cuda_visible_devices
from .store import (
    StateStore,
    child_process_environment,
    secure_binary_append,
    utc_now,
)


STAGE_PATTERNS = (
    (re.compile(r"\[06b\]", re.IGNORECASE), "06b Retrieval"),
    (re.compile(r"\[06a\]", re.IGNORECASE), "06a Alignment"),
    (re.compile(r"\[0?1\]"), "01 数据与 Embedding"),
    (re.compile(r"\[0?2\]"), "02 层级 Tokenizer"),
    (re.compile(r"\[0?3\]"), "03 Code 导出与质量门禁"),
    (re.compile(r"\[0?4\]"), "04 Router 数据"),
    (re.compile(r"\[0?5\]"), "05 Memorization"),
    (re.compile(r"\[0?6\]"), "06 Retrieval"),
    (re.compile(r"\[0?7\]"), "07 评估"),
    (re.compile(r"\[0?8\]"), "08 诊断"),
    (re.compile(r"\[0?9\]"), "09 遗忘诊断"),
    (re.compile(r"\[10\]"), "10 导出 Web Bundle"),
)
CHECKPOINT_RE = re.compile(r"(?P<path>[^\s\"']*checkpoint[-_]\d+[^\s\"']*)")
PROGRESS_RE = re.compile(r"(?P<done>\d[\d,]*)\s*/\s*(?P<total>\d[\d,]*)")
STOP_GRACE_SECONDS = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("run config must be an object")
    if payload.get("dataset") not in DATASETS:
        raise ValueError("run snapshot contains an invalid dataset")
    if payload.get("command") not in PIPELINE_COMMANDS:
        raise ValueError("run snapshot contains an invalid command")
    if not isinstance(payload.get("resolved"), dict):
        raise ValueError("run snapshot has no resolved environment")
    if not isinstance(payload.get("runtime_env", {}), dict):
        raise ValueError("run snapshot has an invalid runtime environment")
    return payload


def _progress_from_text(text: str, stage: str, checkpoint: str) -> tuple[str, str, str]:
    next_stage = stage
    next_checkpoint = checkpoint
    progress = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for pattern, label in STAGE_PATTERNS:
            if pattern.search(line):
                next_stage = label
                break
        checkpoint_match = CHECKPOINT_RE.search(line)
        if checkpoint_match:
            next_checkpoint = checkpoint_match.group("path").rstrip(".,:;)")
        progress_match = PROGRESS_RE.search(line)
        if progress_match:
            progress = (
                f"{progress_match.group('done')} / "
                f"{progress_match.group('total')}"
            )
        elif any(
            marker in line.lower()
            for marker in ("loss", "epoch", "eval_", "saving", "checkpoint")
        ):
            progress = line[-240:]
    return next_stage, next_checkpoint, progress


def _read_new_text(path: Path, offset: int) -> tuple[str, int]:
    if not path.is_file():
        return "", offset
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
        return data.decode("utf-8", errors="replace"), stream.tell()


def _latest_checkpoint(artifact_run_dir: str, repo_root: Path) -> str:
    if not artifact_run_dir:
        return ""
    root = Path(artifact_run_dir).expanduser()
    if not root.is_absolute():
        root = repo_root / root
    if not root.is_dir():
        return ""
    candidates = [
        path
        for path in root.glob("router/**/checkpoint-*")
        if path.is_dir()
    ]
    if not candidates:
        return ""
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        return str(latest.relative_to(repo_root))
    except ValueError:
        return str(latest)


def _stop_requested(store: StateStore, run_id: str) -> bool:
    return bool(
        store.get_run(run_id, observe=False).get("stop_requested_at")
    )


def _signal_process_group(process: subprocess.Popen[bytes], sig: int) -> None:
    """Signal only the training session created by this runner."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def _mark_stopped(
    store: StateStore,
    run_id: str,
    *,
    exit_code: int | None,
    progress_text: str,
) -> None:
    finished_at = utc_now()
    store.update_run(
        run_id,
        status="stopped",
        stage="用户停止",
        progress_text=progress_text,
        exit_code=exit_code,
        stopped_at=finished_at,
        finished_at=finished_at,
    )


def run(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.expanduser().resolve()
    store = StateStore(args.state_root)
    run_meta = store.get_run(args.run_id, observe=False)
    config = _load_config(Path(run_meta["config_path"]))
    resolved = {
        key: str(value)
        for key, value in config["resolved"].items()
        if key in ALLOWED_KEYS and key not in {"DATASET", "PIPELINE_COMMAND"}
    }
    env = child_process_environment(
        config.get("runtime_env", {}),
        os.environ,
    )
    env.update(resolved)
    gpu_resolution: dict[str, Any] | None = None
    if resolved.get("DEVICE", "").startswith("cuda"):
        env["CUDA_DEVICE_ORDER"] = resolved.get(
            "CUDA_DEVICE_ORDER",
            "PCI_BUS_ID",
        )
        requested_gpus = resolved.get("CUDA_VISIBLE_DEVICES", "")
        if requested_gpus:
            gpu_resolution = resolve_cuda_visible_devices(
                requested_gpus,
                probe_gpu_metrics(include_processes=False),
            )
            env["CUDA_VISIBLE_DEVICES"] = gpu_resolution["runtime_value"]
    argv = [
        "bash",
        "scripts/router_pipeline.sh",
        config["dataset"],
        config["command"],
    ]
    log_path = Path(run_meta["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    store.update_run(
        args.run_id,
        status="starting",
        stage="启动现有训练 CLI",
        runner_pid=os.getpid(),
        started_at=utc_now(),
        command_argv=argv,
        progress_text="独立运行器已加载不可变运行快照",
        gpu_bindings=(
            gpu_resolution["bindings"] if gpu_resolution is not None else []
        ),
        runtime_visible_devices=env.get("CUDA_VISIBLE_DEVICES", ""),
        gpu_binding_verified=bool(
            gpu_resolution and gpu_resolution["verified"]
        ),
        cuda_device_order=env.get("CUDA_DEVICE_ORDER", ""),
    )
    if _stop_requested(store, args.run_id):
        _mark_stopped(
            store,
            args.run_id,
            exit_code=None,
            progress_text="训练进程启动前收到停止请求",
        )
        return 0
    print(
        f"launching detached training process for {args.run_id}: {argv!r}",
        flush=True,
    )
    with secure_binary_append(log_path) as log_stream:
        log_stream.write(
            (
                f"\n[training-console] run_id={args.run_id}\n"
                f"[training-console] config={run_meta['config_path']}\n"
                f"[training-console] argv={argv!r}\n"
                "[training-console] "
                f"CUDA_DEVICE_ORDER={env.get('CUDA_DEVICE_ORDER', '')}\n"
                "[training-console] requested CUDA_VISIBLE_DEVICES="
                f"{resolved.get('CUDA_VISIBLE_DEVICES', '')}\n"
                "[training-console] runtime CUDA_VISIBLE_DEVICES="
                f"{env.get('CUDA_VISIBLE_DEVICES', '')}\n"
            ).encode("utf-8")
        )
        process = subprocess.Popen(
            argv,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    store.update_run(
        args.run_id,
        status="running",
        training_pid=process.pid,
        training_pgid=process.pid,
        progress_text="训练进程已脱离 Web 服务运行",
    )
    offset = 0
    stage = "训练已启动"
    checkpoint = ""
    progress = ""
    stop_signal_at: float | None = None
    force_stop_sent = False
    while process.poll() is None:
        text, offset = _read_new_text(log_path, offset)
        stage, checkpoint, parsed_progress = _progress_from_text(
            text,
            stage,
            checkpoint,
        )
        if parsed_progress:
            progress = parsed_progress
        if _stop_requested(store, args.run_id):
            if stop_signal_at is None:
                _signal_process_group(process, signal.SIGTERM)
                stop_signal_at = time.monotonic()
                store.update_run(
                    args.run_id,
                    status="stopping",
                    stage="正在停止",
                    latest_checkpoint=checkpoint,
                    progress_text="已向训练进程组发送 SIGTERM",
                )
            elif (
                not force_stop_sent
                and time.monotonic() - stop_signal_at >= STOP_GRACE_SECONDS
            ):
                _signal_process_group(process, signal.SIGKILL)
                force_stop_sent = True
                store.update_run(
                    args.run_id,
                    status="stopping",
                    stage="正在强制停止",
                    latest_checkpoint=checkpoint,
                    progress_text="优雅退出超时，已向训练进程组发送 SIGKILL",
                )
        else:
            store.update_run(
                args.run_id,
                status="running",
                stage=stage,
                latest_checkpoint=checkpoint,
                progress_text=progress or "等待新的训练日志",
            )
        time.sleep(max(0.25, args.poll_seconds))

    text, _ = _read_new_text(log_path, offset)
    stage, checkpoint, parsed_progress = _progress_from_text(
        text,
        stage,
        checkpoint,
    )
    if parsed_progress:
        progress = parsed_progress
    checkpoint = checkpoint or _latest_checkpoint(
        str(run_meta.get("artifact_run_dir", "")),
        repo_root,
    )
    return_code = int(
        process.returncode if process.returncode is not None else 0
    )
    if _stop_requested(store, args.run_id):
        _mark_stopped(
            store,
            args.run_id,
            exit_code=return_code,
            progress_text="训练进程已按用户请求停止",
        )
        print("training process stopped by user request", flush=True)
        return 0
    store.update_run(
        args.run_id,
        status="succeeded" if return_code == 0 else "failed",
        stage=stage,
        latest_checkpoint=checkpoint,
        progress_text=progress or (
            "训练完成" if return_code == 0 else f"训练失败 · exit {return_code}"
        ),
        exit_code=return_code,
        finished_at=utc_now(),
    )
    print(f"training process exited with {return_code}", flush=True)
    return return_code


def main() -> None:
    args = parse_args()
    try:
        return_code = run(args)
    except Exception as exc:
        try:
            StateStore(args.state_root).update_run(
                args.run_id,
                status="failed_to_start",
                progress_text=f"独立运行器失败：{exc}",
                finished_at=utc_now(),
            )
        except Exception:
            pass
        print(f"training runner failed: {exc}", file=sys.stderr, flush=True)
        raise
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
