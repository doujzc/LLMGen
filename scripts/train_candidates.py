#!/usr/bin/env python3
"""CLI boundary for the generic candidate-to-router pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llmgen.pipeline.config import PipelineConfigError, load_pipeline_config
from llmgen.pipeline.artifacts import ArtifactError
from llmgen.pipeline.runner import (
    PipelineRunner,
    PipelineRunnerError,
    create_pipeline_run,
)
from llmgen.pipeline.state import PipelineStateError


def _add_range_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from", dest="from_stage", help="first stage to execute")
    parser.add_argument("--to", dest="to_stage", help="last stage to execute")
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help="invalidate and rerun this stage and descendants; may be repeated",
    )


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="strict YAML value override; may be repeated",
    )


def _load_new_runner(args: argparse.Namespace) -> PipelineRunner:
    config = load_pipeline_config(
        args.config,
        overrides=args.overrides,
        candidates=args.candidates,
        output=args.output,
    )
    return create_pipeline_run(config, repo_root=REPO_ROOT)


def _runner_for_run(args: argparse.Namespace) -> PipelineRunner:
    if args.run_dir:
        if args.overrides:
            raise PipelineRunnerError("--set is only valid while creating a Run or forking")
        ignored = [
            name
            for name in ("candidates", "config", "output")
            if getattr(args, name, None) is not None
        ]
        if ignored:
            raise PipelineRunnerError(
                "an existing --run-dir cannot be combined with "
                + ", ".join(f"--{name}" for name in ignored)
            )
        return PipelineRunner.open(args.run_dir, repo_root=REPO_ROOT)
    if not (args.candidates and args.config and args.output):
        raise PipelineRunnerError(
            "a new run requires --candidates, --config, and --output; "
            "use --run-dir to resume an existing Run"
        )
    return _load_new_runner(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="create or resume and execute a stage range")
    run.add_argument("--run-dir", type=Path, help="existing run directory")
    run.add_argument("--candidates", type=Path, help="candidate JSONL for a new Run")
    run.add_argument("--config", type=Path, help="pipeline YAML for a new Run")
    run.add_argument("--output", type=Path, help="new Run directory")
    _add_range_arguments(run)
    _add_overrides(run)
    run.add_argument("--resume-checkpoint", help="only valid when one stage is selected")
    run.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="allow an explicit or discovered checkpoint without pipeline lineage",
    )
    run.add_argument("--allow-failed-gates", action="store_true", help="set export.allow_failed_gates=true for a new Run")

    stage = commands.add_parser("stage", help="execute exactly one stage")
    stage.add_argument("stage")
    stage.add_argument("--run-dir", type=Path, required=True)
    stage.add_argument("--resume-checkpoint")
    stage.add_argument("--allow-legacy-checkpoint", action="store_true")

    status = commands.add_parser("status", help="show Run and stage state")
    status.add_argument("--run-dir", type=Path, required=True)

    fork = commands.add_parser("fork", help="derive a Run with compatible artifacts copied forward")
    fork.add_argument("--from-run", type=Path, required=True)
    fork.add_argument("--output", type=Path, required=True)
    _add_overrides(fork)
    return parser


def _run_command(args: argparse.Namespace) -> int:
    if args.allow_failed_gates:
        if args.run_dir:
            raise PipelineRunnerError(
                "--allow-failed-gates cannot mutate an existing Run; fork it with "
                "--set export.allow_failed_gates=true"
            )
        args.overrides.append("export.allow_failed_gates=true")
    runner = _runner_for_run(args)
    executions = runner.run(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        force_stage=args.force_stage or None,
        resume_checkpoint=args.resume_checkpoint,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    print(json.dumps([execution.__dict__ for execution in executions], ensure_ascii=False))
    return 0


def _fork_command(args: argparse.Namespace) -> int:
    parent = PipelineRunner.open(args.from_run, repo_root=REPO_ROOT)
    candidate = parent.run_dir / "source" / "candidates.input.jsonl"
    if not candidate.is_file():
        raise PipelineRunnerError(
            "parent Run has no frozen candidate snapshot; complete ingest before forking"
        )
    overrides = list(args.overrides)
    overrides_manual_alignment = any(
        value.split("=", 1)[0].strip()
        == "data_generation.manual_alignment_path"
        for value in overrides
    )
    manual_fingerprint_path = (
        parent.run_dir / "config" / "manual_alignment_input.json"
    )
    if not overrides_manual_alignment and manual_fingerprint_path.is_file():
        manual_fingerprint = json.loads(
            manual_fingerprint_path.read_text(encoding="utf-8")
        )
        if bool(manual_fingerprint.get("enabled")):
            frozen_manual = parent.run_dir / str(
                manual_fingerprint.get("frozen_path")
                or "source/manual_alignment.input.jsonl"
            )
            if not frozen_manual.is_file():
                raise PipelineRunnerError(
                    "parent Run has no frozen manual-alignment snapshot"
                )
            overrides.append(
                "data_generation.manual_alignment_path="
                + json.dumps(str(frozen_manual))
            )
    config = load_pipeline_config(
        parent.run_dir / "config" / "pipeline.resolved.yaml",
        overrides=overrides,
        candidates=candidate,
        output=args.output,
    )
    child = create_pipeline_run(
        config,
        repo_root=REPO_ROOT,
        parent_run_id=str(parent.state.read_run()["run_id"]),
    )
    reused = child.reuse_from(parent)
    print(json.dumps({"run_dir": str(child.run_dir), "reused_artifacts": list(reused)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "stage":
            result = PipelineRunner.open(args.run_dir, repo_root=REPO_ROOT).stage(
                args.stage,
                resume_checkpoint=args.resume_checkpoint,
                allow_legacy_checkpoint=args.allow_legacy_checkpoint,
            )
            print(json.dumps(result.__dict__))
            return 0
        if args.command == "status":
            print(json.dumps(PipelineRunner.open(args.run_dir, repo_root=REPO_ROOT).status(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "fork":
            return _fork_command(args)
        raise AssertionError(f"unhandled command: {args.command}")
    except (
        ArtifactError,
        FileNotFoundError,
        PipelineConfigError,
        PipelineRunnerError,
        PipelineStateError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
