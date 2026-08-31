#!/usr/bin/env python
"""Inspect an installed vLLM's constrained-decoding implementation paths.

This probe is read-only and does not load a model or import torch/torch_npu.  It
finds the installed ``vllm`` package, scans its Python/native source files for
request-level logits processors and nearby alternative constraint mechanisms,
then prints bounded source context and enclosing Python scopes.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import platform
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence


_MARKER = "[[LLMGEN-VLLM-SOURCE-PROBE]]"
_DEFAULT_TERMS = (
    "logits_processors",
    "logits_processor",
    "greedy_token_ids",
    "allowed_token_ids",
    "guided_decoding",
    "structured_output",
    "grammar",
)
_SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".cu",
    ".cuh",
}


def _emit(event: str, **fields: Any) -> None:
    rendered = " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
        for key, value in fields.items()
    )
    print(f"{_MARKER} event={event} {rendered}".rstrip(), flush=True)


def _error_details(exc: BaseException) -> dict[str, str]:
    return {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        help="Installed vllm package directory; auto-detected by default.",
    )
    parser.add_argument(
        "--term",
        action="append",
        dest="terms",
        help="Case-sensitive search term. Repeat to override the defaults.",
    )
    parser.add_argument("--context-lines", type=int, default=3)
    parser.add_argument("--max-matches", type=int, default=500)
    parser.add_argument("--max-file-bytes", type=int, default=8 * 1024 * 1024)
    return parser.parse_args(argv)


def _resolve_package_dir(override: str | None) -> tuple[Path, dict[str, Any]]:
    if override:
        package_dir = Path(override).expanduser().resolve()
        origin = None
    else:
        specification = importlib.util.find_spec("vllm")
        if specification is None:
            raise RuntimeError("cannot locate the installed vllm package")
        origin = specification.origin
        locations = list(specification.submodule_search_locations or ())
        if locations:
            package_dir = Path(locations[0]).resolve()
        elif origin:
            package_dir = Path(origin).resolve().parent
        else:
            raise RuntimeError("the vllm module has no filesystem location")
    if not package_dir.is_dir():
        raise RuntimeError(f"vllm package directory does not exist: {package_dir}")
    return package_dir, {
        "override": override,
        "detected_origin": origin,
        "resolved_package_dir": str(package_dir),
    }


def _scope_ranges(source: str) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    ranges: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        next_parents = parents
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            label = (
                f"class {node.name}"
                if isinstance(node, ast.ClassDef)
                else f"function {node.name}"
            )
            qualified = ".".join((*parents, label))
            end_line = getattr(node, "end_lineno", node.lineno)
            ranges.append((node.lineno, int(end_line), qualified))
            next_parents = (*parents, node.name)
        for child in ast.iter_child_nodes(node):
            visit(child, next_parents)

    visit(tree, ())
    return ranges


def _enclosing_scope(
    ranges: Sequence[tuple[int, int, str]], line_number: int
) -> str | None:
    candidates = [
        item for item in ranges if item[0] <= line_number <= item[1]
    ]
    if not candidates:
        return None
    # The narrowest source range is the innermost class/function.
    return min(candidates, key=lambda item: item[1] - item[0])[2]


def _source_files(package_dir: Path) -> Iterable[Path]:
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES:
            yield path


def _matching_terms(line: str, terms: Sequence[str]) -> list[str]:
    return [term for term in terms if term in line]


def _scan_sources(
    package_dir: Path,
    *,
    terms: Sequence[str],
    context_lines: int,
    max_matches: int,
    max_file_bytes: int,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    term_totals = {term: 0 for term in terms}
    matching_files: set[str] = set()
    scanned_files = 0
    skipped_large_files: list[dict[str, Any]] = []
    read_errors: list[dict[str, str]] = []
    truncated = False

    for path in _source_files(package_dir):
        try:
            size = path.stat().st_size
        except OSError as exc:
            read_errors.append({"path": str(path), "error": str(exc)})
            continue
        if size > max_file_bytes:
            skipped_large_files.append({"path": str(path), "size_bytes": size})
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            read_errors.append({"path": str(path), "error": str(exc)})
            continue
        scanned_files += 1
        lines = source.splitlines()
        line_hits = [
            (line_number, _matching_terms(line, terms))
            for line_number, line in enumerate(lines, start=1)
            if _matching_terms(line, terms)
        ]
        if not line_hits:
            continue
        relative_path = str(path.relative_to(package_dir))
        matching_files.add(relative_path)
        ranges = _scope_ranges(source) if path.suffix == ".py" else []
        for line_number, hit_terms in line_hits:
            for term in hit_terms:
                term_totals[term] += 1
            if len(matches) >= max_matches:
                truncated = True
                continue
            start = max(1, line_number - context_lines)
            end = min(len(lines), line_number + context_lines)
            matches.append(
                {
                    "path": relative_path,
                    "line": line_number,
                    "terms": hit_terms,
                    "scope": _enclosing_scope(ranges, line_number),
                    "context_start_line": start,
                    "context": [
                        f"{number}: {lines[number - 1]}"
                        for number in range(start, end + 1)
                    ],
                }
            )

    return {
        "terms": list(terms),
        "term_totals": term_totals,
        "scanned_files": scanned_files,
        "matching_file_count": len(matching_files),
        "matching_files": sorted(matching_files),
        "match_count_returned": len(matches),
        "truncated": truncated,
        "matches": matches,
        "skipped_large_files": skipped_large_files,
        "read_errors": read_errors,
    }


def _classify_paths(scan: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize likely API, transfer, scheduler, and sampler occurrences."""

    buckets: dict[str, list[dict[str, Any]]] = {
        "sampling_params_definition": [],
        "request_ingress_or_copy": [],
        "scheduler_or_sequence": [],
        "sampler_or_model_runner": [],
        "api_server": [],
        "tests_or_examples": [],
        "other": [],
    }
    for match in scan.get("matches", []):
        path = str(match.get("path", "")).lower()
        scope = str(match.get("scope", "")).lower()
        if "test" in path or "example" in path:
            bucket = "tests_or_examples"
        elif "sampling_params" in path:
            bucket = "sampling_params_definition"
        elif "entrypoint" in path or "api_server" in path:
            bucket = "api_server"
        elif any(name in path for name in ("sampler", "model_runner", "sampling_metadata")):
            bucket = "sampler_or_model_runner"
        elif any(name in path for name in ("scheduler", "sequence")):
            bucket = "scheduler_or_sequence"
        elif any(
            name in path or name in scope
            for name in ("engine", "request", "protocol", "input")
        ):
            bucket = "request_ingress_or_copy"
        else:
            bucket = "other"
        buckets[bucket].append(
            {
                "path": match.get("path"),
                "line": match.get("line"),
                "terms": match.get("terms"),
                "scope": match.get("scope"),
            }
        )
    return {
        name: {"count": len(items), "locations": items}
        for name, items in buckets.items()
    }


def run(argv: Sequence[str] | None = None) -> tuple[dict[str, Any], int]:
    args = _parse_args(argv)
    if args.context_lines < 0:
        raise ValueError("--context-lines must be non-negative")
    if args.max_matches < 1 or args.max_file_bytes < 1:
        raise ValueError("--max-matches and --max-file-bytes must be positive")
    terms = tuple(dict.fromkeys(args.terms or _DEFAULT_TERMS))
    package_dir, discovery = _resolve_package_dir(args.package_dir)
    _emit("scan.begin", package_dir=str(package_dir), terms=terms)
    scan = _scan_sources(
        package_dir,
        terms=terms,
        context_lines=args.context_lines,
        max_matches=args.max_matches,
        max_file_bytes=args.max_file_bytes,
    )
    report = {
        "probe_version": 1,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "discovery": discovery,
        "scan": scan,
        "classification": _classify_paths(scan),
    }
    _emit(
        "scan.complete",
        scanned_files=scan["scanned_files"],
        matching_files=scan["matching_file_count"],
        matches=scan["match_count_returned"],
        truncated=scan["truncated"],
        term_totals=scan["term_totals"],
    )
    return report, 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report, exit_code = run(argv)
    except Exception as exc:
        report = {"probe_version": 1, "fatal_error": _error_details(exc)}
        exit_code = 1
        _emit("scan.failed", error_type=type(exc).__name__, message=str(exc))
    print(
        f"{_MARKER} FINAL_JSON_BEGIN\n"
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n"
        f"{_MARKER} FINAL_JSON_END",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
