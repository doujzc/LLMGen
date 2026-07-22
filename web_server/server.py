#!/usr/bin/env python3
"""Serve the local manual-testing UI and JSON inference API."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

from llmgen.router import RouterDataError
from web_server.runtime import RouterRuntime


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ROUTES = {
    "/": STATIC_DIR / "index.html",
    "/static/app.js": STATIC_DIR / "app.js",
    "/static/styles.css": STATIC_DIR / "styles.css",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model-name-or-path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--max-code-paths", type=int, default=8)
    parser.add_argument("--max-input-length", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def handler_class(runtime: RouterRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LLMGenWeb/1.0"

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._json(HTTPStatus.OK, runtime.health())
                return
            if parsed.path == "/api/catalog":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                try:
                    limit = int(params.get("limit", ["100"])[0])
                except ValueError:
                    self._error(HTTPStatus.BAD_REQUEST, "limit must be an integer")
                    return
                self._json(HTTPStatus.OK, runtime.catalog(query, limit))
                return
            if parsed.path == "/api/skill":
                params = parse_qs(parsed.query)
                skill_id = params.get("id", [""])[0]
                if not skill_id:
                    self._error(HTTPStatus.BAD_REQUEST, "id is required")
                    return
                try:
                    detail = runtime.skill_detail(skill_id)
                except RouterDataError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json(HTTPStatus.OK, detail)
                return
            static_path = STATIC_ROUTES.get(parsed.path)
            if static_path is None or not static_path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            body = static_path.read_bytes()
            content_type, _ = mimetypes.guess_type(static_path.name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if urlparse(self.path).path != "/api/infer":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
                return
            if content_length < 1 or content_length > 1_000_000:
                self._error(HTTPStatus.BAD_REQUEST, "invalid request body size")
                return
            try:
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                result = runtime.infer(
                    str(payload.get("query", "")),
                    max_code_paths=int(payload.get("max_code_paths", 4)),
                    top_k=int(payload.get("top_k", 10)),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except RouterDataError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:  # pragma: no cover - hardware/model failures
                print(f"inference failed: {exc}", file=sys.stderr)
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "inference failed")
                return
            self._json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    return Handler


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    print(f"Loading router from {Path(args.model_dir).expanduser()} ...", flush=True)
    runtime = RouterRuntime(
        model_dir=args.model_dir,
        base_model_name_or_path=args.base_model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        max_code_paths=args.max_code_paths,
        max_input_length=args.max_input_length,
        trust_remote_code=args.trust_remote_code,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class(runtime))
    print(
        f"LLMGen manual test UI: http://{args.host}:{args.port} "
        f"({runtime.decode_map['num_skills']} skills)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
