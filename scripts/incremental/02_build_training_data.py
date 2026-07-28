#!/usr/bin/env python3
"""Build one memorization row and about ten direct queries for a new skill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmgen.clawhub_alignment import generate_alignment_query_rows
from llmgen.clawhub_dataset import (
    DOMAINS,
    ROLES,
    ChatBatchClient,
    load_api_config,
)
from llmgen.incremental import (
    load_candidate_state,
    write_incremental_training_data,
)
from llmgen.router import RouterDataError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-state-dir", type=Path, required=True)
    parser.add_argument(
        "--skill-id",
        help="Defaults to the single skill in the state's latest add operation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--queries-txt",
        type=Path,
        help="Optional UTF-8 file with one manually written query per non-empty line.",
    )
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--api-config", type=Path, default=Path("~/llm_api.txt"))
    parser.add_argument("--model", default="Qwen3.6-Plus")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve_skill_id(
    payload: dict,
    requested: str | None,
) -> str:
    if requested:
        return requested
    state = payload.get("incremental_state")
    operation = state.get("last_operation") if isinstance(state, dict) else None
    skill_ids = operation.get("skill_ids") if isinstance(operation, dict) else None
    if (
        isinstance(operation, dict)
        and operation.get("type") == "add"
        and isinstance(skill_ids, list)
        and len(skill_ids) == 1
    ):
        return str(skill_ids[0])
    raise RouterDataError(
        "--skill-id is required unless the latest candidate-state operation adds one skill"
    )


def _manual_queries(path: Path) -> list[str]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise RouterDataError(f"query TXT does not exist: {source}")
    queries = [
        line.strip()
        for line in source.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not queries:
        raise RouterDataError("query TXT contains no non-empty lines")
    return queries


def _profile(skill_id: str, metadata: dict) -> dict:
    raw_roles = metadata.get("roles")
    roles = (
        [str(value) for value in raw_roles if str(value) in ROLES]
        if isinstance(raw_roles, list)
        else []
    )
    raw_domain = str(metadata.get("domain") or "")
    capability = str(
        metadata.get("capability_zh")
        or metadata.get("description")
        or metadata.get("summary")
        or metadata.get("name")
        or skill_id
    ).strip()
    return {
        **metadata,
        "skill_id": skill_id,
        "capability_zh": capability[:500],
        "domain": raw_domain if raw_domain in DOMAINS else "agent_system_automation",
        "roles": roles[:3] or ["act"],
        "mobile_fit": str(metadata.get("mobile_fit") or "medium"),
        "unsafe_action": bool(metadata.get("unsafe_action", False)),
    }


def main() -> None:
    args = parse_args()
    if args.num_queries < 1:
        raise RouterDataError("--num-queries must be positive")
    payload, decode_path, _ = load_candidate_state(args.candidate_state_dir)
    skill_id = _resolve_skill_id(payload, args.skill_id)
    if skill_id not in payload["skills"]:
        raise RouterDataError(f"candidate state has no skill {skill_id!r}")

    if args.queries_txt is not None:
        queries = _manual_queries(args.queries_txt)
        provenance = {
            "provider": "manual_txt",
            "path": str(args.queries_txt.expanduser().resolve()),
        }
    else:
        config = load_api_config(args.api_config, model=args.model)
        client = ChatBatchClient(
            config,
            workers=1,
            temperature=0.55,
        )
        rows = generate_alignment_query_rows(
            [_profile(skill_id, payload["skills"][skill_id])],
            client,
            variants=args.num_queries,
        )
        queries = [str(row["query"]) for row in rows]
        provenance = {
            "provider": "openai_chat",
            "model": config.model,
            "base_url": config.base_url,
            "usage": client.usage_dict(),
        }
    result = write_incremental_training_data(
        args.output_dir,
        decode_map=payload,
        decode_map_path=decode_path,
        skill_id=skill_id,
        queries=queries,
        query_provenance=provenance,
        overwrite=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
